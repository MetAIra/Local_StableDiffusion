"""ControlNet パイプラインモジュール

ポーズ・エッジ・深度マップ・線画・落書き・タイルで構図制御を行う画像生成
複数ControlNet同時使用にも対応
"""
import os
import gc
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.models import AutoencoderKL

from config import (
    MODEL_FILES, DEFAULT_MODEL, VAE_FILES, NEGATIVE_EMBEDDING, DEVICE,
    CONTROLNET_MODELS, CONTROLNET_MODELS_SDXL, PROMPT_PREFIX, PROMPT_PREFIX_SDXL,
    SCHEDULERS, DEFAULT_SCHEDULER, is_sdxl_model, get_model_type,
    clear_stop, is_stop_requested
)
from utils.preprocessors import preprocessors
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.long_prompt import encode_prompt_long, encode_prompt_long_sdxl, needs_long_encoding
from .lora_utils import load_loras_into, unload_loras_from
from .scheduler_factory import create_scheduler


class ControlNetPipelineManager:
    """ControlNetパイプラインを管理するクラス"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.controlnet_models = {}
        self.pipelines = {}
        self.current_model = None
        self.current_vae = None
        self.current_scheduler = None
        self.current_loras = []  # 複数LoRA対応: [(name, weight), ...]
        self.lora_error = None  # LoRAエラーメッセージを保持
        self.device = DEVICE

    def get_scheduler(self, scheduler_name: str, config=None):
        """スケジューラインスタンスを作成"""
        return create_scheduler(scheduler_name, config)

    def _load_controlnet_model(self, controlnet_type: str, model_type: str = "sd15") -> ControlNetModel:
        """ControlNetモデルをロード

        Args:
            controlnet_type: ControlNetの種類（openpose, canny, depth等）
            model_type: モデルタイプ（"sd15" または "sdxl"）
        """
        cache_key = f"{controlnet_type}_{model_type}"

        if cache_key not in self.controlnet_models:
            # モデルタイプに応じたControlNetモデル辞書を選択
            if model_type == "sdxl":
                controlnet_dict = CONTROLNET_MODELS_SDXL
            else:
                controlnet_dict = CONTROLNET_MODELS

            if controlnet_type not in controlnet_dict:
                available = list(controlnet_dict.keys())
                raise ValueError(f"ControlNet '{controlnet_type}' は {model_type} モデルでは利用できません。利用可能: {available}")

            model_info = controlnet_dict[controlnet_type]
            print(f"Loading ControlNet model ({model_type}): {model_info['model_id']}")

            controlnet = ControlNetModel.from_pretrained(
                model_info['model_id'],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            controlnet.to(self.device)
            self.controlnet_models[cache_key] = controlnet
            print(f"ControlNet {controlnet_type} ({model_type}) loaded successfully")

        return self.controlnet_models[cache_key]

    def get_pipeline(
        self,
        controlnet_type: str,
        vae_name: str,
        model_name: str = None,
        scheduler_name: str = None
    ):
        """ControlNetパイプラインを取得"""
        if model_name is None:
            model_name = DEFAULT_MODEL
        if scheduler_name is None:
            scheduler_name = DEFAULT_SCHEDULER

        # モデルタイプを検出
        model_type = get_model_type(model_name)

        cache_key = f"{controlnet_type}_{vae_name}_{model_name}"

        if (cache_key in self.pipelines and
            self.current_vae == vae_name and
            self.current_model == model_name):
            # スケジューラのみ変更の場合
            if self.current_scheduler != scheduler_name:
                pipe = self.pipelines[cache_key]
                pipe.scheduler = self.get_scheduler(scheduler_name, pipe.scheduler.config)
                self.current_scheduler = scheduler_name
                print(f"Scheduler changed to: {scheduler_name}")
            return self.pipelines[cache_key]

        print(f"Creating ControlNet pipeline ({model_type}): {controlnet_type} with Model: {model_name}, VAE: {vae_name}, Scheduler: {scheduler_name}")

        # モデルパスを取得
        model_path = MODEL_FILES.get(model_name)
        if model_path is None:
            raise ValueError(f"Model not found: {model_name}")

        # ControlNetモデルをロード（モデルタイプに応じたControlNetを使用）
        controlnet = self._load_controlnet_model(controlnet_type, model_type)

        # VAEのロード
        vae = None
        if VAE_FILES[vae_name] is not None:
            vae = AutoencoderKL.from_single_file(
                VAE_FILES[vae_name],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        # スケジューラを作成
        scheduler = self.get_scheduler(scheduler_name)

        # パイプラインを作成（モデルタイプに応じて適切なパイプラインクラスを使用）
        if model_type == "sdxl":
            pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                load_safety_checker=False
            )

        if vae is not None:
            pipe.vae = vae

        pipe.scheduler = scheduler
        pipe.to(self.device)

        # EasyNegativeV2をロード（SD1.5のみ）
        if model_type != "sdxl":
            try:
                pipe.load_textual_inversion(
                    pretrained_model_name_or_path=NEGATIVE_EMBEDDING,
                    token='EasyNegative'
                )
                print("EasyNegativeV2 loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load EasyNegativeV2: {e}")

        # メモリ効率化
        if self.device == "cuda":
            pipe.enable_attention_slicing()

        self.pipelines[cache_key] = pipe
        self.current_model = model_name
        self.current_vae = vae_name
        self.current_scheduler = scheduler_name
        self.current_loras = []

        return pipe

    def get_multi_pipeline(
        self,
        controlnet_types: list[str],
        vae_name: str,
        model_name: str = None,
        scheduler_name: str = None
    ):
        """複数ControlNetを使用するパイプラインを取得

        Args:
            controlnet_types: ControlNetの種類のリスト
            vae_name: VAE名
            model_name: モデル名
            scheduler_name: スケジューラ名

        Returns:
            複数ControlNet対応パイプライン
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        if scheduler_name is None:
            scheduler_name = DEFAULT_SCHEDULER

        # モデルタイプを検出
        model_type = get_model_type(model_name)

        # モデルパスを取得
        model_path = MODEL_FILES.get(model_name)
        if model_path is None:
            raise ValueError(f"Model not found: {model_name}")

        # キャッシュキーを生成
        sorted_types = sorted(controlnet_types)
        cache_key = f"multi_{'_'.join(sorted_types)}_{vae_name}_{model_name}"

        if (cache_key in self.pipelines and
            self.current_vae == vae_name and
            self.current_model == model_name):
            # スケジューラのみ変更の場合
            if self.current_scheduler != scheduler_name:
                pipe = self.pipelines[cache_key]
                pipe.scheduler = self.get_scheduler(scheduler_name, pipe.scheduler.config)
                self.current_scheduler = scheduler_name
                print(f"Scheduler changed to: {scheduler_name}")
            return self.pipelines[cache_key]

        print(f"Creating Multi-ControlNet pipeline ({model_type}): {controlnet_types} with Model: {model_name}, VAE: {vae_name}, Scheduler: {scheduler_name}")

        # 複数のControlNetモデルをロード（モデルタイプに応じたControlNetを使用）
        controlnets = []
        for cn_type in controlnet_types:
            controlnet = self._load_controlnet_model(cn_type, model_type)
            controlnets.append(controlnet)

        # VAEのロード
        vae = None
        if VAE_FILES[vae_name] is not None:
            vae = AutoencoderKL.from_single_file(
                VAE_FILES[vae_name],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        # スケジューラを作成
        scheduler = self.get_scheduler(scheduler_name)

        # パイプラインを作成（複数ControlNet、モデルタイプに応じて適切なパイプラインクラスを使用）
        if model_type == "sdxl":
            pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                model_path,
                controlnet=controlnets,  # リストで渡す
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                model_path,
                controlnet=controlnets,  # リストで渡す
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                load_safety_checker=False
            )

        if vae is not None:
            pipe.vae = vae

        pipe.scheduler = scheduler
        pipe.to(self.device)

        # EasyNegativeV2をロード（SD1.5のみ）
        if model_type != "sdxl":
            try:
                pipe.load_textual_inversion(
                    pretrained_model_name_or_path=NEGATIVE_EMBEDDING,
                    token='EasyNegative'
                )
                print("EasyNegativeV2 loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load EasyNegativeV2: {e}")

        # メモリ効率化
        if self.device == "cuda":
            pipe.enable_attention_slicing()

        self.pipelines[cache_key] = pipe
        self.current_model = model_name
        self.current_vae = vae_name
        self.current_scheduler = scheduler_name
        self.current_loras = []

        return pipe

    def get_multi_pipeline_with_loras(
        self,
        controlnet_types: list[str],
        vae_name: str,
        model_name: str = None,
        lora_configs: list = None,
        scheduler_name: str = None
    ) -> StableDiffusionControlNetPipeline:
        """複数LoRA適用済み複数ControlNetパイプラインを取得"""
        pipe = self.get_multi_pipeline(controlnet_types, vae_name, model_name, scheduler_name)
        if lora_configs:
            self.load_loras(pipe, lora_configs)
        return pipe

    def clear_cache(self):
        """GPUメモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def load_loras(self, pipe, lora_configs: list):
        """複数のLoRAをロード（fuse方式で確実に適用）

        Args:
            pipe: パイプライン
            lora_configs: [(lora_name, weight), ...] のリスト
        """
        self.current_loras, self.lora_error = load_loras_into(
            pipe, lora_configs, self.current_loras, log_label=" for ControlNet"
        )

    def unload_all_loras(self, pipe):
        """すべてのLoRAをアンロード"""
        self.current_loras = unload_loras_from(pipe, self.current_loras)

    def get_pipeline_with_loras(
        self,
        controlnet_type: str,
        vae_name: str,
        model_name: str = None,
        lora_configs: list = None,
        scheduler_name: str = None
    ) -> StableDiffusionControlNetPipeline:
        """複数LoRA適用済みControlNetパイプラインを取得"""
        pipe = self.get_pipeline(controlnet_type, vae_name, model_name, scheduler_name)
        if lora_configs:
            self.load_loras(pipe, lora_configs)
        return pipe


# グローバルインスタンス
controlnet_manager = ControlNetPipelineManager()


def generate_with_controlnet(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    controlnet_type: str,
    num_images: int,
    seed: int,
    steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    canny_low: int = 100,
    canny_high: int = 200,
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0
) -> tuple[list[Image.Image], str]:
    """ControlNetを使用して画像を生成

    Args:
        image: 制御画像（プリプロセス前の元画像）
        prompt: プロンプト
        negative_prompt: ネガティブプロンプト
        controlnet_type: ControlNetの種類 ("openpose", "canny", "depth")
        num_images: 生成枚数
        seed: シード値
        steps: ステップ数
        guidance_scale: CFG Scale
        controlnet_scale: ControlNet強度
        vae_name: VAE名
        model_name: モデル名
        scheduler_name: スケジューラ名
        canny_low: Cannyの下限閾値
        canny_high: Cannyの上限閾値
        lora1-3: LoRA名
        weight1-3: LoRA適用強度

    Returns:
        (生成画像リスト, ステータスメッセージ)
    """
    if image is None:
        return [], "画像をアップロードしてください"

    if not prompt.strip():
        return [], "プロンプトを入力してください"

    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    try:
        # パイプラインを取得（複数LoRA適用、スケジューラ指定）
        pipe = controlnet_manager.get_pipeline_with_loras(controlnet_type, vae_name, model_name, lora_configs, scheduler_name)

        # LoRAエラーチェック
        if controlnet_manager.lora_error:
            return [], f"⚠️ {controlnet_manager.lora_error}\nLoRAなしで生成を続行するか、別のLoRAを選択してください。"

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, f"controlnet_{controlnet_type}", prompt)

        # SDXLモデルかどうかを判定
        is_sdxl = is_sdxl_model(model_name)

        # プロンプトを構成（SDXLの場合は別のプレフィックス）
        if is_sdxl:
            full_prompt = PROMPT_PREFIX_SDXL + prompt
            full_negative = negative_prompt
        else:
            full_prompt = PROMPT_PREFIX + prompt
            full_negative = "EasyNegative, " + negative_prompt if negative_prompt else "EasyNegative"

        # 制御画像をプリプロセス
        print(f"Preprocessing image with {controlnet_type}...")
        if controlnet_type == "canny":
            control_image = preprocessors.preprocess(
                image, controlnet_type,
                low_threshold=canny_low,
                high_threshold=canny_high
            )
        else:
            control_image = preprocessors.preprocess(image, controlnet_type)

        # 画像サイズを64の倍数に調整
        width, height = image.size
        width = (width // 64) * 64
        height = (height // 64) * 64
        control_image = control_image.resize((width, height), Image.LANCZOS)

        # 制御画像も保存
        control_filepath = os.path.join(output_dir, f"control_{controlnet_type}.png")
        control_image.save(control_filepath)

        # Long Prompt対応
        use_long = needs_long_encoding(pipe, full_prompt, full_negative)
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if use_long:
            from utils.long_prompt import get_token_count
            print(f"Long Prompt detected: {get_token_count(pipe, full_prompt)} tokens. Using chunked encoding.")
            if is_sdxl:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    encode_prompt_long_sdxl(pipe, full_prompt, full_negative, DEVICE)
                )
            else:
                prompt_embeds, negative_prompt_embeds = (
                    encode_prompt_long(pipe, full_prompt, full_negative, DEVICE)
                )

        results = []
        csv_rows = []  # CSV用のパラメータリスト
        clear_stop()

        for i in range(int(num_images)):
            if is_stop_requested():
                if results:
                    csv_path = save_batch_generation_params(output_dir, csv_rows)
                    return results, f"中止しました（{len(results)}/{int(num_images)}枚生成済み）\nCSV: {csv_path}\n保存先: {output_dir}"
                return [], "生成が中止されました"

            current_seed = int(seed) + i
            generator = torch.Generator(device=DEVICE).manual_seed(current_seed)

            print(f"Generating image {i + 1}/{num_images} with seed {current_seed}...")

            if use_long:
                gen_kwargs = dict(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image=control_image,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    controlnet_conditioning_scale=float(controlnet_scale),
                    generator=generator,
                    width=width,
                    height=height,
                )
                if is_sdxl and pooled_prompt_embeds is not None:
                    gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
                    gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
                output = pipe(**gen_kwargs)
            else:
                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=full_negative,
                    image=control_image,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    controlnet_conditioning_scale=float(controlnet_scale),
                    generator=generator,
                    width=width,
                    height=height,
                )

            result_image = output.images[0]

            # 画像を保存
            filename = f"image_{i+1:03d}_seed{current_seed}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_safe(result_image, filepath)
            results.append(result_image)

            # CSV用のパラメータを記録
            csv_rows.append({
                "filename": filename,
                "seed": current_seed,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": guidance_scale,
                "controlnet_type": controlnet_type,
                "controlnet_scale": controlnet_scale,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "model": model_name,
                "vae": vae_name,
                "scheduler": scheduler_name,
                "lora1": lora1,
                "lora1_weight": weight1,
                "lora2": lora2,
                "lora2_weight": weight2,
                "lora3": lora3,
                "lora3_weight": weight3
            })

            print(f"Image saved to: {filepath}")

            # メモリ解放
            controlnet_manager.clear_cache()
            gc.collect()

        # 生成パラメータをCSVに保存
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        print(f"Generation params saved to: {csv_path}")

        return results, f"{num_images}枚の画像を生成しました（{controlnet_type}）\n保存先: {output_dir}"

    except Exception as e:
        import traceback
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [], error_msg


def get_control_image(
    image: Image.Image,
    controlnet_type: str,
    canny_low: int = 100,
    canny_high: int = 200
) -> Image.Image:
    """制御画像のプレビューを取得

    Args:
        image: 入力画像
        controlnet_type: ControlNetの種類
        canny_low: Cannyの下限閾値
        canny_high: Cannyの上限閾値

    Returns:
        プリプロセス結果の画像
    """
    if image is None:
        return None

    if controlnet_type == "canny":
        return preprocessors.preprocess(
            image, controlnet_type,
            low_threshold=canny_low,
            high_threshold=canny_high
        )
    else:
        return preprocessors.preprocess(image, controlnet_type)


def generate_with_multi_controlnet(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    controlnet_types: list[str],
    controlnet_scales: list[float],
    num_images: int,
    seed: int,
    steps: int,
    guidance_scale: float,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    canny_low: int = 100,
    canny_high: int = 200,
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0
) -> tuple[list[Image.Image], str]:
    """複数ControlNetを使用して画像を生成

    Args:
        image: 制御画像（プリプロセス前の元画像）
        prompt: プロンプト
        negative_prompt: ネガティブプロンプト
        controlnet_types: ControlNetの種類のリスト
        controlnet_scales: 各ControlNetの強度リスト
        num_images: 生成枚数
        seed: シード値
        steps: ステップ数
        guidance_scale: CFG Scale
        vae_name: VAE名
        model_name: モデル名
        scheduler_name: スケジューラ名
        canny_low: Cannyの下限閾値
        canny_high: Cannyの上限閾値
        lora1-3: LoRA名
        weight1-3: LoRA適用強度

    Returns:
        (生成画像リスト, ステータスメッセージ)
    """
    if image is None:
        return [], "画像をアップロードしてください"

    if not prompt.strip():
        return [], "プロンプトを入力してください"

    if not controlnet_types:
        return [], "少なくとも1つのControlNetを選択してください"

    # リストの長さを揃える
    while len(controlnet_scales) < len(controlnet_types):
        controlnet_scales.append(1.0)

    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    try:
        # パイプラインを取得（複数LoRA適用、スケジューラ指定）
        pipe = controlnet_manager.get_multi_pipeline_with_loras(
            controlnet_types, vae_name, model_name, lora_configs, scheduler_name
        )

        # LoRAエラーチェック
        if controlnet_manager.lora_error:
            return [], f"⚠️ {controlnet_manager.lora_error}\nLoRAなしで生成を続行するか、別のLoRAを選択してください。"

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        cn_names = "_".join(controlnet_types)
        output_dir = create_output_dir(base_path, f"controlnet_multi_{cn_names}", prompt)

        # SDXLモデルかどうかを判定
        is_sdxl = is_sdxl_model(model_name)

        # プロンプトを構成（SDXLの場合は別のプレフィックス）
        if is_sdxl:
            full_prompt = PROMPT_PREFIX_SDXL + prompt
            full_negative = negative_prompt
        else:
            full_prompt = PROMPT_PREFIX + prompt
            full_negative = "EasyNegative, " + negative_prompt if negative_prompt else "EasyNegative"

        # 画像サイズを64の倍数に調整
        width, height = image.size
        width = (width // 64) * 64
        height = (height // 64) * 64

        # 各ControlNet用の制御画像をプリプロセス
        control_images = []
        for cn_type in controlnet_types:
            print(f"Preprocessing image with {cn_type}...")
            if cn_type == "canny":
                control_img = preprocessors.preprocess(
                    image, cn_type,
                    low_threshold=canny_low,
                    high_threshold=canny_high
                )
            else:
                control_img = preprocessors.preprocess(image, cn_type)

            control_img = control_img.resize((width, height), Image.LANCZOS)
            control_images.append(control_img)

            # 制御画像を保存
            control_filepath = os.path.join(output_dir, f"control_{cn_type}.png")
            control_img.save(control_filepath)

        # Long Prompt対応
        use_long = needs_long_encoding(pipe, full_prompt, full_negative)
        prompt_embeds_multi = None
        negative_prompt_embeds_multi = None
        pooled_prompt_embeds_multi = None
        negative_pooled_prompt_embeds_multi = None

        if use_long:
            from utils.long_prompt import get_token_count
            print(f"Long Prompt detected: {get_token_count(pipe, full_prompt)} tokens. Using chunked encoding.")
            if is_sdxl:
                prompt_embeds_multi, negative_prompt_embeds_multi, pooled_prompt_embeds_multi, negative_pooled_prompt_embeds_multi = (
                    encode_prompt_long_sdxl(pipe, full_prompt, full_negative, DEVICE)
                )
            else:
                prompt_embeds_multi, negative_prompt_embeds_multi = (
                    encode_prompt_long(pipe, full_prompt, full_negative, DEVICE)
                )

        results = []
        csv_rows = []  # CSV用のパラメータリスト
        clear_stop()

        for i in range(int(num_images)):
            if is_stop_requested():
                if results:
                    csv_path = save_batch_generation_params(output_dir, csv_rows)
                    return results, f"中止しました（{len(results)}/{int(num_images)}枚生成済み）\nCSV: {csv_path}\n保存先: {output_dir}"
                return [], "生成が中止されました"

            current_seed = int(seed) + i
            generator = torch.Generator(device=DEVICE).manual_seed(current_seed)

            print(f"Generating image {i + 1}/{num_images} with seed {current_seed}...")

            if use_long:
                gen_kwargs = dict(
                    prompt_embeds=prompt_embeds_multi,
                    negative_prompt_embeds=negative_prompt_embeds_multi,
                    image=control_images,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    controlnet_conditioning_scale=controlnet_scales[:len(controlnet_types)],
                    generator=generator,
                    width=width,
                    height=height,
                )
                if is_sdxl and pooled_prompt_embeds_multi is not None:
                    gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds_multi
                    gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds_multi
                output = pipe(**gen_kwargs)
            else:
                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=full_negative,
                    image=control_images,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    controlnet_conditioning_scale=controlnet_scales[:len(controlnet_types)],
                    generator=generator,
                    width=width,
                    height=height,
                )

            result_image = output.images[0]

            # 画像を保存
            filename = f"image_{i+1:03d}_seed{current_seed}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_safe(result_image, filepath)
            results.append(result_image)

            # CSV用のパラメータを記録
            cn_types_str = ",".join(controlnet_types)
            cn_scales_str = ",".join([str(s) for s in controlnet_scales[:len(controlnet_types)]])
            csv_rows.append({
                "filename": filename,
                "seed": current_seed,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": guidance_scale,
                "controlnet_types": cn_types_str,
                "controlnet_scales": cn_scales_str,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "model": model_name,
                "vae": vae_name,
                "scheduler": scheduler_name,
                "lora1": lora1,
                "lora1_weight": weight1,
                "lora2": lora2,
                "lora2_weight": weight2,
                "lora3": lora3,
                "lora3_weight": weight3
            })

            print(f"Image saved to: {filepath}")

            # メモリ解放
            controlnet_manager.clear_cache()
            gc.collect()

        # 生成パラメータをCSVに保存
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        print(f"Generation params saved to: {csv_path}")

        cn_desc = " + ".join(controlnet_types)
        return results, f"{num_images}枚の画像を生成しました（{cn_desc}）\n保存先: {output_dir}"

    except Exception as e:
        import traceback
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [], error_msg
