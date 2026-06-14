"""IP-Adapter パイプラインモジュール

参考画像から顔やスタイルを保持しながら新しい画像を生成
"""
import os
import gc
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models import AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from config import (
    MODEL_FILES, DEFAULT_MODEL, VAE_FILES, NEGATIVE_EMBEDDING, DEVICE,
    PROMPT_PREFIX, PROMPT_PREFIX_SDXL, LORA_FILES, SCHEDULERS, DEFAULT_SCHEDULER,
    IP_ADAPTER_REPO, IP_ADAPTER_MODELS_SD15, IP_ADAPTER_MODELS_SDXL,
    get_model_type,
    clear_stop, is_stop_requested
)
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.long_prompt import encode_prompt_long, encode_prompt_long_sdxl, needs_long_encoding
from .scheduler_factory import create_scheduler


class IPAdapterPipelineManager:
    """IP-Adapterパイプラインを管理するシングルトンクラス"""

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

        self.pipe = None
        self.current_model = None
        self.current_vae = None
        self.current_scheduler = None
        self.current_model_type = None
        self.current_ip_adapter = None
        self.current_loras = []
        self.lora_error = None
        self.device = DEVICE

    def get_scheduler(self, scheduler_name: str, config=None):
        """スケジューラインスタンスを作成"""
        return create_scheduler(scheduler_name, config)

    def clear_cache(self):
        """GPUメモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def get_ip_adapter_models(self, model_type: str) -> dict:
        """モデルタイプに応じたIP-Adapterモデル一覧を取得"""
        if model_type == "sdxl":
            return IP_ADAPTER_MODELS_SDXL
        return IP_ADAPTER_MODELS_SD15

    def load_pipeline(
        self,
        vae_name: str,
        model_name: str = None,
        scheduler_name: str = None,
        ip_adapter_name: str = None
    ):
        """IP-Adapter付きパイプラインをロード"""
        if model_name is None:
            model_name = DEFAULT_MODEL
        if scheduler_name is None:
            scheduler_name = DEFAULT_SCHEDULER

        model_type = get_model_type(model_name)
        ip_adapter_models = self.get_ip_adapter_models(model_type)

        # デフォルトのIP-Adapterを設定
        if ip_adapter_name is None:
            if model_type == "sdxl":
                ip_adapter_name = "ip-adapter-plus-face_sdxl_vit-h"
            else:
                ip_adapter_name = "ip-adapter-plus-face_sd15"

        # キャッシュチェック
        if (self.pipe is not None and
            self.current_model == model_name and
            self.current_vae == vae_name and
            self.current_ip_adapter == ip_adapter_name):
            # スケジューラのみ変更
            if self.current_scheduler != scheduler_name:
                self.pipe.scheduler = self.get_scheduler(scheduler_name, self.pipe.scheduler.config)
                self.current_scheduler = scheduler_name
            return self.pipe

        print(f"Loading IP-Adapter pipeline: Model={model_name}, VAE={vae_name}, IP-Adapter={ip_adapter_name}")

        # モデルパスを取得
        model_path = MODEL_FILES.get(model_name)
        if model_path is None:
            raise ValueError(f"Model not found: {model_name}")

        # 既存パイプラインをクリア
        if self.pipe is not None:
            self.unload_all_loras()
            del self.pipe
            self.clear_cache()

        # VAEのロード
        vae = None
        if VAE_FILES.get(vae_name) is not None:
            vae = AutoencoderKL.from_single_file(
                VAE_FILES[vae_name],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        # スケジューラを作成
        scheduler = self.get_scheduler(scheduler_name)

        # パイプラインをロード
        if model_type == "sdxl":
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                use_safetensors=True
            )
        else:
            self.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                use_safetensors=True
            )

        if vae is not None:
            self.pipe.vae = vae

        self.pipe.scheduler = scheduler
        self.pipe.to(self.device)

        # EasyNegativeV2をロード（SD1.5のみ）
        if model_type != "sdxl":
            try:
                self.pipe.load_textual_inversion(
                    pretrained_model_name_or_path=NEGATIVE_EMBEDDING,
                    token='EasyNegative'
                )
                print("EasyNegativeV2 loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load EasyNegativeV2: {e}")

        # IP-Adapterをロード（attention slicingより先に実行する必要あり）
        ip_adapter_config = ip_adapter_models.get(ip_adapter_name)
        if ip_adapter_config:
            try:
                print(f"Loading IP-Adapter: {ip_adapter_name}")
                print(f"  Repo: {IP_ADAPTER_REPO}")
                print(f"  Subfolder: {ip_adapter_config['subfolder']}")
                print(f"  Weight: {ip_adapter_config['weight_name']}")

                # IP-Adapter用のImage EncoderとFeature Extractorをロード
                # from_single_fileで読み込んだモデルにはこれらがないため別途ロードが必要
                #
                # エンコーダーの選択:
                # - SD1.5: models/image_encoder (ViT-H/14, 1280次元)
                # - SDXL base: sdxl_models/image_encoder (ViT-bigG, 1664次元)
                # - SDXL vit-h variants: models/image_encoder (ViT-H/14, 1280次元)
                #
                if model_type == "sdxl":
                    # SDXL: vit-h バリアントはSD1.5と同じimage encoderを使用
                    if "vit-h" in ip_adapter_name:
                        image_encoder_subfolder = "models/image_encoder"
                    else:
                        image_encoder_subfolder = "sdxl_models/image_encoder"
                else:
                    image_encoder_subfolder = "models/image_encoder"

                image_encoder_path = "h94/IP-Adapter"

                print(f"  Loading image encoder from: {image_encoder_path}/{image_encoder_subfolder}")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    image_encoder_path,
                    subfolder=image_encoder_subfolder,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

                feature_extractor = CLIPImageProcessor()

                # パイプラインにimage_encoderとfeature_extractorを設定
                self.pipe.image_encoder = image_encoder
                self.pipe.feature_extractor = feature_extractor

                # IP-Adapter weightsをロード
                self.pipe.load_ip_adapter(
                    IP_ADAPTER_REPO,
                    subfolder=ip_adapter_config["subfolder"],
                    weight_name=ip_adapter_config["weight_name"],
                    low_cpu_mem_usage=True
                )
                print(f"IP-Adapter loaded successfully: {ip_adapter_name}")
            except Exception as e:
                import traceback
                print(f"Error loading IP-Adapter: {e}")
                print(traceback.format_exc())
                # パイプラインをクリーンアップ
                del self.pipe
                self.pipe = None
                self.clear_cache()
                raise RuntimeError(f"IP-Adapterのロードに失敗しました: {str(e)}")

        # 注意: IP-Adapter使用時はenable_attention_slicing()を使用しない
        # attention slicingはIP-Adapterのattention processorsと互換性がない
        # if self.device == "cuda":
        #     self.pipe.enable_attention_slicing()

        self.current_model = model_name
        self.current_vae = vae_name
        self.current_scheduler = scheduler_name
        self.current_model_type = model_type
        self.current_ip_adapter = ip_adapter_name
        self.current_loras = []

        return self.pipe

    def load_loras(self, lora_configs: list):
        """複数のLoRAをロード"""
        self.lora_error = None

        if self.pipe is None:
            return

        valid_loras = [
            (name, weight) for name, weight in lora_configs
            if name and name != "なし" and LORA_FILES.get(name)
        ]

        if valid_loras == self.current_loras:
            return

        self.unload_all_loras()

        if not valid_loras:
            return

        adapter_names = []
        adapter_weights = []
        errors = []

        for lora_name, lora_weight in valid_loras:
            lora_path = LORA_FILES[lora_name]
            try:
                print(f"Loading LoRA: {lora_name} (weight: {lora_weight})")
                self.pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                adapter_names.append(lora_name)
                adapter_weights.append(lora_weight)
            except Exception as e:
                error_msg = str(e)
                print(f"Error loading LoRA {lora_name}: {error_msg}")
                if "Target modules" in error_msg and "not found" in error_msg:
                    errors.append(f"LoRA '{lora_name}' はこのモデルと互換性がありません")
                else:
                    errors.append(f"LoRA '{lora_name}' のロード失敗: {error_msg[:50]}")

        if adapter_names:
            try:
                self.pipe.set_adapters(adapter_names, adapter_weights)
                self.pipe.fuse_lora(adapter_names=adapter_names)
                self.current_loras = list(zip(adapter_names, adapter_weights))
                print(f"LoRAs fused: {adapter_names}")
            except Exception as e:
                print(f"Error fusing LoRAs: {e}")
                errors.append(f"LoRA融合エラー: {str(e)[:50]}")
                self.current_loras = []

        if errors:
            self.lora_error = "\n".join(errors)

    def unload_all_loras(self):
        """すべてのLoRAをアンロード"""
        if self.pipe is not None and self.current_loras:
            try:
                try:
                    self.pipe.unfuse_lora()
                except Exception:
                    pass
                self.pipe.unload_lora_weights()
                self.current_loras = []
            except Exception as e:
                print(f"Error unloading LoRAs: {e}")

    def get_pipeline_with_loras(
        self,
        vae_name: str,
        model_name: str = None,
        lora_configs: list = None,
        scheduler_name: str = None,
        ip_adapter_name: str = None
    ):
        """LoRA適用済みパイプラインを取得"""
        pipe = self.load_pipeline(vae_name, model_name, scheduler_name, ip_adapter_name)
        if lora_configs:
            self.load_loras(lora_configs)
        return pipe


# グローバルインスタンス
ip_adapter_manager = IPAdapterPipelineManager()


def generate_with_ip_adapter(
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    ip_adapter_scale: float,
    ip_adapter_name: str,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0
) -> tuple[list[Image.Image], str]:
    """IP-Adapterを使用して画像を生成

    Args:
        reference_image: 参考画像（顔/スタイルを抽出）
        prompt: プロンプト
        negative_prompt: ネガティブプロンプト
        num_images: 生成枚数
        seed: シード値
        width: 出力幅
        height: 出力高さ
        steps: ステップ数
        guidance_scale: CFG Scale
        ip_adapter_scale: IP-Adapter強度 (0-1)
        ip_adapter_name: IP-Adapterモデル名
        vae_name: VAE名
        model_name: ベースモデル名
        scheduler_name: スケジューラ名
        lora1-3: LoRA名
        weight1-3: LoRA適用強度

    Returns:
        (生成画像リスト, ステータスメッセージ)
    """
    if reference_image is None:
        return [], "参考画像をアップロードしてください"

    if not prompt.strip():
        return [], "プロンプトを入力してください"

    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    try:
        # パイプラインを取得
        pipe = ip_adapter_manager.get_pipeline_with_loras(
            vae_name, model_name, lora_configs, scheduler_name, ip_adapter_name
        )

        # LoRAエラーチェック
        if ip_adapter_manager.lora_error:
            return [], f"⚠️ {ip_adapter_manager.lora_error}\nLoRAなしで生成を続行するか、別のLoRAを選択してください。"

        # IP-Adapterスケールを設定
        pipe.set_ip_adapter_scale(float(ip_adapter_scale))

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "ip_adapter", prompt)

        # 参考画像を保存
        ref_path = os.path.join(output_dir, "reference.png")
        reference_image.save(ref_path)

        # プロンプトを構成（SDXLの場合は別のプレフィックス）
        model_type = get_model_type(model_name)
        is_sdxl = model_type == "sdxl"
        if model_type == "sdxl":
            full_prompt = PROMPT_PREFIX_SDXL + prompt
            full_negative = negative_prompt
        else:
            full_prompt = PROMPT_PREFIX + prompt
            full_negative = "EasyNegative, " + negative_prompt if negative_prompt else "EasyNegative"

        # IP-Adapter用の画像エンベディングを事前計算
        print("Computing IP-Adapter image embeddings...")
        print(f"  Reference image size: {reference_image.size}")
        print(f"  Pipe has image_encoder: {hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None}")
        print(f"  Pipe has feature_extractor: {hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is not None}")

        # UNetのattention processorsを確認
        attn_procs = pipe.unet.attn_processors
        print(f"  Number of attention processors: {len(attn_procs)}")
        # Attention processorの種類を確認
        proc_types = set(type(p).__name__ for p in attn_procs.values())
        print(f"  Attention processor types: {proc_types}")
        # IP-Adapter用のprocessorがあるか確認
        ip_adapter_procs = [k for k, v in attn_procs.items() if 'IPAdapter' in type(v).__name__]
        print(f"  IP-Adapter processors found: {len(ip_adapter_procs)}")

        ip_adapter_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=reference_image,
            ip_adapter_image_embeds=None,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0
        )

        print(f"  IP-Adapter embeds type: {type(ip_adapter_image_embeds)}")
        if isinstance(ip_adapter_image_embeds, list):
            print(f"  IP-Adapter embeds length: {len(ip_adapter_image_embeds)}")
            for idx, emb in enumerate(ip_adapter_image_embeds):
                if hasattr(emb, 'shape'):
                    print(f"    Embed {idx} shape: {emb.shape}")
                else:
                    print(f"    Embed {idx} type: {type(emb)}")

        # Long Prompt対応
        use_long = needs_long_encoding(pipe, full_prompt, full_negative)
        prompt_embeds_ip = None
        negative_prompt_embeds_ip = None
        pooled_prompt_embeds_ip = None
        negative_pooled_prompt_embeds_ip = None

        if use_long:
            from utils.long_prompt import get_token_count
            print(f"Long Prompt detected: {get_token_count(pipe, full_prompt)} tokens. Using chunked encoding.")
            if is_sdxl:
                prompt_embeds_ip, negative_prompt_embeds_ip, pooled_prompt_embeds_ip, negative_pooled_prompt_embeds_ip = (
                    encode_prompt_long_sdxl(pipe, full_prompt, full_negative, DEVICE)
                )
            else:
                prompt_embeds_ip, negative_prompt_embeds_ip = (
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
                    prompt_embeds=prompt_embeds_ip,
                    negative_prompt_embeds=negative_prompt_embeds_ip,
                    ip_adapter_image_embeds=ip_adapter_image_embeds,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    width=int(width),
                    height=int(height),
                )
                if is_sdxl and pooled_prompt_embeds_ip is not None:
                    gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds_ip
                    gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds_ip
                output = pipe(**gen_kwargs)
            else:
                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=full_negative,
                    ip_adapter_image_embeds=ip_adapter_image_embeds,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    width=int(width),
                    height=int(height),
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
                "ip_adapter_name": ip_adapter_name,
                "ip_adapter_scale": ip_adapter_scale,
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
            ip_adapter_manager.clear_cache()
            gc.collect()

        # 生成パラメータをCSVに保存
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        print(f"Generation params saved to: {csv_path}")

        return results, f"{num_images}枚の画像を生成しました（IP-Adapter: {ip_adapter_name}）\n保存先: {output_dir}"

    except Exception as e:
        import traceback
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [], error_msg


def get_ip_adapter_choices(model_name: str) -> list[str]:
    """モデルタイプに応じたIP-Adapterの選択肢を取得"""
    model_type = get_model_type(model_name)
    models = ip_adapter_manager.get_ip_adapter_models(model_type)
    return list(models.keys())


def get_ip_adapter_description(ip_adapter_name: str, model_name: str) -> str:
    """IP-Adapterの説明を取得"""
    model_type = get_model_type(model_name)
    models = ip_adapter_manager.get_ip_adapter_models(model_type)
    if ip_adapter_name in models:
        return models[ip_adapter_name]["description"]
    return ""
