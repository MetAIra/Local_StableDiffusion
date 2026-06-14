"""パイプライン管理モジュール"""
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

from config import (
    MODEL_FILES, DEFAULT_MODEL, VAE_FILES, NEGATIVE_EMBEDDING, DEVICE,
    SCHEDULERS, DEFAULT_SCHEDULER, get_model_type
)
from .lora_utils import load_loras_into, unload_loras_from
from .scheduler_factory import create_scheduler


class PipelineManager:
    """Stable Diffusionパイプラインを管理するシングルトンクラス"""

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
        self.pipe_img2img = None
        self.pipe_inpaint = None
        self.current_model = None
        self.current_vae = None
        self.current_scheduler = None
        self.current_model_type = None  # "sd15" or "sdxl"
        self.current_loras = []  # 複数LoRA対応: [(name, weight), ...]
        self.lora_error = None  # LoRAエラーメッセージを保持
        self.device = DEVICE

    def get_scheduler(self, scheduler_name: str, config=None):
        """スケジューラインスタンスを作成"""
        return create_scheduler(scheduler_name, config)

    def set_scheduler(self, scheduler_name: str):
        """現在のパイプラインのスケジューラを変更"""
        if self.pipe is None:
            return

        scheduler = self.get_scheduler(scheduler_name, self.pipe.scheduler.config)
        self.pipe.scheduler = scheduler

        if self.pipe_img2img is not None:
            self.pipe_img2img.scheduler = scheduler

        if self.pipe_inpaint is not None:
            self.pipe_inpaint.scheduler = scheduler

        self.current_scheduler = scheduler_name
        print(f"Scheduler changed to: {scheduler_name}")

    def load_pipeline(self, vae_name: str, model_name: str = None, scheduler_name: str = None):
        """パイプラインをロード（VAE・モデル・スケジューラ選択対応）"""
        if model_name is None:
            model_name = DEFAULT_MODEL

        if scheduler_name is None:
            scheduler_name = DEFAULT_SCHEDULER

        # モデルタイプを判定
        model_type = get_model_type(model_name)

        # モデルまたはVAEが変わった場合のみ再ロード
        if (self.pipe is not None and
            self.current_vae == vae_name and
            self.current_model == model_name):
            # スケジューラのみ変更の場合
            if self.current_scheduler != scheduler_name:
                self.set_scheduler(scheduler_name)
            return self.pipe, self.pipe_img2img, self.pipe_inpaint

        print(f"Loading pipeline with Model: {model_name} ({model_type}), VAE: {vae_name}, Scheduler: {scheduler_name}")

        # モデルパスを取得
        model_path = MODEL_FILES.get(model_name)
        if model_path is None:
            raise ValueError(f"Model not found: {model_name}")

        # VAEのロード
        vae = None
        if VAE_FILES[vae_name] is not None:
            vae = AutoencoderKL.from_single_file(VAE_FILES[vae_name])

        # スケジューラを作成
        scheduler = self.get_scheduler(scheduler_name)

        # 既存パイプラインをクリア
        if self.pipe is not None:
            self.unload_all_loras()
            del self.pipe
            del self.pipe_img2img
            del self.pipe_inpaint
            self.pipe = None
            self.pipe_img2img = None
            self.pipe_inpaint = None
            self.current_model = None
            self.current_vae = None
            self.current_model_type = None
            self.clear_cache()

        # パイプラインのロード
        try:
            # Fluxモデルは未サポート
            if model_type == "flux":
                raise ValueError(
                    f"Fluxモデル '{model_name}' はサポートされていません。"
                    "FluxはStable Diffusionとは異なるアーキテクチャを使用しており、"
                    "別のパイプライン（FluxPipeline）が必要です。"
                )

            if model_type == "sdxl":
                # SDXL用パイプラインをロード
                self.pipe = download_from_original_stable_diffusion_ckpt(
                    checkpoint_path_or_dict=model_path,
                    from_safetensors=True,
                    vae=vae,
                    local_files_only=False,
                    device=self.device,
                    load_safety_checker=False,
                    pipeline_class=None,  # 自動判別
                    model_type="SDXL"
                )
            else:
                # SD1.5用パイプラインをロード
                self.pipe = download_from_original_stable_diffusion_ckpt(
                    checkpoint_path_or_dict=model_path,
                    from_safetensors=True,
                    vae=vae,
                    local_files_only=False,
                    device=self.device,
                    load_safety_checker=False
                )

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

            # メモリ効率化
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()

            # img2imgパイプラインを作成
            if model_type == "sdxl":
                self.pipe_img2img = StableDiffusionXLImg2ImgPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                )
            else:
                self.pipe_img2img = StableDiffusionImg2ImgPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False
                )
            self.pipe_img2img.to(self.device)

            if self.device == "cuda":
                self.pipe_img2img.enable_attention_slicing()

            print("img2img pipeline created successfully")

            # Inpaintパイプラインを作成
            if model_type == "sdxl":
                self.pipe_inpaint = StableDiffusionXLInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                )
            else:
                self.pipe_inpaint = StableDiffusionInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False
                )
            self.pipe_inpaint.to(self.device)

            if self.device == "cuda":
                self.pipe_inpaint.enable_attention_slicing()

            print("inpaint pipeline created successfully")

            self.current_model = model_name
            self.current_vae = vae_name
            self.current_scheduler = scheduler_name
            self.current_model_type = model_type
            self.current_loras = []  # LoRAをリセット

        except Exception as e:
            # ロード失敗時は状態をクリーンアップ
            print(f"Error loading model {model_name}: {e}")
            if self.pipe is not None:
                del self.pipe
            if self.pipe_img2img is not None:
                del self.pipe_img2img
            if self.pipe_inpaint is not None:
                del self.pipe_inpaint
            self.pipe = None
            self.pipe_img2img = None
            self.pipe_inpaint = None
            self.current_model = None
            self.current_vae = None
            self.current_model_type = None
            self.current_loras = []
            self.clear_cache()
            raise  # エラーを再送出

        return self.pipe, self.pipe_img2img, self.pipe_inpaint

    def get_img2img_pipeline(self, vae_name: str, model_name: str = None, scheduler_name: str = None):
        """img2imgパイプラインを取得"""
        self.load_pipeline(vae_name, model_name, scheduler_name)
        return self.pipe_img2img

    def clear_cache(self):
        """GPUメモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def load_loras(self, lora_configs: list):
        """複数のLoRAをロード（fuse方式で確実に適用）

        Args:
            lora_configs: [(lora_name, weight), ...] のリスト
        """
        self.lora_error = None  # エラーをリセット

        if self.pipe is None:
            print("Warning: Pipeline not loaded, cannot load LoRA")
            return

        self.current_loras, self.lora_error = load_loras_into(
            self.pipe, lora_configs, self.current_loras
        )

    def unload_all_loras(self):
        """すべてのLoRAをアンロード"""
        if self.pipe is not None:
            self.current_loras = unload_loras_from(self.pipe, self.current_loras)

    def get_txt2img_pipeline_with_loras(self, vae_name: str, model_name: str = None, lora_configs: list = None, scheduler_name: str = None):
        """複数LoRA適用済みtxt2imgパイプラインを取得

        Args:
            vae_name: VAE名
            model_name: モデル名
            lora_configs: [(lora_name, weight), ...] のリスト
            scheduler_name: スケジューラ名
        """
        self.load_pipeline(vae_name, model_name, scheduler_name)
        if lora_configs:
            self.load_loras(lora_configs)
        return self.pipe

    def get_img2img_pipeline_with_loras(self, vae_name: str, model_name: str = None, lora_configs: list = None, scheduler_name: str = None):
        """複数LoRA適用済みimg2imgパイプラインを取得"""
        self.load_pipeline(vae_name, model_name, scheduler_name)
        if lora_configs:
            self.load_loras(lora_configs)
        return self.pipe_img2img

    def get_inpaint_pipeline_with_loras(self, vae_name: str, model_name: str = None, lora_configs: list = None, scheduler_name: str = None):
        """複数LoRA適用済みinpaintパイプラインを取得"""
        self.load_pipeline(vae_name, model_name, scheduler_name)
        if lora_configs:
            self.load_loras(lora_configs)
        return self.pipe_inpaint


# グローバルインスタンス
pipeline_manager = PipelineManager()
