"""音声パイプライン管理モジュール"""
import torch
from typing import Any

from config import DEVICE, MUSIC_GEN_MODELS, AUDIOLDM_MODELS


class AudioPipelineManager:
    """音声生成パイプラインを管理するシングルトンクラス

    VRAMを効率的に使用するため、各種音声モデルの排他的ロードを管理
    """

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

        # TTS モデル
        self.bark_model = None
        self.bark_processor = None
        self.xtts_model = None
        self.current_tts_model = None

        # Music Generation モデル
        self.musicgen_model = None
        self.musicgen_processor = None
        self.audioldm_model = None
        self.current_music_model = None

        # Voice Conversion モデル
        self.rvc_model = None
        self.current_rvc_model = None

        self.device = DEVICE
        self.error = None

    # =========================================================================
    # TTS Model Management
    # =========================================================================

    def load_bark(self) -> tuple:
        """Barkモデルをロード

        Returns:
            (model, processor) タプル
        """
        if self.bark_model is not None:
            return self.bark_model, self.bark_processor

        # 他の音声モデルをアンロード
        self.unload_music_models()
        self.unload_voice_conversion_models()

        print("Loading Bark model...")
        try:
            from transformers import AutoProcessor, BarkModel

            self.bark_processor = AutoProcessor.from_pretrained("suno/bark")
            self.bark_model = BarkModel.from_pretrained(
                "suno/bark",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # デバイスに配置（CPU offloadは使わない - デバイス不整合を避けるため）
            self.bark_model.to(self.device)

            self.current_tts_model = "bark"
            print("Bark model loaded successfully")
            return self.bark_model, self.bark_processor

        except Exception as e:
            self.error = f"Bark モデルのロードに失敗: {str(e)}"
            print(f"Error loading Bark: {e}")
            raise

    def load_xtts(self) -> Any:
        """XTTS v2モデルをロード

        Returns:
            TTSモデル
        """
        if self.xtts_model is not None:
            return self.xtts_model

        # 他の音声モデルをアンロード
        self.unload_tts_models()
        self.unload_music_models()
        self.unload_voice_conversion_models()

        print("Loading XTTS v2 model...")
        try:
            from TTS.api import TTS
        except ImportError as e:
            if "BeamSearchScorer" in str(e) or "cannot import name" in str(e):
                self.error = "XTTS v2はtransformersライブラリとの互換性問題があります。\nBarkモデルを使用してください。"
            else:
                self.error = "XTTS v2を使用するにはCoqui TTSライブラリが必要です。\npip install TTS を実行してください。"
            raise ImportError(self.error)

        try:
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if self.device == "cuda":
                self.xtts_model.to(self.device)

            self.current_tts_model = "xtts_v2"
            print("XTTS v2 model loaded successfully")
            return self.xtts_model

        except ImportError as e:
            if "BeamSearchScorer" in str(e) or "cannot import name" in str(e):
                self.error = "XTTS v2はtransformersライブラリ(v4.57)との互換性問題があります。\nBarkモデルを使用してください。"
            else:
                self.error = f"XTTS v2 モジュールエラー: {str(e)}"
            print(f"Error loading XTTS v2: {e}")
            raise
        except Exception as e:
            self.error = f"XTTS v2 モデルのロードに失敗: {str(e)}"
            print(f"Error loading XTTS v2: {e}")
            raise

    def unload_tts_models(self):
        """TTSモデルをアンロード"""
        if self.bark_model is not None:
            print("Unloading Bark model...")
            del self.bark_model
            del self.bark_processor
            self.bark_model = None
            self.bark_processor = None

        if self.xtts_model is not None:
            print("Unloading XTTS model...")
            del self.xtts_model
            self.xtts_model = None

        self.current_tts_model = None
        self.clear_cache()

    # =========================================================================
    # Music Generation Model Management
    # =========================================================================

    def load_musicgen(self, model_name: str = "musicgen-small") -> tuple:
        """MusicGenモデルをロード

        Args:
            model_name: モデル名（musicgen-small, musicgen-medium, musicgen-large, musicgen-melody）

        Returns:
            (model, processor) タプル
        """
        if self.musicgen_model is not None and self.current_music_model == model_name:
            return self.musicgen_model, self.musicgen_processor

        # 他のモデルをアンロード
        self.unload_tts_models()
        self.unload_music_models()
        self.unload_voice_conversion_models()

        print(f"Loading MusicGen model: {model_name}...")
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            model_config = MUSIC_GEN_MODELS.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown MusicGen model: {model_name}")

            repo_id = model_config["repo_id"]

            self.musicgen_processor = AutoProcessor.from_pretrained(repo_id)
            self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.musicgen_model.to(self.device)

            self.current_music_model = model_name
            print(f"MusicGen model loaded successfully: {model_name}")
            return self.musicgen_model, self.musicgen_processor

        except Exception as e:
            self.error = f"MusicGen モデルのロードに失敗: {str(e)}"
            print(f"Error loading MusicGen: {e}")
            raise

    def load_audioldm(self, model_name: str = "audioldm2-music") -> Any:
        """AudioLDM2モデルをロード

        Args:
            model_name: モデル名（audioldm2-music, audioldm2）

        Returns:
            パイプライン
        """
        if self.audioldm_model is not None and self.current_music_model == model_name:
            return self.audioldm_model

        # 他のモデルをアンロード
        self.unload_tts_models()
        self.unload_music_models()
        self.unload_voice_conversion_models()

        print(f"Loading AudioLDM2 model: {model_name}...")
        try:
            from diffusers import AudioLDM2Pipeline

            model_config = AUDIOLDM_MODELS.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown AudioLDM model: {model_name}")

            repo_id = model_config["repo_id"]

            self.audioldm_model = AudioLDM2Pipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.audioldm_model.to(self.device)

            self.current_music_model = model_name
            print(f"AudioLDM2 model loaded successfully: {model_name}")
            return self.audioldm_model

        except Exception as e:
            error_str = str(e)
            if "_get_initial_cache_position" in error_str or "language_model" in error_str:
                self.error = "AudioLDM2は現在のdiffusers/transformersバージョンとの互換性問題があります。\nMusicGenを使用してください。"
            else:
                self.error = f"AudioLDM2 モデルのロードに失敗: {error_str}"
            print(f"Error loading AudioLDM2: {e}")
            raise

    def unload_music_models(self):
        """Music Generationモデルをアンロード"""
        if self.musicgen_model is not None:
            print("Unloading MusicGen model...")
            del self.musicgen_model
            del self.musicgen_processor
            self.musicgen_model = None
            self.musicgen_processor = None

        if self.audioldm_model is not None:
            print("Unloading AudioLDM model...")
            del self.audioldm_model
            self.audioldm_model = None

        self.current_music_model = None
        self.clear_cache()

    # =========================================================================
    # Voice Conversion Model Management
    # =========================================================================

    def unload_voice_conversion_models(self):
        """Voice Conversionモデルをアンロード"""
        if self.rvc_model is not None:
            print("Unloading RVC model...")
            del self.rvc_model
            self.rvc_model = None
            self.current_rvc_model = None

        self.clear_cache()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_cache(self):
        """GPUメモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("GPU cache cleared")


# グローバルインスタンス
audio_pipeline_manager = AudioPipelineManager()
