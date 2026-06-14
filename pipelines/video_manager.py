"""動画生成パイプライン管理モジュール

8GB VRAM で動作させることを前提に、CPU offload と VAE tiling を駆使する。
画像生成側 (PipelineManager) や音声側 (AudioPipelineManager) と独立した
シングルトンとして VRAM を管理。
"""
import os
import gc
from typing import Optional, Any

import torch

from config import (
    DEVICE, VIDEO_MODELS, VIDEO_MODEL_DIR,
    get_video_model_path, get_video_base_model_path,
)


def _is_drive_path(path: str) -> bool:
    """Google Drive 上のパスか判定（Windows G: または "google drive" 含む）"""
    try:
        s = str(path).lower()
        return s.startswith("g:") or "googledrive" in s or "google drive" in s
    except Exception:
        return False


class VideoPipelineManager:
    """動画生成パイプラインを管理するシングルトンクラス"""

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

        self.pipeline = None
        self.current_model_key: Optional[str] = None
        self.device = DEVICE
        self.error: Optional[str] = None

        # 8GB VRAM 環境では基本 bf16 を使う（fp16 より安定するモデルが多い）
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Google Drive (G:) 上のファイルは mmap が動かないことがある（OSError 22）。
        # モデルが G ドライブにあるなら mmap を無効化してロード。
        self.disable_mmap = self._should_disable_mmap()

    def _should_disable_mmap(self) -> bool:
        """モデル格納先が Google Drive 上なら mmap を無効化"""
        return _is_drive_path(VIDEO_MODEL_DIR)

    # =========================================================================
    # 共通ローダー
    # =========================================================================

    def _resolve_model_source(self, model_key: str) -> str:
        """ローカルに重みがDL済みなら local パス、未DLなら HF repo_id を返す

        フォルダだけ作られて重みファイルが入っていない（README/ライセンスのみ）状態を
        検出するため、is_video_model_downloaded() の重みチェックを使う。
        """
        from config import is_video_model_downloaded
        local = get_video_model_path(model_key)
        if local and os.path.isdir(local) and is_video_model_downloaded(model_key):
            return local
        cfg = VIDEO_MODELS.get(model_key)
        if not cfg:
            raise ValueError(f"Unknown video model: {model_key}")
        return cfg["repo_id"]

    def _resolve_base_model_source(self, model_key: str) -> str:
        """AnimateDiff用ベースSD1.5モデルのソースを解決

        ローカルに必須サブフォルダ(unet/vae/text_encoder)の重みが揃っていなければ
        HF リポID にフォールバック。
        """
        from config import is_video_base_model_downloaded
        cfg = VIDEO_MODELS.get(model_key, {})
        local_base = get_video_base_model_path(model_key)
        if local_base and os.path.isdir(local_base) and is_video_base_model_downloaded(model_key):
            return local_base
        return cfg.get("base_model_repo", "runwayml/stable-diffusion-v1-5")

    def _apply_low_vram_optimizations(self, pipeline, mode: str = "model_offload", enable_vae_tiling: bool = True):
        """8GB VRAM向けの省メモリ設定を適用

        Args:
            pipeline: パイプライン
            mode: "model_offload" (推奨) or "sequential" (より省メモリ・低速)
            enable_vae_tiling: VAE tiling を有効化するか。AnimateDiff のように 512x512
                ネイティブで小さい解像度のモデルでは tiling の境界がモザイク状の
                アーティファクトとして出るため False にする。
        """
        if self.device != "cuda":
            return pipeline

        # VAE 最適化
        if hasattr(pipeline, "vae"):
            try:
                pipeline.vae.enable_slicing()
            except Exception:
                pass
            if enable_vae_tiling:
                try:
                    pipeline.vae.enable_tiling()
                except Exception:
                    pass

        # メモリオフロード
        if mode == "sequential":
            try:
                pipeline.enable_sequential_cpu_offload()
                print("  [低VRAM] sequential CPU offload 有効")
            except Exception as e:
                print(f"  sequential offload 失敗、model offload にフォールバック: {e}")
                try:
                    pipeline.enable_model_cpu_offload()
                except Exception:
                    pipeline.to(self.device)
        else:
            try:
                pipeline.enable_model_cpu_offload()
                print("  [低VRAM] model CPU offload 有効")
            except Exception as e:
                print(f"  model offload 失敗、to(device) にフォールバック: {e}")
                pipeline.to(self.device)

        # attention slicing
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

        return pipeline

    # =========================================================================
    # 各モデル固有ローダー
    # =========================================================================

    def _make_slim_sd15_config(self, local_sd15: "Path") -> "Path":
        """ローカル SD1.5 から safety_checker/feature_extractor を抜いた config 用 folder を作る

        from_single_file() が config フォルダの model_index.json を読み、そこに含まれる
        safety_checker / feature_extractor のクラスを HF から取得しようとして失敗するため、
        該当エントリを除いた slim な model_index.json を C: に用意する。
        """
        import json
        import shutil
        import tempfile
        from pathlib import Path

        cache_root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "video_gen_sd15_slim"
        cache_root.mkdir(parents=True, exist_ok=True)
        slim_dir = cache_root / "stable-diffusion-v1-5"

        if (slim_dir / "model_index.json").exists() and (slim_dir / "unet" / "config.json").exists():
            return slim_dir

        print(f"  slim SD1.5 config テンプレートを作成: {slim_dir}")
        slim_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
            src_sub = local_sd15 / sub
            dst_sub = slim_dir / sub
            if not src_sub.exists():
                continue
            # config.json などの軽量ファイルだけコピー（重みファイルは不要なので除外）
            dst_sub.mkdir(parents=True, exist_ok=True)
            for f in src_sub.iterdir():
                if not f.is_file():
                    continue
                # 重みファイルはスキップ（テンプレートには不要）
                if f.suffix in (".safetensors", ".bin"):
                    continue
                shutil.copyfile(str(f), str(dst_sub / f.name))

        # model_index.json を slim 化
        with open(local_sd15 / "model_index.json", "r", encoding="utf-8") as f:
            idx = json.load(f)
        idx.pop("safety_checker", None)
        idx.pop("feature_extractor", None)
        with open(slim_dir / "model_index.json", "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)

        return slim_dir

    def _stage_file_to_local(self, src_path: str) -> str:
        """Google Drive 上の大きい safetensors を C: の一時領域にコピー

        Google Drive のストリーミング(仮想)ファイルは Python の `open().read()` /
        mmap が大きいファイル（>1-2GB）でしばしば OSError 22 を出すため、
        ロード前に C: にコピーしておくと安定する。

        既に staging 済みで mtime/サイズが変わっていなければそのままパスを返す。
        """
        import shutil
        import tempfile
        from pathlib import Path

        if not _is_drive_path(src_path):
            return src_path

        src = Path(src_path)
        if not src.exists():
            return src_path

        stage_root = Path(tempfile.gettempdir()) / "video_gen_staged"
        stage_root.mkdir(parents=True, exist_ok=True)
        # サブフォルダ無しでファイル名が衝突する可能性があるためサイズで一意化
        stat = src.stat()
        staged = stage_root / f"{src.stem}_{stat.st_size}{src.suffix}"

        if staged.exists() and staged.stat().st_size == stat.st_size:
            return str(staged)

        print(f"  Google Drive 上のファイルをローカルにステージング: {src.name} ({stat.st_size/(1024**3):.2f}GB)")
        print(f"    → {staged}")
        shutil.copyfile(str(src), str(staged))
        print(f"    ステージング完了")
        return str(staged)

    def _convert_sd15_single_file_to_diffusers(self, single_file_path: str) -> str:
        """SD1.5 single-file safetensors を diffusers folder 形式に変換してキャッシュ

        AnimateDiff は diffusers folder 形式の SD1.5 を要求するため、
        models/model/*.safetensors を一旦変換しておく。
        既に変換済みの場合はキャッシュフォルダパスを返すだけ。

        Returns: 変換済み diffusers folder のパス
        """
        from pathlib import Path
        import tempfile
        from diffusers import StableDiffusionPipeline

        single_file = Path(single_file_path)
        # 変換キャッシュは Google Drive ではなくローカル C: の永続領域に置く。
        # Drive 上に置くと書き込みは出来ても読み込み (mmap/big-read) で再び OSError 22
        # を起こすため、AnimateDiff のロード自体が失敗する。
        if _is_drive_path(VIDEO_MODEL_DIR):
            local_cache_root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "video_gen_converted_sd15"
        else:
            local_cache_root = Path(VIDEO_MODEL_DIR) / "_converted_sd15"
        cache_dir = local_cache_root / single_file.stem

        # 既に変換済みなら何もしない（VAE/UNet の重みも揃っているか確認）
        required_weights = [
            cache_dir / "model_index.json",
            cache_dir / "unet" / "config.json",
            cache_dir / "vae" / "config.json",
        ]
        if all(p.exists() and p.stat().st_size > 0 for p in required_weights):
            return str(cache_dir)

        # Google Drive 上の safetensors は OSError 22 で読めないため、ローカルへステージング
        load_path = self._stage_file_to_local(str(single_file))

        print(f"  SD1.5 single-file → diffusers folder に変換中: {single_file.name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # from_single_file() はデフォルトで HF から SD1.5 config テンプレートを取得しに行く。
        # オフライン運用ではここで失敗するので、ローカルの SD1.5 diffusers folder を
        # config テンプレートとして渡す。
        # 加えて、safety_checker/feature_extractor のエントリを除いた slim 版を C: に
        # 用意する。これらが残っていると `from_single_file` が CompVis の image_processor
        # を取得しに行ってオフライン環境で失敗する。
        local_sd15 = Path(VIDEO_MODEL_DIR) / "stable-diffusion-v1-5"
        config_kwargs = {}
        if (local_sd15 / "model_index.json").exists():
            slim_sd15 = self._make_slim_sd15_config(local_sd15)
            config_kwargs["config"] = str(slim_sd15)

        # SD1.5 系は fp16 で訓練されているため bf16 だと出力が壊れる
        sd_dtype = torch.float16 if self.device == "cuda" else torch.float32
        # 注意: `load_safety_checker=False` は最新の diffusers で legacy パスを
        # 強制的に発動させてしまい、オフラインで CompVis/stable-diffusion-safety-checker
        # を取りに行って失敗する。deprecation 警告で指示されている通り、
        # `safety_checker=None` + `feature_extractor=None` だけで渡す。
        sd_pipe = StableDiffusionPipeline.from_single_file(
            load_path,
            torch_dtype=sd_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            disable_mmap=self.disable_mmap,
            local_files_only=True,  # HF アクセスを完全にブロック
            **config_kwargs,
        )
        sd_pipe.save_pretrained(str(cache_dir), safe_serialization=True)
        del sd_pipe
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"  変換完了 → {cache_dir}")
        return str(cache_dir)

    def load_animatediff(self, custom_base_path: Optional[str] = None) -> Any:
        """AnimateDiff (SD1.5 + Motion Adapter) パイプラインをロード

        Args:
            custom_base_path: SD1.5 single-file safetensors のパス（任意）。
                指定すると既存の AnimateDiff 用 SD1.5 ベースの代わりに使う。
                models/model/ 内の高品質 SD1.5 モデル（bluePencil, meinamix 等）を
                ベースにすることで生成品質を大きく改善できる。
        """
        # キャッシュキーは custom_base を含める
        cache_key = "animatediff" if not custom_base_path else f"animatediff::{custom_base_path}"
        if self.pipeline is not None and self.current_model_key == cache_key:
            return self.pipeline

        self.unload()
        print(f"Loading AnimateDiff pipeline (base={custom_base_path or 'default SD1.5'})...")

        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

            # dtype 戦略:
            # - UNet/MotionAdapter/text_encoder: bf16
            #   (Blackwell GPU [SM 12.0, RTX 50 系] では fp16 の matmul が大きな
            #    アキュムレーションでオーバーフローして Inf/NaN になり、UNet 出力が
            #    壊れて VAE 復号が砂嵐/モザイクになるため、bf16 を選択。fp16 で
            #    訓練された SD1.5 でも bf16 推論で十分動く)
            # - VAE: fp32
            #   (bf16 で VAE 復号すると過去に砂嵐ノイズが出た経緯があり、VAE は
            #    精度を優先して fp32 に固定。決定的に安定。VRAM コストはわずか
            #    [~330MB] で 8GB でも問題なし)
            ad_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            vae_dtype = torch.float32

            adapter_src = self._resolve_model_source("animatediff")
            if custom_base_path and os.path.isfile(custom_base_path):
                base_src = self._convert_sd15_single_file_to_diffusers(custom_base_path)
                use_custom_base = True
            else:
                base_src = self._resolve_base_model_source("animatediff")
                use_custom_base = False

            adapter = MotionAdapter.from_pretrained(
                adapter_src,
                torch_dtype=ad_dtype,
                use_safetensors=True,
                disable_mmap=self.disable_mmap,
            )
            # SD1.5ベースは fp16 variant を優先。8GB VRAMで十分かつDLサイズも半分。
            # use_safetensors=True を必ず渡す（diffusers のデフォルトが .bin を探そうとし、
            # 一部サブフォルダに .bin が無いとロード失敗するため）。
            # fp16 variant が一部サブフォルダにしか無い場合は variant 指定なしにフォールバック。
            base_kwargs = dict(
                motion_adapter=adapter,
                torch_dtype=ad_dtype,
                use_safetensors=True,
                disable_mmap=self.disable_mmap,
            )
            # カスタム単体ファイル変換キャッシュには fp16 variant が無いので、
            # variant 無しから始めて重複試行を避ける
            try:
                if use_custom_base:
                    pipe = AnimateDiffPipeline.from_pretrained(
                        base_src,
                        **base_kwargs,
                    )
                else:
                    pipe = AnimateDiffPipeline.from_pretrained(
                        base_src,
                        variant="fp16",
                        **base_kwargs,
                    )
            except (OSError, ValueError) as e:
                print(f"  variant=fp16 でのロード失敗 → 通常 variant でリトライ: {str(e)[:200]}")
                try:
                    pipe = AnimateDiffPipeline.from_pretrained(
                        base_src,
                        **base_kwargs,
                    )
                except (OSError, ValueError) as e2:
                    # ローカルベースモデルの重みが不完全 → HF リポからオンラインで再試行
                    cfg = VIDEO_MODELS.get("animatediff", {})
                    online_base = cfg.get("base_model_repo", "runwayml/stable-diffusion-v1-5")
                    if base_src != online_base:
                        print(f"  ローカルベース読込失敗 → HF '{online_base}' から再取得: {str(e2)[:200]}")
                        try:
                            pipe = AnimateDiffPipeline.from_pretrained(
                                online_base,
                                variant="fp16",
                                **base_kwargs,
                            )
                        except (OSError, ValueError):
                            pipe = AnimateDiffPipeline.from_pretrained(
                                online_base,
                                **base_kwargs,
                            )
                    else:
                        raise
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )

            # VAE のみ fp32 にキャスト（Blackwell の fp16 オーバーフロー対策）。
            # offload 適用前に行う必要がある。bf16 では VAE 復号が砂嵐になる経緯あり。
            if self.device == "cuda":
                pipe.vae.to(vae_dtype)
                # UNet は bf16 / VAE は fp32 の混在になるため、VAE.decode 呼び出し時
                # に入力潜在を fp32 にキャストするフックを差し込む。これをしないと
                # post_quant_conv で「Input bf16 vs bias fp32」エラーになる。
                _orig_vae_decode = pipe.vae.decode
                def _vae_decode_autocast(z, *args, **kwargs):
                    return _orig_vae_decode(z.to(vae_dtype), *args, **kwargs)
                pipe.vae.decode = _vae_decode_autocast

            # AnimateDiff は 512x512 ネイティブで VAE tiling の境界がモザイクとして
            # 残るため tiling は無効化（slicing のみ）。公式 diffusers の例も同様。
            self.pipeline = self._apply_low_vram_optimizations(pipe, mode="model_offload", enable_vae_tiling=False)
            self.current_model_key = cache_key
            print("AnimateDiff loaded.")
            return self.pipeline

        except Exception as e:
            self.error = f"AnimateDiff のロードに失敗: {e}"
            print(self.error)
            raise

    def load_svd_xt(self) -> Any:
        """Stable Video Diffusion XT (image→video) パイプラインをロード"""
        if self.pipeline is not None and self.current_model_key == "svd_xt":
            return self.pipeline

        self.unload()
        print("Loading SVD-XT pipeline...")

        try:
            from diffusers import StableVideoDiffusionPipeline

            src = self._resolve_model_source("svd_xt")
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    src,
                    torch_dtype=torch.float16,  # SVDは fp16 推奨
                    variant="fp16",
                    use_safetensors=True,
                    disable_mmap=self.disable_mmap,
                )
            except (OSError, ValueError) as e:
                print(f"  variant=fp16 失敗 → variant 指定なしでリトライ: {str(e)[:200]}")
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    src,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    disable_mmap=self.disable_mmap,
                )

            self.pipeline = self._apply_low_vram_optimizations(pipe, mode="model_offload")
            self.current_model_key = "svd_xt"
            print("SVD-XT loaded.")
            return self.pipeline

        except Exception as e:
            self.error = f"SVD-XT のロードに失敗: {e}\n（HFログイン+ライセンス同意が必要かも）"
            print(self.error)
            raise

    def load_ltx_video(self, mode: str = "t2v") -> Any:
        """LTX-Video パイプラインをロード

        Args:
            mode: "t2v" (text→video) or "i2v" (image→video)
        """
        cache_key = f"ltx_video_{mode}"
        if self.pipeline is not None and self.current_model_key == cache_key:
            return self.pipeline

        self.unload()
        print(f"Loading LTX-Video pipeline ({mode})...")

        try:
            if mode == "i2v":
                from diffusers import LTXImageToVideoPipeline as PipeCls
            else:
                from diffusers import LTXPipeline as PipeCls

            src = self._resolve_model_source("ltx_video")

            # transformers 4.57+ では fast の T5TokenizerFast が
            # spiece.model を tiktoken 形式と誤判定して失敗するため、
            # 明示的に slow の T5Tokenizer をロードして PipeCls に渡す。
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(src, subfolder="tokenizer")

            pipe = PipeCls.from_pretrained(
                src,
                tokenizer=tokenizer,
                torch_dtype=self.dtype,
                use_safetensors=True,
                disable_mmap=self.disable_mmap,
            )

            self.pipeline = self._apply_low_vram_optimizations(pipe, mode="model_offload")
            self.current_model_key = cache_key
            print(f"LTX-Video loaded ({mode}).")
            return self.pipeline

        except Exception as e:
            self.error = f"LTX-Video のロードに失敗: {e}"
            print(self.error)
            raise

    def load_cogvideox(self) -> Any:
        """CogVideoX-2B (text→video) パイプラインをロード"""
        if self.pipeline is not None and self.current_model_key == "cogvideox_2b":
            return self.pipeline

        self.unload()
        print("Loading CogVideoX-2B pipeline...")

        try:
            from diffusers import CogVideoXPipeline

            src = self._resolve_model_source("cogvideox_2b")
            pipe = CogVideoXPipeline.from_pretrained(
                src,
                torch_dtype=self.dtype,
                use_safetensors=True,
                disable_mmap=self.disable_mmap,
            )

            # 8GB では sequential offload が必要
            self.pipeline = self._apply_low_vram_optimizations(pipe, mode="sequential")
            self.current_model_key = "cogvideox_2b"
            print("CogVideoX-2B loaded.")
            return self.pipeline

        except Exception as e:
            self.error = f"CogVideoX-2B のロードに失敗: {e}"
            print(self.error)
            raise

    def load_wan21(self) -> Any:
        """Wan2.1 1.3B (text→video) パイプラインをロード"""
        if self.pipeline is not None and self.current_model_key == "wan21_1_3b":
            return self.pipeline

        self.unload()
        print("Loading Wan2.1 1.3B pipeline...")

        try:
            from diffusers import WanPipeline, AutoencoderKLWan

            src = self._resolve_model_source("wan21_1_3b")
            # Wan は VAE を別途 fp32 で読み込むのが推奨
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    src, subfolder="vae", torch_dtype=torch.float32,
                    use_safetensors=True, disable_mmap=self.disable_mmap,
                )
                pipe = WanPipeline.from_pretrained(
                    src, vae=vae, torch_dtype=self.dtype,
                    use_safetensors=True, disable_mmap=self.disable_mmap,
                )
            except Exception:
                # VAE 分離ロードに失敗したらまるごとロード
                pipe = WanPipeline.from_pretrained(
                    src, torch_dtype=self.dtype,
                    use_safetensors=True, disable_mmap=self.disable_mmap,
                )

            self.pipeline = self._apply_low_vram_optimizations(pipe, mode="model_offload")
            self.current_model_key = "wan21_1_3b"
            print("Wan2.1 1.3B loaded.")
            return self.pipeline

        except Exception as e:
            self.error = f"Wan2.1 のロードに失敗: {e}\n（diffusers バージョンに WanPipeline が必要）"
            print(self.error)
            raise

    # =========================================================================
    # アンロード/キャッシュクリア
    # =========================================================================

    def unload(self):
        """現在のパイプラインをアンロード"""
        if self.pipeline is not None:
            print(f"Unloading video pipeline: {self.current_model_key}")
            try:
                # CPU offload を解除しつつ削除
                del self.pipeline
            except Exception:
                pass
            self.pipeline = None
            self.current_model_key = None
            self.clear_cache()

    def clear_cache(self):
        """GPUメモリキャッシュをクリア"""
        gc.collect()
        if self.device == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()


# グローバルインスタンス
video_pipeline_manager = VideoPipelineManager()
