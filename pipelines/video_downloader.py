"""動画モデルダウンローダー（CLI スクリプトと UI から共通利用）

設計方針:
- HF snapshot_download をリトライ付きでラップ
- ignore_patterns を一元管理して bloat（単体ckptや fp16 重複等）を排除
- xet/CAS バックエンドは Google Drive と相性が悪いので無効化推奨
- progress_callback を通じて Gradio 側に進捗を返せる
"""
import gc
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from config import (
    VIDEO_MODELS, VIDEO_MODEL_DIR,
    is_video_model_downloaded, is_video_base_model_downloaded,
    get_video_model_path,
)

# Google Drive で xet/CAS バックエンドがリソース不足エラー (WinError 1450) を
# 引き起こすため、importされた時点で従来HTTP DLに切替。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


# diffusersのみで使うため、bloat（単体ckpt, fp16重複, サンプル等）を一元除外
VIDEO_DL_IGNORE_PATTERNS = [
    # 不要フォーマット
    "*.msgpack", "*.onnx", "*.onnx_data",
    "*.fp16.safetensors",
    "*.flax", "*flax_model*",
    "*.ckpt",  # diffusers では .safetensors を優先
    # 単体（非diffusers）チェックポイント
    "svd_xt.safetensors", "svd_xt_image_decoder.safetensors",
    "v1-5-pruned*.safetensors",
    # LTX-Video の各バージョン単体チェックポイント
    "ltx-video-*.safetensors", "ltxv-*.safetensors",
    "*-13b-*.safetensors",   # LTXの13Bモデル（巨大）
    "*-2b-v*.safetensors",
    # サンプル/プレビュー
    "comparison.png", "*comparison*",
    "output_tile.gif", "*.webp",
    "*.license.txt",
]


# =============================================================================
# 低レベル: snapshot_download ラッパー
# =============================================================================

def _is_google_drive_path(path: Path) -> bool:
    """Google Drive 上のパスかどうかを判定"""
    try:
        s = str(path).lower()
        return s.startswith("g:") or "googledrive" in s or "google drive" in s
    except Exception:
        return False


def _safe_snapshot_download(
    repo_id: str,
    local_dir: Path,
    ignore_patterns: Optional[list] = None,
    max_retries: int = 3,
    max_workers: int = 2,
) -> Tuple[bool, str]:
    """リトライ付き snapshot_download

    Google Drive 上に直接DLすると `.incomplete` ファイルのリネームが
    Drive 同期と衝突して失敗する（OSError 22 / "ファイル名を変更できません"）。
    対策: 一時的にローカル C: の tempdir にDL → 完了後にまるごとコピーする。

    Returns:
        (成功フラグ, エラーメッセージ or "")
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub がインストールされていません"

    if ignore_patterns is None:
        ignore_patterns = VIDEO_DL_IGNORE_PATTERNS

    use_temp_staging = _is_google_drive_path(local_dir)
    if use_temp_staging:
        import tempfile
        import shutil
        # tempdir 名にリポID由来の文字を入れて衝突を避ける
        repo_safe = repo_id.replace("/", "_").replace("\\", "_")
        staging = Path(tempfile.gettempdir()) / f"hf_dl_{repo_safe}"
        staging.mkdir(parents=True, exist_ok=True)
        print(f"  Google Drive 同期回避: 一時DL先 = {staging}")
        download_target = staging
    else:
        download_target = local_dir

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(download_target),
                ignore_patterns=ignore_patterns,
                max_workers=max_workers,
            )
            # ステージング → Google Drive へコピー
            if use_temp_staging:
                import shutil
                print(f"  ステージングから Google Drive へコピー中 → {local_dir}")
                local_dir.mkdir(parents=True, exist_ok=True)
                # dirs_exist_ok=True で既存と統合
                shutil.copytree(str(download_target), str(local_dir), dirs_exist_ok=True)
                shutil.rmtree(str(download_target), ignore_errors=True)
                print(f"  コピー完了")
            return True, ""
        except Exception as e:
            err = str(e)
            last_err = err
            print(f"  [試行{attempt}/{max_retries}] {repo_id} 失敗: {err[:200]}")
            if "gated" in err.lower() or "401" in err or "403" in err:
                hint = (
                    f"ゲーティング/認証エラー。"
                    f"https://huggingface.co/{repo_id} で同意 + huggingface-cli login が必要"
                )
                return False, hint
            if attempt < max_retries:
                wait = 30 * attempt  # 30s, 60s, 90s
                print(f"  -> {wait}秒後にリトライ...")
                gc.collect()
                time.sleep(wait)
    return False, f"{max_retries}回リトライ失敗: {last_err[:200]}"


# =============================================================================
# 高レベル: モデル単位のDL
# =============================================================================

def download_video_model(
    model_key: str,
    include_base: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """指定された動画モデル（と必要ならベースモデル）をDL

    Args:
        model_key: VIDEO_MODELS のキー
        include_base: AnimateDiff用のベースSD1.5もDLするか
        progress_callback: 進捗メッセージを通知するコールバック（UI用）

    Returns:
        (成功フラグ, ステータスメッセージ)
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return False, f"未知のモデル: {model_key}"

    def _notify(msg: str):
        print(msg)
        if progress_callback is not None:
            try:
                progress_callback(msg)
            except Exception:
                pass

    target_dir = Path(VIDEO_MODEL_DIR) / cfg["local_subdir"]
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"ディレクトリ作成失敗: {e}"

    _notify(f"=== {cfg['label']} をダウンロード中 ===")
    _notify(f"  repo: {cfg['repo_id']}  size: 約{cfg.get('size_gb', 0)}GB")
    _notify(f"  to:   {target_dir}")

    ok, err = _safe_snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=target_dir,
    )
    if not ok:
        return False, f"{cfg['label']} DL失敗: {err}"
    _notify(f"[OK] {model_key} DL完了")

    # AnimateDiff用ベースSD1.5モデル
    if include_base and "base_model_repo" in cfg:
        base_target = Path(VIDEO_MODEL_DIR) / cfg["base_model_subdir"]
        if is_video_base_model_downloaded(model_key):
            _notify(f"  ベースモデルは既にDL済み: {base_target}")
        else:
            try:
                base_target.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return True, f"{cfg['label']} は完了。ただしベースモデル DL 失敗: {e}"
            _notify(f"--- ベースモデル {cfg['base_model_repo']} をDL中 (fp16 variant) ---")
            # SD1.5 base は fp16 variant 優先 + VAE は fp16 variant が無いリポもあるため
            # fp32 safetensors も許可。.bin は除外（safetensors で十分）。
            # 旧 ignore_patterns で diffusion_pytorch_model.safetensors / model.safetensors を
            # 除外していたが、これだと VAE/text_encoder で fp16 variant が無いリポで重みが
            # ゼロ件になり、AnimateDiff のロードに失敗していた。
            base_ignore = [
                "*.msgpack", "*.onnx", "*.onnx_data",
                "*.flax", "*flax_model*",
                "*.ckpt",
                # .bin は不要（safetensors で十分）
                "*.bin",
                # 単体（非diffusers）チェックポイント
                "v1-5-pruned*.safetensors",
                "v1-5-pruned-emaonly*.safetensors",
                "*non_ema*",
                "comparison.png", "*comparison*",
                "*.webp", "*.license.txt",
            ]
            base_ok, base_err = _safe_snapshot_download(
                repo_id=cfg["base_model_repo"],
                local_dir=base_target,
                ignore_patterns=base_ignore,
            )
            if base_ok:
                _notify(f"[OK] ベースモデル {cfg['base_model_repo']} DL完了")
            else:
                return True, f"{cfg['label']} は完了。ベースモデル失敗: {base_err}"

    return True, f"{cfg['label']} DL完了"


def ensure_video_model_downloaded(
    model_key: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """未DLならDLする。DL済みなら何もしない

    UI から生成前に呼ぶための薄いラッパー。
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return False, f"未知のモデル: {model_key}"

    main_done = is_video_model_downloaded(model_key)
    base_done = is_video_base_model_downloaded(model_key)

    if main_done and base_done:
        return True, f"{cfg['label']} は既にDL済み"

    if progress_callback is not None:
        progress_callback(f"モデル '{cfg['label']}' が未DLです。HFからDL開始...")

    return download_video_model(
        model_key,
        include_base=("base_model_repo" in cfg),
        progress_callback=progress_callback,
    )


def get_dl_status_summary() -> dict:
    """全モデルのDL状況サマリーを返す（UI表示用）"""
    summary = {}
    for key, cfg in VIDEO_MODELS.items():
        local = get_video_model_path(key)
        main_ok = is_video_model_downloaded(key)
        base_ok = is_video_base_model_downloaded(key)
        summary[key] = {
            "label": cfg["label"],
            "main_downloaded": main_ok,
            "base_downloaded": base_ok,
            "ready": main_ok and base_ok,
            "local_path": local,
            "size_gb": cfg.get("size_gb", 0),
        }
    return summary
