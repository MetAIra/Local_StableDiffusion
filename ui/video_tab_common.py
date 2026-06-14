"""動画系タブ共通ヘルパー

video_gen_tab / video_batch_tab / video_chain_tab で重複していた
モデル選択肢・DL状況テキスト・DL確認プリアンブル・動き補間 kwargs 構築を共通化する。
"""
from config import VIDEO_MODELS, is_video_model_downloaded
from pipelines.video_downloader import ensure_video_model_downloaded


def model_choices(supported_keys=None):
    """VIDEO_MODELS から (label, key) のドロップダウン選択肢を作る

    supported_keys を渡すと、そのキー集合に含まれるモデルだけに絞り込む
    （動画連鎖タブの VIDEO_CHAIN_SUPPORTED_MODELS 用）。
    """
    return [
        (cfg["label"], key)
        for key, cfg in VIDEO_MODELS.items()
        if supported_keys is None or key in supported_keys
    ]


def model_status_text(model_key: str, style: str = "full") -> str:
    """モデルのDL状況とスペックを表すマークダウンテキスト

    style:
        'full'    — 動画生成タブの複数行表示（vram / size / repo 付き）
        'oneline' — 動画バッチタブの1行表示（タイプ + デフォルト解像度）
        'chain'   — 動画連鎖タブの1行表示（解像度 + fps + max frames）
    """
    cfg = VIDEO_MODELS.get(model_key, {})
    if not cfg:
        return ""
    dl_ok = is_video_model_downloaded(model_key)
    if style == "full":
        status = "✅ ダウンロード済み" if dl_ok else "⚠️ 未ダウンロード（HFから自動DL試行）"
        return (
            f"**{cfg['label']}**　{status}\n\n"
            f"- タイプ: `{cfg['type']}`　目安VRAM: `{cfg['vram']}GB`　サイズ: `{cfg['size_gb']}GB`\n"
            f"- {cfg['description']}\n"
            f"- repo: `{cfg['repo_id']}`"
        )
    status = "✅ DL済" if dl_ok else "⚠️ 未DL（生成時に自動DL）"
    if style == "chain":
        return (
            f"**{cfg['label']}** {status} ・ デフォルト解像度 "
            f"{cfg['default_width']}×{cfg['default_height']} / "
            f"native fps {cfg['default_fps']} / max frames {cfg.get('max_frames', '-')}"
        )
    # 'oneline'
    return f"**{cfg['label']}** {status} ・ {cfg['type']} ・ デフォルト解像度 {cfg['default_width']}×{cfg['default_height']}"


def ensure_model_ready(model_key, progress):
    """生成前のモデルDL確認プリアンブル

    progress に DL確認中メッセージを出してから ensure_video_model_downloaded を呼ぶ。
    Returns: (ok: bool, dl_status: str) — 失敗時の戻り値の形は呼び出し側が決める。
    """
    cfg = VIDEO_MODELS.get(model_key, {})
    label = cfg.get("label", model_key)
    progress(0.0, desc=f"モデル '{label}' DL確認中...")
    return ensure_video_model_downloaded(model_key)


def build_smooth_kwargs(smooth_enable, smooth_fps, fps, fmt, smooth_method, smooth_mode=None) -> dict:
    """動き補間（生成FPSより目標FPSが大きい mp4 のときのみ意味あり）の kwargs を構築"""
    kwargs = {}
    if smooth_enable and int(smooth_fps) > int(fps) and fmt == "mp4":
        kwargs["smooth_target_fps"] = int(smooth_fps)
        kwargs["smooth_method"] = str(smooth_method)
        if smooth_mode is not None:
            kwargs["smooth_mode"] = str(smooth_mode)
    return kwargs
