"""動画のフレーム補間ユーティリティ

低fps動画を ffmpeg の minterpolate フィルタで滑らかにする。
- imageio-ffmpeg がバンドルしている ffmpeg バイナリを使うので追加 DL 不要
- mi_mode=mci (動き補正補間) が品質的にバランス良い

xfade 連結（複数 mp4 をクロスフェードで滑らかに繋ぐ）と最終フレーム抽出も同居。
"""
import os
import subprocess
from typing import Optional, List

from PIL import Image


def _get_ffmpeg_exe() -> Optional[str]:
    """利用可能な ffmpeg バイナリのパスを返す。なければ None"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # PATH の ffmpeg を最後の手段として試す
        from shutil import which
        return which("ffmpeg")


def interpolate_video_fps(
    input_path: str,
    output_path: str,
    target_fps: int,
    mode: str = "mci",
    crf: int = 20,
) -> str:
    """ffmpeg minterpolate でフレーム補間して滑らかにする

    Args:
        input_path: 入力 mp4 のパス
        output_path: 出力 mp4 のパス
        target_fps: 目標 FPS（元の FPS より大きい値）
        mode: minterpolate の mi_mode
              - "mci": 動き補正補間（品質 OK・標準）
              - "blend": 単純ブレンド（高速・品質低）
              - "dup": 重複フレーム（最速・品質低）
        crf: H.264 の CRF（小さいほど高品質）

    Returns:
        出力先パス。失敗時は input_path をそのまま返す。
    """
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        print("[interpolate_video_fps] ffmpeg が見つかりません -> 補間スキップ")
        return input_path

    if not os.path.exists(input_path):
        print(f"[interpolate_video_fps] 入力ファイルなし: {input_path}")
        return input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # mi_mode=mci のとき me_mode=bidir (双方向動き推定) で品質向上、
    # vsbmc=1 で時間方向のスムージングを有効化
    if mode == "mci":
        vf = f"minterpolate=fps={int(target_fps)}:mi_mode=mci:me_mode=bidir:vsbmc=1"
    else:
        vf = f"minterpolate=fps={int(target_fps)}:mi_mode={mode}"

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[interpolate_video_fps] ffmpeg 失敗: {result.stderr[-500:]}")
            return input_path
        return output_path
    except subprocess.TimeoutExpired:
        print("[interpolate_video_fps] ffmpeg タイムアウト")
        return input_path
    except Exception as e:
        print(f"[interpolate_video_fps] 例外: {e}")
        return input_path


# =============================================================================
# 最終フレーム抽出（シーン連鎖の i2v 受け渡し用）
# =============================================================================

def extract_last_frame(video_path: str, output_image_path: Optional[str] = None) -> Optional[Image.Image]:
    """動画から最終フレームを PIL.Image として返す

    output_image_path を指定すると同時にディスクにも保存（JPEG）。
    抽出に失敗したら None を返す。
    """
    if not os.path.exists(video_path):
        return None

    # まず imageio で試行（pyav/ffmpeg バックエンド自動切替）
    try:
        import imageio.v3 as iio
        arr = iio.imread(video_path, index=-1, plugin="FFMPEG")
        img = Image.fromarray(arr).convert("RGB")
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            img.save(output_image_path, quality=95)
        return img
    except Exception as e:
        print(f"[extract_last_frame] imageio 失敗、ffmpeg 直叩きにフォールバック: {e}")

    # フォールバック: ffmpeg で末尾 1 秒に seek して 1 枚抽出
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        print("[extract_last_frame] ffmpeg なし -> 抽出不能")
        return None

    tmp_path = output_image_path or os.path.join(os.path.dirname(video_path), "_last_tmp.jpg")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-sseof", "-1.0", "-i", video_path,
        "-update", "1", "-frames:v", "1",
        "-q:v", "2",
        tmp_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not os.path.exists(tmp_path):
            print(f"[extract_last_frame] ffmpeg 失敗: {result.stderr[-300:]}")
            return None
        img = Image.open(tmp_path).convert("RGB")
        if output_image_path is None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return img
    except Exception as e:
        print(f"[extract_last_frame] 例外: {e}")
        return None


# =============================================================================
# xfade 連結（複数 mp4 を滑らかに繋ぐ）
# =============================================================================

def _probe_duration(video_path: str) -> Optional[float]:
    """ffprobe (imageio-ffmpeg バンドルにはない) または ffmpeg 経由で秒数取得"""
    # imageio が pyav 経由で読めるならそれが最速
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            # フォールバック: ストリーム全体の duration
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass

    # フォールバック: ffmpeg -i で stderr から duration を拾う
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-i", video_path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stderr
        # "Duration: 00:00:05.20, ..." を探す
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mi * 60 + s
    except Exception:
        pass
    return None


def concat_videos_xfade(
    input_paths: List[str],
    output_path: str,
    crossfade_seconds: float = 0.3,
    transition: str = "fade",
    crf: int = 20,
    target_fps: Optional[int] = None,
) -> str:
    """複数 mp4 を ffmpeg xfade フィルタでクロスフェード連結

    Args:
        input_paths: 連結する mp4 のパスリスト（順序通り）
        output_path: 出力 mp4 のパス
        crossfade_seconds: シーン間のクロスフェード秒数
        transition: xfade の transition 種類（fade / fadeblack / dissolve / smoothleft 等）
        crf: H.264 CRF
        target_fps: 出力 FPS（指定なしは入力1本目のまま）

    Returns:
        出力先パス。失敗時は input_paths[0] を返す（呼び出し側で気づけるように）。

    Notes:
        - 全クリップは同じ解像度想定（LTX-Video チェーン前提）。違う場合は事前にリサイズしておく
        - クロスフェードは無音前提（音声トラックなし）。音声付きが必要になったら acrossfade を追加
        - クリップ長 < crossfade_seconds のときは crossfade を縮める
    """
    if not input_paths:
        return output_path
    if len(input_paths) == 1:
        # 1本だけなら何もせずコピー（ただし呼び出し側はそのまま使えば良いので、安全のためコピー）
        try:
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_paths[0], output_path)
            return output_path
        except Exception:
            return input_paths[0]

    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        print("[concat_videos_xfade] ffmpeg が見つかりません -> 連結スキップ")
        return input_paths[0]

    # 各クリップの長さを取得
    durations: List[float] = []
    for p in input_paths:
        d = _probe_duration(p)
        if d is None or d <= 0:
            print(f"[concat_videos_xfade] duration 取得失敗: {p}")
            return input_paths[0]
        durations.append(d)

    # crossfade はクリップ最短長の半分以下にクランプ（過剰な fade で全部消えるのを防ぐ）
    safe_xf = max(0.05, min(float(crossfade_seconds), 0.49 * min(durations)))

    # filter_complex を構築
    # [0:v][1:v]xfade=transition=fade:duration=xf:offset=O1[v1];
    # [v1][2:v]xfade=...:offset=O2[v2]; ...
    parts = []
    last_label = "[0:v]"
    cum = durations[0]  # [v0..vk] の累積長
    for i in range(1, len(input_paths)):
        offset = cum - safe_xf
        new_label = f"[v{i}]"
        parts.append(
            f"{last_label}[{i}:v]xfade=transition={transition}:"
            f"duration={safe_xf:.4f}:offset={offset:.4f}{new_label}"
        )
        cum += durations[i] - safe_xf
        last_label = new_label

    filter_complex = ";".join(parts)

    cmd = [ffmpeg, "-y"]
    for p in input_paths:
        cmd.extend(["-i", p])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", last_label,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
    ])
    if target_fps:
        cmd.extend(["-r", str(int(target_fps))])
    cmd.append(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # 長尺の連結は時間がかかるためタイムアウト広め (60分)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"[concat_videos_xfade] ffmpeg 失敗: {result.stderr[-800:]}")
            return input_paths[0]
        print(
            f"[concat_videos_xfade] 連結成功: {len(input_paths)}本 / "
            f"xfade={safe_xf:.2f}s / total~={cum:.1f}s / -> {output_path}"
        )
        return output_path
    except subprocess.TimeoutExpired:
        print("[concat_videos_xfade] ffmpeg タイムアウト")
        return input_paths[0]
    except Exception as e:
        print(f"[concat_videos_xfade] 例外: {e}")
        return input_paths[0]
