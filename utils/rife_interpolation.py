"""RIFE (Real-Time Intermediate Flow Estimation) によるニューラル動画フレーム補間

minterpolate のブロックマッチング由来の歪み（ぐにゃぐにゃ）を解消するため、
学習済みの IFNet を使って中間フレームを推定する。

モデル: MonsterMMORPG/Practical-RIFE (RIFE 4.25 系)
モデルファイル: ~/AppData/Local/video_models/rife/{train_log,model}/...
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# RIFE モデル/コードの設置先。Google Drive 由来の mmap 問題を避けるためローカルに置く。
# 環境変数 RIFE_HOME で上書き可。未設定時は OS ごとの既定を使う
# （Windows: %LOCALAPPDATA%\video_models\rife、Linux/Colab: ~/.cache/video_models/rife）。
def _default_rife_home() -> str:
    env = os.environ.get("RIFE_HOME")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "video_models", "rife")
    return os.path.expanduser("~/.cache/video_models/rife")


RIFE_HOME = Path(_default_rife_home())


_rife_model = None  # シングルトン


def _ensure_rife_paths_on_syspath():
    """RIFE_HOME を sys.path に追加して `train_log.IFNet_HDv3` 等を import 可能にする"""
    rife_dir = str(RIFE_HOME)
    if rife_dir not in sys.path:
        sys.path.insert(0, rife_dir)


def _load_rife_model(device: str = "cuda"):
    """IFNet を読み込んで eval モードに切り替えたモデルを返す（シングルトン）"""
    global _rife_model
    if _rife_model is not None:
        return _rife_model

    _ensure_rife_paths_on_syspath()
    try:
        from train_log.IFNet_HDv3 import IFNet
    except Exception as e:
        raise RuntimeError(
            f"RIFE モデルコードの import に失敗: {e}\n"
            f"RIFE_HOME = {RIFE_HOME} に train_log/IFNet_HDv3.py 等が存在するか確認してください"
        )

    weights_path = RIFE_HOME / "train_log" / "flownet.pkl"
    if not weights_path.exists():
        raise FileNotFoundError(f"RIFE 重みが見つかりません: {weights_path}")

    flownet = IFNet()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    # キー名が "module." 始まりの場合を吸収
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    flownet.load_state_dict(cleaned, strict=False)
    flownet.eval().to(device)

    _rife_model = (flownet, device)
    return _rife_model


def _pad_to_multiple(t: torch.Tensor, multiple: int = 64):
    """高さ・幅を `multiple` の倍数になるよう右下に reflect pad（IFNet の要件）

    IFNet 内部では F.interpolate(scale=1/16) → conv stride2 ×2 → ConvTranspose ×2 →
    PixelShuffle ×2 → F.interpolate(scale=16) という round-trip があり、
    入力サイズが 64 の倍数でないと中間 round 誤差で復元サイズがずれる
    （例: 480 → 512 で warp の grid_sample サイズが合わずに RuntimeError）。
    """
    _, _, h, w = t.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return t, (0, 0)
    return F.pad(t, (0, pw, 0, ph), mode="replicate"), (ph, pw)


def _interpolate_pair(model, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5, scale: float = 1.0):
    """img0, img1 (B,3,H,W, 0-1 float) → 中間フレーム (B,3,H,W)"""
    imgs = torch.cat((img0, img1), dim=1)
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    with torch.no_grad():
        _, _, merged = model(imgs, timestep, scale_list)
    return merged[-1].clamp(0.0, 1.0)


def _read_video_frames(video_path: str):
    """mp4 から numpy フレーム配列のリストを取り出す（pyav を使用）"""
    import av
    frames = []
    src_fps = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 8.0
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            frames.append(arr)
    return frames, src_fps


def _write_video_frames(frames: list, output_path: str, fps: int, crf: int = 20):
    """numpy フレーム → mp4（pyav h264）"""
    import av
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(output_path, mode="w") as container:
        stream = container.add_stream("h264", rate=int(fps))
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for arr in frames:
            av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def interpolate_video_rife(
    input_path: str,
    output_path: str,
    target_fps: int,
    multiplier: Optional[int] = None,
    scale: float = 1.0,
    device: str = "cuda",
    use_amp: bool = True,
) -> str:
    """RIFE による動画フレーム補間

    Args:
        input_path: 入力 mp4
        output_path: 出力 mp4
        target_fps: 目標 FPS
        multiplier: 中間フレームを 2^k 倍に増やす場合に明示指定。None の場合
                    target_fps / src_fps から自動算出（最も近い 2 のべき乗）。
        scale: IFNet の探索スケール（小さいほど精細・大きいほど大きい動きに強い）。1.0 標準
        device: "cuda" or "cpu"
        use_amp: fp16 推論を使うか（VRAM 削減）

    Returns:
        実際の出力先パス（失敗時は input_path を返す）
    """
    if not os.path.exists(input_path):
        print(f"[RIFE] 入力なし: {input_path}")
        return input_path

    try:
        # フレームと FPS を取り出し
        frames, src_fps = _read_video_frames(input_path)
        n = len(frames)
        if n < 2:
            return input_path

        # 補間倍率を決定: 整数の最も近い倍率を採用（RIFE 4.x は任意の timestep を扱える）。
        # 例: 8fps → 24fps なら 3x、8fps → 16fps なら 2x、16fps → 24fps なら 2x。
        ratio = max(1.0, float(target_fps) / float(src_fps))
        if multiplier is None:
            multiplier = max(1, int(round(ratio)))
            # 過大な倍率は品質低下するので 8x でクランプ
            multiplier = min(multiplier, 8)
        if multiplier <= 1:
            print(f"[RIFE] 補間不要 (src={src_fps}fps, target={target_fps}fps)")
            return input_path

        print(f"[RIFE] フレーム数={n}  src={src_fps:.1f}fps  target={target_fps}fps  multiplier={multiplier}x  scale={scale}")

        # モデル読み込み
        model, device = _load_rife_model(device=device)

        # フレームを (1,3,H,W) tensor 化（[0,1]）
        def to_tensor(arr):
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0).to(device)

        def to_array(t):
            t = t.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (t * 255.0).astype(np.uint8)

        # 補間: 隣接ペアごとに中間 (multiplier-1) フレームを生成して挟み込む
        # 例 multiplier=4: t=0.25, 0.5, 0.75
        result_frames = [frames[0]]
        timesteps = [i / multiplier for i in range(1, multiplier)]

        amp_dtype = torch.float16 if (use_amp and device == "cuda") else torch.float32

        for i in range(n - 1):
            a = to_tensor(frames[i])
            b = to_tensor(frames[i + 1])
            a, _ = _pad_to_multiple(a, 64)
            b_pad, (ph, pw) = _pad_to_multiple(b, 64)
            a = a.to(amp_dtype) if amp_dtype == torch.float16 else a
            b_pad = b_pad.to(amp_dtype) if amp_dtype == torch.float16 else b_pad
            for t in timesteps:
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype, enabled=(device == "cuda" and use_amp)):
                    mid = _interpolate_pair(model, a, b_pad, timestep=float(t), scale=float(scale))
                # crop pad
                if ph > 0 or pw > 0:
                    h_real = b_pad.shape[2] - ph
                    w_real = b_pad.shape[3] - pw
                    mid = mid[:, :, :h_real, :w_real]
                mid = mid.float()
                result_frames.append(to_array(mid))
            result_frames.append(frames[i + 1])

        # 補間後の実 FPS （src × multiplier）が target_fps とずれる場合は
        # ffmpeg で fps だけ揃える（フレーム複製/間引きで微調整）
        new_native_fps = src_fps * multiplier
        save_fps = int(round(target_fps))

        # まず native FPS で書き出し
        tmp_path = output_path + ".tmp.mp4"
        _write_video_frames(result_frames, tmp_path, fps=int(round(new_native_fps)))

        if abs(new_native_fps - save_fps) < 0.5:
            os.replace(tmp_path, output_path)
        else:
            # ffmpeg で目標FPSに微調整
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                from shutil import which
                ffmpeg = which("ffmpeg")
            if ffmpeg is None:
                os.replace(tmp_path, output_path)
            else:
                cmd = [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-i", tmp_path,
                    "-vf", f"fps={save_fps}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    output_path,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    print(f"[RIFE] fps 調整失敗、native fps で保存: {r.stderr[-200:]}")
                    os.replace(tmp_path, output_path)
                else:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # GPU メモリ解放
        if device == "cuda":
            torch.cuda.empty_cache()

        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[RIFE] 補間に失敗: {e}")
        return input_path


def unload_rife_model():
    """RIFE モデルを VRAM から開放（動画生成パイプラインと VRAM を共有するとき用）"""
    global _rife_model
    if _rife_model is not None:
        del _rife_model
        _rife_model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
