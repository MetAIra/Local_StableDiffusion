"""動画生成モジュール

各モデル別の生成関数 + 統合エントリポイント。
すべての関数は (output_path, status_message) タプルを返す。
"""
import os
import gc
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT, VIDEO_OUTPUT_DIR_PREFIX,
    VIDEO_WAN_NEGATIVE, VIDEO_ANIMATEDIFF_NEGATIVE,
)
from utils.file import create_output_dir
from .video_manager import video_pipeline_manager

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 動画保存ユーティリティ
# =============================================================================

def _write_gif(pil_frames: list, path: str, fps: int, optimize: bool = False):
    """PIL フレームリストを gif として保存"""
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=optimize,
    )


def _save_video(
    frames: list,
    output_path: str,
    fps: int = 8,
    fmt: str = "mp4",
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
):
    """フレームリストを mp4/gif で保存

    Args:
        frames: PIL.Image または numpy 配列のリスト
        output_path: 出力ファイルパス（拡張子は .mp4 / .gif）
        fps: フレームレート（元の生成フレームレート）
        fmt: "mp4" または "gif"
        smooth_target_fps: 指定すると補間してこの FPS の滑らかな mp4 を生成（fmt='mp4' のみ）。
                           None なら補間しない。
        smooth_method: "rife" (ニューラル補間・推奨) または "minterpolate" (ffmpeg)
        smooth_mode: minterpolate のモード（smooth_method="minterpolate" 時のみ意味あり）
                     - "mci": 動き補正補間（シャープだが大きい動きで歪む）
                     - "blend": フレームブレンド（少しぼやけるが破綻しない・アニメ向き）
                     - "dup": フレーム複製（補間効果なし・カクカクのまま）
    """
    # PIL Image に統一
    pil_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            pil_frames.append(f.convert("RGB"))
        elif isinstance(f, np.ndarray):
            arr = f
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(arr).convert("RGB"))
        else:
            raise TypeError(f"Unsupported frame type: {type(f)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if fmt == "gif":
        if not output_path.lower().endswith(".gif"):
            output_path = os.path.splitext(output_path)[0] + ".gif"
        _write_gif(pil_frames, output_path, fps, optimize=False)
    else:  # mp4
        if not output_path.lower().endswith(".mp4"):
            output_path = os.path.splitext(output_path)[0] + ".mp4"

        frames_arr = np.stack([np.array(f) for f in pil_frames])

        # 試行順:
        # (1) pyav 直書き（pyav 17.x が入っている環境）
        # (2) imageio.v3 の ffmpeg バックエンド（imageio-ffmpeg が入っている環境）
        # (3) diffusers.utils.export_to_video
        # (4) gif フォールバック
        last_err = None
        write_ok = False

        # (1) pyav
        try:
            import av
            with av.open(output_path, mode="w") as container:
                stream = container.add_stream("h264", rate=int(fps))
                stream.width = int(frames_arr.shape[2])
                stream.height = int(frames_arr.shape[1])
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20"}
                for frame in frames_arr:
                    av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                # flush
                for packet in stream.encode():
                    container.mux(packet)
            write_ok = True
        except Exception as e:
            last_err = e
            print(f"  pyav による mp4 書き込み失敗 → imageio を試行: {str(e)[:200]}")

        # (2) imageio.v3 + ffmpeg backend
        if not write_ok:
            try:
                import imageio.v3 as iio
                iio.imwrite(
                    output_path,
                    frames_arr,
                    fps=int(fps),
                    codec="libx264",
                    plugin="FFMPEG",
                    output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
                )
                write_ok = True
            except Exception as e:
                print(f"  imageio による mp4 書き込み失敗 → diffusers を試行: {str(e)[:200]}")

                # (3) diffusers fallback
                try:
                    from diffusers.utils import export_to_video
                    export_to_video(pil_frames, output_path, fps=int(fps))
                    write_ok = True
                except Exception as e2:
                    # (4) gif fallback
                    print(f"mp4 保存失敗 ({last_err or e}, {e2})、gif にフォールバック")
                    gif_path = os.path.splitext(output_path)[0] + ".gif"
                    _write_gif(pil_frames, gif_path, fps)
                    output_path = gif_path

    # 動き補間（mp4 で生成元 FPS より高い目標 FPS が指定されたとき）
    # ※ どのライターで保存しても最後に一度だけ実行されるよう、return を後回しにしてここで補間処理を行う
    if (
        fmt == "mp4"
        and smooth_target_fps
        and int(smooth_target_fps) > int(fps)
        and output_path.lower().endswith(".mp4")
    ):
        if smooth_method == "rife":
            from utils.rife_interpolation import interpolate_video_rife
            suffix = f"_smooth{int(smooth_target_fps)}_RIFE.mp4"
            smoothed_path = output_path.replace(".mp4", suffix)
            print(f"  [Smooth] RIFE 補間開始 ({fps}fps -> {smooth_target_fps}fps)")
            result_path = interpolate_video_rife(
                output_path, smoothed_path,
                target_fps=int(smooth_target_fps),
            )
        else:
            from utils.video_interpolation import interpolate_video_fps
            suffix = f"_smooth{int(smooth_target_fps)}_{smooth_mode}.mp4"
            smoothed_path = output_path.replace(".mp4", suffix)
            print(f"  [Smooth] minterpolate 補間開始 ({fps}fps -> {smooth_target_fps}fps, mode={smooth_mode})")
            result_path = interpolate_video_fps(
                output_path, smoothed_path,
                target_fps=int(smooth_target_fps),
                mode=smooth_mode,
            )

        if result_path != output_path:
            print(f"  [Smooth] 補間動画を保存: {result_path}")
            return result_path
        else:
            print("  [Smooth] 補間に失敗したため元動画を返却")

    return output_path


def _make_generator(seed: int) -> Tuple[torch.Generator, int]:
    if seed is None or seed < 0:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return gen, int(seed)


def _finalize_video(
    frames: list,
    out_dir: str,
    prefix: str,
    used_seed,
    fps: int,
    fmt: str,
    smooth_target_fps: Optional[int],
    smooth_mode: str,
    smooth_method: str,
) -> Tuple[str, str]:
    """保存 + キャッシュ解放 + 補間タグ生成の共通エピローグ

    Returns:
        (out_path, smooth_tag)
    """
    out_path = os.path.join(out_dir, f"{prefix}_seed{used_seed}.{fmt}")
    out_path = _save_video(frames, out_path, fps=fps, fmt=fmt, smooth_target_fps=smooth_target_fps, smooth_mode=smooth_mode, smooth_method=smooth_method)

    gc.collect()
    video_pipeline_manager.clear_cache()

    smooth_tag = f" -> {smooth_target_fps}fps補間" if smooth_target_fps and smooth_target_fps > fps else ""
    return out_path, smooth_tag


def _normalize_image_input(image: Union[Image.Image, str, np.ndarray], width: int, height: int) -> Image.Image:
    """str/ndarray/PIL の入力画像を RGB の PIL Image に正規化してリサイズ"""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    return image.resize((int(width), int(height)))


# =============================================================================
# AnimateDiff (text → video, SD1.5)
# =============================================================================

def generate_animatediff(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 16,
    fps: int = 8,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    custom_base_path: Optional[str] = None,
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    # ネガティブ未指定 or 短すぎる場合は専用ネガに差し替え（Wan2.1 と同パターン）。
    # AnimateDiff は SD1.5 系で顔・手の崩壊が一番の弱点なので、重み付き構文で強めに抑える。
    if not negative_prompt or len(negative_prompt.strip()) < 30:
        negative_prompt = VIDEO_ANIMATEDIFF_NEGATIVE

    try:
        pipe = video_pipeline_manager.load_animatediff(custom_base_path=custom_base_path)
        gen, used_seed = _make_generator(seed)

        out_dir = output_dir or create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_animatediff", prompt[:30], category="video")

        # FreeNoise: AnimateDiff の motion adapter は 16 フレームまで訓練されている。
        # それ以上のフレーム数を生成する場合は FreeNoise (sliding window) を有効化して
        # 連続性を保つ。16 以下のときは無効化（純粋な短尺生成）。
        free_noise_threshold = VIDEO_MODELS.get("animatediff", {}).get("free_noise_threshold", 16)
        free_noise_enabled = False
        if int(num_frames) > free_noise_threshold and hasattr(pipe, "enable_free_noise"):
            try:
                pipe.enable_free_noise(context_length=16, context_stride=4)
                free_noise_enabled = True
                print(f"  [FreeNoise] {num_frames}フレーム生成のため context_length=16, stride=4 で有効化")
            except Exception as fn_e:
                print(f"  [FreeNoise] 有効化失敗: {fn_e} -> そのまま続行")
        else:
            # 念のため毎回 disable しておく（前回 enable 状態が残っていると挙動が変わる）
            if hasattr(pipe, "disable_free_noise"):
                try:
                    pipe.disable_free_noise()
                except Exception:
                    pass

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_frames=int(num_frames),
                width=int(width),
                height=int(height),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=gen,
            )
        finally:
            # 次回生成（短尺など）に影響しないよう必ず無効化
            if free_noise_enabled and hasattr(pipe, "disable_free_noise"):
                try:
                    pipe.disable_free_noise()
                except Exception:
                    pass

        frames = result.frames[0]
        prefix = filename_prefix or "animatediff"
        out_path, smooth_tag = _finalize_video(frames, out_dir, prefix, used_seed, fps, fmt, smooth_target_fps, smooth_mode, smooth_method)

        msg = (
            f"AnimateDiff 生成完了: {len(frames)}フレーム / {fps}fps{smooth_tag}"
            f"{' [FreeNoise]' if free_noise_enabled else ''}\n"
            f"Seed: {used_seed}\n保存先: {out_path}"
        )
        return out_path, msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"AnimateDiff 生成エラー: {e}"


# =============================================================================
# Stable Video Diffusion XT (image → video)
# =============================================================================

def generate_svd_xt(
    image: Union[Image.Image, str, np.ndarray],
    num_frames: int = 25,
    fps: int = 7,
    width: int = 1024,
    height: int = 576,
    num_inference_steps: int = 25,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 2,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    num_chunks: int = 1,
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """SVD-XT で image→video 生成

    SVD-XT は 25 フレーム固定で訓練されているため、それ以上は num_chunks で
    連続生成（前回の最終フレームを次回の入力に渡す）して連結する。
    結果は 25 × num_chunks フレーム。
    """
    if image is None:
        return None, "入力画像を指定してください"

    try:
        # 入力正規化
        image = _normalize_image_input(image, width, height)

        pipe = video_pipeline_manager.load_svd_xt()

        out_dir = output_dir or create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_svd", "img2vid", category="video")

        chunks = max(1, int(num_chunks))
        all_frames = []
        last_used_seed = None
        cur_image = image
        for i in range(chunks):
            # チャンクごとに seed を変えて多様性を出す（基本シードがあれば +i のオフセット）
            if seed is None or seed < 0:
                gen, used_seed = _make_generator(-1)
            else:
                gen, used_seed = _make_generator(int(seed) + i)
            last_used_seed = used_seed

            print(f"  [SVD-XT] chunk {i+1}/{chunks}  seed={used_seed}")
            result = pipe(
                image=cur_image,
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                min_guidance_scale=float(min_guidance_scale),
                max_guidance_scale=float(max_guidance_scale),
                motion_bucket_id=int(motion_bucket_id),
                noise_aug_strength=float(noise_aug_strength),
                decode_chunk_size=int(decode_chunk_size),
                generator=gen,
            )
            chunk_frames = result.frames[0]

            if i == 0:
                all_frames.extend(chunk_frames)
            else:
                # 連続性のため重複する初フレームを除いて結合
                all_frames.extend(chunk_frames[1:])

            # 次チャンクの入力 = 今回の最終フレーム
            if i < chunks - 1:
                last = chunk_frames[-1]
                if not isinstance(last, Image.Image):
                    last = Image.fromarray(np.array(last)).convert("RGB")
                cur_image = last.resize((int(width), int(height)))

        prefix = filename_prefix or "svd_xt"
        out_path, smooth_tag = _finalize_video(all_frames, out_dir, prefix, f"{last_used_seed}_x{chunks}", fps, fmt, smooth_target_fps, smooth_mode, smooth_method)

        msg = (
            f"SVD-XT 生成完了: {len(all_frames)}フレーム / {fps}fps{smooth_tag}"
            f"{f' (連続生成 x{chunks})' if chunks > 1 else ''}\n"
            f"Seed: {last_used_seed}\n保存先: {out_path}"
        )
        return out_path, msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"SVD-XT 生成エラー: {e}"


# =============================================================================
# LTX-Video (text→video / image→video)
# =============================================================================

def generate_ltx_video(
    prompt: str,
    negative_prompt: str = "",
    image: Optional[Union[Image.Image, str, np.ndarray]] = None,
    num_frames: int = 65,
    fps: int = 24,
    width: int = 704,
    height: int = 480,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.0,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    try:
        mode = "i2v" if image is not None else "t2v"
        pipe = video_pipeline_manager.load_ltx_video(mode=mode)
        gen, used_seed = _make_generator(seed)

        out_dir = output_dir or create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_ltx_{mode}", prompt[:30], category="video")

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "worst quality, blurry",
            num_frames=int(num_frames),
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen,
        )

        if image is not None:
            kwargs["image"] = _normalize_image_input(image, width, height)

        result = pipe(**kwargs)
        frames = result.frames[0]

        prefix = filename_prefix or f"ltx_{mode}"
        out_path, smooth_tag = _finalize_video(frames, out_dir, prefix, used_seed, fps, fmt, smooth_target_fps, smooth_mode, smooth_method)

        msg = (
            f"LTX-Video 生成完了 ({mode}): {len(frames)}フレーム / {fps}fps{smooth_tag}\n"
            f"Seed: {used_seed}\n保存先: {out_path}"
        )
        return out_path, msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"LTX-Video 生成エラー: {e}"


# =============================================================================
# CogVideoX-2B (text → video)
# =============================================================================

def generate_cogvideox(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 49,
    fps: int = 8,
    width: int = 720,
    height: int = 480,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    try:
        pipe = video_pipeline_manager.load_cogvideox()
        gen, used_seed = _make_generator(seed)

        out_dir = output_dir or create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_cogvideox", prompt[:30], category="video")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen,
        )
        frames = result.frames[0]

        prefix = filename_prefix or "cogvideox"
        out_path, smooth_tag = _finalize_video(frames, out_dir, prefix, used_seed, fps, fmt, smooth_target_fps, smooth_mode, smooth_method)

        msg = (
            f"CogVideoX-2B 生成完了: {len(frames)}フレーム / {fps}fps{smooth_tag}\n"
            f"Seed: {used_seed}\n保存先: {out_path}"
        )
        return out_path, msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"CogVideoX-2B 生成エラー: {e}"


# =============================================================================
# Wan2.1 1.3B (text → video)
# =============================================================================

def generate_wan21(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 81,
    fps: int = 16,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    smooth_target_fps: Optional[int] = None,
    smooth_mode: str = "mci",
    smooth_method: str = "rife",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    # Wan は標準ネガでは砂嵐になるケースがあるので、未指定または短すぎるなら公式推奨に置換。
    if not negative_prompt or len(negative_prompt.strip()) < 30:
        negative_prompt = VIDEO_WAN_NEGATIVE

    try:
        pipe = video_pipeline_manager.load_wan21()
        gen, used_seed = _make_generator(seed)

        out_dir = output_dir or create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_wan21", prompt[:30], category="video")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=int(num_frames),
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen,
        )
        frames = result.frames[0]

        prefix = filename_prefix or "wan21"
        out_path, smooth_tag = _finalize_video(frames, out_dir, prefix, used_seed, fps, fmt, smooth_target_fps, smooth_mode, smooth_method)

        msg = (
            f"Wan2.1 1.3B 生成完了: {len(frames)}フレーム / {fps}fps{smooth_tag}\n"
            f"Seed: {used_seed}\n保存先: {out_path}"
        )
        return out_path, msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Wan2.1 生成エラー: {e}"


# =============================================================================
# 統合エントリポイント
# =============================================================================

def generate_video(
    model_key: str,
    prompt: str = "",
    negative_prompt: str = "",
    image: Optional[Union[Image.Image, str, np.ndarray]] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: int = -1,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    **kwargs,
) -> Tuple[Optional[str], str]:
    """統合動画生成関数

    Args:
        model_key: VIDEO_MODELS のキー
        prompt: テキストプロンプト
        negative_prompt: ネガティブプロンプト
        image: 画像入力（SVD/LTX i2v）
        num_frames, fps, width, height, num_inference_steps, guidance_scale: モデル別パラメータ（None=デフォルト）
        seed: シード（-1=ランダム）
        fmt: 出力フォーマット ("mp4" or "gif")
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return None, f"未知のモデル: {model_key}"

    # デフォルト値の埋め込み
    num_frames = num_frames if num_frames is not None else cfg["default_frames"]
    fps = fps if fps is not None else cfg["default_fps"]
    width = width if width is not None else cfg["default_width"]
    height = height if height is not None else cfg["default_height"]
    num_inference_steps = num_inference_steps if num_inference_steps is not None else cfg["default_steps"]
    guidance_scale = guidance_scale if guidance_scale is not None else cfg["default_guidance"]

    common = dict(
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        seed=seed,
        fmt=fmt,
        smooth_target_fps=kwargs.get("smooth_target_fps"),
        smooth_mode=kwargs.get("smooth_mode", "mci"),
        smooth_method=kwargs.get("smooth_method", "rife"),
        output_dir=kwargs.get("output_dir"),
        filename_prefix=kwargs.get("filename_prefix"),
    )

    if model_key == "animatediff":
        return generate_animatediff(
            prompt=prompt, negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            custom_base_path=kwargs.get("custom_base_path"),
            **common,
        )
    if model_key == "svd_xt":
        return generate_svd_xt(
            image=image,
            motion_bucket_id=kwargs.get("motion_bucket_id", 127),
            noise_aug_strength=kwargs.get("noise_aug_strength", 0.02),
            decode_chunk_size=kwargs.get("decode_chunk_size", 2),
            min_guidance_scale=kwargs.get("min_guidance_scale", 1.0),
            max_guidance_scale=guidance_scale,
            num_chunks=kwargs.get("num_chunks", 1),
            **common,
        )
    if model_key == "ltx_video":
        return generate_ltx_video(
            prompt=prompt, negative_prompt=negative_prompt,
            image=image, guidance_scale=guidance_scale, **common,
        )
    if model_key == "cogvideox_2b":
        return generate_cogvideox(
            prompt=prompt, negative_prompt=negative_prompt,
            guidance_scale=guidance_scale, **common,
        )
    if model_key == "wan21_1_3b":
        return generate_wan21(
            prompt=prompt, negative_prompt=negative_prompt,
            guidance_scale=guidance_scale, **common,
        )

    return None, f"未対応のモデル: {model_key}"


# 動画生成のプロンプト例
VIDEO_PROMPT_EXAMPLES = {
    "[風景] 海の波": "ocean waves crashing on a sandy beach, slow motion, sunset lighting, cinematic",
    "[風景] 桜": "cherry blossom petals falling in slow motion, spring breeze, japanese garden",
    "[風景] 滝": "majestic waterfall in lush forest, mist rising, sunlight through trees",
    "[風景] 雪山": "snowy mountain peaks, clouds drifting, golden hour, drone shot",
    "[人物] 歩く女性": "a woman walking through a busy city street, cinematic, shallow depth of field",
    "[人物] ダンス": "a dancer performing on stage, dynamic movement, dramatic stage lighting",
    "[動物] 走る犬": "a golden retriever running through a meadow, slow motion, sunny day",
    "[動物] 飛ぶ鳥": "a flock of birds flying across a colorful sunset sky, slow motion",
    "[ファンタジー] 魔法": "magical glowing particles swirling, fantasy atmosphere, dark background",
    "[ファンタジー] ドラゴン": "a dragon flying over a medieval castle, epic fantasy, dramatic clouds",
    "[サイバーパンク] 街": "neon-lit cyberpunk city at night, rain reflections, futuristic vehicles",
    "[アニメ] 走るキャラ": "anime style girl running through a flower field, ghibli inspired, sunlight",
    "[宇宙] 銀河": "spiral galaxy slowly rotating, deep space, stars twinkling, cosmic",
    "[抽象] 流体": "abstract colorful fluid simulation, ink in water, smooth motion",
    "[料理] パン": "freshly baked bread steaming, close-up, warm lighting, appetizing",
}
