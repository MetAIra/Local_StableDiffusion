"""動画バッチ生成モジュール

3種類のバッチ生成を提供:
- Seed バッチ: 同じプロンプトで N 個の seed バリエーション
- Variable Prompt: 固定 + テンプレート + 変数定義 で全組み合わせ
- X/Y/Z Plot: 軸ごとにパラメータを変化させて格子状に生成

すべて pipelines.video_gen.generate_video() の上に乗っているので、
モデル別の補間/FreeNoise/SVD連続生成などはそのまま継承される。
"""
import os
import gc
import re
import traceback
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT, VIDEO_OUTPUT_DIR_PREFIX,
    is_stop_requested, clear_stop,
)
from utils.batch_parsing import (
    parse_values, parse_variable_definitions, generate_combinations, save_metadata_csv
)
from utils.file import create_output_dir
from .video_gen import generate_video
from .video_manager import video_pipeline_manager

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 上限（暴走防止）
VIDEO_BATCH_LIMIT = 1000

# X/Y/Z Plot で変化させられるパラメータ
VIDEO_PLOT_PARAMETERS = {
    "なし": None,
    "Seed": "seed",
    "Guidance Scale": "guidance_scale",
    "Steps": "num_inference_steps",
    "Frames": "num_frames",
    "Width": "width",
    "Height": "height",
}

# 整数として扱うパラメータ
_INT_PARAMS = {"seed", "num_inference_steps", "num_frames", "width", "height"}


# Variable Prompt 用テンプレート例
VIDEO_VARIABLE_PROMPT_EXAMPLES = {
    "動物 × 場所": {
        "fixed": "cinematic, masterpiece, high quality, 4k",
        "template": "a {animal} in {place}, walking",
        "variables": (
            "animal: cat, dog, fox, rabbit\n"
            "place: forest, beach, mountain"
        ),
    },
    "天気 × 時間帯": {
        "fixed": "cinematic landscape, beautiful, detailed",
        "template": "{weather} {time}, atmospheric",
        "variables": (
            "weather: sunny, rainy, snowy, foggy\n"
            "time: morning, evening, night"
        ),
    },
    "キャラクター × アクション": {
        "fixed": "anime style, masterpiece, dynamic",
        "template": "a {character} {action}, side view",
        "variables": (
            "character: warrior, mage, archer\n"
            "action: running, jumping, fighting"
        ),
    },
    "車 × 環境": {
        "fixed": "cinematic, ultra realistic, motion blur",
        "template": "a {car} driving through {env}",
        "variables": (
            "car: sports car, motorcycle, truck\n"
            "env: tunnel, city street, highway, mountain road"
        ),
    },
    "ファンタジー風景": {
        "fixed": "fantasy art, ethereal, magical",
        "template": "a {creature} flying over {terrain}, {weather}",
        "variables": (
            "creature: dragon, phoenix, griffin\n"
            "terrain: castle, ocean, forest, volcano\n"
            "weather: stormy, sunny, foggy"
        ),
    },
}


# =============================================================================
# 1. Seed バッチ
# =============================================================================

def generate_video_seed_batch(
    model_key: str,
    prompt: str = "",
    negative_prompt: str = "",
    image: Optional[Union[Image.Image, str, np.ndarray]] = None,
    num_videos: int = 4,
    base_seed: int = -1,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    custom_base_path: Optional[str] = None,
    smooth_target_fps: Optional[int] = None,
    smooth_method: str = "rife",
    smooth_mode: str = "mci",
    num_chunks: int = 1,
) -> tuple[list[str], str]:
    """同じプロンプトで N 個の seed 違い動画を生成

    Args:
        model_key: VIDEO_MODELS のキー
        prompt: テキストプロンプト
        num_videos: 生成本数（1〜VIDEO_BATCH_LIMIT）
        base_seed: 基準 seed。-1 の場合はランダム seed を毎回振る。
                   それ以外なら base_seed, base_seed+1, base_seed+2 ... と連番
        その他: generate_video と同じ

    Returns:
        (生成された動画パスのリスト, ステータスメッセージ)
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return [], f"未知のモデル: {model_key}"

    num_videos = max(1, min(int(num_videos), VIDEO_BATCH_LIMIT))
    label_prompt = (prompt or "img2vid")[:30]
    out_dir = create_output_dir(_PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_batch_seed_{model_key}", label_prompt, category="video")

    print(f"[Seed バッチ] {cfg['label']}  本数={num_videos}  base_seed={base_seed}")

    csv_rows = []
    output_paths: list[str] = []
    generated = 0
    stopped = False
    clear_stop()

    for i in range(num_videos):
        if is_stop_requested():
            stopped = True
            break

        cur_seed = -1 if int(base_seed) < 0 else int(base_seed) + i
        prefix = f"v{i:03d}"

        try:
            out_path, msg = generate_video(
                model_key=model_key,
                prompt=prompt, negative_prompt=negative_prompt,
                image=image,
                num_frames=num_frames, fps=fps, width=width, height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=cur_seed, fmt=fmt,
                custom_base_path=custom_base_path,
                smooth_target_fps=smooth_target_fps,
                smooth_method=smooth_method,
                smooth_mode=smooth_mode,
                num_chunks=num_chunks,
                output_dir=out_dir,
                filename_prefix=prefix,
            )
        except Exception as e:
            traceback.print_exc()
            return output_paths, f"生成中にエラー: {e}\n（{generated}/{num_videos}件完了）"

        if out_path and os.path.exists(out_path):
            output_paths.append(out_path)
            csv_rows.append({
                "filename": os.path.basename(out_path),
                "model": model_key,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": cur_seed,
                "num_frames": num_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "smooth_target_fps": smooth_target_fps,
                "smooth_method": smooth_method,
            })
            generated += 1
            print(f"  [{generated}/{num_videos}] {os.path.basename(out_path)}")
        else:
            print(f"  [{i+1}/{num_videos}] FAIL: {msg[:200]}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    if csv_rows:
        save_metadata_csv(csv_path, csv_rows, list(csv_rows[0].keys()))

    head = "中止" if stopped else "完了"
    msg = (
        f"Seed バッチ {head}（{generated}/{num_videos}件）\n"
        f"モデル: {cfg['label']}\n"
        f"CSV: {csv_path if csv_rows else '(なし)'}\n"
        f"保存先: {out_dir}"
    )
    return output_paths, msg


# =============================================================================
# 2. Variable Prompt
# =============================================================================

def generate_video_variable_prompt(
    model_key: str,
    fixed_prompt: str,
    variable_template: str,
    variable_definitions: str,
    negative_prompt: str = "",
    image: Optional[Union[Image.Image, str, np.ndarray]] = None,
    num_seed_variations: int = 1,
    base_seed: int = -1,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    fmt: str = VIDEO_DEFAULT_FORMAT,
    custom_base_path: Optional[str] = None,
    smooth_target_fps: Optional[int] = None,
    smooth_method: str = "rife",
    smooth_mode: str = "mci",
    num_chunks: int = 1,
) -> tuple[list[str], str]:
    """変数プロンプトで動画をバッチ生成

    Args:
        fixed_prompt: 全動画に共通する固定プロンプト
        variable_template: {var} 形式のテンプレート（例: "a {animal} in {place}"）
        variable_definitions: 変数定義（例: "animal: cat, dog\\nplace: forest, beach"）
        num_seed_variations: 各組み合わせあたり何個の seed で生成するか
        base_seed: ベースシード
        その他: generate_video と同じ

    Returns:
        (生成された動画パスのリスト, ステータスメッセージ)
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return [], f"未知のモデル: {model_key}"

    if not fixed_prompt.strip() and not variable_template.strip():
        return [], "固定プロンプトまたは変数テンプレートを入力してください"

    variables = parse_variable_definitions(variable_definitions)
    if not variables:
        return [], "変数が定義されていません。形式: 変数名: 値1, 値2, 値3"

    # テンプレート内の変数をチェック
    template_vars = set(re.findall(r"\{(\w+)\}", variable_template))
    undefined = template_vars - set(variables.keys())
    if undefined:
        return [], f"未定義の変数があります: {', '.join(undefined)}"

    combinations = list(generate_combinations(variables))
    total_combos = len(combinations)
    num_seeds = max(1, int(num_seed_variations))
    total = total_combos * num_seeds

    if total > VIDEO_BATCH_LIMIT:
        return [], (
            f"生成数が多すぎます: {total}件 (上限: {VIDEO_BATCH_LIMIT}件)\n"
            f"組み合わせ {total_combos} × Seed {num_seeds}"
        )

    print(f"[Variable Prompt] {cfg['label']}  {total_combos} 組合せ × {num_seeds} seeds = {total} 本")
    for k, v in variables.items():
        print(f"  {k}: {v}")

    label = (fixed_prompt or variable_template)[:30]
    out_dir = create_output_dir(
        _PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_batch_varprompt_{model_key}", label, category="video"
    )

    csv_rows = []
    output_paths: list[str] = []
    generated = 0
    stopped = False
    clear_stop()

    for combo_idx, combo in enumerate(combinations):
        if is_stop_requested():
            stopped = True
            break
        try:
            filled = variable_template.format(**combo) if variable_template.strip() else ""
        except KeyError as e:
            return [], f"テンプレートエラー: 変数 {e} が見つかりません"

        if fixed_prompt.strip() and filled.strip():
            full_prompt = fixed_prompt.strip() + ", " + filled.strip()
        elif fixed_prompt.strip():
            full_prompt = fixed_prompt.strip()
        else:
            full_prompt = filled.strip()

        for seed_offset in range(num_seeds):
            if is_stop_requested():
                stopped = True
                break

            cur_seed = -1 if int(base_seed) < 0 else int(base_seed) + seed_offset
            prefix = f"c{combo_idx:03d}_s{seed_offset:02d}"

            try:
                out_path, msg = generate_video(
                    model_key=model_key,
                    prompt=full_prompt, negative_prompt=negative_prompt,
                    image=image,
                    num_frames=num_frames, fps=fps, width=width, height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=cur_seed, fmt=fmt,
                    custom_base_path=custom_base_path,
                    smooth_target_fps=smooth_target_fps,
                    smooth_method=smooth_method,
                    smooth_mode=smooth_mode,
                    num_chunks=num_chunks,
                    output_dir=out_dir,
                    filename_prefix=prefix,
                )
            except Exception as e:
                traceback.print_exc()
                return output_paths, f"生成中にエラー: {e}\n（{generated}/{total}件完了）"

            if out_path and os.path.exists(out_path):
                output_paths.append(out_path)
                row = {
                    "filename": os.path.basename(out_path),
                    "model": model_key,
                    "fixed_prompt": fixed_prompt,
                    "variable_template": variable_template,
                    "full_prompt": full_prompt,
                    "negative_prompt": negative_prompt,
                    "seed": cur_seed,
                }
                # 変数値もCSVに含める
                for k, v in combo.items():
                    row[f"var_{k}"] = v
                row.update({
                    "num_frames": num_frames, "fps": fps,
                    "width": width, "height": height,
                    "steps": num_inference_steps, "guidance_scale": guidance_scale,
                    "smooth_target_fps": smooth_target_fps,
                    "smooth_method": smooth_method,
                })
                csv_rows.append(row)
                generated += 1
                print(f"  [{generated}/{total}] {os.path.basename(out_path)} ({combo})")
            else:
                print(f"  FAIL: {msg[:200]}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    if csv_rows:
        # 全行のキーをマージしたヘッダー
        all_keys = []
        seen = set()
        for r in csv_rows:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        save_metadata_csv(csv_path, csv_rows, all_keys)

    head = "中止" if stopped else "完了"
    msg = (
        f"Variable Prompt {head}（{generated}/{total}件）\n"
        f"モデル: {cfg['label']}  {total_combos}組合せ × {num_seeds}seeds\n"
        f"CSV: {csv_path if csv_rows else '(なし)'}\n"
        f"保存先: {out_dir}"
    )
    return output_paths, msg


# =============================================================================
# 3. X/Y/Z Plot
# =============================================================================

def generate_video_xyz_plot(
    model_key: str,
    prompt: str,
    negative_prompt: str = "",
    image: Optional[Union[Image.Image, str, np.ndarray]] = None,
    base_seed: int = 42,
    base_num_frames: Optional[int] = None,
    base_fps: Optional[int] = None,
    base_width: Optional[int] = None,
    base_height: Optional[int] = None,
    base_num_inference_steps: Optional[int] = None,
    base_guidance_scale: Optional[float] = None,
    x_param: str = "なし", x_values: str = "",
    y_param: str = "なし", y_values: str = "",
    z_param: str = "なし", z_values: str = "",
    fmt: str = VIDEO_DEFAULT_FORMAT,
    custom_base_path: Optional[str] = None,
    smooth_target_fps: Optional[int] = None,
    smooth_method: str = "rife",
    smooth_mode: str = "mci",
    num_chunks: int = 1,
) -> tuple[list[str], str]:
    """X/Y/Z 軸でパラメータを変化させて格子状に動画生成

    Returns:
        (動画パスのリスト, ステータスメッセージ)
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return [], f"未知のモデル: {model_key}"

    if not prompt.strip() and image is None:
        return [], "プロンプトまたは入力画像を指定してください"

    x_key = VIDEO_PLOT_PARAMETERS.get(x_param)
    y_key = VIDEO_PLOT_PARAMETERS.get(y_param)
    z_key = VIDEO_PLOT_PARAMETERS.get(z_param)

    x_vals = parse_values(x_values, x_key, _INT_PARAMS) if x_key else [None]
    y_vals = parse_values(y_values, y_key, _INT_PARAMS) if y_key else [None]
    z_vals = parse_values(z_values, z_key, _INT_PARAMS) if z_key else [None]

    if x_vals == [None] and y_vals == [None] and z_vals == [None]:
        return [], "少なくとも1つの軸にパラメータと値を設定してください"

    total = len(x_vals) * len(y_vals) * len(z_vals)
    if total > VIDEO_BATCH_LIMIT:
        return [], f"生成数が多すぎます: {total}件 (上限: {VIDEO_BATCH_LIMIT}件)"

    print(f"[X/Y/Z Plot] {cfg['label']}  total={total}")
    print(f"  X: {x_param} = {x_vals}")
    print(f"  Y: {y_param} = {y_vals}")
    print(f"  Z: {z_param} = {z_vals}")

    out_dir = create_output_dir(
        _PROJECT_ROOT, f"{VIDEO_OUTPUT_DIR_PREFIX}_batch_xyz_{model_key}", prompt[:30] or "img2vid", category="video"
    )

    csv_rows = []
    output_paths: list[str] = []
    generated = 0
    stopped = False
    clear_stop()

    for z_idx, z_val in enumerate(z_vals):
        if is_stop_requested(): stopped = True; break
        for y_idx, y_val in enumerate(y_vals):
            if is_stop_requested(): stopped = True; break
            for x_idx, x_val in enumerate(x_vals):
                if is_stop_requested(): stopped = True; break

                params = {
                    "seed": int(base_seed),
                    "num_frames": base_num_frames,
                    "fps": base_fps,
                    "width": base_width,
                    "height": base_height,
                    "num_inference_steps": base_num_inference_steps,
                    "guidance_scale": base_guidance_scale,
                }
                if x_key and x_val is not None: params[x_key] = x_val
                if y_key and y_val is not None: params[y_key] = y_val
                if z_key and z_val is not None: params[z_key] = z_val

                prefix = f"x{x_idx:02d}_y{y_idx:02d}_z{z_idx:02d}"

                try:
                    out_path, msg = generate_video(
                        model_key=model_key,
                        prompt=prompt, negative_prompt=negative_prompt,
                        image=image,
                        num_frames=params["num_frames"], fps=params["fps"],
                        width=params["width"], height=params["height"],
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        seed=int(params["seed"]), fmt=fmt,
                        custom_base_path=custom_base_path,
                        smooth_target_fps=smooth_target_fps,
                        smooth_method=smooth_method,
                        smooth_mode=smooth_mode,
                        num_chunks=num_chunks,
                        output_dir=out_dir,
                        filename_prefix=prefix,
                    )
                except Exception as e:
                    traceback.print_exc()
                    return output_paths, f"生成中にエラー: {e}\n（{generated}/{total}件完了）"

                if out_path and os.path.exists(out_path):
                    output_paths.append(out_path)
                    csv_rows.append({
                        "filename": os.path.basename(out_path),
                        "x_param": x_param, "x_value": x_val,
                        "y_param": y_param, "y_value": y_val,
                        "z_param": z_param, "z_value": z_val,
                        "seed": params["seed"],
                        "num_frames": params["num_frames"],
                        "fps": params["fps"],
                        "width": params["width"],
                        "height": params["height"],
                        "steps": params["num_inference_steps"],
                        "guidance_scale": params["guidance_scale"],
                        "model": model_key, "prompt": prompt,
                        "smooth_target_fps": smooth_target_fps,
                        "smooth_method": smooth_method,
                    })
                    generated += 1
                    print(f"  [{generated}/{total}] {os.path.basename(out_path)} (x={x_val}, y={y_val}, z={z_val})")
                else:
                    print(f"  FAIL: {msg[:200]}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    if csv_rows:
        save_metadata_csv(csv_path, csv_rows, list(csv_rows[0].keys()))

    head = "中止" if stopped else "完了"
    msg = (
        f"X/Y/Z Plot {head}（{generated}/{total}件）\n"
        f"モデル: {cfg['label']}\n"
        f"X軸: {x_param} = {x_vals}\n"
        f"Y軸: {y_param} = {y_vals}\n"
        f"Z軸: {z_param} = {z_vals}\n"
        f"CSV: {csv_path if csv_rows else '(なし)'}\n"
        f"保存先: {out_dir}"
    )
    return output_paths, msg
