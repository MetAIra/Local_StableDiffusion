"""X/Y/Zプロット生成モジュール

同じseedで複数のパラメータを変化させながら比較画像を生成
"""
import os
import gc
import torch
from PIL import Image, ImageDraw, ImageFont

from config import DEVICE, PROMPT_PREFIX, PROMPT_PREFIX_SDXL, is_sdxl_model, clear_stop, is_stop_requested
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.long_prompt import encode_prompt_long, encode_prompt_long_sdxl, needs_long_encoding
from .manager import pipeline_manager


# 対応するパラメータ
PLOT_PARAMETERS = {
    "なし": None,
    "CFG Scale": "cfg_scale",
    "Steps": "steps",
    "Seed": "seed",
    "Width": "width",
    "Height": "height",
    "LoRA 1 Weight": "lora1_weight",
    "LoRA 2 Weight": "lora2_weight",
    "LoRA 3 Weight": "lora3_weight",
}


def parse_values(value_str: str, param_type: str) -> list:
    """カンマ区切りの値をパース

    形式:
    - 単純なカンマ区切り: 1, 2, 3, 4
    - 範囲指定: 1-10:2 (1から10まで2刻み)
    """
    if not value_str or not value_str.strip():
        return [None]

    values = []
    parts = [p.strip() for p in value_str.split(",")]

    for part in parts:
        if "-" in part and ":" in part:
            # 範囲指定: start-end:step
            try:
                range_part, step_str = part.split(":")
                start_str, end_str = range_part.split("-")
                start = float(start_str)
                end = float(end_str)
                step = float(step_str)
                if step <= 0:  # ステップ0以下は無限ループになるためスキップ
                    continue

                current = start
                while current <= end + 0.0001:  # 浮動小数点誤差対策
                    if param_type in ["steps", "seed", "width", "height"]:
                        values.append(int(current))
                    else:
                        values.append(round(current, 2))
                    current += step
            except ValueError:
                continue
        else:
            # 単純な値
            try:
                val = float(part)
                if param_type in ["steps", "seed", "width", "height"]:
                    values.append(int(val))
                else:
                    values.append(round(val, 2))
            except ValueError:
                continue

    return values if values else [None]


def create_grid_image(images: list[list[Image.Image]],
                     x_labels: list[str],
                     y_labels: list[str],
                     x_param_name: str,
                     y_param_name: str) -> Image.Image:
    """画像グリッドを作成

    Args:
        images: 2D配列 [y][x] の画像リスト
        x_labels: X軸のラベル
        y_labels: Y軸のラベル
        x_param_name: X軸パラメータ名
        y_param_name: Y軸パラメータ名
    """
    if not images or not images[0]:
        return None

    # 画像サイズを取得
    img_width = images[0][0].width
    img_height = images[0][0].height

    # ラベル用のマージン
    label_margin_top = 60  # 上部のラベル用
    label_margin_left = 80  # 左側のラベル用

    # グリッドサイズを計算
    grid_width = label_margin_left + img_width * len(x_labels)
    grid_height = label_margin_top + img_height * len(y_labels)

    # グリッド画像を作成
    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)

    # フォントの設定（デフォルトフォントを使用）
    try:
        font_small = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font_small = ImageFont.load_default()

    # X軸ラベルを描画
    for i, label in enumerate(x_labels):
        x = label_margin_left + i * img_width + img_width // 2
        y = 10
        text = f"{x_param_name}={label}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_width = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            text_width = len(text) * 6
        draw.text((x - text_width // 2, y), text, fill="black", font=font_small)

    # Y軸ラベルを描画
    for j, label in enumerate(y_labels):
        x = 5
        y = label_margin_top + j * img_height + img_height // 2
        text = f"{y_param_name}={label}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_height = bbox[3] - bbox[1]
        except (AttributeError, TypeError):
            text_height = 12
        draw.text((x, y - text_height // 2), text, fill="black", font=font_small)

    # 画像を配置
    for j, row in enumerate(images):
        for i, img in enumerate(row):
            x = label_margin_left + i * img_width
            y = label_margin_top + j * img_height
            grid.paste(img, (x, y))

    return grid


def generate_xyz_plot(
    prompt: str,
    negative_prompt: str,
    base_seed: int,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0,
    x_param: str = "なし",
    x_values: str = "",
    y_param: str = "なし",
    y_values: str = "",
    z_param: str = "なし",
    z_values: str = ""
) -> tuple[list[Image.Image], str]:
    """X/Y/Zプロット画像を生成

    同じseedで異なるパラメータ値を使用して比較画像を生成
    """
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    # パラメータ値をパース
    x_param_key = PLOT_PARAMETERS.get(x_param)
    y_param_key = PLOT_PARAMETERS.get(y_param)
    z_param_key = PLOT_PARAMETERS.get(z_param)

    x_vals = parse_values(x_values, x_param_key) if x_param_key else [None]
    y_vals = parse_values(y_values, y_param_key) if y_param_key else [None]
    z_vals = parse_values(z_values, z_param_key) if z_param_key else [None]

    # 有効なパラメータがない場合
    if x_vals == [None] and y_vals == [None] and z_vals == [None]:
        return None, "少なくとも1つの軸にパラメータと値を設定してください"

    # 総生成枚数を計算
    total_images = len([v for v in x_vals if v is not None] or [1]) * \
                   len([v for v in y_vals if v is not None] or [1]) * \
                   len([v for v in z_vals if v is not None] or [1])

    print(f"X/Y/Z Plot: Total images to generate: {total_images}")
    print(f"X: {x_param} = {x_vals}")
    print(f"Y: {y_param} = {y_vals}")
    print(f"Z: {z_param} = {z_vals}")

    # 出力フォルダを作成
    base_path = os.path.dirname(os.path.dirname(__file__))
    output_dir = create_output_dir(base_path, "xyz_plot", prompt)

    # SDXLモデルかどうかを判定
    is_sdxl = is_sdxl_model(model_name)

    # Negative promptの設定（SDXLの場合はEasyNegativeを追加しない）
    if is_sdxl:
        full_negative = negative_prompt
    else:
        full_negative = negative_prompt
        if "EasyNegative" not in negative_prompt:
            full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

    # プロンプトにプレフィックスを追加（SDXLの場合は別のプレフィックス）
    if is_sdxl:
        full_prompt = PROMPT_PREFIX_SDXL + prompt if prompt else PROMPT_PREFIX_SDXL
    else:
        full_prompt = PROMPT_PREFIX + prompt if prompt else PROMPT_PREFIX

    # 結果を格納
    all_grids = []
    csv_rows = []  # CSV用のパラメータリスト
    generated_count = 0
    stopped = False

    clear_stop()

    # Z軸でループ
    for z_idx, z_val in enumerate(z_vals):
        if is_stop_requested():
            stopped = True
            break

        # 2D画像配列 [y][x]
        grid_images = []

        # Y軸でループ
        for y_idx, y_val in enumerate(y_vals):
            if is_stop_requested():
                stopped = True
                break

            row_images = []

            # X軸でループ
            for x_idx, x_val in enumerate(x_vals):
                if is_stop_requested():
                    stopped = True
                    break
                # 現在のパラメータを設定
                current_params = {
                    "cfg_scale": float(guidance_scale),
                    "steps": int(num_inference_steps),
                    "seed": int(base_seed),
                    "width": int(width),
                    "height": int(height),
                    "lora1_weight": float(weight1),
                    "lora2_weight": float(weight2),
                    "lora3_weight": float(weight3),
                }

                # X/Y/Zの値を適用
                if x_param_key and x_val is not None:
                    current_params[x_param_key] = x_val
                if y_param_key and y_val is not None:
                    current_params[y_param_key] = y_val
                if z_param_key and z_val is not None:
                    current_params[z_param_key] = z_val

                # LoRA設定を作成
                lora_configs = [
                    (lora1, current_params["lora1_weight"]),
                    (lora2, current_params["lora2_weight"]),
                    (lora3, current_params["lora3_weight"])
                ]

                # パイプラインを取得（スケジューラ指定）
                pipeline = pipeline_manager.get_txt2img_pipeline_with_loras(
                    vae_name, model_name, lora_configs, scheduler_name
                )

                if pipeline_manager.lora_error:
                    return None, f"LoRAエラー: {pipeline_manager.lora_error}"

                # ジェネレータを作成
                generator = torch.Generator(DEVICE).manual_seed(current_params["seed"])

                # Long Prompt対応
                use_long = needs_long_encoding(pipeline, full_prompt, full_negative)
                if use_long:
                    if is_sdxl:
                        pe, npe, ppe, nppe = encode_prompt_long_sdxl(pipeline, full_prompt, full_negative, DEVICE)
                        gen_kwargs = dict(
                            prompt_embeds=pe,
                            negative_prompt_embeds=npe,
                            pooled_prompt_embeds=ppe,
                            negative_pooled_prompt_embeds=nppe,
                            width=current_params["width"],
                            height=current_params["height"],
                            num_inference_steps=current_params["steps"],
                            guidance_scale=current_params["cfg_scale"],
                            generator=generator,
                        )
                    else:
                        pe, npe = encode_prompt_long(pipeline, full_prompt, full_negative, DEVICE)
                        gen_kwargs = dict(
                            prompt_embeds=pe,
                            negative_prompt_embeds=npe,
                            width=current_params["width"],
                            height=current_params["height"],
                            num_inference_steps=current_params["steps"],
                            guidance_scale=current_params["cfg_scale"],
                            generator=generator,
                        )
                    image = pipeline(**gen_kwargs).images[0]
                else:
                    # 画像を生成
                    image = pipeline(
                        prompt=full_prompt,
                        negative_prompt=full_negative,
                        width=current_params["width"],
                        height=current_params["height"],
                        num_inference_steps=current_params["steps"],
                        guidance_scale=current_params["cfg_scale"],
                        generator=generator,
                    ).images[0]

                row_images.append(image)
                generated_count += 1

                # 個別画像を保存
                params_str = f"x{x_idx}_y{y_idx}_z{z_idx}"
                filename = f"image_{params_str}_seed{current_params['seed']}.png"
                filepath = os.path.join(output_dir, filename)
                save_image_safe(image, filepath)

                # CSV用のパラメータを記録
                csv_rows.append({
                    "filename": filename,
                    "prompt": prompt,
                    "full_prompt": full_prompt,
                    "negative_prompt": negative_prompt,
                    "seed": current_params["seed"],
                    "width": current_params["width"],
                    "height": current_params["height"],
                    "steps": current_params["steps"],
                    "cfg_scale": current_params["cfg_scale"],
                    "x_param": x_param,
                    "x_value": x_val,
                    "y_param": y_param,
                    "y_value": y_val,
                    "z_param": z_param,
                    "z_value": z_val,
                    "model": model_name,
                    "vae": vae_name,
                    "scheduler": scheduler_name,
                    "lora1": lora1,
                    "lora1_weight": current_params["lora1_weight"],
                    "lora2": lora2,
                    "lora2_weight": current_params["lora2_weight"],
                    "lora3": lora3,
                    "lora3_weight": current_params["lora3_weight"]
                })

                print(f"Generated {generated_count}/{total_images}: "
                      f"X={x_val}, Y={y_val}, Z={z_val}")

                # メモリ解放
                pipeline_manager.clear_cache()
                gc.collect()

            if row_images:
                grid_images.append(row_images)

            if stopped:
                break

        # グリッド画像を作成
        x_labels = [str(v) if v is not None else "-" for v in x_vals]
        y_labels = [str(v) if v is not None else "-" for v in y_vals]

        x_name = x_param if x_param != "なし" else ""
        y_name = y_param if y_param != "なし" else ""

        if len(x_vals) > 1 or len(y_vals) > 1:
            grid = create_grid_image(grid_images, x_labels, y_labels, x_name, y_name)

            # グリッド画像を保存
            z_label = f"_z{z_val}" if z_val is not None else ""
            grid_filename = f"grid{z_label}.png"
            grid_filepath = os.path.join(output_dir, grid_filename)
            save_image_safe(grid, grid_filepath)

            all_grids.append(grid)
        else:
            # 単一画像の場合
            all_grids.extend([img for row in grid_images for img in row])

    # 中止チェック
    if stopped and generated_count == 0:
        return None, "生成が中止されました"

    # 生成パラメータをCSVに保存
    csv_path = save_batch_generation_params(output_dir, csv_rows)
    print(f"Generation params saved to: {csv_path}")

    z_info = f"\nZ軸: {z_param} = {z_vals}" if z_param != "なし" else ""
    if stopped:
        return all_grids, f"中止しました（{generated_count}/{total_images}枚生成済み）\n" \
                          f"X軸: {x_param} = {x_vals}\n" \
                          f"Y軸: {y_param} = {y_vals}" \
                          f"{z_info}\n" \
                          f"CSV: {csv_path}\n" \
                          f"保存先: {output_dir}"

    return all_grids, f"{total_images}枚の画像を生成しました\n" \
                      f"X軸: {x_param} = {x_vals}\n" \
                      f"Y軸: {y_param} = {y_vals}" \
                      f"{z_info}\n" \
                      f"保存先: {output_dir}"
