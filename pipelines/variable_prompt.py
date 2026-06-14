"""Variable Prompt バッチ生成モジュール

固定プロンプト + 変数テンプレートで大量の画像を生成

使用例:
固定プロンプト: "(best quality:1.2), anime style, 1 boy"
変数テンプレート: "{age} years old NPC, {outfit_style}"
変数定義:
    age: 20, 25, 30, 35
    outfit_style: casual clothes, formal suit, military uniform
"""
import os
import gc
import re
import torch
from PIL import Image, ImageDraw, ImageFont

from config import DEVICE, PROMPT_PREFIX, PROMPT_PREFIX_SDXL, is_sdxl_model, clear_stop, is_stop_requested
# parse_variable_definitions / generate_combinations は ui/variable_prompt_tab.py が
# このモジュールから import しているため再エクスポートする
from utils.batch_parsing import parse_variable_definitions, generate_combinations, save_metadata_csv
from utils.file import create_output_dir, sanitize_filename, save_image_safe
from utils.long_prompt import encode_prompt_long, encode_prompt_long_sdxl, needs_long_encoding, get_token_count
from .manager import pipeline_manager


def create_variable_grid(
    images_data: list[tuple[dict, int, Image.Image]],
    group_name: str,
    x_var_name: str,
    x_values: list[str],
    y_label: str = "seed"
) -> Image.Image:
    """変数別のグリッド画像を作成

    Args:
        images_data: [(combo_dict, seed, image), ...] のリスト
        group_name: グループ名（ラベル用）
        x_var_name: X軸の変数名
        x_values: X軸の値リスト
        y_label: Y軸のラベル（通常は "seed"）
    """
    if not images_data:
        return None

    # 画像サイズを取得
    img_width = images_data[0][2].width
    img_height = images_data[0][2].height

    # Y軸（シード）の値を抽出
    seeds = sorted(set(seed for _, seed, _ in images_data))

    # 2D配列に整理 [y][x]
    grid_images = []
    for seed in seeds:
        row = []
        for x_val in x_values:
            # この組み合わせの画像を探す
            img = None
            for combo, s, image in images_data:
                if s == seed and combo.get(x_var_name) == x_val:
                    img = image
                    break
            if img:
                row.append(img)
        if row:
            grid_images.append(row)

    if not grid_images:
        return None

    # ラベル用のマージン
    label_margin_top = 80
    label_margin_left = 100

    # グリッドサイズを計算
    num_cols = len(x_values)
    num_rows = len(seeds)
    grid_width = label_margin_left + img_width * num_cols
    grid_height = label_margin_top + img_height * num_rows

    # グリッド画像を作成
    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)

    # フォントの設定
    try:
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_title = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font_small = ImageFont.load_default()
        font_title = font_small

    # タイトル
    title = f"Group: {group_name}"
    draw.text((10, 5), title, fill="black", font=font_title)

    # X軸ラベルを描画
    for i, x_val in enumerate(x_values):
        x = label_margin_left + i * img_width + img_width // 2
        y = 30
        text = f"{x_var_name}={x_val}"
        # 長いテキストは短縮
        if len(text) > 25:
            text = text[:22] + "..."
        try:
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_width = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            text_width = len(text) * 6
        draw.text((x - text_width // 2, y), text, fill="black", font=font_small)

    # Y軸ラベルを描画
    for j, seed in enumerate(seeds):
        x = 5
        y = label_margin_top + j * img_height + img_height // 2
        text = f"{y_label}={seed}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_height = bbox[3] - bbox[1]
        except (AttributeError, TypeError):
            text_height = 12
        draw.text((x, y - text_height // 2), text, fill="black", font=font_small)

    # 画像を配置
    for j, row in enumerate(grid_images):
        for i, img in enumerate(row):
            x = label_margin_left + i * img_width
            y = label_margin_top + j * img_height
            grid.paste(img, (x, y))

    return grid


def generate_variable_prompt(
    fixed_prompt: str,
    variable_template: str,
    variable_definitions: str,
    negative_prompt: str,
    num_seed_variations: int,
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
    generate_grids: bool = True,
    grid_groupby: str = "first_variable",
    use_prompt_prefix: bool = False
) -> tuple[list[Image.Image], str]:
    """変数プロンプトで大量の画像をバッチ生成

    Args:
        fixed_prompt: 固定プロンプト部分
        variable_template: 変数テンプレート（例: "{age} years old, {outfit}"）
        variable_definitions: 変数定義テキスト
        negative_prompt: ネガティブプロンプト
        num_seed_variations: シード変化数（1-1000）
        base_seed: ベースシード値
        width, height: 画像サイズ
        num_inference_steps: 推論ステップ数
        guidance_scale: CFGスケール
        vae_name, model_name, scheduler_name: モデル設定
        lora1-3, weight1-3: LoRA設定
        generate_grids: グリッド画像を生成するか
        grid_groupby: グリッドのグループ化方法
        use_prompt_prefix: 品質タグプレフィックスを追加するか（デフォルト: False）

    Returns:
        (images, status_message)
    """
    # 入力検証
    if not fixed_prompt.strip() and not variable_template.strip():
        return None, "固定プロンプトまたは変数テンプレートを入力してください"

    # 変数定義をパース
    variables = parse_variable_definitions(variable_definitions)
    if not variables:
        return None, "変数が定義されていません。形式: 変数名: 値1, 値2, 値3"

    # テンプレート内の変数を検証
    for var_name in variables.keys():
        placeholder = f"{{{var_name}}}"
        if placeholder not in variable_template:
            print(f"Warning: 変数 {placeholder} がテンプレートに見つかりません")

    # テンプレート内の未定義変数をチェック
    template_vars = set(re.findall(r'\{(\w+)\}', variable_template))
    undefined_vars = template_vars - set(variables.keys())
    if undefined_vars:
        return None, f"未定義の変数があります: {', '.join(undefined_vars)}"

    # 組み合わせを生成
    combinations = list(generate_combinations(variables))
    total_combinations = len(combinations)
    total_images = total_combinations * int(num_seed_variations)

    # 上限チェック
    if total_images > 10000:
        return None, f"生成画像数が多すぎます: {total_images}枚 (上限: 10000枚)\n" \
                     f"組み合わせ: {total_combinations} x シード: {num_seed_variations}"

    print(f"Variable Prompt: {total_combinations} combinations x {num_seed_variations} seeds = {total_images} images")
    for var_name, values in variables.items():
        print(f"  {var_name}: {values}")

    # 出力フォルダを作成
    base_path = os.path.dirname(os.path.dirname(__file__))
    output_dir = create_output_dir(base_path, "variable_prompt", fixed_prompt)

    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    # パイプラインを取得
    pipeline = pipeline_manager.get_txt2img_pipeline_with_loras(
        vae_name, model_name, lora_configs, scheduler_name
    )

    if pipeline_manager.lora_error:
        return None, f"LoRAエラー: {pipeline_manager.lora_error}"

    # SDXLモデルかどうかを判定
    is_sdxl = is_sdxl_model(model_name)

    # Negative promptの設定（SDXLの場合はEasyNegativeを追加しない）
    if is_sdxl:
        full_negative = negative_prompt
    else:
        full_negative = negative_prompt
        if "EasyNegative" not in negative_prompt:
            full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

    # CSV準備
    csv_path = os.path.join(output_dir, "metadata.csv")
    csv_headers = [
        "filename", "seed", "full_prompt", "fixed_prompt", "variable_template"
    ] + list(variables.keys()) + [
        "width", "height", "steps", "cfg_scale",
        "model", "vae", "scheduler",
        "lora1", "lora1_weight", "lora2", "lora2_weight", "lora3", "lora3_weight"
    ]
    csv_rows = []

    # 生成結果を格納
    all_images = []
    images_by_group = {}  # グリッド生成用
    generated_count = 0

    # 生成ループ
    clear_stop()

    for combo_idx, combo in enumerate(combinations):
        if is_stop_requested():
            break
        # テンプレートに変数を適用
        try:
            filled_template = variable_template.format(**combo)
        except KeyError as e:
            return None, f"テンプレートエラー: 変数 {e} が見つかりません"

        # フルプロンプトを構築（SDXLの場合は別のプレフィックス）
        if use_prompt_prefix:
            prefix = PROMPT_PREFIX_SDXL if is_sdxl else PROMPT_PREFIX
        else:
            prefix = ""
        if fixed_prompt.strip() and filled_template.strip():
            full_prompt = prefix + fixed_prompt.strip() + ", " + filled_template.strip()
        elif fixed_prompt.strip():
            full_prompt = prefix + fixed_prompt.strip()
        else:
            full_prompt = prefix + filled_template.strip()

        # Long Prompt対応: 77トークン超の場合は事前エンコード
        use_long = needs_long_encoding(pipeline, full_prompt, full_negative)
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if use_long:
            token_count = get_token_count(pipeline, full_prompt)
            print(f"  Long Prompt: {token_count} tokens, using chunked encoding")
            if is_sdxl:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    encode_prompt_long_sdxl(pipeline, full_prompt, full_negative, DEVICE)
                )
            else:
                prompt_embeds, negative_prompt_embeds = (
                    encode_prompt_long(pipeline, full_prompt, full_negative, DEVICE)
                )

        # シード変化ループ
        for seed_offset in range(int(num_seed_variations)):
            if is_stop_requested():
                break

            current_seed = int(base_seed) + seed_offset
            generator = torch.Generator(DEVICE).manual_seed(current_seed)

            # 画像生成
            if use_long:
                gen_kwargs = dict(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    width=int(width),
                    height=int(height),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                )
                if is_sdxl and pooled_prompt_embeds is not None:
                    gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
                    gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
                image = pipeline(**gen_kwargs).images[0]
            else:
                image = pipeline(
                    prompt=full_prompt,
                    negative_prompt=full_negative,
                    width=int(width),
                    height=int(height),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                ).images[0]

            # ファイル名を作成（変数値を含める、パス長制限あり）
            combo_parts = []
            for k, v in combo.items():
                safe_k = sanitize_filename(str(k), 10)
                safe_v = sanitize_filename(str(v), 15)
                combo_parts.append(f"{safe_k}-{safe_v}")
            combo_str = "_".join(combo_parts)
            # ファイル名全体を制限（img_0000_ + combo + _seed + .png で最大50文字程度に）
            if len(combo_str) > 40:
                combo_str = combo_str[:40]
            filename = f"img_{combo_idx:04d}_{combo_str}_seed{current_seed}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_safe(image, filepath)

            # CSV行を追加
            csv_rows.append({
                "filename": filename,
                "seed": current_seed,
                "full_prompt": full_prompt,
                "fixed_prompt": fixed_prompt,
                "variable_template": variable_template,
                **combo,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "cfg_scale": guidance_scale,
                "model": model_name,
                "vae": vae_name,
                "scheduler": scheduler_name,
                "lora1": lora1,
                "lora1_weight": weight1,
                "lora2": lora2,
                "lora2_weight": weight2,
                "lora3": lora3,
                "lora3_weight": weight3
            })

            # グリッド用に画像を分類
            if generate_grids and variables:
                first_var_name = list(variables.keys())[0]
                group_key = combo[first_var_name]
                if group_key not in images_by_group:
                    images_by_group[group_key] = []
                images_by_group[group_key].append((combo, current_seed, image))

            # ギャラリー用（最大100枚まで保持）
            if len(all_images) < 100:
                all_images.append(image)

            generated_count += 1
            print(f"Generated {generated_count}/{total_images}: {combo_str}, seed={current_seed}")

            # メモリ解放
            pipeline_manager.clear_cache()
            gc.collect()

            # 50枚ごとに強力なメモリクリア
            if generated_count % 50 == 0:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

    # 中止チェック（ループ終了後）
    stopped = is_stop_requested()
    if stopped and generated_count == 0:
        return None, "生成が中止されました"

    # CSVを保存（BOM付きUTF-8 + 全フィールドクォートでExcelでも正しく開ける）
    save_metadata_csv(csv_path, csv_rows, csv_headers)

    # グリッド画像を生成
    grid_images = []
    if generate_grids and len(images_by_group) > 0:
        first_var_name = list(variables.keys())[0]

        # 2番目の変数があればX軸に使用
        if len(variables) > 1:
            second_var_name = list(variables.keys())[1]
            second_var_values = variables[second_var_name]
        else:
            second_var_name = first_var_name
            second_var_values = variables[first_var_name]

        for group_name, group_data in images_by_group.items():
            grid = create_variable_grid(
                group_data,
                f"{first_var_name}={group_name}",
                second_var_name if len(variables) > 1 else "seed",
                second_var_values if len(variables) > 1 else [str(int(base_seed) + i) for i in range(int(num_seed_variations))],
                y_label="seed"
            )
            if grid:
                safe_var_name = sanitize_filename(str(first_var_name), 15)
                safe_group_name = sanitize_filename(str(group_name), 20)
                grid_filename = f"grid_{safe_var_name}-{safe_group_name}.png"
                grid_filepath = os.path.join(output_dir, grid_filename)
                save_image_safe(grid, grid_filepath)
                grid_images.append(grid)
                print(f"Created grid: {grid_filename}")

    # グリッド画像を先頭に追加
    result_images = grid_images + all_images

    # 変数サマリーを作成
    var_summary = ", ".join(f"{k}({len(v)})" for k, v in variables.items())

    if stopped:
        status_msg = f"中止しました（{generated_count}/{total_images}枚生成済み）\n" \
                     f"変数: {var_summary}\n" \
                     f"CSV: {csv_path}\n" \
                     f"保存先: {output_dir}"
    else:
        status_msg = f"{total_images}枚の画像を生成しました\n" \
                     f"変数: {var_summary}\n" \
                     f"組み合わせ: {total_combinations}通り x {num_seed_variations}シード\n" \
                     f"CSV: {csv_path}\n" \
                     f"保存先: {output_dir}"

    if grid_images:
        status_msg += f"\nグリッド画像: {len(grid_images)}枚"

    if len(all_images) >= 100:
        status_msg += f"\n※ ギャラリーには最初の100枚のみ表示（全画像は保存済み）"

    return result_images, status_msg
