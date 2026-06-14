"""モデル比較バッチ生成モジュール

複数モデルをそれぞれの最適パラメータで画像生成し、比較グリッドを出力する。

テキスト定義フォーマット:
    # SD1.5モデル
    [bluePencil_v10]
    scheduler: Euler a
    steps: 40
    cfg: 7
    width: 512
    height: 512

    # SDXLモデル（省略可 → デフォルト値使用）
    [waiIllustriousSDXL_v160]
    scheduler: Euler a
    steps: 20
    cfg: 5
    width: 1024
    height: 1024
"""
import os
import gc
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from config import (
    DEVICE, PROMPT_PREFIX, PROMPT_PREFIX_SDXL, MODEL_FILES,
    SCHEDULERS, is_sdxl_model, is_flux_model, clear_stop, is_stop_requested,
    get_catalog_info,
)
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe, sanitize_filename
from utils.long_prompt import encode_prompt_long, encode_prompt_long_sdxl, needs_long_encoding
from .manager import pipeline_manager


# プリフライト VRAM 見積もり（GB）
# 実測ベース: SDXL pipeline+txt2img/img2img/inpaint(weight共有)で約6.5GB,
# 生成中の中間テンソル+CLIP埋め込みで +0.5GB ほど。安全マージン込み。
_VRAM_REQUIRED_SDXL_GB = 7.0
_VRAM_REQUIRED_SD15_GB = 3.5
_VRAM_SAFETY_MARGIN_GB = 0.5


def _free_vram_gb() -> float:
    """driver視点で実際にfreeなVRAMをGB単位で返す。

    torch.cuda.empty_cache() を呼んでから mem_get_info() を取るため、
    PyTorchが保持していたが未使用の reserved も解放後の値が返る。
    """
    if not torch.cuda.is_available():
        return float("inf")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _required_vram_gb(is_sdxl: bool) -> float:
    """モデルロード+生成に必要なVRAMの見積もり（GB）"""
    return _VRAM_REQUIRED_SDXL_GB if is_sdxl else _VRAM_REQUIRED_SD15_GB


# デフォルト値（SD1.5 / SDXL）
DEFAULT_SETTINGS_SD15 = {
    "scheduler": "DPM++ 2M Karras",
    "steps": 20,
    "cfg": 7.0,
    "width": 512,
    "height": 512,
}

DEFAULT_SETTINGS_SDXL = {
    "scheduler": "DPM++ 2M Karras",
    "steps": 20,
    "cfg": 5.0,
    "width": 1024,
    "height": 1024,
}


def parse_model_configs(text: str) -> tuple[list[dict], list[str]]:
    """モデル設定テキストをパースしてモデル設定リストとエラーリストを返却

    Args:
        text: INI形式のモデル設定テキスト

    Returns:
        (configs, errors) タプル
        configs: [{"model": "モデル名", "scheduler": ..., "steps": ..., "cfg": ..., "width": ..., "height": ...}, ...]
        errors: エラーメッセージのリスト
    """
    configs = []
    errors = []

    if not text or not text.strip():
        errors.append("モデル設定が空です")
        return configs, errors

    current_model = None
    current_config = {}

    lines = text.strip().split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # 空行・コメント行はスキップ
        if not line or line.startswith('#'):
            continue

        # モデル名の開始 [モデル名]
        if line.startswith('[') and line.endswith(']'):
            # 前のモデル設定を保存
            if current_model:
                configs.append(_finalize_config(current_model, current_config))

            # 新しいモデル開始
            current_model = line[1:-1].strip()
            current_config = {}

            # モデルの存在確認
            if current_model not in MODEL_FILES:
                errors.append(f"行{line_num}: モデル '{current_model}' が見つかりません")
                current_model = None
                continue

            # Fluxモデルはサポート外
            if is_flux_model(current_model):
                errors.append(f"行{line_num}: Fluxモデル '{current_model}' はサポートされていません（SD1.5/SDXLのみ対応）")
                current_model = None

            continue

        # パラメータ行: key: value
        if ':' in line and current_model:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "scheduler":
                if value in SCHEDULERS:
                    current_config["scheduler"] = value
                else:
                    errors.append(f"行{line_num}: 不明なスケジューラ '{value}'")
            elif key == "steps":
                try:
                    current_config["steps"] = int(value)
                except ValueError:
                    errors.append(f"行{line_num}: steps は整数で指定してください")
            elif key == "cfg":
                try:
                    current_config["cfg"] = float(value)
                except ValueError:
                    errors.append(f"行{line_num}: cfg は数値で指定してください")
            elif key == "width":
                try:
                    current_config["width"] = int(value)
                except ValueError:
                    errors.append(f"行{line_num}: width は整数で指定してください")
            elif key == "height":
                try:
                    current_config["height"] = int(value)
                except ValueError:
                    errors.append(f"行{line_num}: height は整数で指定してください")
            else:
                errors.append(f"行{line_num}: 不明なパラメータ '{key}'")

    # 最後のモデル設定を保存
    if current_model:
        configs.append(_finalize_config(current_model, current_config))

    return configs, errors


def _finalize_config(model_name: str, partial_config: dict) -> dict:
    """部分的な設定にデフォルト値を適用して完全な設定を返す"""
    is_sdxl = is_sdxl_model(model_name)
    defaults = DEFAULT_SETTINGS_SDXL if is_sdxl else DEFAULT_SETTINGS_SD15

    return {
        "model": model_name,
        "scheduler": partial_config.get("scheduler", defaults["scheduler"]),
        "steps": partial_config.get("steps", defaults["steps"]),
        "cfg": partial_config.get("cfg", defaults["cfg"]),
        "width": partial_config.get("width", defaults["width"]),
        "height": partial_config.get("height", defaults["height"]),
        "is_sdxl": is_sdxl,
    }


def generate_config_from_catalog(model_name: str) -> str:
    """カタログの推奨設定からINI形式ブロックを生成

    Args:
        model_name: モデル名（MODEL_FILESのキー）

    Returns:
        INI形式の設定ブロック文字列
    """
    info = get_catalog_info(model_name)
    is_sdxl = is_sdxl_model(model_name)
    defaults = DEFAULT_SETTINGS_SDXL if is_sdxl else DEFAULT_SETTINGS_SD15

    lines = [f"[{model_name}]"]

    # カタログに推奨設定があればパース
    if info and info.get('settings') and info['settings'] != '-':
        settings_text = info['settings']
        # 例: "Euler a, Steps 20, CFG 5" 形式をパース
        parsed = _parse_catalog_settings(settings_text)
        if parsed.get("scheduler"):
            lines.append(f"scheduler: {parsed['scheduler']}")
        if parsed.get("steps"):
            lines.append(f"steps: {parsed['steps']}")
        if parsed.get("cfg"):
            lines.append(f"cfg: {parsed['cfg']}")
    else:
        # デフォルト値を使用
        lines.append(f"scheduler: {defaults['scheduler']}")
        lines.append(f"steps: {defaults['steps']}")
        lines.append(f"cfg: {defaults['cfg']}")

    lines.append(f"width: {defaults['width']}")
    lines.append(f"height: {defaults['height']}")

    return '\n'.join(lines)


def _parse_catalog_settings(settings_text: str) -> dict:
    """カタログの推奨設定文字列をパース

    例: "Euler a, Steps 20, CFG 5" → {"scheduler": "Euler a", "steps": 20, "cfg": 5.0}
    """
    result = {}

    # スケジューラ名を検出（長い名前から順にマッチさせる）
    scheduler_names = sorted(SCHEDULERS.keys(), key=len, reverse=True)
    for scheduler_name in scheduler_names:
        if scheduler_name.lower() in settings_text.lower():
            result["scheduler"] = scheduler_name
            break

    # Steps を検出
    steps_match = re.search(r'steps?\s*[:=]?\s*(\d+)', settings_text, re.IGNORECASE)
    if steps_match:
        result["steps"] = int(steps_match.group(1))

    # CFG を検出
    cfg_match = re.search(r'cfg\s*[:=]?\s*([\d.]+)', settings_text, re.IGNORECASE)
    if cfg_match:
        result["cfg"] = float(cfg_match.group(1))

    return result


def create_model_comparison_grid(
    images_data: list[tuple[str, int, Image.Image, dict]],
    model_names: list[str],
    seeds: list[int]
) -> Optional[Image.Image]:
    """比較グリッド画像を作成（X軸=モデル、Y軸=シード）

    Args:
        images_data: [(model_name, seed, image, config), ...] のリスト
        model_names: モデル名のリスト（X軸の順序）
        seeds: シード値のリスト（Y軸の順序）

    Returns:
        グリッド画像（PIL.Image）またはNone
    """
    if not images_data:
        return None

    # 2D辞書で整理
    image_dict = {}
    for model_name, seed, image, config in images_data:
        key = (model_name, seed)
        image_dict[key] = (image, config)

    # 最小サイズを求める（サイズが異なる場合にリサイズ）
    min_width = min(img.width for _, _, img, _ in images_data)
    min_height = min(img.height for _, _, img, _ in images_data)

    # ラベル用のマージン
    label_margin_top = 100
    label_margin_left = 80

    # グリッドサイズを計算
    num_cols = len(model_names)
    num_rows = len(seeds)
    grid_width = label_margin_left + min_width * num_cols
    grid_height = label_margin_top + min_height * num_rows

    # グリッド画像を作成
    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)

    # フォントの設定
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # X軸ラベルを描画（モデル名 + 設定）
    for i, model_name in enumerate(model_names):
        x_center = label_margin_left + i * min_width + min_width // 2

        # モデル名が長い場合は短縮
        display_name = model_name
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."

        # モデル名
        try:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            text_width = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            text_width = len(display_name) * 7
        draw.text((x_center - text_width // 2, 5), display_name, fill="black", font=font)

        # 設定情報（最初の画像から取得）
        for seed in seeds:
            if (model_name, seed) in image_dict:
                _, config = image_dict[(model_name, seed)]
                model_type = "SDXL" if config.get("is_sdxl") else "SD1.5"
                settings_text = f"{model_type}, {config['scheduler']}"
                settings_text2 = f"s:{config['steps']}, cfg:{config['cfg']}, {config['width']}x{config['height']}"

                try:
                    bbox1 = draw.textbbox((0, 0), settings_text, font=font_small)
                    tw1 = bbox1[2] - bbox1[0]
                    bbox2 = draw.textbbox((0, 0), settings_text2, font=font_small)
                    tw2 = bbox2[2] - bbox2[0]
                except (AttributeError, TypeError):
                    tw1 = len(settings_text) * 6
                    tw2 = len(settings_text2) * 6

                draw.text((x_center - tw1 // 2, 22), settings_text, fill="gray", font=font_small)
                draw.text((x_center - tw2 // 2, 36), settings_text2, fill="gray", font=font_small)
                break

    # Y軸ラベルを描画（シード値）
    for j, seed in enumerate(seeds):
        x = 5
        y = label_margin_top + j * min_height + min_height // 2
        text = f"seed={seed}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
        except (AttributeError, TypeError):
            text_height = 12
        draw.text((x, y - text_height // 2), text, fill="black", font=font)

    # 画像を配置
    for j, seed in enumerate(seeds):
        for i, model_name in enumerate(model_names):
            key = (model_name, seed)
            if key in image_dict:
                img, _ = image_dict[key]
                # リサイズが必要な場合
                if img.width != min_width or img.height != min_height:
                    img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
                x = label_margin_left + i * min_width
                y = label_margin_top + j * min_height
                grid.paste(img, (x, y))

    return grid


def generate_model_comparison(
    prompt: str,
    negative_prompt: str,
    model_configs_text: str,
    base_seed: int,
    num_seed_variations: int,
    vae_name: str,
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0,
    use_prompt_prefix: bool = True,
    generate_grid: bool = True
) -> tuple[list[Image.Image], str]:
    """モデル比較バッチ生成を実行

    Args:
        prompt: プロンプト
        negative_prompt: ネガティブプロンプト
        model_configs_text: INI形式のモデル設定テキスト
        base_seed: ベースシード値
        num_seed_variations: シード変化数（1-100）
        vae_name: VAE名
        lora1-3, weight1-3: LoRA設定
        use_prompt_prefix: 品質タグプレフィックスを使用するか
        generate_grid: 比較グリッドを生成するか

    Returns:
        (images, status_message)
    """
    # 入力検証
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    # モデル設定をパース
    configs, errors = parse_model_configs(model_configs_text)

    if errors:
        return None, "モデル設定エラー:\n" + "\n".join(errors)

    if not configs:
        return None, "有効なモデル設定がありません"

    # 総画像数を計算
    total_images = len(configs) * int(num_seed_variations)

    # 上限チェック
    if total_images > 1000:
        return None, f"生成画像数が多すぎます: {total_images}枚 (上限: 1000枚)\n" \
                     f"モデル: {len(configs)} x シード: {num_seed_variations}"

    print(f"Model Comparison: {len(configs)} models x {num_seed_variations} seeds = {total_images} images")
    for cfg in configs:
        print(f"  {cfg['model']}: {cfg['scheduler']}, steps={cfg['steps']}, cfg={cfg['cfg']}, {cfg['width']}x{cfg['height']}")

    # 出力フォルダを作成
    base_path = os.path.dirname(os.path.dirname(__file__))
    output_dir = create_output_dir(base_path, "model_comparison", prompt)

    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    # シード値のリスト
    seeds = [int(base_seed) + i for i in range(int(num_seed_variations))]

    # 結果を格納
    all_images = []
    images_data = []  # グリッド生成用: (model_name, seed, image, config)
    csv_rows = []
    generated_count = 0
    skipped_models: list[str] = []  # VRAM不足等でスキップしたモデルの記録

    # 生成ループ
    clear_stop()

    model_names = [cfg["model"] for cfg in configs]

    for config in configs:
        if is_stop_requested():
            break

        model_name = config["model"]
        scheduler_name = config["scheduler"]
        steps = config["steps"]
        cfg_scale = config["cfg"]
        width = config["width"]
        height = config["height"]
        is_sdxl = config["is_sdxl"]

        # プリフライト VRAM チェック: 必要量+マージンより空きが少ない場合はスキップ。
        # 過去モデルのキャッシュを解放してから測定するので、フラグメント込みの実空き値。
        if torch.cuda.is_available():
            required_gb = _required_vram_gb(is_sdxl)
            free_gb = _free_vram_gb()
            if free_gb < required_gb + _VRAM_SAFETY_MARGIN_GB:
                msg = (
                    f"{model_name}: VRAM不足のためスキップ "
                    f"(必要 {required_gb:.1f}GB + マージン {_VRAM_SAFETY_MARGIN_GB:.1f}GB, "
                    f"実空き {free_gb:.2f}GB)"
                )
                print(f"⚠️ {msg}")
                skipped_models.append(msg)
                continue

        print(f"\n--- Loading model: {model_name} ---")

        # プロンプトにプレフィックスを追加
        if use_prompt_prefix:
            prefix = PROMPT_PREFIX_SDXL if is_sdxl else PROMPT_PREFIX
            full_prompt = prefix + prompt
        else:
            full_prompt = prompt

        # ネガティブプロンプトの設定（SD1.5の場合はEasyNegativeを追加）
        if is_sdxl:
            full_negative = negative_prompt
        else:
            if "EasyNegative" not in negative_prompt:
                full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"
            else:
                full_negative = negative_prompt

        # パイプラインを取得
        try:
            pipeline = pipeline_manager.get_txt2img_pipeline_with_loras(
                vae_name, model_name, lora_configs, scheduler_name
            )
        except torch.cuda.OutOfMemoryError as e:
            # プリフライトをすり抜けたOOM。ここでも明示メッセージでスキップ。
            msg = f"{model_name}: ロード中にOOMが発生したためスキップ ({e})"
            print(f"⚠️ {msg}")
            skipped_models.append(msg)
            # 残骸を解放してから次へ
            pipeline_manager.clear_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            skipped_models.append(f"{model_name}: ロード失敗 ({e})")
            continue

        if pipeline_manager.lora_error:
            print(f"LoRA warning for {model_name}: {pipeline_manager.lora_error}")

        # Long Prompt対応
        use_long = needs_long_encoding(pipeline, full_prompt, full_negative)
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if use_long:
            print(f"  Using long prompt encoding")
            if is_sdxl:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    encode_prompt_long_sdxl(pipeline, full_prompt, full_negative, DEVICE)
                )
            else:
                prompt_embeds, negative_prompt_embeds = (
                    encode_prompt_long(pipeline, full_prompt, full_negative, DEVICE)
                )

        # シード変化ループ
        for seed in seeds:
            if is_stop_requested():
                break

            # 画像生成（Generator 作成も含めて try 内に置く: OOM は非同期で
            # この行で表面化することがあるため）
            try:
                generator = torch.Generator(DEVICE).manual_seed(seed)
                if use_long:
                    gen_kwargs = dict(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
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
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                    ).images[0]
            except torch.cuda.OutOfMemoryError as e:
                # このモデル/シードでOOM。残りシードも危険なのでこのモデルは中断。
                msg = f"{model_name}: seed={seed} でOOM。このモデルの残りシードはスキップ"
                print(f"⚠️ {msg} ({e})")
                skipped_models.append(msg)
                # キャッシュを解放して次モデルへ進める
                pipeline_manager.clear_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break  # このモデルのシードループを抜ける
            except Exception as e:
                print(f"Error generating image for {model_name}, seed={seed}: {e}")
                continue

            # ファイル名を作成
            safe_model = sanitize_filename(model_name, 30)
            filename = f"{safe_model}_seed{seed}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_safe(image, filepath)

            # CSV行を追加
            csv_rows.append({
                "filename": filename,
                "model": model_name,
                "model_type": "SDXL" if is_sdxl else "SD1.5",
                "seed": seed,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "scheduler": scheduler_name,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "vae": vae_name,
                "lora1": lora1,
                "lora1_weight": weight1,
                "lora2": lora2,
                "lora2_weight": weight2,
                "lora3": lora3,
                "lora3_weight": weight3,
            })

            # グリッド用データ
            images_data.append((model_name, seed, image, config))

            # ギャラリー用（最大100枚まで保持）
            if len(all_images) < 100:
                all_images.append(image)

            generated_count += 1
            print(f"Generated {generated_count}/{total_images}: {model_name}, seed={seed}")

            # メモリ解放
            pipeline_manager.clear_cache()
            gc.collect()

        # モデル切り替え前にキャッシュクリア
        pipeline_manager.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # 中止チェック
    stopped = is_stop_requested()
    if stopped and generated_count == 0:
        return None, "生成が中止されました"

    # CSVを保存
    csv_path = save_batch_generation_params(output_dir, csv_rows)

    # グリッド画像を生成
    grid_images = []
    if generate_grid and images_data:
        grid = create_model_comparison_grid(images_data, model_names, seeds)
        if grid:
            grid_filename = "comparison_grid.png"
            grid_filepath = os.path.join(output_dir, grid_filename)
            save_image_safe(grid, grid_filepath)
            grid_images.append(grid)
            print(f"Created grid: {grid_filename}")

    # 結果を返す
    result_images = grid_images + all_images

    if stopped:
        status_msg = f"中止しました（{generated_count}/{total_images}枚生成済み）\n" \
                     f"モデル: {len(configs)}個\n" \
                     f"CSV: {csv_path}\n" \
                     f"保存先: {output_dir}"
    else:
        status_msg = f"{total_images}枚の画像を生成しました\n" \
                     f"モデル: {len(configs)}個 x シード: {num_seed_variations}\n" \
                     f"CSV: {csv_path}\n" \
                     f"保存先: {output_dir}"

    if grid_images:
        status_msg += f"\n比較グリッド: {len(grid_images)}枚"

    if len(all_images) >= 100:
        status_msg += f"\n※ ギャラリーには最初の100枚のみ表示（全画像は保存済み）"

    if skipped_models:
        status_msg += f"\n\n⚠️ スキップ {len(skipped_models)}件:"
        for s in skipped_models:
            status_msg += f"\n  - {s}"

    return result_images, status_msg
