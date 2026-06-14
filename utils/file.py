"""ファイル操作ユーティリティ"""
import os
import re
import csv
import time
from datetime import datetime
from PIL import Image


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """ファイル名に使えない文字を除去し、長さを制限"""
    # Windows禁止文字 + カンマ、セミコロン等の問題文字を除去
    bad_chars = set('\\/:*?"<>|,;(){}[]\'&!@#$%^~`')
    sanitized = ''.join(c for c in text if c not in bad_chars)
    sanitized = sanitized.replace(' ', '_')
    # 連続するアンダースコアを1つにまとめる
    sanitized = re.sub(r'_+', '_', sanitized)
    # 先頭・末尾のアンダースコアを除去
    sanitized = sanitized.strip('_')
    return sanitized[:max_length]


def create_output_dir(base_path: str, prefix: str, prompt: str = "", category: str = "image") -> str:
    """出力ディレクトリを作成して返す

    category により、base_path 配下にさらに output/{image_output,audio_output,video_output}/ を挟む。
    既存呼び出しは category="image" のままで output/image_output/ 配下に集約される。

    Windows MAX_PATH (260文字) を考慮し、パス長を制限する。
    ファイル名の余裕として60文字を確保する。

    Args:
        base_path: 出力先のベースディレクトリ（通常はプロジェクトルート）
        prefix: パイプライン名（txt2img, video_animatediff など）
        prompt: ディレクトリ名末尾に付与する短縮プロンプト
        category: "image" / "audio" / "video" / "raw"
                  "raw" は base_path をそのまま使用（後方互換用）
    """
    import config

    if category == "image":
        subdir = config.IMAGE_OUTPUT_DIR
    elif category == "audio":
        subdir = config.AUDIO_OUTPUT_DIR
    elif category == "video":
        subdir = config.VIDEO_OUTPUT_DIR
    else:
        subdir = ""

    full_base = os.path.join(base_path, subdir) if subdir else base_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ベース部分の長さ: full_base + separator + "output_" + prefix + "_" + timestamp + "_"
    base_dir_prefix = f"output_{prefix}_{timestamp}_"
    fixed_len = len(full_base) + 1 + len(base_dir_prefix)

    # ファイル名用に60文字の余裕を確保（例: img_0000_var=value_seed100.png）
    max_path = 255
    filename_reserve = 60
    available_for_prompt = max(10, max_path - fixed_len - filename_reserve)

    safe_prompt = sanitize_filename(prompt, max_length=min(50, available_for_prompt)) if prompt else ""

    if safe_prompt:
        dir_name = f"output_{prefix}_{timestamp}_{safe_prompt}"
    else:
        dir_name = f"output_{prefix}_{timestamp}"

    output_dir = os.path.join(full_base, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_image_safe(image: Image.Image, filepath: str, max_retries: int = 3):
    """Google Drive等でのファイル保存失敗に対応するリトライ付き画像保存"""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            image.save(filepath)
            return
        except (FileNotFoundError, OSError) as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 2
                print(f"Save failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def save_batch_generation_params(output_dir: str, params_list: list[dict], filename: str = "generation_params.csv") -> str:
    """複数画像の生成パラメータをCSVファイルに保存（バッチ用）

    Args:
        output_dir: 出力ディレクトリ
        params_list: パラメータ辞書のリスト（各画像ごと）
        filename: CSVファイル名

    Returns:
        保存したCSVファイルのパス
    """
    if not params_list:
        return None

    csv_path = os.path.join(output_dir, filename)

    # タイムスタンプを追加
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for params in params_list:
        params["timestamp"] = timestamp

    # 全キーを収集（順序を保持）
    all_keys = []
    for params in params_list:
        for key in params.keys():
            if key not in all_keys:
                all_keys.append(key)

    # CSVに書き込み（BOM付きUTF-8 + 全フィールドクォートでExcelでも正しく開ける）
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(params_list)

    return csv_path
