"""バッチ生成共通パーサー / メタデータCSV書き出しモジュール

image / music / video のバッチ系パイプラインで重複していた
変数定義パース・組み合わせ生成・値リストパース・metadata.csv 書き出しを共通化。

注意: pipelines/xyz_plot.py の parse_values は丸め桁数・epsilon・int判定が
異なる別実装のため、ここには統合しない（グリッドラベル/ファイル名が変わるため）。
"""
import os
import csv
import itertools
from typing import Generator, Optional


def parse_variable_definitions(text: str) -> dict[str, list[str]]:
    """変数定義テキストをパース

    形式:
        変数名: 値1, 値2, 値3
        age: 20, 25, 30, 35
        outfit_style: casual clothes, formal suit

    Returns:
        {"変数名": ["値1", "値2", ...], ...}
    """
    variables = {}

    if not text or not text.strip():
        return variables

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        # 最初の : で分割（値の中に : が含まれる場合も対応）
        name, values_str = line.split(':', 1)
        name = name.strip()

        # カンマで分割して各値をトリム
        values = [v.strip() for v in values_str.split(',') if v.strip()]

        if name and values:
            variables[name] = values

    return variables


def generate_combinations(variables: dict[str, list[str]]) -> Generator[dict, None, None]:
    """変数の全組み合わせを生成

    Args:
        variables: {"age": ["20", "25"], "outfit": ["casual", "formal"]}

    Yields:
        {"age": "20", "outfit": "casual"}, {"age": "20", "outfit": "formal"}, ...
    """
    if not variables:
        return

    keys = list(variables.keys())
    value_lists = [variables[k] for k in keys]

    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def parse_values(value_str: str, param_key: Optional[str], int_params: set) -> list:
    """カンマ区切りの値をパース

    形式:
    - 単純なカンマ区切り: 1, 2, 3
    - 範囲指定: 1-10:2 (1から10まで2刻み)
    """
    if not value_str or not value_str.strip():
        return [None]

    is_int = param_key in int_params
    values = []

    for part in (p.strip() for p in value_str.split(",")):
        if not part:
            continue
        # 範囲指定: start-end:step (負数は今回サポートしない)
        if "-" in part and ":" in part:
            try:
                range_part, step_str = part.split(":")
                start_str, end_str = range_part.split("-")
                start = float(start_str)
                end = float(end_str)
                step = float(step_str)
                if step <= 0:
                    continue
                current = start
                while current <= end + 1e-6:
                    values.append(int(current) if is_int else round(current, 4))
                    current += step
            except ValueError:
                continue
        else:
            try:
                val = float(part)
                values.append(int(val) if is_int else round(val, 4))
            except ValueError:
                continue

    return values if values else [None]


def save_metadata_csv(csv_path: str, rows: list, fieldnames: Optional[list] = None):
    """metadata.csv を保存（BOM付きUTF-8 + 全フィールドクォートでExcelでも正しく開ける）"""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
