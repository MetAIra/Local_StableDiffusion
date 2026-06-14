"""動画生成モデルダウンロードスクリプト

UI 側からも使える共通ロジックは pipelines/video_downloader.py にあり、
このスクリプトは CLI ラッパーに過ぎない。

使い方:
    python scripts/download_video_models.py            # 全モデルDL
    python scripts/download_video_models.py animatediff svd_xt  # 指定のみ
    python scripts/download_video_models.py --list     # モデル一覧表示
    python scripts/download_video_models.py --skip-base  # AnimateDiff用SD1.5ベースをスキップ

注意:
- 合計 約30GB の容量が必要（bloat除外後）
- SVDなど一部モデルは HuggingFace アカウントとライセンス同意が必要
  事前に: huggingface-cli login
"""
import argparse
import os
import sys
from pathlib import Path

# Windows コンソール (cp932) でもUnicodeを安全に出すための保険
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# config.py を import するために親ディレクトリを sys.path に追加
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import VIDEO_MODELS, VIDEO_MODEL_DIR  # noqa: E402
from pipelines.video_downloader import (  # noqa: E402
    download_video_model,
    get_dl_status_summary,
)


def list_models():
    """モデル一覧と DL 状況を表示"""
    print("\n=== 利用可能な動画モデル ===")
    print(f"ダウンロード先: {VIDEO_MODEL_DIR}\n")
    summary = get_dl_status_summary()
    total = 0
    for key, info in summary.items():
        size = info["size_gb"]
        total += size
        if info["ready"]:
            status = "[DL済]"
        elif info["main_downloaded"]:
            status = "[本体のみDL]"
        else:
            status = "[未DL]"
        print(f"{status} {key:15s} ({size:5.1f}GB) - {info['label']}")
        cfg = VIDEO_MODELS[key]
        print(f"        repo: {cfg['repo_id']}")
        print(f"        {cfg['description']}")
        print()
    print(f"合計サイズ目安: 約 {total:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="動画生成モデルをダウンロード")
    parser.add_argument(
        "models",
        nargs="*",
        help="DL対象モデルキー (省略時は全モデル)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="利用可能モデル一覧を表示して終了",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="AnimateDiff用ベースSD1.5モデルのDLをスキップ",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    os.makedirs(VIDEO_MODEL_DIR, exist_ok=True)

    target_keys = args.models if args.models else list(VIDEO_MODELS.keys())
    invalid = [k for k in target_keys if k not in VIDEO_MODELS]
    if invalid:
        print(f"[NG] 未知のモデルキー: {invalid}")
        print(f"利用可能: {list(VIDEO_MODELS.keys())}")
        return 1

    print(f"DL先: {VIDEO_MODEL_DIR}")
    print(f"対象: {target_keys}\n")

    results = {}
    for key in target_keys:
        ok, msg = download_video_model(
            key,
            include_base=(not args.skip_base),
        )
        results[key] = (ok, msg)

    print("\n=== 結果サマリー ===")
    for key, (ok, msg) in results.items():
        marker = "[OK]" if ok else "[NG]"
        print(f"  {marker} {key}: {msg}")

    failed = [k for k, (ok, _) in results.items() if not ok]
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
