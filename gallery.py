"""生成画像ギャラリービューア

output/image_output/ 配下の output_* ディレクトリ（および後方互換でプロジェクト直下）の
生成画像を一覧表示する独立アプリ。
使い方: python gallery.py
"""
import os
import re
import glob
import csv
import gradio as gr
from datetime import datetime

# このスクリプトのあるディレクトリを基準にする
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 新しい画像出力ディレクトリ（output/image_output/）
IMAGE_OUTPUT_BASE = os.path.join(BASE_DIR, "output", "image_output")

# 1ページあたりの画像数
IMAGES_PER_PAGE = 50

# 対応する画像拡張子
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _scan_output_dirs(root: str) -> list[os.DirEntry]:
    """root 直下の output_* サブディレクトリを返す"""
    if not os.path.isdir(root):
        return []
    return [e for e in os.scandir(root) if e.is_dir() and e.name.startswith("output_")]


def find_output_dirs() -> list[dict]:
    """output_* ディレクトリを検索し、情報を返す

    新パス: output/image_output/output_*
    旧パス: プロジェクトルート直下の output_*（移動前ファイルへの後方互換）
    """
    dirs = []
    seen = set()
    # 新パスを優先、続いて旧パス（重複除外）
    for entry in _scan_output_dirs(IMAGE_OUTPUT_BASE) + _scan_output_dirs(BASE_DIR):
        if entry.path in seen:
            continue
        seen.add(entry.path)

        # 画像ファイル数をカウント
        image_files = []
        for ext in IMAGE_EXTS:
            image_files += glob.glob(os.path.join(entry.path, f"*{ext}"))
        image_count = len(image_files)

        if image_count == 0:
            continue

        # ディレクトリ名からタイムスタンプを抽出
        ts_match = re.search(r"(\d{8}_\d{6})", entry.name)
        if ts_match:
            try:
                ts = datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M%S")
                date_str = ts.strftime("%Y/%m/%d %H:%M")
            except ValueError:
                date_str = "不明"
        else:
            date_str = "不明"

        # パイプライン名を抽出 (output_{pipeline}_{timestamp}_{prompt})
        name_parts = entry.name.split("_")
        if len(name_parts) >= 3:
            # "output" の次からタイムスタンプの前まで
            pipeline = ""
            for i, part in enumerate(name_parts[1:], 1):
                if re.match(r"^\d{8}$", part):
                    pipeline = "_".join(name_parts[1:i])
                    break
            if not pipeline:
                pipeline = name_parts[1] if len(name_parts) > 1 else "unknown"
        else:
            pipeline = "unknown"

        dirs.append({
            "name": entry.name,
            "path": entry.path,
            "image_count": image_count,
            "date_str": date_str,
            "pipeline": pipeline,
            "display": f"{date_str}  |  {pipeline}  |  {image_count}枚  |  {entry.name}",
        })

    # 日付の新しい順にソート
    dirs.sort(key=lambda d: d["name"], reverse=True)
    return dirs


def get_variable_values(dir_path: str) -> list[str]:
    """ディレクトリ内の画像ファイル名から変数値を抽出"""
    values = set()
    for f in os.listdir(dir_path):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        # img_0000_zodiac_animal=mouse_seed42.png
        # img_0000_zodiac_ani-mouse_seed42.png
        # パターン: img_NNNN_{var_name}={value}_seed or img_NNNN_{var_name}-{value}_seed
        match = re.search(r"img_\d+_(.+?)_seed\d+", f)
        if match:
            var_part = match.group(1)
            # "=" または最後の "-" で分割して値を取得
            if "=" in var_part:
                val = var_part.split("=", 1)[1]
            elif "-" in var_part:
                val = var_part.rsplit("-", 1)[1]
            else:
                val = var_part
            values.add(val)

    return sorted(values)


def load_metadata(dir_path: str) -> str:
    """ディレクトリ内のメタデータを読み込む"""
    metadata_parts = []

    # プロンプト.txt を読む
    prompt_file = os.path.join(dir_path, "プロンプト.txt")
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata_parts.append("### プロンプト情報\n```\n" + content + "\n```")
        except Exception:
            pass

    # generation_params.csv を読む
    csv_file = os.path.join(dir_path, "generation_params.csv")
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                first = rows[0]
                lines = [f"- **{k}**: {v}" for k, v in first.items() if v]
                metadata_parts.append("### 生成パラメータ\n" + "\n".join(lines))
                if len(rows) > 1:
                    metadata_parts.append(f"\n*（他 {len(rows) - 1} 件のパラメータあり）*")
        except Exception:
            pass

    if not metadata_parts:
        return "*メタデータなし*"

    return "\n\n".join(metadata_parts)


def get_images_for_page(dir_path: str, page: int, filter_value: str = "") -> tuple[list[str], int, int, int]:
    """指定ページの画像パスリスト・トータルページ数・総画像数・クランプ済みページ番号を返す"""
    all_images = []
    for f in sorted(os.listdir(dir_path)):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        if filter_value and filter_value != "すべて":
            # フィルター: ファイル名に値が含まれるか
            if filter_value.lower() not in f.lower():
                continue
        all_images.append(os.path.join(dir_path, f))

    total = len(all_images)
    total_pages = max(1, (total + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    return all_images[start:end], total_pages, total, page


def _render_page(dir_path: str, page: int, filter_value: str):
    """指定ページを描画するための6要素タプルを返す

    戻り値: (images, page, page_text, page_nav, prev_update, next_update)
    """
    if not dir_path:
        return [], 0, "", "1 / 1", gr.update(interactive=False), gr.update(interactive=False)

    images, total_pages, total_count, page = get_images_for_page(dir_path, page, filter_value)

    page_text = f"**{total_count}枚** の画像  |  ページ {page + 1} / {total_pages}"
    page_nav = f"{page + 1} / {total_pages}"

    return (
        images,
        page,
        page_text,
        page_nav,
        gr.update(interactive=page > 0),
        gr.update(interactive=page < total_pages - 1),
    )


def create_gallery_app():
    """ギャラリーアプリを作成"""
    output_dirs = find_output_dirs()

    if not output_dirs:
        with gr.Blocks() as demo:
            gr.Markdown("# 画像ギャラリー")
            gr.Markdown("output_* ディレクトリが見つかりませんでした。")
        return demo

    dir_choices = [d["display"] for d in output_dirs]
    dir_map = {d["display"]: d for d in output_dirs}

    with gr.Blocks() as demo:
        gr.Markdown("# 生成画像ギャラリー")

        # State
        current_page = gr.State(0)
        current_dir_path = gr.State("")

        with gr.Row():
            # 左サイドバー
            with gr.Column(scale=1):
                gr.Markdown("### 出力ディレクトリ")
                dir_selector = gr.Dropdown(
                    choices=dir_choices,
                    label="ディレクトリ選択",
                    value=dir_choices[0] if dir_choices else None,
                    interactive=True,
                )
                refresh_btn = gr.Button("ディレクトリ一覧を更新", variant="secondary", size="sm")

                filter_dropdown = gr.Dropdown(
                    choices=["すべて"],
                    label="変数値フィルター",
                    value="すべて",
                    interactive=True,
                )

                metadata_display = gr.Markdown("")

            # メイン表示エリア
            with gr.Column(scale=3):
                with gr.Row():
                    page_info = gr.Markdown("", elem_id="page-info")

                gallery = gr.Gallery(
                    label="生成画像",
                    columns=5,
                    height="auto",
                    object_fit="cover",
                )

                with gr.Row():
                    prev_btn = gr.Button("< 前のページ", size="sm", interactive=False)
                    page_display = gr.Markdown("1 / 1", elem_id="page-display")
                    next_btn = gr.Button("次のページ >", size="sm", interactive=False)

        # --- イベントハンドラ ---

        def on_dir_selected(dir_display):
            """ディレクトリ選択時"""
            if not dir_display or dir_display not in dir_map:
                return [], "", gr.update(choices=["すべて"], value="すべて"), "*ディレクトリを選択してください*", 0, "", "1 / 1", gr.update(interactive=False), gr.update(interactive=False)

            d = dir_map[dir_display]
            dir_path = d["path"]

            # 変数値を取得
            values = get_variable_values(dir_path)
            filter_choices = ["すべて"] + values

            # メタデータ読み込み
            metadata = load_metadata(dir_path)

            # 最初のページの画像を取得
            images, page, page_text, page_nav, prev_update, next_update = _render_page(dir_path, 0, "")

            return (
                images,                                                          # gallery
                dir_path,                                                        # current_dir_path
                gr.update(choices=filter_choices, value="すべて"),                # filter_dropdown
                metadata,                                                        # metadata_display
                page,                                                            # current_page
                page_text,                                                       # page_info
                page_nav,                                                        # page_display
                prev_update,                                                     # prev_btn
                next_update,                                                     # next_btn
            )

        def on_filter_changed(filter_value, dir_path):
            """フィルター変更時"""
            return _render_page(dir_path, 0, filter_value)

        def on_page_change(page, dir_path, filter_value, direction):
            """ページ移動"""
            return _render_page(dir_path, page + direction, filter_value)

        def on_refresh():
            """ディレクトリ一覧を更新"""
            nonlocal output_dirs, dir_choices, dir_map
            output_dirs = find_output_dirs()
            dir_choices = [d["display"] for d in output_dirs]
            dir_map = {d["display"]: d for d in output_dirs}
            return gr.update(choices=dir_choices, value=dir_choices[0] if dir_choices else None)

        # イベントバインド
        dir_selector.change(
            fn=on_dir_selected,
            inputs=[dir_selector],
            outputs=[gallery, current_dir_path, filter_dropdown, metadata_display, current_page, page_info, page_display, prev_btn, next_btn],
        )

        filter_dropdown.change(
            fn=on_filter_changed,
            inputs=[filter_dropdown, current_dir_path],
            outputs=[gallery, current_page, page_info, page_display, prev_btn, next_btn],
        )

        prev_btn.click(
            fn=lambda p, d, f: on_page_change(p, d, f, -1),
            inputs=[current_page, current_dir_path, filter_dropdown],
            outputs=[gallery, current_page, page_info, page_display, prev_btn, next_btn],
        )

        next_btn.click(
            fn=lambda p, d, f: on_page_change(p, d, f, 1),
            inputs=[current_page, current_dir_path, filter_dropdown],
            outputs=[gallery, current_page, page_info, page_display, prev_btn, next_btn],
        )

        refresh_btn.click(
            fn=on_refresh,
            inputs=[],
            outputs=[dir_selector],
        )

        # 初期表示
        demo.load(
            fn=on_dir_selected,
            inputs=[dir_selector],
            outputs=[gallery, current_dir_path, filter_dropdown, metadata_display, current_page, page_info, page_display, prev_btn, next_btn],
        )

    return demo


if __name__ == "__main__":
    demo = create_gallery_app()
    demo.launch(server_port=7861, server_name="0.0.0.0")
