"""背景除去 タブ"""
import gradio as gr

from pipelines.background_remove import (
    remove_background_image,
    REMBG_MODELS,
    BG_COLOR_OPTIONS
)
from .common import create_output_components


def create_background_remove_tab():
    """背景除去タブを作成"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="入力画像",
                type="numpy"
            )

            gr.Markdown("### 背景除去設定")

            model_input = gr.Dropdown(
                label="モデル",
                choices=list(REMBG_MODELS.values()),
                value=REMBG_MODELS["u2net_human_seg"],
                info="人物切り抜きにはu2net_human_segがおすすめ"
            )

            bg_color_input = gr.Radio(
                label="背景色",
                choices=list(BG_COLOR_OPTIONS.keys()),
                value="透明",
                info="透明背景はPNG形式で保存されます"
            )

            gr.Markdown("### 詳細設定")

            alpha_matting_input = gr.Checkbox(
                label="Alpha Matting（髪の毛などの細かい部分を綺麗に）",
                value=False,
                info="処理時間が長くなりますが、境界が滑らかになります"
            )

            output_mask_input = gr.Checkbox(
                label="マスク画像も出力する",
                value=False,
                info="切り抜きに使用したマスク画像を同時に出力"
            )

            generate_btn = gr.Button("背景を除去", variant="primary")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    # イベント設定
    generate_btn.click(
        fn=remove_background_image,
        inputs=[
            input_image,
            model_input,
            bg_color_input,
            alpha_matting_input,
            output_mask_input
        ],
        outputs=[output_gallery, output_message]
    )
