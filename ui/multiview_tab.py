"""Multi-View生成 タブ (Zero123++)

Zero123++: 1枚の画像から多視点画像を生成
"""
import gradio as gr

from pipelines.multiview import generate_multiview
from config import ZERO123_DEFAULT_STEPS, ZERO123_DEFAULT_CFG


def create_multiview_tab():
    """Multi-View生成タブを作成"""

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Zero123++ (画像→多視点)")

            input_image = gr.Image(
                label="入力画像",
                type="numpy",
                height=350
            )

            gr.Markdown("""
**Zero123++** は1枚の画像から多視点画像を生成します。

- 入力画像は最小320×320px必要（自動リサイズ）
- **背景除去済みの画像を推奨**（Background Removeタブで処理可能）
- 出力: Front-Right, Right, Back の3視点 + 入力画像(Front)
            """)

            gr.Markdown("### 生成設定")

            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="ステップ数",
                    minimum=25,
                    maximum=150,
                    value=ZERO123_DEFAULT_STEPS,
                    step=5,
                    info="推奨: 75"
                )
                guidance_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=ZERO123_DEFAULT_CFG,
                    step=0.5,
                    info="推奨: 4.0"
                )

            seed = gr.Number(
                label="Seed",
                value=42,
                precision=0
            )

            generate_btn = gr.Button("多視点を生成", variant="primary")

        with gr.Column():
            gr.Markdown("### 出力")
            gr.Markdown("*最初の画像がラベル付きグリッド、以降が個別の視点画像です*")
            output_gallery = gr.Gallery(
                label="生成結果（グリッド + 個別視点）",
                columns=4,
                height="auto",
                object_fit="contain"
            )
            output_message = gr.Textbox(
                label="ステータス",
                interactive=False
            )

    def on_generate(input_image, num_inference_steps, guidance_scale, seed):
        return generate_multiview(
            input_image,
            int(num_inference_steps),
            float(guidance_scale),
            int(seed)
        )

    generate_btn.click(
        fn=on_generate,
        inputs=[input_image, num_inference_steps, guidance_scale, seed],
        outputs=[output_gallery, output_message]
    )
