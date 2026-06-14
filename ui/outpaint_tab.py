"""Outpainting タブ"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, request_stop
from pipelines.outpaint import generate_outpaint
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector


def create_outpaint_tab():
    """Outpaintingタブを作成"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="入力画像",
                type="numpy"
            )

            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: 1girl, beautiful, masterpiece",
                lines=3,
                value=""
            )
            gr.Markdown(f"*自動で先頭に追加: `{PROMPT_PREFIX}`*")

            negative_prompt_input = gr.Textbox(
                label="Negative プロンプト",
                value=DEFAULT_NEGATIVE,
                lines=4
            )
            gr.Markdown("*EasyNegativeは自動で追加されます*")

            gr.Markdown("### 拡張設定（ピクセル）")
            gr.Markdown("*64の倍数に自動調整されます*")

            with gr.Row():
                left = gr.Slider(
                    label="左",
                    minimum=0,
                    maximum=512,
                    value=0,
                    step=64
                )
                right = gr.Slider(
                    label="右",
                    minimum=0,
                    maximum=512,
                    value=0,
                    step=64
                )

            with gr.Row():
                top = gr.Slider(
                    label="上",
                    minimum=0,
                    maximum=512,
                    value=0,
                    step=64
                )
                bottom = gr.Slider(
                    label="下",
                    minimum=0,
                    maximum=512,
                    value=0,
                    step=64
                )

            feather = gr.Slider(
                label="フェザリング（境界のぼかし）",
                minimum=8,
                maximum=64,
                value=32,
                step=4,
                info="大きいほど境界が滑らかになります"
            )

            with gr.Row():
                num_images_input = gr.Slider(
                    label="生成枚数",
                    minimum=1,
                    maximum=10000,
                    value=4,
                    step=1
                )
                seed_input = gr.Number(
                    label="ベースSeed値",
                    value=42
                )

            with gr.Row():
                steps_input = gr.Slider(
                    label="ステップ数",
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=1
                )
                guidance_input = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=20,
                    value=7.5,
                    step=0.5
                )

            vae_input = gr.Dropdown(
                label="VAE",
                choices=list(VAE_FILES.keys()),
                value="CleanVAE"
            )

            model_input = create_model_selector()
            scheduler_input = create_scheduler_selector()

            lora_components = create_multi_lora_selector(num_slots=3)

            with gr.Row():
                generate_btn = gr.Button("生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_outpaint,
        inputs=[
            input_image,
            prompt_input,
            negative_prompt_input,
            num_images_input,
            seed_input,
            left,
            right,
            top,
            bottom,
            feather,
            steps_input,
            guidance_input,
            vae_input,
            model_input,
            scheduler_input,
            *lora_components  # lora1, weight1, lora2, weight2, lora3, weight3
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
