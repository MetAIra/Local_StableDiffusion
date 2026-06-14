"""Image to Image タブ"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, request_stop
from pipelines.img2img import generate_images_img2img
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector, create_face_restore_section


def create_img2img_tab():
    """Image to Imageタブを作成"""
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

            strength_input = gr.Slider(
                label="Strength（元画像からの変化度）",
                minimum=0.1,
                maximum=1.0,
                value=0.75,
                step=0.05,
                info="0.1=ほぼ元画像、1.0=完全に再生成"
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

            face_restore_enabled, face_restore_method, face_restore_weight = create_face_restore_section()

            with gr.Row():
                generate_btn = gr.Button("生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_images_img2img,
        inputs=[
            input_image,
            prompt_input,
            negative_prompt_input,
            num_images_input,
            seed_input,
            strength_input,
            steps_input,
            guidance_input,
            vae_input,
            model_input,
            scheduler_input,
            *lora_components,  # lora1, weight1, lora2, weight2, lora3, weight3
            face_restore_enabled,
            face_restore_method,
            face_restore_weight
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
