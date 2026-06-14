"""Text to Image タブ"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, DEFAULT_NEGATIVE_SDXL, PROMPT_PREFIX, PROMPT_PREFIX_SDXL, is_sdxl_model, request_stop
from pipelines.txt2img import generate_images
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector, create_face_restore_section


def create_txt2img_tab():
    """Text to Imageタブを作成"""
    with gr.Row():
        with gr.Column():
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
            prompt_info = gr.Markdown("*EasyNegativeは自動で追加されます（SD1.5のみ）*")

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
                width_input = gr.Slider(
                    label="幅",
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64
                )
                height_input = gr.Slider(
                    label="高さ",
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64
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

    # モデル変更時に推奨設定を適用
    def on_model_change(model_name):
        """モデル変更時のコールバック"""
        if is_sdxl_model(model_name):
            # SDXL: 1024x1024推奨、VAEはデフォルト
            return [
                gr.update(value=1024),  # width
                gr.update(value=1024),  # height
                gr.update(value="なし（デフォルト）"),  # VAE
                gr.update(value=DEFAULT_NEGATIVE_SDXL),  # negative prompt
                gr.update(value=f"*SDXLモデル検出: プレフィックス `{PROMPT_PREFIX_SDXL}` を使用*")  # info
            ]
        else:
            # SD1.5: 512x512推奨
            return [
                gr.update(value=512),  # width
                gr.update(value=512),  # height
                gr.update(value="CleanVAE"),  # VAE
                gr.update(value=DEFAULT_NEGATIVE),  # negative prompt
                gr.update(value=f"*SD1.5モデル: EasyNegative自動追加、プレフィックス `{PROMPT_PREFIX}`*")  # info
            ]

    model_input.change(
        fn=on_model_change,
        inputs=[model_input],
        outputs=[width_input, height_input, vae_input, negative_prompt_input, prompt_info]
    )

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt_input,
            negative_prompt_input,
            num_images_input,
            seed_input,
            width_input,
            height_input,
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
