"""Upscale タブ"""
import gradio as gr

from config import REALESRGAN_MODELS, UPSCALE_MODES, DEFAULT_NEGATIVE, PROMPT_PREFIX
from pipelines.upscale import upscale_image
from .common import create_output_components


def create_upscale_tab():
    """Upscaleタブを作成"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="入力画像",
                type="numpy"
            )

            gr.Markdown("### アップスケール設定")

            mode_input = gr.Radio(
                label="Upscale方式",
                choices=list(UPSCALE_MODES.keys()),
                value="Real-ESRGAN Only",
                info="Real-ESRGAN: 高速、SD: ディテール追加可能"
            )

            realesrgan_model_input = gr.Dropdown(
                label="Real-ESRGANモデル",
                choices=list(REALESRGAN_MODELS.keys()),
                value="RealESRGAN_x4plus_anime_6B",
                info="anime_6B: アニメ/イラスト向け、x4plus: 汎用/実写向け"
            )

            gr.Markdown("---")
            gr.Markdown("### SD Upscaler設定（SD使用時のみ）")

            prompt_input = gr.Textbox(
                label="プロンプト（任意）",
                placeholder="例: highly detailed, sharp focus",
                lines=2,
                value=""
            )
            gr.Markdown(f"*自動で先頭に追加: `{PROMPT_PREFIX}`*")

            negative_prompt_input = gr.Textbox(
                label="Negative プロンプト",
                value=DEFAULT_NEGATIVE,
                lines=3
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

            denoising_input = gr.Slider(
                label="Denoising Strength",
                minimum=0.1,
                maximum=1.0,
                value=0.3,
                step=0.05,
                info="小さいほど元画像を維持"
            )

            seed_input = gr.Number(
                label="Seed値",
                value=42
            )

            generate_btn = gr.Button("アップスケール実行", variant="primary")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    def on_upscale(
        input_image,
        mode,
        realesrgan_model,
        prompt,
        negative_prompt,
        steps,
        guidance,
        denoising,
        seed
    ):
        # モード名を内部値に変換
        mode_value = UPSCALE_MODES[mode]
        return upscale_image(
            input_image,
            mode_value,
            realesrgan_model,
            prompt,
            negative_prompt,
            steps,
            guidance,
            denoising,
            int(seed)
        )

    # イベント設定
    generate_btn.click(
        fn=on_upscale,
        inputs=[
            input_image,
            mode_input,
            realesrgan_model_input,
            prompt_input,
            negative_prompt_input,
            steps_input,
            guidance_input,
            denoising_input,
            seed_input
        ],
        outputs=[output_gallery, output_message]
    )

    # モードに応じてSD設定を表示/非表示
    def update_sd_visibility(mode):
        is_sd = mode in ["SD Upscaler Only", "Real-ESRGAN → SD"]
        return [
            gr.update(visible=is_sd),  # prompt
            gr.update(visible=is_sd),  # negative_prompt
            gr.update(visible=is_sd),  # steps
            gr.update(visible=is_sd),  # guidance
            gr.update(visible=is_sd),  # denoising
        ]

    mode_input.change(
        fn=update_sd_visibility,
        inputs=[mode_input],
        outputs=[
            prompt_input,
            negative_prompt_input,
            steps_input,
            guidance_input,
            denoising_input
        ]
    )
