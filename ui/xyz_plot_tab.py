"""X/Y/Zプロット タブ

同じseedでパラメータを変えながら比較画像を生成
"""
import gradio as gr

from config import VAE_FILES, request_stop
from pipelines.xyz_plot import generate_xyz_plot, PLOT_PARAMETERS
from .common import create_prompt_inputs, create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector


def create_xyz_plot_tab():
    """X/Y/Zプロットタブを作成"""
    with gr.Row():
        with gr.Column():
            gr.Markdown("### プロンプト設定")
            prompt_input, negative_prompt_input = create_prompt_inputs()

            gr.Markdown("---")
            gr.Markdown("### 基本パラメータ（デフォルト値）")

            with gr.Row():
                seed_input = gr.Number(
                    label="Seed値",
                    value=42,
                    info="同じseedで各パラメータを変化させます"
                )

            with gr.Row():
                width_input = gr.Slider(
                    label="幅",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64
                )
                height_input = gr.Slider(
                    label="高さ",
                    minimum=256,
                    maximum=1024,
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

        with gr.Column():
            gr.Markdown("### X/Y/Z プロット設定")
            gr.Markdown("""
**値の指定方法:**
- カンマ区切り: `5, 7, 9, 11`
- 範囲指定: `5-15:2` (5から15まで2刻み)
- 混合: `5, 7-11:2, 15`
            """)

            with gr.Group():
                gr.Markdown("#### X軸（横方向）")
                with gr.Row():
                    x_param = gr.Dropdown(
                        label="パラメータ",
                        choices=list(PLOT_PARAMETERS.keys()),
                        value="CFG Scale",
                        scale=1
                    )
                    x_values = gr.Textbox(
                        label="値（カンマ区切り）",
                        placeholder="例: 5, 7, 9, 11",
                        value="5, 7, 9, 11",
                        scale=2
                    )

            with gr.Group():
                gr.Markdown("#### Y軸（縦方向）")
                with gr.Row():
                    y_param = gr.Dropdown(
                        label="パラメータ",
                        choices=list(PLOT_PARAMETERS.keys()),
                        value="Steps",
                        scale=1
                    )
                    y_values = gr.Textbox(
                        label="値（カンマ区切り）",
                        placeholder="例: 15, 20, 25, 30",
                        value="15, 20, 25, 30",
                        scale=2
                    )

            with gr.Group():
                gr.Markdown("#### Z軸（複数グリッド生成）")
                with gr.Row():
                    z_param = gr.Dropdown(
                        label="パラメータ",
                        choices=list(PLOT_PARAMETERS.keys()),
                        value="なし",
                        scale=1
                    )
                    z_values = gr.Textbox(
                        label="値（カンマ区切り）",
                        placeholder="例: 42, 100, 200",
                        value="",
                        scale=2
                    )

            # 生成ボタン
            with gr.Row():
                generate_btn = gr.Button("X/Y/Zプロット生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

            gr.Markdown("---")
            output_gallery, output_message = create_output_components()

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_xyz_plot,
        inputs=[
            prompt_input,
            negative_prompt_input,
            seed_input,
            width_input,
            height_input,
            steps_input,
            guidance_input,
            vae_input,
            model_input,
            scheduler_input,
            *lora_components,  # lora1, weight1, lora2, weight2, lora3, weight3
            x_param,
            x_values,
            y_param,
            y_values,
            z_param,
            z_values
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
