"""Inpainting タブ - マスク描画で部分再生成"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, request_stop
from pipelines.inpaint import generate_inpaint
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector, create_face_restore_section


def create_inpaint_tab():
    """Inpaintingタブを作成（マスク描画対応）"""
    gr.Markdown("""
    ### 使い方
    1. 画像をアップロード
    2. ブラシツールで再生成したい領域を塗りつぶす（白い部分が再生成されます）
    3. プロンプトを入力して生成
    """)

    with gr.Row():
        with gr.Column():
            # マスク描画対応のImageEditor
            input_image = gr.ImageEditor(
                label="入力画像（ブラシで再生成する領域をマスク）",
                type="numpy",
                brush=gr.Brush(
                    default_size=30,
                    colors=["#FFFFFF"],
                    default_color="#FFFFFF",
                    color_mode="fixed"
                ),
                eraser=gr.Eraser(default_size=30),
                height=512,
                layers=True,
                transforms=[]
            )

            gr.Markdown("**ヒント**: ブラシで白く塗った領域が再生成されます。消しゴムでマスクを修正できます。")

            prompt_input = gr.Textbox(
                label="プロンプト（再生成する内容）",
                placeholder="例: beautiful face, detailed eyes, smile",
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
                label="Strength（変化の強度）",
                minimum=0.1,
                maximum=1.0,
                value=0.75,
                step=0.05,
                info="0.1=元画像に近い、1.0=完全に再生成"
            )

            mask_blur_input = gr.Slider(
                label="マスクぼかし（境界の滑らかさ）",
                minimum=0,
                maximum=20,
                value=4,
                step=1,
                info="大きいほど境界が自然にブレンド"
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
                generate_btn = gr.Button("Inpaint 生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_inpaint,
        inputs=[
            input_image,  # ImageEditorからはdict形式で入力画像とマスク両方が渡される
            input_image,  # 同じ入力からマスクを抽出
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
            face_restore_weight,
            mask_blur_input
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
