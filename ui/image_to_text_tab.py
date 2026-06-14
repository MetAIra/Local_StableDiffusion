"""Image to Text タブ UI

画像から Stable Diffusion 向けプロンプトを生成するUI。
Claude API + CLIP Interrogator を使用。
"""
import gradio as gr

from config import (
    I2T_STYLES,
    DEFAULT_I2T_STYLE,
    CLIP_MODELS,
    DEFAULT_CLIP_MODEL,
)
from pipelines.image_to_text import (
    generate_prompt_from_image,
    format_settings_display,
)


def create_image_to_text_tab():
    """Image to Text タブを作成"""

    with gr.Row():
        # 左カラム: 入力
        with gr.Column(scale=1):
            gr.Markdown("### 入力画像")

            input_image = gr.Image(
                label="分析する画像",
                type="pil",
                height=400
            )

            gr.Markdown("### 分析設定")

            style_input = gr.Dropdown(
                label="スタイルヒント",
                choices=[(v, k) for k, v in I2T_STYLES.items()],
                value=DEFAULT_I2T_STYLE,
                info="画像のスタイルを指定（自動検出も可能）"
            )

            with gr.Accordion("CLIP Interrogator 設定", open=True):
                use_clip = gr.Checkbox(
                    label="CLIP Interrogator を使用",
                    value=True,
                    info="SD特化タグ（アーティスト名等）を補完"
                )

                clip_model_input = gr.Dropdown(
                    label="CLIP モデル",
                    choices=[(v, k) for k, v in CLIP_MODELS.items()],
                    value=DEFAULT_CLIP_MODEL,
                    info="標準: バランス型 / 高精度: SDXL向け",
                    visible=True
                )

                clip_mode_input = gr.Radio(
                    label="CLIP モード",
                    choices=[("高速", "fast"), ("詳細", "full")],
                    value="fast",
                    info="高速: アーティスト・スタイルのみ / 詳細: 全タグ",
                    visible=True
                )

            # CLIP設定の表示切り替え
            def toggle_clip_options(use_clip_val):
                return [
                    gr.update(visible=use_clip_val),
                    gr.update(visible=use_clip_val)
                ]

            use_clip.change(
                fn=toggle_clip_options,
                inputs=[use_clip],
                outputs=[clip_model_input, clip_mode_input]
            )

            generate_btn = gr.Button("プロンプト生成", variant="primary", size="lg")

        # 右カラム: 出力
        with gr.Column(scale=1):
            gr.Markdown("### 生成結果")

            output_prompt = gr.Textbox(
                label="生成されたプロンプト（Ctrl+A → Ctrl+C でコピー）",
                lines=8,
                interactive=True
            )

            output_negative = gr.Textbox(
                label="ネガティブプロンプト",
                lines=4,
                interactive=True
            )

            output_settings = gr.Textbox(
                label="推奨設定・分析結果",
                lines=10,
                interactive=False
            )

            output_status = gr.Textbox(
                label="ステータス",
                interactive=False
            )

            gr.Markdown("---")
            gr.Markdown("### txt2img へ転送")

            with gr.Row():
                copy_to_txt2img_btn = gr.Button(
                    "txt2img にコピー",
                    variant="secondary"
                )
                copy_status = gr.Textbox(
                    label="",
                    interactive=False,
                    show_label=False,
                    max_lines=1
                )

    # 生成ボタンのイベントハンドラ
    def on_generate(image, style, use_clip_val, clip_model, clip_mode):
        if image is None:
            return "", "", "", "画像をアップロードしてください"

        main_prompt, negative_prompt, settings, status = generate_prompt_from_image(
            image=image,
            style=style,
            use_clip=use_clip_val,
            clip_model=clip_model,
            clip_mode=clip_mode
        )

        settings_display = format_settings_display(settings)

        return main_prompt, negative_prompt, settings_display, status

    generate_btn.click(
        fn=on_generate,
        inputs=[
            input_image,
            style_input,
            use_clip,
            clip_model_input,
            clip_mode_input
        ],
        outputs=[
            output_prompt,
            output_negative,
            output_settings,
            output_status
        ]
    )

    # txt2img への State（タブ間共有用）
    # 注: 実際のコピー機能は app.py で実装
    # ここでは State を返してタブ間で共有できるようにする

    # コピーボタンは、生成されたプロンプトを State に保存するだけ
    # 実際の転送は app.py 側で txt2img の入力に接続

    # State を使用した簡易コピー機能
    shared_prompt_state = gr.State(value="")
    shared_negative_state = gr.State(value="")

    def on_copy_click(prompt, negative):
        if not prompt:
            return "", "", "プロンプトが空です"
        return prompt, negative, "コピーしました（txt2img タブで Ctrl+V でペースト）"

    copy_to_txt2img_btn.click(
        fn=on_copy_click,
        inputs=[output_prompt, output_negative],
        outputs=[shared_prompt_state, shared_negative_state, copy_status]
    )
