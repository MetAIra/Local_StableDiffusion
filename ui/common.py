"""共通UIコンポーネント"""
import gradio as gr

from config import (
    DEFAULT_NEGATIVE, PROMPT_PREFIX,
    LORA_FILES, DEFAULT_MODEL,
    SCHEDULERS, DEFAULT_SCHEDULER,
    FACE_RESTORE_METHODS,
    format_catalog_info, get_model_dropdown_choices,
)


def create_prompt_inputs():
    """プロンプト入力コンポーネントを作成"""
    prompt = gr.Textbox(
        label="プロンプト",
        placeholder="例: 1girl, beautiful, masterpiece",
        lines=3,
        value=""
    )
    gr.Markdown(f"*自動で先頭に追加: `{PROMPT_PREFIX}`*")

    negative_prompt = gr.Textbox(
        label="Negative プロンプト",
        value=DEFAULT_NEGATIVE,
        lines=4
    )
    gr.Markdown("*EasyNegativeは自動で追加されます*")

    return prompt, negative_prompt


def create_multi_lora_selector(num_slots: int = 3):
    """複数LoRA選択コンポーネントを作成

    Args:
        num_slots: LoRAスロット数（デフォルト3）

    Returns:
        [(lora1, weight1), (lora2, weight2), ...] のフラットなリスト
    """
    components = []
    with gr.Accordion("LoRA設定（複数選択可）", open=True):
        gr.Markdown("*複数のLoRAを重ねがけできます。使用しないスロットは「なし」のままにしてください。*")
        for i in range(num_slots):
            with gr.Row():
                lora = gr.Dropdown(
                    label=f"LoRA {i+1}",
                    choices=list(LORA_FILES.keys()),
                    value="なし",
                    scale=2
                )
                weight = gr.Slider(
                    label=f"Weight {i+1}",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    scale=1
                )
            components.extend([lora, weight])
    return components


def create_model_selector():
    """モデル選択コンポーネントを作成（カタログ情報付き）"""
    model = gr.Dropdown(
        label="ベースモデル",
        choices=get_model_dropdown_choices(),
        value=DEFAULT_MODEL,
        info="使用するベースモデルを選択"
    )
    model_info = gr.Markdown(
        value=format_catalog_info(DEFAULT_MODEL) if DEFAULT_MODEL else "",
        elem_classes=["model-info"]
    )

    model.change(
        fn=lambda name: format_catalog_info(name),
        inputs=[model],
        outputs=[model_info]
    )

    return model


def create_scheduler_selector():
    """スケジューラ選択コンポーネントを作成"""
    scheduler = gr.Dropdown(
        label="サンプラー/スケジューラ",
        choices=list(SCHEDULERS.keys()),
        value=DEFAULT_SCHEDULER,
        info="サンプリング方式を選択"
    )
    return scheduler


def create_face_restore_section():
    """顔修正（自動補正）セクションを作成"""
    gr.Markdown("---")
    gr.Markdown("### 顔修正（自動補正）")

    face_restore_enabled = gr.Checkbox(
        label="生成後に顔を自動補正する",
        value=False
    )

    face_restore_method = gr.Radio(
        label="修正方式",
        choices=list(FACE_RESTORE_METHODS.keys()),
        value="GFPGAN",
        visible=False
    )

    face_restore_weight = gr.Slider(
        label="修正強度",
        minimum=0.0,
        maximum=1.0,
        value=0.5,
        step=0.1,
        visible=False,
        info="0=修正なし、1=最大修正"
    )

    # 顔修正オプションの表示/非表示
    def toggle_face_restore_options(enabled):
        return [
            gr.update(visible=enabled),
            gr.update(visible=enabled)
        ]

    face_restore_enabled.change(
        fn=toggle_face_restore_options,
        inputs=[face_restore_enabled],
        outputs=[face_restore_method, face_restore_weight]
    )

    return face_restore_enabled, face_restore_method, face_restore_weight


def create_output_components():
    """出力コンポーネントを作成"""
    gallery = gr.Gallery(
        label="生成された画像",
        columns=4,
        height="auto"
    )
    message = gr.Textbox(label="ステータス", interactive=False)
    return gallery, message
