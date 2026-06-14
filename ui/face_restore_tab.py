"""顔修正 タブ"""
import gradio as gr

from config import FACE_RESTORE_METHODS, GFPGAN_VERSIONS
from pipelines.face_restore import restore_face_image
from .common import create_output_components


def create_face_restore_tab():
    """顔修正タブを作成"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="入力画像",
                type="numpy"
            )

            gr.Markdown("### 顔修正設定")

            method_input = gr.Radio(
                label="修正方式",
                choices=list(FACE_RESTORE_METHODS.keys()),
                value="GFPGAN",
                info="GFPGAN: 高速で安定、CodeFormer: より自然な結果"
            )

            gr.Markdown("---")
            gr.Markdown("### GFPGAN設定")

            gfpgan_version_input = gr.Dropdown(
                label="GFPGANバージョン",
                choices=list(GFPGAN_VERSIONS.keys()),
                value="v1.4 (推奨)"
            )

            gfpgan_weight_input = gr.Slider(
                label="修正強度 (weight)",
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                info="0=修正なし、1=最大修正"
            )

            gr.Markdown("---")
            gr.Markdown("### CodeFormer設定")

            codeformer_fidelity_input = gr.Slider(
                label="忠実度 (fidelity)",
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                info="0=品質重視、1=元画像忠実"
            )

            generate_btn = gr.Button("顔修正実行", variant="primary")

        with gr.Column():
            output_gallery, output_message = create_output_components()

    def on_restore(
        input_image,
        method,
        gfpgan_version,
        gfpgan_weight,
        codeformer_fidelity
    ):
        # 設定値を内部値に変換
        method_value = FACE_RESTORE_METHODS[method]
        version_value = GFPGAN_VERSIONS[gfpgan_version]

        return restore_face_image(
            input_image,
            method_value,
            version_value,
            gfpgan_weight,
            codeformer_fidelity
        )

    # イベント設定
    generate_btn.click(
        fn=on_restore,
        inputs=[
            input_image,
            method_input,
            gfpgan_version_input,
            gfpgan_weight_input,
            codeformer_fidelity_input
        ],
        outputs=[output_gallery, output_message]
    )

    # メソッドに応じて設定を表示/非表示
    def update_settings_visibility(method):
        is_gfpgan = method == "GFPGAN"
        return [
            gr.update(visible=is_gfpgan),   # gfpgan_version
            gr.update(visible=is_gfpgan),   # gfpgan_weight
            gr.update(visible=not is_gfpgan),  # codeformer_fidelity
        ]

    method_input.change(
        fn=update_settings_visibility,
        inputs=[method_input],
        outputs=[
            gfpgan_version_input,
            gfpgan_weight_input,
            codeformer_fidelity_input
        ]
    )
