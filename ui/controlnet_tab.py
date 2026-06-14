"""ControlNet タブ（拡張版）

Lineart、Scribble、Tile追加 + 複数ControlNet同時使用対応
"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, request_stop
from pipelines.controlnet import (
    generate_with_controlnet,
    generate_with_multi_controlnet,
    get_control_image
)
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector


# ControlNetタイプの選択肢
CONTROLNET_CHOICES = [
    ("OpenPose（ポーズ制御）", "openpose"),
    ("Canny（エッジ検出）", "canny"),
    ("Depth（深度マップ）", "depth"),
    ("Lineart（線画）", "lineart"),
    ("Scribble（落書き）", "scribble"),
    ("Tile（ディテール維持）", "tile"),
]


def _create_canny_params():
    """Canny用パラメータのアコーディオンを作成"""
    with gr.Accordion("Canny パラメータ", open=False):
        with gr.Row():
            canny_low = gr.Slider(
                label="下限閾値",
                minimum=0,
                maximum=255,
                value=100,
                step=1
            )
            canny_high = gr.Slider(
                label="上限閾値",
                minimum=0,
                maximum=255,
                value=200,
                step=1
            )
    return canny_low, canny_high


def create_controlnet_tab():
    """ControlNetタブを作成（シングルモード）"""
    with gr.Row():
        with gr.Column():
            # 画像入力
            input_image = gr.Image(
                label="入力画像",
                type="pil",
                height=300
            )

            # ControlNetタイプ選択
            controlnet_type = gr.Dropdown(
                label="ControlNet タイプ",
                choices=CONTROLNET_CHOICES,
                value="canny"
            )

            # Canny用パラメータ（Canny選択時のみ表示）
            canny_low, canny_high = _create_canny_params()

            # プレビューボタン
            preview_btn = gr.Button("制御画像をプレビュー", variant="secondary")

            # プロンプト入力
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

            # 生成パラメータ
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

            # ControlNet強度
            controlnet_scale_input = gr.Slider(
                label="ControlNet 強度",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="1.0が標準。高いほど制御画像に忠実"
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
            # 制御画像プレビュー
            control_preview = gr.Image(
                label="制御画像プレビュー",
                type="pil",
                height=300
            )

            # 出力
            output_gallery, output_message = create_output_components()

    # プレビューイベント（ボタン押下・入力画像変更・ControlNetタイプ変更で更新）
    for trigger in (preview_btn.click, input_image.change, controlnet_type.change):
        trigger(
            fn=get_control_image,
            inputs=[
                input_image,
                controlnet_type,
                canny_low,
                canny_high
            ],
            outputs=[control_preview]
        )

    # 生成イベント
    generate_event = generate_btn.click(
        fn=generate_with_controlnet,
        inputs=[
            input_image,
            prompt_input,
            negative_prompt_input,
            controlnet_type,
            num_images_input,
            seed_input,
            steps_input,
            guidance_input,
            controlnet_scale_input,
            vae_input,
            model_input,
            scheduler_input,
            canny_low,
            canny_high,
            *lora_components  # lora1, weight1, lora2, weight2, lora3, weight3
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])


def create_multi_controlnet_tab():
    """複数ControlNet同時使用タブを作成"""

    def get_multi_previews(image, cn1, cn2, cn3, canny_low, canny_high):
        """複数ControlNetのプレビューを取得"""
        previews = []
        for cn_type in [cn1, cn2, cn3]:
            if cn_type and cn_type != "なし":
                preview = get_control_image(image, cn_type, canny_low, canny_high)
                previews.append(preview)
            else:
                previews.append(None)
        return previews[0], previews[1], previews[2]

    def generate_multi(
        image, prompt, negative_prompt,
        cn1, cn2, cn3, scale1, scale2, scale3,
        num_images, seed, steps, guidance,
        vae_name, model_name, scheduler_name, canny_low, canny_high,
        lora1, weight1, lora2, weight2, lora3, weight3
    ):
        """複数ControlNetで生成"""
        # 選択されたControlNetと強度をリストにまとめる
        controlnet_types = []
        controlnet_scales = []

        for cn_type, scale in [(cn1, scale1), (cn2, scale2), (cn3, scale3)]:
            if cn_type and cn_type != "なし":
                controlnet_types.append(cn_type)
                controlnet_scales.append(scale)

        if not controlnet_types:
            return [], "少なくとも1つのControlNetを選択してください"

        return generate_with_multi_controlnet(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_types=controlnet_types,
            controlnet_scales=controlnet_scales,
            num_images=num_images,
            seed=seed,
            steps=steps,
            guidance_scale=guidance,
            vae_name=vae_name,
            model_name=model_name,
            scheduler_name=scheduler_name,
            canny_low=canny_low,
            canny_high=canny_high,
            lora1=lora1, weight1=weight1,
            lora2=lora2, weight2=weight2,
            lora3=lora3, weight3=weight3
        )

    with gr.Row():
        with gr.Column():
            # 画像入力
            input_image = gr.Image(
                label="入力画像",
                type="pil",
                height=300
            )

            gr.Markdown("### ControlNet 1")
            with gr.Row():
                cn1_type = gr.Dropdown(
                    label="タイプ",
                    choices=[("なし", "なし")] + CONTROLNET_CHOICES,
                    value="openpose"
                )
                cn1_scale = gr.Slider(
                    label="強度",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05
                )

            gr.Markdown("### ControlNet 2")
            with gr.Row():
                cn2_type = gr.Dropdown(
                    label="タイプ",
                    choices=[("なし", "なし")] + CONTROLNET_CHOICES,
                    value="depth"
                )
                cn2_scale = gr.Slider(
                    label="強度",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.5,
                    step=0.05
                )

            gr.Markdown("### ControlNet 3")
            with gr.Row():
                cn3_type = gr.Dropdown(
                    label="タイプ",
                    choices=[("なし", "なし")] + CONTROLNET_CHOICES,
                    value="なし"
                )
                cn3_scale = gr.Slider(
                    label="強度",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.5,
                    step=0.05
                )

            # Canny用パラメータ
            canny_low, canny_high = _create_canny_params()

            # プレビューボタン
            preview_btn = gr.Button("制御画像をプレビュー", variant="secondary")

            # プロンプト入力
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

            # 生成パラメータ
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
            # 制御画像プレビュー（3つ）
            gr.Markdown("### 制御画像プレビュー")
            with gr.Row():
                control_preview1 = gr.Image(
                    label="ControlNet 1",
                    type="pil",
                    height=200
                )
                control_preview2 = gr.Image(
                    label="ControlNet 2",
                    type="pil",
                    height=200
                )
                control_preview3 = gr.Image(
                    label="ControlNet 3",
                    type="pil",
                    height=200
                )

            # 出力
            output_gallery, output_message = create_output_components()

    # プレビューイベント
    preview_btn.click(
        fn=get_multi_previews,
        inputs=[
            input_image,
            cn1_type, cn2_type, cn3_type,
            canny_low, canny_high
        ],
        outputs=[control_preview1, control_preview2, control_preview3]
    )

    # 生成イベント
    generate_event = generate_btn.click(
        fn=generate_multi,
        inputs=[
            input_image,
            prompt_input,
            negative_prompt_input,
            cn1_type, cn2_type, cn3_type,
            cn1_scale, cn2_scale, cn3_scale,
            num_images_input,
            seed_input,
            steps_input,
            guidance_input,
            vae_input,
            model_input,
            scheduler_input,
            canny_low,
            canny_high,
            *lora_components  # lora1, weight1, lora2, weight2, lora3, weight3
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
