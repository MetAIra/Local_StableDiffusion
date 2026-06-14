"""IP-Adapter タブ

参考画像から顔やスタイルを保持しながら新しい画像を生成
"""
import gradio as gr

from config import (
    VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, DEFAULT_MODEL,
    IP_ADAPTER_MODELS_SD15, IP_ADAPTER_MODELS_SDXL,
    IP_ADAPTER_DEFAULT_SCALE, IP_ADAPTER_DEFAULT_STEPS,
    is_sdxl_model, format_catalog_info, get_model_dropdown_choices,
    request_stop
)
from pipelines.ip_adapter import generate_with_ip_adapter
from .common import create_output_components, create_multi_lora_selector, create_scheduler_selector


def get_ip_adapter_dropdown_choices(model_name: str) -> list[tuple[str, str]]:
    """モデルに応じたIP-Adapterの選択肢を取得（説明付き）"""
    if is_sdxl_model(model_name):
        models = IP_ADAPTER_MODELS_SDXL
    else:
        models = IP_ADAPTER_MODELS_SD15

    choices = []
    for name, config in models.items():
        label = f"{name} - {config['description']}"
        choices.append((label, name))
    return choices


def get_default_ip_adapter(model_name: str) -> str:
    """デフォルトのIP-Adapterを取得"""
    if is_sdxl_model(model_name):
        return "ip-adapter-plus-face_sdxl_vit-h"
    return "ip-adapter-plus-face_sd15"


def create_ip_adapter_tab():
    """IP-Adapterタブを作成"""
    gr.Markdown("""
    ### IP-Adapter: 参考画像から顔・スタイルを保持して生成

    参考画像の顔の特徴やスタイルを保持しながら、新しい画像を生成します。
    - **顔特化モデル**: 顔の特徴を強く保持（ポートレート向け）
    - **スタイル転写モデル**: 画像全体のスタイルを転写
    """)

    with gr.Row():
        with gr.Column():
            # 参考画像
            reference_image = gr.Image(
                label="参考画像（顔/スタイルを抽出）",
                type="pil",
                height=300
            )

            gr.Markdown("*この画像から顔やスタイルの特徴を抽出します*")

            # プロンプト
            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: 1girl, beautiful, in a garden, sunny day",
                lines=3,
                value=""
            )
            gr.Markdown(f"*自動で先頭に追加: `{PROMPT_PREFIX}`*")

            negative_prompt_input = gr.Textbox(
                label="Negative プロンプト",
                value=DEFAULT_NEGATIVE,
                lines=3
            )

            # モデル選択
            model_input = gr.Dropdown(
                label="ベースモデル",
                choices=get_model_dropdown_choices(),
                value=DEFAULT_MODEL,
                info="使用するベースモデルを選択（✓ = commercial_safe_models.md 掲載の商用OKモデル）"
            )
            model_info_display = gr.Markdown(
                value=format_catalog_info(DEFAULT_MODEL) if DEFAULT_MODEL else "",
                elem_classes=["model-info"]
            )

            # IP-Adapter選択（モデルに応じて動的に変更）
            initial_choices = get_ip_adapter_dropdown_choices(DEFAULT_MODEL)
            ip_adapter_input = gr.Dropdown(
                label="IP-Adapterモデル",
                choices=[c[0] for c in initial_choices],
                value=initial_choices[2][0] if len(initial_choices) > 2 else initial_choices[0][0],  # 顔特化をデフォルト
                info="顔特化 or スタイル転写を選択"
            )

            # IP-Adapter内部名を保持する隠しコンポーネント
            ip_adapter_name_state = gr.State(value=get_default_ip_adapter(DEFAULT_MODEL))

            # IP-Adapter強度
            ip_adapter_scale = gr.Slider(
                label="IP-Adapter強度",
                minimum=0.0,
                maximum=1.5,
                value=IP_ADAPTER_DEFAULT_SCALE,
                step=0.05,
                info="0=影響なし、0.7=推奨、1.0+=強い影響"
            )

            with gr.Row():
                num_images_input = gr.Slider(
                    label="生成枚数",
                    minimum=1,
                    maximum=20,
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
                    value=IP_ADAPTER_DEFAULT_STEPS,
                    step=1
                )
                guidance_input = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=15,
                    value=7.0,
                    step=0.5
                )

            vae_input = gr.Dropdown(
                label="VAE",
                choices=list(VAE_FILES.keys()),
                value="CleanVAE"
            )

            scheduler_input = create_scheduler_selector()

            lora_components = create_multi_lora_selector(num_slots=3)

            with gr.Row():
                generate_btn = gr.Button("生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_gallery, output_message = create_output_components()

            gr.Markdown("""
            ---
            ### Tips
            - **顔を保持したい場合**: `ip-adapter-plus-face` または `ip-adapter-full-face` を選択
            - **スタイルを転写したい場合**: `ip-adapter-plus` を選択
            - **強度調整**: 0.5-0.8が自然な結果になりやすい。1.0以上は参考画像に強く引っ張られる
            - **プロンプトとの併用**: IP-Adapterは参考画像の特徴を抽出し、プロンプトで新しい要素を追加できる
            """)

    # モデル変更時にIP-Adapter選択肢とモデル情報を更新
    def update_ip_adapter_choices(model_name):
        choices = get_ip_adapter_dropdown_choices(model_name)
        choice_labels = [c[0] for c in choices]
        # 顔特化モデルをデフォルトに
        default_idx = 2 if len(choices) > 2 else 0
        default_value = choice_labels[default_idx]
        default_name = choices[default_idx][1]
        info_text = format_catalog_info(model_name)
        return gr.update(choices=choice_labels, value=default_value), default_name, info_text

    model_input.change(
        fn=update_ip_adapter_choices,
        inputs=[model_input],
        outputs=[ip_adapter_input, ip_adapter_name_state, model_info_display]
    )

    # IP-Adapter選択時に内部名を更新
    def update_ip_adapter_name(ip_adapter_display, model_name):
        choices = get_ip_adapter_dropdown_choices(model_name)
        for label, name in choices:
            if label == ip_adapter_display:
                return name
        return get_default_ip_adapter(model_name)

    ip_adapter_input.change(
        fn=update_ip_adapter_name,
        inputs=[ip_adapter_input, model_input],
        outputs=[ip_adapter_name_state]
    )

    # 生成ボタンのラッパー関数
    def generate_wrapper(
        reference_image, prompt, negative_prompt, num_images, seed,
        width, height, steps, guidance, ip_adapter_scale, ip_adapter_name,
        vae, model, scheduler,
        lora1, weight1, lora2, weight2, lora3, weight3
    ):
        return generate_with_ip_adapter(
            reference_image=reference_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            ip_adapter_scale=ip_adapter_scale,
            ip_adapter_name=ip_adapter_name,
            vae_name=vae,
            model_name=model,
            scheduler_name=scheduler,
            lora1=lora1, weight1=weight1,
            lora2=lora2, weight2=weight2,
            lora3=lora3, weight3=weight3
        )

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            reference_image,
            prompt_input,
            negative_prompt_input,
            num_images_input,
            seed_input,
            width_input,
            height_input,
            steps_input,
            guidance_input,
            ip_adapter_scale,
            ip_adapter_name_state,
            vae_input,
            model_input,
            scheduler_input,
            *lora_components
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
