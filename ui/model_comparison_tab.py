"""モデル比較タブ

複数モデルをそれぞれの最適パラメータで画像生成し、比較グリッドを出力する。
"""
import gradio as gr

from config import (
    VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, DEFAULT_MODEL,
    request_stop, format_catalog_info, is_sdxl_model, get_model_dropdown_choices,
)
from pipelines.model_comparison import (
    generate_model_comparison, parse_model_configs, generate_config_from_catalog,
)
from .common import create_multi_lora_selector, create_output_components


def create_model_comparison_tab():
    """モデル比較タブを作成"""

    def update_catalog_info(model_name: str) -> str:
        """カタログ情報を更新"""
        if not model_name:
            return ""
        info = format_catalog_info(model_name)
        model_type = "SDXL" if is_sdxl_model(model_name) else "SD1.5"
        type_info = f"**モデルタイプ:** {model_type}\n\n"
        return type_info + info if info else type_info

    def add_model_config(current_text: str, model_name: str) -> str:
        """カタログからモデル設定を追加"""
        if not model_name:
            return current_text

        new_config = generate_config_from_catalog(model_name)

        if current_text.strip():
            return current_text.strip() + "\n\n" + new_config
        else:
            return new_config

    def update_preview(config_text: str, num_seeds: int) -> str:
        """プレビュー情報を更新"""
        if not config_text.strip():
            return "モデル設定を入力してください"

        configs, errors = parse_model_configs(config_text)

        if errors:
            return "**エラー:**\n" + "\n".join(f"- {e}" for e in errors)

        if not configs:
            return "有効なモデル設定がありません"

        total_images = len(configs) * int(num_seeds)

        lines = [f"**モデル数:** {len(configs)}"]
        lines.append(f"**シード数:** {int(num_seeds)}")
        lines.append(f"**総画像数:** {total_images}枚")

        if total_images > 1000:
            lines.append("\n**警告:** 生成画像数が上限（1000枚）を超えています")

        lines.append("\n**モデル一覧:**")
        for cfg in configs:
            model_type = "SDXL" if cfg.get("is_sdxl") else "SD1.5"
            lines.append(f"- {cfg['model']} ({model_type})")
            lines.append(f"  - {cfg['scheduler']}, steps={cfg['steps']}, cfg={cfg['cfg']}")
            lines.append(f"  - {cfg['width']}x{cfg['height']}")

        return "\n".join(lines)

    with gr.Row():
        # 左カラム: 共通設定
        with gr.Column(scale=1):
            gr.Markdown("### 共通設定")

            # プロンプト
            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: 1girl, beautiful, masterpiece",
                lines=3,
                value=""
            )

            use_prefix = gr.Checkbox(
                label="品質タグプレフィックスを追加",
                value=True,
                info=f"SD1.5: {PROMPT_PREFIX[:30]}... / SDXL用も自動切替"
            )

            negative_prompt_input = gr.Textbox(
                label="Negative プロンプト",
                value=DEFAULT_NEGATIVE,
                lines=4
            )
            gr.Markdown("*SD1.5の場合、EasyNegativeは自動で追加されます*")

            gr.Markdown("---")
            gr.Markdown("### シード設定")

            with gr.Row():
                seed_input = gr.Number(
                    label="ベースSeed値",
                    value=42,
                    precision=0
                )
                num_seeds_input = gr.Slider(
                    label="シード変化数",
                    minimum=1,
                    maximum=100,
                    value=3,
                    step=1,
                    info="各モデルで生成するシード数"
                )

            gr.Markdown("---")
            gr.Markdown("### VAE設定")

            vae_input = gr.Dropdown(
                label="VAE",
                choices=list(VAE_FILES.keys()),
                value="CleanVAE",
                info="全モデル共通で使用"
            )

            gr.Markdown("---")
            lora_components = create_multi_lora_selector(num_slots=3)

            gr.Markdown("---")
            gr.Markdown("### 出力オプション")

            generate_grid_input = gr.Checkbox(
                label="比較グリッドを生成",
                value=True,
                info="X軸=モデル、Y軸=シード の比較画像"
            )

        # 右カラム: モデル別設定
        with gr.Column(scale=1):
            gr.Markdown("### モデル別設定")

            config_text_input = gr.Textbox(
                label="モデル設定（INI形式）",
                placeholder="""# 例:
[bluePencil_v10]
scheduler: Euler a
steps: 40
cfg: 7
width: 512
height: 512

[waiIllustriousSDXL_v160]
scheduler: Euler a
steps: 20
cfg: 5
width: 1024
height: 1024""",
                lines=15
            )

            gr.Markdown("---")
            gr.Markdown("### カタログからモデルを追加")

            with gr.Row():
                catalog_model_input = gr.Dropdown(
                    label="モデル選択（✓ = 商用OK）",
                    choices=get_model_dropdown_choices(),
                    value=DEFAULT_MODEL,
                    scale=3
                )
                add_model_btn = gr.Button("追加", scale=1)

            catalog_info_display = gr.Markdown(
                value=update_catalog_info(DEFAULT_MODEL) if DEFAULT_MODEL else "",
                elem_classes=["catalog-info"]
            )

            # ヘルプ
            with gr.Accordion("設定フォーマットのヘルプ", open=False):
                gr.Markdown("""
**設定フォーマット:**
```ini
# コメント行（#で始まる）

[モデル名]
scheduler: スケジューラ名
steps: ステップ数
cfg: CFGスケール値
width: 幅
height: 高さ
```

**利用可能なスケジューラ:**
DDIM, Euler, Euler a, DPM++ 2M, DPM++ 2M Karras, DPM++ SDE, DPM++ SDE Karras, UniPC, LMS, PNDM, Heun

**省略時のデフォルト値:**
- SD1.5: steps=20, cfg=7.0, 512x512
- SDXL: steps=20, cfg=5.0, 1024x1024

**注意:**
- モデル名はファイル名（拡張子なし）と完全一致が必要
- モデルタイプ（SD1.5/SDXL）は自動判定
- 品質タグはモデルタイプに応じて自動切替
                """)

            gr.Markdown("---")
            gr.Markdown("### プレビュー")

            preview_display = gr.Markdown(
                value="モデル設定を入力してください"
            )

            gr.Markdown("---")

            # 生成ボタン
            with gr.Row():
                generate_btn = gr.Button("モデル比較生成", variant="primary", scale=2)
                stop_btn = gr.Button("停止", variant="stop", scale=1)

            # 出力
            output_gallery, output_message = create_output_components()

    # イベント設定

    # カタログモデル選択変更 → 情報表示更新
    catalog_model_input.change(
        fn=update_catalog_info,
        inputs=[catalog_model_input],
        outputs=[catalog_info_display]
    )

    # 追加ボタン → テキストエリアに設定追加
    add_model_btn.click(
        fn=add_model_config,
        inputs=[config_text_input, catalog_model_input],
        outputs=[config_text_input]
    )

    # テキストエリア/シード数変更 → プレビュー更新
    config_text_input.change(
        fn=update_preview,
        inputs=[config_text_input, num_seeds_input],
        outputs=[preview_display]
    )

    num_seeds_input.change(
        fn=update_preview,
        inputs=[config_text_input, num_seeds_input],
        outputs=[preview_display]
    )

    # 生成ボタン
    generate_event = generate_btn.click(
        fn=generate_model_comparison,
        inputs=[
            prompt_input,
            negative_prompt_input,
            config_text_input,
            seed_input,
            num_seeds_input,
            vae_input,
            *lora_components,  # lora1, weight1, lora2, weight2, lora3, weight3
            use_prefix,
            generate_grid_input,
        ],
        outputs=[output_gallery, output_message]
    )

    # 停止ボタン
    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
