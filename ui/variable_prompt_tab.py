"""Variable Prompt バッチ生成 タブ

固定プロンプト + 変数テンプレートで大量の画像を生成
"""
import gradio as gr

from config import VAE_FILES, DEFAULT_NEGATIVE, PROMPT_PREFIX, request_stop
from pipelines.variable_prompt import generate_variable_prompt, parse_variable_definitions
from .common import create_output_components, create_multi_lora_selector, create_model_selector, create_scheduler_selector


def create_variable_prompt_tab():
    """Variable Promptタブを作成"""
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 固定プロンプト")
            fixed_prompt_input = gr.Textbox(
                label="固定部分（全画像に適用）",
                placeholder="例: (best quality:1.2), ultra detailed, anime style, 1 boy, human-like android, full body, standing",
                lines=3,
                value=""
            )
            use_prefix_input = gr.Checkbox(
                label="品質タグを追加",
                value=False,
                info="チェックするとプレフィックスが追加されます（トークン数に注意）"
            )
            gr.Markdown(f"*品質タグ: `{PROMPT_PREFIX}`*")

            gr.Markdown("### 変数テンプレート")
            variable_template_input = gr.Textbox(
                label="変数部分（{変数名}で参照）",
                placeholder="例: {age} years old male NPC, {outfit_style}",
                lines=2,
                value=""
            )
            gr.Markdown("*`{変数名}` の形式で変数を参照します*")

            gr.Markdown("### Negative プロンプト")
            negative_prompt_input = gr.Textbox(
                label="Negative プロンプト",
                value=DEFAULT_NEGATIVE,
                lines=3
            )
            gr.Markdown("*EasyNegativeは自動で追加されます*")

            gr.Markdown("---")
            gr.Markdown("### 生成パラメータ")

            with gr.Row():
                num_seed_variations_input = gr.Slider(
                    label="シード変化数",
                    minimum=1,
                    maximum=10000,
                    value=10,
                    step=1,
                    info="各組み合わせに対して生成するシード数（上限10000）"
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

            with gr.Accordion("出力オプション", open=False):
                generate_grids_input = gr.Checkbox(
                    label="グリッド画像を生成",
                    value=True,
                    info="変数ごとにグリッド比較画像を生成します"
                )
                grid_groupby_input = gr.Radio(
                    label="グリッドのグループ化",
                    choices=["first_variable", "seed", "none"],
                    value="first_variable",
                    info="first_variable: 最初の変数でグループ化"
                )

        with gr.Column():
            gr.Markdown("### 変数定義")
            variable_definitions_input = gr.Textbox(
                label="変数と値（1行1変数）",
                placeholder="""age: 20, 25, 30, 35
outfit_style: casual clothes, formal suit, military uniform, fantasy armor
hair_color: black hair, blonde hair, silver hair""",
                lines=10,
                value=""
            )
            gr.Markdown("""
**形式:** `変数名: 値1, 値2, 値3`

**例:**
```
age: 20, 25, 30, 35
outfit_style: casual, formal, military
```

上記の例では 4 x 3 = 12通りの組み合わせが生成されます。
            """)

            gr.Markdown("---")
            gr.Markdown("### プレビュー")
            preview_info = gr.Markdown("変数を定義してください")

            # 生成ボタン
            with gr.Row():
                generate_btn = gr.Button("バッチ生成開始", variant="primary", size="lg")
                stop_btn = gr.Button("停止", variant="stop")

            gr.Markdown("---")
            output_gallery, output_message = create_output_components()

    # プレビュー更新関数
    def update_preview(variable_defs: str, num_seeds: int):
        """変数定義とシード数から総生成枚数をプレビュー"""
        variables = parse_variable_definitions(variable_defs)

        if not variables:
            return "変数が定義されていません"

        # 組み合わせ数を計算
        num_combos = 1
        for values in variables.values():
            num_combos *= len(values)

        total = num_combos * num_seeds

        # 警告メッセージ
        warning = ""
        if total > 10000:
            warning = "\n\n**Warning:** 生成枚数が10000枚を超えています。上限は10000枚です。"
        elif total > 1000:
            warning = "\n\n**Note:** 生成枚数が多いため、時間がかかります。"

        # 変数サマリー
        var_lines = []
        for k, v in variables.items():
            var_lines.append(f"- **{k}**: {len(v)}通り ({', '.join(v[:3])}{'...' if len(v) > 3 else ''})")

        var_summary = "\n".join(var_lines)

        return f"""**変数:**
{var_summary}

**計算:**
- 組み合わせ数: **{num_combos}通り**
- シード変化数: **x {num_seeds}**
- **総生成枚数: {total}枚**{warning}"""

    # プレビュー更新イベント
    variable_definitions_input.change(
        fn=update_preview,
        inputs=[variable_definitions_input, num_seed_variations_input],
        outputs=[preview_info]
    )
    num_seed_variations_input.change(
        fn=update_preview,
        inputs=[variable_definitions_input, num_seed_variations_input],
        outputs=[preview_info]
    )

    # 生成ボタンイベント
    generate_event = generate_btn.click(
        fn=generate_variable_prompt,
        inputs=[
            fixed_prompt_input,
            variable_template_input,
            variable_definitions_input,
            negative_prompt_input,
            num_seed_variations_input,
            seed_input,
            width_input,
            height_input,
            steps_input,
            guidance_input,
            vae_input,
            model_input,
            scheduler_input,
            *lora_components,  # lora1, weight1, lora2, weight2, lora3, weight3
            generate_grids_input,
            grid_groupby_input,
            use_prefix_input
        ],
        outputs=[output_gallery, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
