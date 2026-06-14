"""Music Batch タブ

MusicGen用に2種類のバッチ生成UIを提供:
- X/Y/Z Plot: パラメータを軸として変化
- Variable Prompt: プロンプトテンプレートを変化
"""
import gradio as gr

from config import MUSIC_GEN_MODELS, request_stop
from pipelines.music_batch import (
    generate_music_xyz_plot,
    generate_music_variable_prompt,
    MUSIC_PLOT_PARAMETERS,
    MUSIC_BATCH_LIMIT,
    MUSIC_VARIABLE_PROMPT_EXAMPLES,
)
from pipelines.music_gen import MUSIC_PROMPT_EXAMPLES


def _create_xyz_subtab():
    """X/Y/Z Plot サブタブ"""
    with gr.Row():
        with gr.Column():
            gr.Markdown("### プロンプト・モデル")
            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: upbeat jazz with piano and saxophone",
                lines=3,
                value=""
            )
            example_dropdown = gr.Dropdown(
                label="プロンプト例",
                choices=list(MUSIC_PROMPT_EXAMPLES.keys()),
                value=None,
                info="選択するとプロンプトに追加（カンマ区切りで結合）"
            )
            model_input = gr.Dropdown(
                label="MusicGenモデル",
                choices=list(MUSIC_GEN_MODELS.keys()),
                value="musicgen-small",
                info="モデルを軸にすると毎回ロードし直すため、ここでは固定です"
            )

            gr.Markdown("### 基本パラメータ（軸で上書きされない値）")
            with gr.Row():
                duration_input = gr.Slider(
                    label="長さ（秒）", minimum=5, maximum=30, value=10, step=1
                )
                seed_input = gr.Number(
                    label="Seed", value=42, info="-1でランダム（軸でSeedを変えない場合）"
                )
            with gr.Row():
                temperature_input = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1
                )
                guidance_input = gr.Slider(
                    label="Guidance Scale", minimum=1.0, maximum=10.0, value=3.0, step=0.5
                )
            with gr.Row():
                top_k_input = gr.Slider(
                    label="Top-K", minimum=0, maximum=500, value=250, step=10
                )
                top_p_input = gr.Slider(
                    label="Top-P", minimum=0.0, maximum=1.0, value=0.0, step=0.05
                )

            melody_input = gr.Audio(
                label="メロディー参照音声 (musicgen-melodyのみ)",
                type="filepath",
                sources=["upload"]
            )

            gr.Markdown("---")
            gr.Markdown(
                "### X/Y/Z 軸設定\n"
                "値の指定形式: カンマ区切り `1,2,3` または範囲 `1-10:2`（1から10まで2刻み）"
            )

            with gr.Row():
                x_param_input = gr.Dropdown(
                    label="X軸パラメータ",
                    choices=list(MUSIC_PLOT_PARAMETERS.keys()),
                    value="Seed"
                )
                x_values_input = gr.Textbox(
                    label="X軸の値", placeholder="例: 1,2,3,4", value="1,2,3,4"
                )
            with gr.Row():
                y_param_input = gr.Dropdown(
                    label="Y軸パラメータ",
                    choices=list(MUSIC_PLOT_PARAMETERS.keys()),
                    value="なし"
                )
                y_values_input = gr.Textbox(label="Y軸の値", value="")
            with gr.Row():
                z_param_input = gr.Dropdown(
                    label="Z軸パラメータ",
                    choices=list(MUSIC_PLOT_PARAMETERS.keys()),
                    value="なし"
                )
                z_values_input = gr.Textbox(label="Z軸の値", value="")

            gr.Markdown(f"*生成上限: {MUSIC_BATCH_LIMIT}件*")

            with gr.Row():
                generate_btn = gr.Button("バッチ生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_files = gr.File(
                label="生成された音声ファイル", file_count="multiple", interactive=False
            )
            output_message = gr.Textbox(label="ステータス", interactive=False, lines=10)

    def on_example_select(example_name, current_prompt):
        if not example_name:
            return current_prompt
        example_prompt = MUSIC_PROMPT_EXAMPLES.get(example_name, "")
        if current_prompt:
            return f"{current_prompt}, {example_prompt}"
        return example_prompt

    example_dropdown.change(
        fn=on_example_select,
        inputs=[example_dropdown, prompt_input],
        outputs=[prompt_input]
    )

    event = generate_btn.click(
        fn=generate_music_xyz_plot,
        inputs=[
            prompt_input, model_input,
            duration_input, temperature_input, guidance_input,
            top_k_input, top_p_input, seed_input, melody_input,
            x_param_input, x_values_input,
            y_param_input, y_values_input,
            z_param_input, z_values_input,
        ],
        outputs=[output_files, output_message]
    )
    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[event])


def _create_variable_prompt_subtab():
    """Variable Prompt サブタブ"""
    with gr.Row():
        with gr.Column():
            gr.Markdown("### テンプレート例")
            template_example_dropdown = gr.Dropdown(
                label="テンプレート例",
                choices=list(MUSIC_VARIABLE_PROMPT_EXAMPLES.keys()),
                value=None,
                info="選択すると下の3項目を一括で埋めます（既存内容は上書きされます）"
            )

            gr.Markdown("### 固定プロンプト")
            fixed_prompt_input = gr.Textbox(
                label="固定部分（全生成に適用）",
                placeholder="例: cinematic orchestral music",
                lines=2,
                value=""
            )

            gr.Markdown("### 変数テンプレート")
            variable_template_input = gr.Textbox(
                label="変数部分（{変数名}で参照）",
                placeholder="例: {instrument} solo, {tempo} tempo",
                lines=2,
                value=""
            )

            gr.Markdown("### 変数定義")
            variable_definitions_input = gr.Textbox(
                label="変数定義",
                placeholder="形式:\n変数名: 値1, 値2, 値3\n\n例:\ninstrument: piano, violin, guitar\ntempo: slow, fast",
                lines=5,
                value=""
            )

            gr.Markdown("---")
            gr.Markdown("### 生成パラメータ")

            model_input = gr.Dropdown(
                label="MusicGenモデル",
                choices=list(MUSIC_GEN_MODELS.keys()),
                value="musicgen-small"
            )

            with gr.Row():
                duration_input = gr.Slider(
                    label="長さ（秒）", minimum=5, maximum=30, value=10, step=1
                )
                num_seed_variations_input = gr.Slider(
                    label="Seed変化数", minimum=1, maximum=10000, value=1, step=1,
                    info="各組み合わせに対して生成するseed数（合計上限10000件）"
                )
            with gr.Row():
                seed_input = gr.Number(label="ベースSeed", value=42)

            with gr.Row():
                temperature_input = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1
                )
                guidance_input = gr.Slider(
                    label="Guidance Scale", minimum=1.0, maximum=10.0, value=3.0, step=0.5
                )
            with gr.Row():
                top_k_input = gr.Slider(
                    label="Top-K", minimum=0, maximum=500, value=250, step=10
                )
                top_p_input = gr.Slider(
                    label="Top-P", minimum=0.0, maximum=1.0, value=0.0, step=0.05
                )

            melody_input = gr.Audio(
                label="メロディー参照音声 (musicgen-melodyのみ)",
                type="filepath",
                sources=["upload"]
            )

            gr.Markdown(f"*生成上限: {MUSIC_BATCH_LIMIT}件*")

            with gr.Row():
                generate_btn = gr.Button("バッチ生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column():
            output_files = gr.File(
                label="生成された音声ファイル", file_count="multiple", interactive=False
            )
            output_message = gr.Textbox(label="ステータス", interactive=False, lines=10)

            with gr.Accordion("使い方", open=False):
                gr.Markdown("""
**例:**
- 固定プロンプト: `cinematic orchestral music`
- 変数テンプレート: `{instrument} solo, {tempo} tempo`
- 変数定義:
  ```
  instrument: piano, violin, guitar
  tempo: slow, fast
  ```
- → `piano solo, slow tempo` / `piano solo, fast tempo` / `violin solo, slow tempo` ... の6通り
- Seed変化数を 2 にすると 6 × 2 = 12件生成されます
                """)

    def on_template_example_select(example_name):
        if not example_name:
            return gr.update(), gr.update(), gr.update()
        example = MUSIC_VARIABLE_PROMPT_EXAMPLES.get(example_name, {})
        return (
            example.get("fixed", ""),
            example.get("template", ""),
            example.get("variables", ""),
        )

    template_example_dropdown.change(
        fn=on_template_example_select,
        inputs=[template_example_dropdown],
        outputs=[fixed_prompt_input, variable_template_input, variable_definitions_input]
    )

    event = generate_btn.click(
        fn=generate_music_variable_prompt,
        inputs=[
            fixed_prompt_input, variable_template_input, variable_definitions_input,
            model_input, duration_input,
            temperature_input, guidance_input, top_k_input, top_p_input,
            seed_input, num_seed_variations_input, melody_input,
        ],
        outputs=[output_files, output_message]
    )
    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[event])


def create_music_batch_tab():
    """Music Batchタブを作成（X/Y/Z PlotとVariable Promptのサブタブ）"""
    with gr.Tabs():
        with gr.TabItem("X/Y/Z Plot"):
            _create_xyz_subtab()
        with gr.TabItem("Variable Prompt"):
            _create_variable_prompt_subtab()
