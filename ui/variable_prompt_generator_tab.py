"""Variable Prompt Generator タブ

Claude APIを使って日本語の説明からVariable Prompt用の
プロンプト一式を自動生成するタブ。
"""
import gradio as gr

from config import (
    T2K_STYLES,
    DEFAULT_T2K_STYLE,
    VPG_CATEGORY_PRESETS,
)
from pipelines.variable_prompt_generator import generate_variable_prompt_template


def create_variable_prompt_generator_tab():
    """Variable Prompt Generator タブを作成"""

    gr.Markdown("""
    ### Variable Prompt Generator
    日本語で「どんなバリエーションの画像を生成したいか」を説明すると、
    Variable Prompt用のプロンプト一式（固定プロンプト・変数テンプレート・変数定義・ネガティブプロンプト）を自動生成します。

    生成結果はそのまま **Variable Prompt** タブにコピーして使えます。

    *Claude API を使用します。ANTHROPIC_API_KEY の設定が必要です。*
    """)

    with gr.Row():
        # 左列: 入力
        with gr.Column(scale=1):
            category_preset = gr.Dropdown(
                label="カテゴリプリセット",
                choices=list(VPG_CATEGORY_PRESETS.keys()),
                value="自由入力",
                info="よく使うテーマを選択、または自由入力"
            )

            description_input = gr.Textbox(
                label="生成したいバリエーションの説明（日本語）",
                placeholder="例: 干支（12種類）の動物をそれぞれ擬人化したアニメキャラクターとして描きたい。各キャラクターは和服を着ている。",
                lines=5,
            )

            with gr.Row():
                style = gr.Dropdown(
                    label="スタイル",
                    choices=list(T2K_STYLES.keys()),
                    value=DEFAULT_T2K_STYLE,
                    info="生成する画像のスタイルを選択"
                )

            additional_instructions = gr.Textbox(
                label="追加の指示（任意）",
                placeholder="例: 背景は日本庭園にしてほしい、全身で描いてほしい",
                lines=2,
            )

            generate_btn = gr.Button("Variable Prompt 生成", variant="primary")

        # 右列: 出力
        with gr.Column(scale=1):
            fixed_prompt_output = gr.Textbox(
                label="固定プロンプト（Fixed Prompt）",
                lines=4,
                info="全画像に共通で適用されるプロンプト"
            )
            variable_template_output = gr.Textbox(
                label="変数テンプレート（Variable Template）",
                lines=3,
                info="{変数名}で変数を参照するテンプレート"
            )
            variable_definitions_output = gr.Textbox(
                label="変数定義（Variable Definitions）",
                lines=10,
                info="変数名: 値1, 値2, 値3 の形式"
            )
            negative_prompt_output = gr.Textbox(
                label="ネガティブプロンプト",
                lines=3,
            )
            status_output = gr.Textbox(
                label="ステータス",
                interactive=False,
            )

    # 使い方
    with gr.Accordion("使い方・入力例", open=False):
        gr.Markdown("""
**使い方:**
1. 左側で生成したいバリエーションを日本語で説明
2. 「Variable Prompt 生成」ボタンをクリック
3. 右側に生成されたプロンプト一式が表示される
4. 結果をコピーして **Variable Prompt** タブに貼り付けて画像生成

**入力例:**

> **干支:** 十二支の動物（ネズミ、牛、虎、うさぎ、龍、蛇、馬、羊、猿、鶏、犬、猪）をそれぞれアニメキャラクターとして描く。和服を着た人間の姿で擬人化。

> **星座:** 12星座をそれぞれファンタジーキャラクターとして描く。各星座の特徴を反映した武器や衣装を持たせる。

> **感情:** 同じキャラクターで7つの感情（喜び、怒り、悲しみ、驚き、恐怖、嫌悪、期待）を表現する。

> **RPG職業 x 種族:** ファンタジーRPGの職業（戦士、魔法使い、僧侶、盗賊、弓使い）と種族（人間、エルフ、ドワーフ）の全組み合わせを描く。

> **1日の時間帯:** 朝・昼・夕方・夜の4つの時間帯で、同じ街角の風景を描く。季節は秋。
        """)

    # プリセット選択時に説明欄を更新
    def on_preset_change(preset_name: str):
        preset_text = VPG_CATEGORY_PRESETS.get(preset_name, "")
        return preset_text

    category_preset.change(
        fn=on_preset_change,
        inputs=[category_preset],
        outputs=[description_input]
    )

    # 生成ボタンイベント
    generate_btn.click(
        fn=generate_variable_prompt_template,
        inputs=[
            description_input,
            style,
            additional_instructions,
        ],
        outputs=[
            fixed_prompt_output,
            variable_template_output,
            variable_definitions_output,
            negative_prompt_output,
            status_output,
        ],
    )
