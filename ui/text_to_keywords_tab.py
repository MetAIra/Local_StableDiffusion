"""Text to Keywords タブ

日本語のテキスト入力からClaude APIを使って
Stable Diffusion用の英語キーワードを生成するタブ。
"""
import gradio as gr

from config import (
    T2K_STYLES,
    DEFAULT_T2K_STYLE,
    T2K_WORD_COUNTS,
    DEFAULT_T2K_WORD_COUNT,
    T2K_DETAIL_LEVELS,
    DEFAULT_T2K_DETAIL_LEVEL,
)
from pipelines.text_to_keywords import generate_keywords_from_text


def create_text_to_keywords_tab():
    """Text to Keywords タブを作成"""

    gr.Markdown("""
    ### Text to Keywords
    日本語で作りたい画像のイメージを入力すると、Stable Diffusion用の英語キーワードを生成します。

    *Claude API を使用します。ANTHROPIC_API_KEY の設定が必要です。*
    """)

    with gr.Row():
        # 左列: 入力
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="画像イメージ（日本語）",
                placeholder="例: 夕焼けの海辺で、白いワンピースを着た女の子が砂浜を歩いている。髪は長くて黒い。夕日が綺麗。",
                lines=5,
            )

            with gr.Row():
                style = gr.Dropdown(
                    label="スタイル",
                    choices=list(T2K_STYLES.keys()),
                    value=DEFAULT_T2K_STYLE,
                    info="生成する画像のスタイルを選択"
                )
                word_count = gr.Dropdown(
                    label="単語数",
                    choices=list(T2K_WORD_COUNTS.keys()),
                    value=DEFAULT_T2K_WORD_COUNT,
                    info="出力するキーワードの量"
                )

            with gr.Row():
                detail_level = gr.Dropdown(
                    label="詳細度",
                    choices=list(T2K_DETAIL_LEVELS.keys()),
                    value=DEFAULT_T2K_DETAIL_LEVEL,
                    info="キーワードの詳細さ"
                )
                include_negative = gr.Checkbox(
                    label="ネガティブプロンプトも生成",
                    value=True,
                    info="避けるべき要素も生成"
                )

            generate_btn = gr.Button("キーワード生成", variant="primary")

        # 右列: 出力
        with gr.Column(scale=1):
            positive_output = gr.Textbox(
                label="Positive プロンプト（英語キーワード）",
                lines=8,
            )
            negative_output = gr.Textbox(
                label="Negative プロンプト",
                lines=4,
            )
            status_output = gr.Textbox(
                label="ステータス",
                interactive=False,
            )

    # スタイル説明を表示
    gr.Markdown("""
    **スタイル説明:**
    - **auto**: 入力内容から自動判断
    - **anime**: アニメ・イラスト向けタグ（1girl, hair color等）
    - **realistic**: 写真・実写向けタグ（photorealistic, 8k等）
    - **artistic**: アート・絵画向けタグ（oil painting, watercolor等）
    - **fantasy**: ファンタジー向けタグ（magical, ethereal等）
    - **cyberpunk**: サイバーパンク向けタグ（neon, futuristic等）
    """)

    # サンプル例
    with gr.Accordion("入力例", open=False):
        gr.Markdown("""
        **アニメ風:**
        > 桜の木の下で、制服を着た女子高生が本を読んでいる。ポニーテールの茶髪。優しい表情。

        **ファンタジー:**
        > 魔法の森の中で、エルフの少女が光る蝶と遊んでいる。長い銀髪、緑色の瞳、白いローブ。

        **サイバーパンク:**
        > 雨が降る夜の東京。ネオンサインが光る路地裏で、サイバーパンクな衣装を着た女性。

        **実写風:**
        > 夕暮れの海辺で、白いドレスを着た女性のポートレート。波が穏やか。ゴールデンアワーの光。
        """)

    # イベントハンドラ
    generate_btn.click(
        fn=generate_keywords_from_text,
        inputs=[
            text_input,
            style,
            word_count,
            detail_level,
            include_negative,
        ],
        outputs=[
            positive_output,
            negative_output,
            status_output,
        ],
    )
