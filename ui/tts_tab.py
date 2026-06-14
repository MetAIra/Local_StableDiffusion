"""Text-to-Speech タブ"""
import gradio as gr

from config import (
    TTS_MODELS, BARK_VOICE_PRESETS, BARK_LANGUAGES, XTTS_LANGUAGES,
    request_stop
)
from pipelines.tts import generate_speech, get_bark_voice_presets_for_language


def create_tts_tab():
    """Text-to-Speechタブを作成"""
    with gr.Row():
        with gr.Column(scale=1):
            # モデル選択
            model_input = gr.Dropdown(
                label="TTSモデル",
                choices=[("Bark（推奨）", "bark"), ("XTTS v2（互換性問題あり）", "xtts_v2")],
                value="bark",
                info="Bark推奨。XTTS v2は現在のtransformersバージョンで動作しません。"
            )

            # モデル説明
            model_info = gr.Markdown(
                f"*{TTS_MODELS['bark']['description']}*"
            )

            # テキスト入力
            text_input = gr.Textbox(
                label="テキスト",
                placeholder="音声に変換するテキストを入力...\n例: こんにちは、音声合成のテストです。",
                lines=5,
                value=""
            )

            # 言語選択（(表示名, 言語コード) の形式）
            language_input = gr.Dropdown(
                label="言語",
                choices=[(v, k) for k, v in BARK_LANGUAGES.items()],
                value="ja",
                info="音声の言語を選択"
            )

            # Bark専用: ボイスプリセット
            with gr.Group() as bark_options:
                gr.Markdown("### Bark設定")

                voice_preset_input = gr.Dropdown(
                    label="ボイスプリセット",
                    choices=[(v, k) for k, v in BARK_VOICE_PRESETS.items()],
                    value="ja_speaker_0",
                    info="使用する声質を選択"
                )

            # XTTS専用: ボイスクローン
            with gr.Group(visible=False) as xtts_options:
                gr.Markdown("### XTTS v2設定")

                reference_audio_input = gr.Audio(
                    label="参照音声（ボイスクローン用、3-10秒推奨）",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                gr.Markdown("*参照音声なしの場合はデフォルトスピーカーを使用*")

            # 共通パラメータ
            with gr.Accordion("詳細設定", open=False):
                temperature_input = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    info="低いと安定、高いと多様性が増す"
                )

                # Bark専用
                semantic_temperature_input = gr.Slider(
                    label="Semantic Temperature (Bark)",
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    visible=True,
                    info="セマンティック生成の温度"
                )

                # XTTS専用
                top_k_input = gr.Slider(
                    label="Top-K (XTTS)",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    visible=False,
                    info="サンプリング候補数"
                )

                top_p_input = gr.Slider(
                    label="Top-P (XTTS)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    visible=False,
                    info="確率累積しきい値"
                )

                repetition_penalty_input = gr.Slider(
                    label="Repetition Penalty (XTTS)",
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    visible=False,
                    info="繰り返し抑制"
                )

            # 生成ボタン
            with gr.Row():
                generate_btn = gr.Button("音声生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=1):
            # 出力
            output_audio = gr.Audio(
                label="生成された音声",
                type="filepath",
                interactive=False
            )

            output_message = gr.Textbox(
                label="ステータス",
                interactive=False,
                lines=4
            )

            # 使用例
            with gr.Accordion("使用例", open=False):
                gr.Markdown("""
### Bark使用例
- 日本語: 「こんにちは、今日は良い天気ですね。」
- 英語: "Hello, how are you today?"
- 感情表現: [laughs] や [sighs] を含めると感情を表現

### XTTS v2使用例
- ボイスクローン: 3-10秒の音声をアップロード
- 対応言語: 16言語（日本語、英語、中国語など）

### ヒント
- 長いテキストは自動で分割されます
- 句読点（、。）を適切に入れると自然な音声になります
                """)

    # モデル変更時のコールバック
    def on_model_change(model_name):
        """モデル変更時にUIを更新"""
        is_bark = model_name == "bark"
        is_xtts = model_name == "xtts_v2"

        model_desc = TTS_MODELS.get(model_name, {}).get('description', '')

        # 言語選択肢を更新（(表示名, 言語コード) の形式）
        if is_bark:
            lang_choices = [(v, k) for k, v in BARK_LANGUAGES.items()]
        else:
            lang_choices = [(v, k) for k, v in XTTS_LANGUAGES.items()]

        return [
            gr.update(value=f"*{model_desc}*"),  # model_info
            gr.update(visible=is_bark),  # bark_options
            gr.update(visible=is_xtts),  # xtts_options
            gr.update(choices=lang_choices, value="ja"),  # language
            gr.update(visible=is_bark),  # semantic_temperature
            gr.update(visible=is_xtts),  # top_k
            gr.update(visible=is_xtts),  # top_p
            gr.update(visible=is_xtts),  # repetition_penalty
        ]

    model_input.change(
        fn=on_model_change,
        inputs=[model_input],
        outputs=[
            model_info,
            bark_options,
            xtts_options,
            language_input,
            semantic_temperature_input,
            top_k_input,
            top_p_input,
            repetition_penalty_input
        ]
    )

    # 言語変更時にボイスプリセットを更新（Barkのみ）
    def on_language_change(language, model):
        """言語変更時にBarkのボイスプリセットを更新"""
        if model != "bark":
            return gr.update()

        presets = get_bark_voice_presets_for_language(language)
        # (表示名, プリセットID) の形式
        preset_choices = [(v, k) for k, v in presets.items()]

        # デフォルト値を設定
        default_value = f"{language}_speaker_0" if f"{language}_speaker_0" in presets else list(presets.keys())[0]

        return gr.update(choices=preset_choices, value=default_value)

    language_input.change(
        fn=on_language_change,
        inputs=[language_input, model_input],
        outputs=[voice_preset_input]
    )

    # 生成関数のラッパー
    def generate_wrapper(
        text, model, voice_preset, language, reference_audio,
        temperature, semantic_temperature, top_k, top_p, repetition_penalty
    ):
        """生成関数のラッパー"""
        return generate_speech(
            text=text,
            model=model,
            voice_preset=voice_preset,
            language=language,
            reference_audio=reference_audio,
            temperature=temperature,
            semantic_temperature=semantic_temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            text_input,
            model_input,
            voice_preset_input,
            language_input,
            reference_audio_input,
            temperature_input,
            semantic_temperature_input,
            top_k_input,
            top_p_input,
            repetition_penalty_input
        ],
        outputs=[output_audio, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
