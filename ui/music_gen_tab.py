"""Music Generation タブ"""
import gradio as gr

from config import MUSIC_GEN_MODELS, AUDIOLDM_MODELS, request_stop
from pipelines.music_gen import generate_music, MUSIC_PROMPT_EXAMPLES


def create_music_gen_tab():
    """Music Generationタブを作成"""
    with gr.Row():
        with gr.Column(scale=1):
            # モデルタイプ選択
            model_type_input = gr.Radio(
                label="モデルタイプ",
                choices=[("MusicGen", "musicgen"), ("AudioLDM2 (互換性問題あり)", "audioldm")],
                value="musicgen",
                info="MusicGen推奨。AudioLDM2は現在のバージョンで動作しません。"
            )

            # MusicGenモデル選択
            musicgen_model_input = gr.Dropdown(
                label="MusicGenモデル",
                choices=list(MUSIC_GEN_MODELS.keys()),
                value="musicgen-small",
                info="小さいほど軽量・高速、大きいほど高品質",
                visible=True
            )

            # AudioLDMモデル選択
            audioldm_model_input = gr.Dropdown(
                label="AudioLDMモデル",
                choices=list(AUDIOLDM_MODELS.keys()),
                value="audioldm2-music",
                visible=False
            )

            # モデル説明
            model_info = gr.Markdown(
                f"*{MUSIC_GEN_MODELS['musicgen-small']['description']}*"
            )

            # プロンプト入力
            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: upbeat jazz with piano and saxophone, relaxing",
                lines=3,
                value=""
            )

            # プロンプト例
            example_dropdown = gr.Dropdown(
                label="プロンプト例",
                choices=list(MUSIC_PROMPT_EXAMPLES.keys()),
                value=None,
                info="選択するとプロンプトに追加"
            )

            # ネガティブプロンプト（AudioLDMのみ）
            negative_prompt_input = gr.Textbox(
                label="ネガティブプロンプト (AudioLDMのみ)",
                placeholder="避けたい要素（例: noise, distortion, vocals）",
                lines=2,
                value="",
                visible=False
            )

            # 基本パラメータ
            with gr.Row():
                duration_input = gr.Slider(
                    label="長さ（秒）",
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=1,
                    info="生成する音楽の長さ"
                )

                seed_input = gr.Number(
                    label="Seed",
                    value=-1,
                    info="-1でランダム"
                )

            # MusicGen詳細設定
            with gr.Accordion("MusicGen詳細設定", open=False, visible=True) as musicgen_settings:
                temperature_input = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="高いほど多様、低いほど安定"
                )

                guidance_scale_musicgen = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                    info="プロンプトへの忠実度"
                )

                top_k_input = gr.Slider(
                    label="Top-K",
                    minimum=0,
                    maximum=500,
                    value=250,
                    step=10,
                    info="サンプリング候補数（0=無効）"
                )

                top_p_input = gr.Slider(
                    label="Top-P",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    info="確率累積しきい値（0=無効）"
                )

                melody_audio_input = gr.Audio(
                    label="メロディー参照音声 (musicgen-melodyで条件付け)",
                    type="filepath",
                    sources=["upload"]
                )

            # AudioLDM詳細設定
            with gr.Accordion("AudioLDM詳細設定", open=False, visible=False) as audioldm_settings:
                num_inference_steps_input = gr.Slider(
                    label="推論ステップ数",
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    info="多いほど高品質だが低速"
                )

                guidance_scale_audioldm = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    value=3.5,
                    step=0.5,
                    info="プロンプトへの忠実度"
                )

            # 生成ボタン
            with gr.Row():
                generate_btn = gr.Button("音楽生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=1):
            # 出力
            output_audio = gr.Audio(
                label="生成された音楽",
                type="filepath",
                interactive=False
            )

            output_message = gr.Textbox(
                label="ステータス",
                interactive=False,
                lines=5
            )

            # 使用ヒント
            with gr.Accordion("使用ヒント", open=False):
                gr.Markdown("""
### プロンプトのコツ

**ジャンル指定:**
- "jazz", "rock", "classical", "electronic", "ambient"

**楽器指定:**
- "with piano", "guitar solo", "orchestral strings"

**雰囲気指定:**
- "upbeat", "relaxing", "dramatic", "melancholic"

**テンポ指定:**
- "fast tempo", "slow and calm", "moderate pace"

### モデル選択ガイド

**MusicGen:**
- small: 軽量・高速（8GB VRAM）
- medium: バランス型（12GB VRAM）
- large: 最高品質（16GB VRAM）
- melody: メロディー条件付け可能

**AudioLDM2:**
- 拡散モデルベース
- より細かい制御が可能
- ネガティブプロンプト対応
                """)

    # モデルタイプ変更時のコールバック
    def on_model_type_change(model_type):
        """モデルタイプ変更時にUIを更新"""
        is_musicgen = model_type == "musicgen"

        if is_musicgen:
            model_desc = MUSIC_GEN_MODELS['musicgen-small']['description']
        else:
            model_desc = AUDIOLDM_MODELS['audioldm2-music']['description']

        return [
            gr.update(visible=is_musicgen),  # musicgen_model
            gr.update(visible=not is_musicgen),  # audioldm_model
            gr.update(value=f"*{model_desc}*"),  # model_info
            gr.update(visible=not is_musicgen),  # negative_prompt
            gr.update(visible=is_musicgen),  # musicgen_settings
            gr.update(visible=not is_musicgen),  # audioldm_settings
        ]

    model_type_input.change(
        fn=on_model_type_change,
        inputs=[model_type_input],
        outputs=[
            musicgen_model_input,
            audioldm_model_input,
            model_info,
            negative_prompt_input,
            musicgen_settings,
            audioldm_settings
        ]
    )

    # MusicGenモデル変更時
    def on_musicgen_model_change(model_name):
        model_desc = MUSIC_GEN_MODELS.get(model_name, {}).get('description', '')
        return gr.update(value=f"*{model_desc}*")

    musicgen_model_input.change(
        fn=on_musicgen_model_change,
        inputs=[musicgen_model_input],
        outputs=[model_info]
    )

    # AudioLDMモデル変更時
    def on_audioldm_model_change(model_name):
        model_desc = AUDIOLDM_MODELS.get(model_name, {}).get('description', '')
        return gr.update(value=f"*{model_desc}*")

    audioldm_model_input.change(
        fn=on_audioldm_model_change,
        inputs=[audioldm_model_input],
        outputs=[model_info]
    )

    # プロンプト例選択時
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

    # 生成関数のラッパー
    def generate_wrapper(
        prompt, model_type, musicgen_model, audioldm_model, negative_prompt,
        duration, seed, temperature, guidance_scale_mg,
        top_k, top_p, melody_audio, num_inference_steps, guidance_scale_al
    ):
        """生成関数のラッパー"""
        if model_type == "musicgen":
            return generate_music(
                prompt=prompt,
                model_type="musicgen",
                model_name=musicgen_model,
                duration=duration,
                seed=int(seed),
                temperature=temperature,
                guidance_scale=guidance_scale_mg,
                top_k=int(top_k),
                top_p=top_p,
                melody_audio=melody_audio,
            )
        else:
            return generate_music(
                prompt=prompt,
                model_type="audioldm",
                model_name=audioldm_model,
                duration=duration,
                seed=int(seed),
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=guidance_scale_al,
            )

    # イベント設定
    generate_event = generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            prompt_input,
            model_type_input,
            musicgen_model_input,
            audioldm_model_input,
            negative_prompt_input,
            duration_input,
            seed_input,
            temperature_input,
            guidance_scale_musicgen,
            top_k_input,
            top_p_input,
            melody_audio_input,
            num_inference_steps_input,
            guidance_scale_audioldm
        ],
        outputs=[output_audio, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])
