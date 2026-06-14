"""Voice Conversion タブ"""
import gradio as gr

from config import (
    RVC_METHODS, RVC_INDEX_RATE, RVC_FILTER_RADIUS,
    RVC_RMS_MIX_RATE, RVC_PROTECT, RVC_PITCH_RANGE,
    request_stop
)
from pipelines.voice_conversion import (
    convert_voice, get_rvc_models, check_rvc_installation
)


def create_voice_conversion_tab():
    """Voice Conversionタブを作成"""

    # RVCモデル一覧を取得
    def refresh_models():
        models = get_rvc_models()
        if not models:
            return ["(モデルなし)"]
        return list(models.keys())

    initial_models = refresh_models()

    with gr.Row():
        with gr.Column(scale=1):
            # モード選択
            mode_input = gr.Radio(
                label="変換モード",
                choices=[
                    ("RVC（高品質ボイスチェンジ）", "rvc"),
                    ("シンプルピッチシフト", "simple")
                ],
                value="rvc",
                info="RVCはモデルが必要、シンプルはピッチのみ変更"
            )

            # 入力音声
            input_audio = gr.Audio(
                label="入力音声（アップロードまたは録音）",
                type="filepath",
                sources=["upload", "microphone"]
            )

            # RVCモデル選択
            with gr.Group() as rvc_group:
                with gr.Row():
                    model_input = gr.Dropdown(
                        label="RVCモデル",
                        choices=initial_models,
                        value=initial_models[0] if initial_models else None,
                        info="使用するボイスモデル"
                    )

                    refresh_btn = gr.Button("🔄 更新", size="sm")

                model_info = gr.Markdown(
                    f"*{len(initial_models)}個のモデルが見つかりました*" if initial_models[0] != "(モデルなし)"
                    else "*models/audio/voice_conversion/rvc/ にモデルを配置してください*"
                )

            # ピッチシフト
            pitch_input = gr.Slider(
                label="ピッチシフト（半音）",
                minimum=RVC_PITCH_RANGE["min"],
                maximum=RVC_PITCH_RANGE["max"],
                value=RVC_PITCH_RANGE["default"],
                step=1,
                info="正: 高く、負: 低く（男声→女声: +12、女声→男声: -12）"
            )

            # RVC詳細設定
            with gr.Accordion("RVC詳細設定", open=False, visible=True) as rvc_settings:
                f0_method_input = gr.Dropdown(
                    label="F0推定方式",
                    choices=[(v, k) for k, v in RVC_METHODS.items()],
                    value="rmvpe",
                    info="音程検出アルゴリズム"
                )

                index_rate_input = gr.Slider(
                    label="インデックス率",
                    minimum=0.0,
                    maximum=1.0,
                    value=RVC_INDEX_RATE,
                    step=0.05,
                    info="モデルの特徴をどれだけ使うか"
                )

                filter_radius_input = gr.Slider(
                    label="フィルタ半径",
                    minimum=0,
                    maximum=7,
                    value=RVC_FILTER_RADIUS,
                    step=1,
                    info="中央値フィルタ（息などのノイズ軽減）"
                )

                rms_mix_rate_input = gr.Slider(
                    label="RMSミックス率",
                    minimum=0.0,
                    maximum=1.0,
                    value=RVC_RMS_MIX_RATE,
                    step=0.05,
                    info="0=元の音量、1=変換後の音量"
                )

                protect_input = gr.Slider(
                    label="プロテクト",
                    minimum=0.0,
                    maximum=0.5,
                    value=RVC_PROTECT,
                    step=0.01,
                    info="子音などを保護（高いほど強く保護）"
                )

            # 生成ボタン
            with gr.Row():
                convert_btn = gr.Button("音声変換", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=1):
            # 出力
            output_audio = gr.Audio(
                label="変換後の音声",
                type="filepath",
                interactive=False
            )

            output_message = gr.Textbox(
                label="ステータス",
                interactive=False,
                lines=5
            )

            # インストール状況
            with gr.Accordion("インストール状況", open=False):
                install_status = gr.Markdown("")

                check_btn = gr.Button("状況確認")

                gr.Markdown("""
### 必要なライブラリ
```
pip install rvc-python
pip install librosa
```

### RVCモデルの配置
モデルファイル（.pth）を以下に配置:
```
models/audio/voice_conversion/rvc/
├── model_name.pth
└── model_name.index (任意)
```
                """)

            # 使用ヒント
            with gr.Accordion("使用ヒント", open=False):
                gr.Markdown("""
### ピッチシフトの目安
- 男声 → 女声: +8 〜 +12
- 女声 → 男声: -8 〜 -12
- 同性間の変換: -4 〜 +4

### F0推定方式
- **RMVPE**: 推奨。高品質で比較的高速
- **Harvest**: 最も正確だが低速
- **Crepe**: GPU使用、高精度
- **PM**: 最も高速だが品質は低め

### 良い結果を得るコツ
1. ノイズの少ないクリアな音声を使用
2. BGMや効果音がない音声が理想的
3. インデックスファイルがあると品質向上
                """)

    # モード変更時のコールバック
    def on_mode_change(mode):
        is_rvc = mode == "rvc"
        return [
            gr.update(visible=is_rvc),  # rvc_group
            gr.update(visible=is_rvc),  # rvc_settings
        ]

    mode_input.change(
        fn=on_mode_change,
        inputs=[mode_input],
        outputs=[rvc_group, rvc_settings]
    )

    # モデル一覧を更新
    def on_refresh_models():
        models = refresh_models()
        count = len(models) if models[0] != "(モデルなし)" else 0
        info = f"*{count}個のモデルが見つかりました*" if count > 0 else "*モデルが見つかりません*"
        return [
            gr.update(choices=models, value=models[0] if models else None),
            gr.update(value=info)
        ]

    refresh_btn.click(
        fn=on_refresh_models,
        inputs=[],
        outputs=[model_input, model_info]
    )

    # インストール状況確認
    def on_check_install():
        status = check_rvc_installation()

        lines = [
            f"- rvc-python: {'✅ インストール済み' if status['rvc_python'] else '❌ 未インストール'}",
            f"- librosa: {'✅ インストール済み' if status['librosa'] else '❌ 未インストール'}",
            f"- fairseq: {'✅ インストール済み' if status['fairseq'] else '❌ 未インストール'}",
            f"- HuBERTモデル: {'✅ 検出' if status['hubert_model'] else '❌ 未検出'}",
            f"- RVCモデル: {status['models_found']}個検出"
        ]

        return "\n".join(lines)

    check_btn.click(
        fn=on_check_install,
        inputs=[],
        outputs=[install_status]
    )

    # 変換関数のラッパー
    def convert_wrapper(
        input_audio, mode, model_name, pitch_shift,
        f0_method, index_rate, filter_radius, rms_mix_rate, protect
    ):
        """変換関数のラッパー"""
        use_rvc = mode == "rvc"

        if use_rvc and (not model_name or model_name == "(モデルなし)"):
            return None, "RVCモデルを選択してください"

        return convert_voice(
            input_audio=input_audio,
            model_name=model_name if use_rvc else None,
            pitch_shift=int(pitch_shift),
            use_rvc=use_rvc,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=int(filter_radius),
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

    # イベント設定
    convert_event = convert_btn.click(
        fn=convert_wrapper,
        inputs=[
            input_audio,
            mode_input,
            model_input,
            pitch_input,
            f0_method_input,
            index_rate_input,
            filter_radius_input,
            rms_mix_rate_input,
            protect_input
        ],
        outputs=[output_audio, output_message]
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[convert_event])
