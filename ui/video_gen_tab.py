"""動画生成タブ"""
import gradio as gr

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT,
    VIDEO_DEFAULT_NEGATIVE, VIDEO_ANIMATEDIFF_NEGATIVE, VIDEO_QUALITY_PREFIX,
    MODEL_FILES, is_sdxl_model, is_flux_model,
    request_stop,
)
from pipelines.video_gen import generate_video, VIDEO_PROMPT_EXAMPLES
from pipelines.video_downloader import ensure_video_model_downloaded
from .video_tab_common import model_choices, model_status_text, build_smooth_kwargs


def _default_negative_for(model_key: str) -> str:
    """モデルキーに応じた既定ネガティブ。AnimateDiff は顔・手強化版を使用"""
    if model_key == "animatediff":
        return VIDEO_ANIMATEDIFF_NEGATIVE
    return VIDEO_DEFAULT_NEGATIVE


def _get_sd15_model_choices() -> list:
    """models/model/ から SD1.5 系モデルを抽出（AnimateDiff のベースに使える）

    Returns: [(label, path), ...] のタプル配列。先頭は「デフォルト(stable-diffusion-v1-5)」
    """
    choices = [("デフォルト (stable-diffusion-v1-5)", "")]
    for name, path in sorted(MODEL_FILES.items()):
        # SDXL / Flux は除外（AnimateDiff は SD1.5 専用）
        if is_sdxl_model(name) or is_flux_model(name):
            continue
        choices.append((name, path))
    return choices


def create_video_gen_tab():
    """動画生成タブを作成"""
    with gr.Row():
        with gr.Column(scale=1):
            # モデル選択
            default_model = "animatediff"
            model_input = gr.Dropdown(
                label="モデル",
                choices=model_choices(),
                value=default_model,
                info="8GB VRAMで動作するモデル。未DLなら自動DLを試行（時間がかかります）",
            )

            model_info = gr.Markdown(model_status_text(default_model, "full"))

            # AnimateDiff 用ベースSD1.5モデル選択
            # （models/model/ の高品質 SD1.5 を選ぶと生成品質が大きく改善）
            sd15_choices = _get_sd15_model_choices()
            base_model_input = gr.Dropdown(
                label="AnimateDiff ベース SD1.5",
                choices=sd15_choices,
                value="",
                visible=(default_model == "animatediff"),
                info="bluePencil・meinamix・dreamshaper など高品質SD1.5を選ぶと品質が大きく向上。SDXL/Flux は AnimateDiff に非対応。初回選択時は変換キャッシュが作成されます",
            )

            # 入力画像（image-to-video系のみ表示）
            image_input = gr.Image(
                label="入力画像 (img→video モデル時のみ)",
                type="pil",
                sources=["upload", "clipboard"],
                visible=(VIDEO_MODELS[default_model]["type"] == "image-to-video"),
            )

            # プロンプト
            prompt_input = gr.Textbox(
                label="プロンプト",
                placeholder="例: a cat walking through a sunlit forest, cinematic",
                lines=3,
                value="",
            )

            quality_prefix_input = gr.Checkbox(
                label="品質向上プレフィックスを自動追加",
                value=True,
                info=f"プロンプト先頭に '{VIDEO_QUALITY_PREFIX[:50]}...' を付与",
            )

            # プロンプト例
            example_dropdown = gr.Dropdown(
                label="プロンプト例",
                choices=list(VIDEO_PROMPT_EXAMPLES.keys()),
                value=None,
                info="選択するとプロンプトに追加",
            )

            negative_prompt_input = gr.Textbox(
                label="ネガティブプロンプト",
                placeholder="例: blurry, low quality, distorted",
                lines=3,
                value=_default_negative_for(default_model),
            )

            # 基本パラメータ
            with gr.Row():
                width_input = gr.Slider(
                    label="幅",
                    minimum=256, maximum=1280,
                    value=VIDEO_MODELS[default_model]["default_width"],
                    step=64,
                )
                height_input = gr.Slider(
                    label="高さ",
                    minimum=256, maximum=720,
                    value=VIDEO_MODELS[default_model]["default_height"],
                    step=64,
                )

            with gr.Row():
                num_frames_input = gr.Slider(
                    label="フレーム数",
                    minimum=8,
                    maximum=VIDEO_MODELS[default_model]["max_frames"],
                    value=VIDEO_MODELS[default_model]["default_frames"],
                    step=1,
                )
                fps_input = gr.Slider(
                    label="FPS",
                    minimum=1, maximum=30,
                    value=VIDEO_MODELS[default_model]["default_fps"],
                    step=1,
                )

            with gr.Row():
                steps_input = gr.Slider(
                    label="推論ステップ数",
                    minimum=10, maximum=100,
                    value=VIDEO_MODELS[default_model]["default_steps"],
                    step=1,
                )
                guidance_input = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0, maximum=15.0,
                    value=VIDEO_MODELS[default_model]["default_guidance"],
                    step=0.5,
                )

            with gr.Row():
                seed_input = gr.Number(label="Seed", value=-1, info="-1でランダム")
                fmt_input = gr.Radio(
                    label="出力フォーマット",
                    choices=["mp4", "gif"],
                    value=VIDEO_DEFAULT_FORMAT,
                )

            # 動き補間（カクカク防止）
            # モデル切替時に自動で auto_smooth フラグに応じて ON/OFF が切り替わる
            with gr.Row():
                smooth_enable_input = gr.Checkbox(
                    label="動き補間で滑らかに",
                    value=VIDEO_MODELS[default_model].get("auto_smooth", False),
                    info="低fps生成（AnimateDiff/SVD-XT）はON推奨。Wan2.1/LTX-Videoはネイティブ滑らかなのでOFF推奨",
                )
                smooth_fps_input = gr.Slider(
                    label="目標FPS",
                    minimum=12, maximum=60,
                    value=24, step=1,
                    info="動画再生時のFPS（生成FPSより大きい値で効果あり）",
                )
            with gr.Row():
                smooth_method_input = gr.Radio(
                    label="補間方法",
                    choices=[
                        ("RIFE (ニューラル補間) - 自然で歪みが少ない・推奨", "rife"),
                        ("ffmpeg minterpolate - 軽量だが大きい動きで歪む", "minterpolate"),
                    ],
                    value="rife",
                    info="RIFE は GPU を使う（モデル ~24MB）。minterpolate は CPU のみ",
                )
                smooth_mode_input = gr.Radio(
                    label="minterpolate のモード（minterpolate 選択時のみ）",
                    choices=[
                        ("動き補正 (mci)", "mci"),
                        ("ブレンド (blend)", "blend"),
                        ("複製のみ (dup)", "dup"),
                    ],
                    value="blend",
                )

            # SVD専用パラメータ
            with gr.Accordion("SVD詳細設定 (SVD-XT使用時のみ)", open=False, visible=False) as svd_panel:
                # 連続生成（25フレーム固定の上限を chain で延長）
                num_chunks_input = gr.Slider(
                    label="連続生成回数 (chunk数)",
                    minimum=1,
                    maximum=VIDEO_MODELS["svd_xt"].get("max_chunks", 8),
                    value=VIDEO_MODELS["svd_xt"].get("default_chunks", 1),
                    step=1,
                    info="末尾フレームを次回の入力に流用して延長。25 × N フレームを生成（N=chunk数）。境界は若干不自然になる場合あり",
                )
                motion_bucket_input = gr.Slider(
                    label="Motion Bucket ID",
                    minimum=1, maximum=255, value=127, step=1,
                    info="動きの大きさ（高いほど大きく動く）",
                )
                noise_aug_input = gr.Slider(
                    label="Noise Aug Strength",
                    minimum=0.0, maximum=1.0, value=0.02, step=0.01,
                    info="入力画像へのノイズ付加量（高いほど自由度↑）",
                )
                decode_chunk_input = gr.Slider(
                    label="Decode Chunk Size",
                    minimum=1, maximum=8, value=2, step=1,
                    info="VRAM節約用（小さいほど省メモリ）",
                )
                min_guidance_input = gr.Slider(
                    label="Min Guidance Scale",
                    minimum=1.0, maximum=10.0, value=1.0, step=0.5,
                )

            with gr.Row():
                generate_btn = gr.Button("動画生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")
                unload_btn = gr.Button("モデルをアンロード", variant="secondary")

            with gr.Row():
                download_btn = gr.Button(
                    "選択モデルを事前DL（生成しない）",
                    variant="secondary",
                )

        with gr.Column(scale=1):
            output_video = gr.Video(
                label="生成された動画",
                interactive=False,
            )
            output_message = gr.Textbox(
                label="ステータス",
                interactive=False,
                lines=6,
            )

            with gr.Accordion("使い方・注意", open=False):
                gr.Markdown("""
### モデル別の特徴と長尺化方法
- **AnimateDiff**: SD1.5ベース。軽量・高速。32フレーム超は **FreeNoise** で自動延長（最大64fr=8秒）
- **SVD-XT**: 画像→動画（25fr固定）。**連続生成回数** で 25×N フレームに延長可能（最大200fr=28秒）
- **LTX-Video**: 高品質。t2v/i2v両対応。最大257フレーム（10.7秒 @ 24fps）
- **CogVideoX-2B**: 高品質だが8GBではsequential offloadで非常に低速
- **Wan2.1 1.3B**: 軽量で8GB余裕あり（832×480 推奨）。最大121フレーム（7.5秒）

### 8GB VRAMでのコツ
- 解像度・フレーム数を伸ばすと VRAM/時間が増える。OOM の場合は減らす
- AnimateDiff の長尺は **FreeNoise** が自動で効くので品質を保てる
- SVD-XT の連続生成は境界が若干不自然になる場合あり
- Wan2.1 は 832×480 + 公式推奨ネガで最も品質が出る（短いネガだと砂嵐になる場合あり）

### カクカク動画（パラパラ漫画化）対策
- 動き補間は **モデル切替時に自動で ON/OFF** が切り替わる
- **AnimateDiff / SVD-XT**（低fpsネイティブ） → 補間 ON 推奨（RIFE で 24fps 化）
- **LTX-Video / Wan2.1**（動画ネイティブ） → 補間 OFF 推奨（既に滑らか・補間で違和感増）
- 補間方法: **RIFE（推奨）** はニューラル補間で歪みが少ない / minterpolate は CPU のみで軽量だが大きい動きで歪む

### モデル選びの目安（品質順）
- 一番滑らか・自然な動きが欲しい → **Wan2.1 1.3B** （16fps, 補間不要、生成 ~6分）
- 高速で 24fps 直接 → **LTX-Video** （生成 1〜3分、解像度高め推奨）
- SD1.5 のキャラ/モデル資産を使いたい → **AnimateDiff + RIFE**（短尺・8fps→24fps）
- 入力画像から動画 → **SVD-XT + RIFE**（連続生成で延長可能）

### 事前ダウンロード
ターミナルで:
```
python scripts/download_video_models.py --list
python scripts/download_video_models.py             # 全DL（約34GB）
python scripts/download_video_models.py animatediff  # 単体DL
```
                """)

    # ===== コールバック =====

    def on_model_change(model_key):
        cfg = VIDEO_MODELS.get(model_key, {})
        is_i2v = cfg.get("type") == "image-to-video"
        is_svd = model_key == "svd_xt"
        is_animatediff = model_key == "animatediff"
        max_frames = cfg.get("max_frames", 32)
        auto_smooth = cfg.get("auto_smooth", False)
        default_smooth_fps = cfg.get("default_smooth_target_fps", 24)

        return (
            model_status_text(model_key, "full"),             # model_info
            gr.update(visible=is_animatediff),                 # base_model_input
            gr.update(visible=is_animatediff),                 # quality_prefix_input
            gr.update(visible=is_i2v),                         # image_input
            gr.update(value=_default_negative_for(model_key)), # negative_prompt
            gr.update(value=cfg.get("default_width", 512)),    # width
            gr.update(value=cfg.get("default_height", 512)),   # height
            gr.update(
                value=cfg.get("default_frames", 16),
                maximum=max_frames,
            ),                                                  # num_frames
            gr.update(value=cfg.get("default_fps", 8)),        # fps
            gr.update(value=cfg.get("default_steps", 25)),     # steps
            gr.update(value=cfg.get("default_guidance", 7.5)), # guidance
            gr.update(visible=is_svd),                          # svd_panel
            gr.update(value=auto_smooth),                       # smooth_enable
            gr.update(value=default_smooth_fps),                # smooth_fps
        )

    model_input.change(
        fn=on_model_change,
        inputs=[model_input],
        outputs=[
            model_info, base_model_input, quality_prefix_input, image_input,
            negative_prompt_input,
            width_input, height_input,
            num_frames_input, fps_input,
            steps_input, guidance_input,
            svd_panel,
            smooth_enable_input, smooth_fps_input,
        ],
    )

    # プロンプト例選択
    def on_example_select(name, current):
        if not name:
            return current
        ex = VIDEO_PROMPT_EXAMPLES.get(name, "")
        if current:
            return f"{current}, {ex}"
        return ex

    example_dropdown.change(
        fn=on_example_select,
        inputs=[example_dropdown, prompt_input],
        outputs=[prompt_input],
    )

    # 生成（未DLなら自動DL → 生成）
    def generate_wrapper(
        model_key, base_model, use_quality_prefix, prompt, negative_prompt, image,
        width, height, num_frames, fps, steps, guidance, seed, fmt,
        motion_bucket, noise_aug, decode_chunk, min_guidance, num_chunks,
        smooth_enable, smooth_fps, smooth_method, smooth_mode,
        progress=gr.Progress(track_tqdm=True),
    ):
        cfg = VIDEO_MODELS.get(model_key, {})
        label = cfg.get("label", model_key)

        # 品質プレフィックスを自動付与（AnimateDiff のときのみ有効）
        effective_prompt = prompt
        if model_key == "animatediff" and use_quality_prefix and prompt and prompt.strip():
            if not prompt.strip().lower().startswith("masterpiece"):
                effective_prompt = f"{VIDEO_QUALITY_PREFIX}, {prompt.strip()}"

        # ステップ1: 必要なら自動DL（既にDL済みなら即座に戻る）
        progress(0.0, desc=f"モデル '{label}' をチェック中...")
        dl_messages = []

        def on_dl_progress(msg: str):
            dl_messages.append(msg)
            progress(0.0, desc=msg[:80])

        ok, dl_status = ensure_video_model_downloaded(
            model_key, progress_callback=on_dl_progress
        )
        if not ok:
            log = "\n".join(dl_messages[-10:])
            return None, f"モデルDL失敗:\n{dl_status}\n\n進捗ログ:\n{log}"

        # ステップ2: 生成
        progress(0.5, desc=f"'{label}' で生成中...")
        kwargs = {}
        if model_key == "svd_xt":
            kwargs.update({
                "motion_bucket_id": int(motion_bucket),
                "noise_aug_strength": float(noise_aug),
                "decode_chunk_size": int(decode_chunk),
                "min_guidance_scale": float(min_guidance),
                "num_chunks": int(num_chunks),
            })
        if model_key == "animatediff" and base_model:
            kwargs["custom_base_path"] = base_model

        # 動き補間（生成FPSより目標FPSが大きいときのみ意味あり）
        kwargs.update(build_smooth_kwargs(
            smooth_enable, smooth_fps, fps, fmt, smooth_method, smooth_mode,
        ))

        return generate_video(
            model_key=model_key,
            prompt=effective_prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=int(num_frames),
            fps=int(fps),
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
            fmt=fmt,
            **kwargs,
        )

    generate_event = generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            model_input, base_model_input, quality_prefix_input,
            prompt_input, negative_prompt_input, image_input,
            width_input, height_input, num_frames_input, fps_input,
            steps_input, guidance_input, seed_input, fmt_input,
            motion_bucket_input, noise_aug_input, decode_chunk_input, min_guidance_input,
            num_chunks_input,
            smooth_enable_input, smooth_fps_input, smooth_method_input, smooth_mode_input,
        ],
        outputs=[output_video, output_message],
    )

    stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[generate_event])

    def do_unload():
        from pipelines.video_manager import video_pipeline_manager
        video_pipeline_manager.unload()
        return "動画モデルをアンロードしました"

    unload_btn.click(fn=do_unload, inputs=None, outputs=output_message)

    # 事前DLボタン
    def do_download(model_key, progress=gr.Progress(track_tqdm=True)):
        cfg = VIDEO_MODELS.get(model_key, {})
        label = cfg.get("label", model_key)
        progress(0.0, desc=f"'{label}' DL開始...")
        dl_messages = []

        def on_msg(msg: str):
            dl_messages.append(msg)
            progress(0.0, desc=msg[:80])

        ok, status = ensure_video_model_downloaded(model_key, progress_callback=on_msg)
        log = "\n".join(dl_messages[-15:])
        if ok:
            return f"[OK] {status}\n\n--- ログ ---\n{log}"
        return f"[NG] {status}\n\n--- ログ ---\n{log}"

    download_btn.click(
        fn=do_download,
        inputs=[model_input],
        outputs=[output_message],
    )
