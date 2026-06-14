"""動画連鎖生成タブ

LTX-Video の i2v チェーン + ffmpeg xfade 連結で最長10分級の長尺動画を作る。

UI構成:
- 共通: モデル(現状LTX固定) / 解像度 / fps / steps / xfade / ベースシード / 補間設定
- シーン定義: タブ区切りテキストで N シーンを記述
    1 行 = "prompt\tnum_frames\tguidance\tseed_offset\tfresh_start(0|1)"
    末尾フィールドは省略可
- プリセット: 風景/都市/海など 3〜6 シーンの例を用意
- 「シーン数概算」「合計秒数概算」をリアルタイム表示
- Stop ボタンで途中中断、できた分まで連結
"""
import os
import gradio as gr

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT, VIDEO_DEFAULT_NEGATIVE,
    request_stop,
)
from pipelines.video_chain import (
    generate_video_chain,
    parse_scenes_text,
    VIDEO_CHAIN_EXAMPLES,
    VIDEO_CHAIN_MAX_SCENES,
    VIDEO_CHAIN_SUPPORTED_MODELS,
)
from .video_tab_common import (
    model_choices, model_status_text, ensure_model_ready, build_smooth_kwargs,
)


def _estimate_duration(scenes_text: str, fps: int, xfade: float) -> str:
    """シーン定義テキストから「シーン数」「秒数概算」を計算"""
    try:
        scenes = parse_scenes_text(scenes_text or "")
    except Exception:
        return "—"
    n = len(scenes)
    if n == 0:
        return "シーン: 0  /  合計: 0秒"
    fps = max(1, int(fps or 1))
    # num_frames=0 のシーンはデフォルト 121 frames 想定（LTX）
    total_frames = sum((s.get("num_frames") or 121) for s in scenes)
    total_sec = total_frames / fps
    # クロスフェード重複ぶん差し引き
    total_sec -= max(0, n - 1) * max(0.0, float(xfade or 0.0))
    minutes = int(total_sec // 60)
    seconds = total_sec - minutes * 60
    warn = "  ⚠️ シーン数上限超過" if n > VIDEO_CHAIN_MAX_SCENES else ""
    return f"シーン: {n}  /  合計: 約 {minutes}分{seconds:.1f}秒 ({total_sec:.1f}s){warn}"


def create_video_chain_tab():
    """動画連鎖生成タブ"""

    default_model = "ltx_video"
    cfg = VIDEO_MODELS[default_model]

    with gr.Column():
        gr.Markdown("## 動画連鎖生成（長尺）")
        gr.Markdown(
            "**LTX-Video の i2v チェーン + xfade 連結で最長10分級の動画を生成します。** "
            "各シーンは前シーンの最終フレームから i2v で続きを生成し、シーン境界では "
            "クロスフェードで滑らかに繋ぎます。シーン単位で `fresh_start` を立てると "
            "「ここから新規開始」（前シーンを引き継がず t2v）にできます。"
            f"  最大シーン数: {VIDEO_CHAIN_MAX_SCENES}"
        )
        gr.Markdown(
            "### キャラクターを扱う時のコツ\n"
            "- **キャラの顔・髪色は3〜4シーンあたりからドリフトします**（LTX-Video には ID 固定の仕組みが無いため）。"
            "完全な一貫性が必要な場合は本タブでは難しいです。\n"
            "- **対策**: 各シーンのプロンプトに「同じ髪色・服装・スタイル」を**毎回明記**してください "
            "（例: `same girl with pink hair and white dress`）。最初の数語を共通にすると i2v が引き継ぎやすくなります。\n"
            "- **場面転換したい時**は `fresh_start=1` を立てて意図的に「別シーン／別人」として再スタート。"
            "境界は xfade でクロスフェードされます。\n"
            "- 表情変化や立ち姿のような**短い動作**は1シーン3秒（73フレーム @ 24fps）程度の方が破綻しにくいです。"
        )

        with gr.Row():
            # ===== 左: 共通パラメータ =====
            with gr.Column(scale=1):
                model_input = gr.Dropdown(
                    label="モデル",
                    choices=model_choices(VIDEO_CHAIN_SUPPORTED_MODELS),
                    value=default_model,
                    info="現状 LTX-Video のみ（プロンプト + 画像入力の両対応モデル）",
                )
                model_info = gr.Markdown(model_status_text(default_model, "chain"))

                with gr.Row():
                    width_input = gr.Slider(
                        label="幅", minimum=256, maximum=1280,
                        value=cfg["default_width"], step=32,
                    )
                    height_input = gr.Slider(
                        label="高さ", minimum=256, maximum=720,
                        value=cfg["default_height"], step=32,
                    )
                with gr.Row():
                    fps_input = gr.Slider(
                        label="FPS (生成時)", minimum=8, maximum=30,
                        value=cfg["default_fps"], step=1,
                        info="LTX-Video の native fps は 24。シーン長 = num_frames / fps",
                    )
                    steps_input = gr.Slider(
                        label="Steps", minimum=10, maximum=80,
                        value=cfg["default_steps"], step=1,
                    )
                with gr.Row():
                    default_frames_input = gr.Slider(
                        label="既定 num_frames (シーン側で 0 のときに採用)",
                        minimum=24, maximum=cfg.get("max_frames", 257),
                        value=121, step=1,
                        info="121 frames @ 24fps ≈ 5秒",
                    )
                    default_guidance_input = gr.Slider(
                        label="既定 Guidance",
                        minimum=1.0, maximum=15.0,
                        value=cfg["default_guidance"], step=0.5,
                    )
                with gr.Row():
                    base_seed_input = gr.Number(
                        label="ベース Seed",
                        value=42,
                        info="-1 で毎シーンランダム / 整数で base+seed_offset",
                    )
                    xfade_input = gr.Slider(
                        label="クロスフェード秒数",
                        minimum=0.0, maximum=1.5, value=0.3, step=0.05,
                        info="0 にすると単純結合（瞬間切替）",
                    )

                negative_input = gr.Textbox(
                    label="ネガティブプロンプト（全シーン共通）",
                    lines=2, value=VIDEO_DEFAULT_NEGATIVE,
                )

                with gr.Row():
                    smooth_enable_input = gr.Checkbox(
                        label="動き補間で滑らかに",
                        value=False,
                        info="生成元 FPS よりも高い目標 FPS に補間。長尺だと所要時間も増える",
                    )
                    smooth_fps_input = gr.Slider(
                        label="目標 FPS", minimum=12, maximum=60, value=30, step=1,
                    )
                    smooth_method_input = gr.Radio(
                        label="補間方法",
                        choices=[("RIFE (推奨)", "rife"), ("minterpolate", "minterpolate")],
                        value="rife",
                    )

                with gr.Row():
                    save_lastframes_input = gr.Checkbox(
                        label="境界フレームを jpg で保存",
                        value=True,
                        info="シーン間で渡される最終フレームを `_lastframe.jpg` に書き出し（確認用）",
                    )
                    fmt_input = gr.Radio(
                        label="出力フォーマット",
                        choices=["mp4"], value=VIDEO_DEFAULT_FORMAT,
                        info="連結は mp4 専用",
                    )

            # ===== 右: 結果表示 =====
            with gr.Column(scale=1):
                final_video = gr.Video(
                    label="連結後の最終動画",
                    interactive=False,
                )
                output_message = gr.Textbox(label="ステータス", interactive=False, lines=10)
                gallery = gr.Gallery(
                    label="シーン別動画パス一覧", columns=2, height="auto",
                )

        gr.Markdown("### シーン定義")
        gr.Markdown(
            "**1 行 1 シーン**。タブ区切りで `プロンプト[TAB]num_frames[TAB]guidance[TAB]seed_offset[TAB]fresh_start(0|1)`。"
            "末尾フィールドは省略可（左欄の既定値が使われます）。`#` で始まる行はコメント。"
            "`fresh_start=1` のシーンは前シーン最終フレームを引き継がず t2v で開始します。"
        )

        with gr.Row():
            example_input = gr.Dropdown(
                label="プリセット例（選ぶと固定/シーン定義が自動入力）",
                choices=list(VIDEO_CHAIN_EXAMPLES.keys()),
                value=None,
                scale=2,
            )
            estimate_label = gr.Markdown("シーン: 0  /  合計: 0秒", elem_id="chain_estimate")

        scenes_input = gr.Textbox(
            label="シーン定義",
            lines=14,
            placeholder=(
                "# 例: '\\t' は実際の TAB 文字を入れてください\n"
                "a calm forest with morning mist\t121\t3.0\t0\t1\n"
                "the same forest, sunlight breaking through leaves\t121\t3.0\t1\t0\n"
                "...\n"
            ),
        )

        with gr.Row():
            generate_btn = gr.Button("連鎖生成スタート", variant="primary", scale=2)
            stop_btn = gr.Button("停止", variant="stop", scale=1)

        # ===== コールバック =====

        def on_model_change(model_key):
            cfg = VIDEO_MODELS.get(model_key, {})
            return (
                model_status_text(model_key, "chain"),
                gr.update(value=cfg.get("default_width", 704)),
                gr.update(value=cfg.get("default_height", 480)),
                gr.update(value=cfg.get("default_fps", 24)),
                gr.update(value=cfg.get("default_steps", 30)),
                gr.update(
                    value=min(121, cfg.get("max_frames", 121)),
                    maximum=cfg.get("max_frames", 257),
                ),
                gr.update(value=cfg.get("default_guidance", 3.0)),
            )

        model_input.change(
            fn=on_model_change, inputs=[model_input],
            outputs=[
                model_info,
                width_input, height_input, fps_input, steps_input,
                default_frames_input, default_guidance_input,
            ],
        )

        def apply_example(name):
            if not name or name not in VIDEO_CHAIN_EXAMPLES:
                return gr.update(), gr.update()
            scenes_text = VIDEO_CHAIN_EXAMPLES[name]
            return scenes_text, VIDEO_DEFAULT_NEGATIVE
        example_input.change(
            fn=apply_example, inputs=[example_input],
            outputs=[scenes_input, negative_input],
        )

        def on_scenes_change(text, fps, xfade):
            return _estimate_duration(text, fps, xfade)
        for _change in (scenes_input.change, fps_input.change, xfade_input.change):
            _change(
                fn=on_scenes_change,
                inputs=[scenes_input, fps_input, xfade_input],
                outputs=[estimate_label],
            )

        def run_chain(
            model_key, scenes_text, negative,
            width, height, fps, steps,
            default_frames, default_guidance,
            base_seed, xfade,
            smooth_enable, smooth_fps, smooth_method,
            save_lastframes, fmt,
            progress=gr.Progress(track_tqdm=True),
        ):
            ok, dl_status = ensure_model_ready(model_key, progress)
            if not ok:
                return None, f"モデルDL失敗:\n{dl_status}", []

            scenes = parse_scenes_text(scenes_text or "")
            if not scenes:
                return None, "シーン定義が空です（1 行 1 シーン）", []

            progress(0.1, desc=f"連鎖生成中（{len(scenes)}シーン）...")

            kwargs = build_smooth_kwargs(smooth_enable, smooth_fps, fps, fmt, smooth_method)

            paths, msg = generate_video_chain(
                scenes=scenes,
                model_key=model_key,
                negative_prompt=negative,
                width=int(width), height=int(height), fps=int(fps),
                default_num_frames=int(default_frames),
                default_guidance=float(default_guidance),
                num_inference_steps=int(steps),
                base_seed=int(base_seed),
                crossfade_seconds=float(xfade),
                fmt=fmt,
                save_lastframes=bool(save_lastframes),
                **kwargs,
            )

            # 最終連結ファイル: _final_concat.mp4 を末尾に積んでいるのでそれを返す
            final = None
            scene_only = paths
            for p in paths[::-1]:
                if p and os.path.basename(p).startswith("_final_concat"):
                    final = p
                    scene_only = [q for q in paths if q != p]
                    break
            if final is None and paths:
                # 1シーンしかなかった場合は scene_paths[0] がそのまま最終
                final = paths[0]
            return final, msg, scene_only

        generate_event = generate_btn.click(
            fn=run_chain,
            inputs=[
                model_input, scenes_input, negative_input,
                width_input, height_input, fps_input, steps_input,
                default_frames_input, default_guidance_input,
                base_seed_input, xfade_input,
                smooth_enable_input, smooth_fps_input, smooth_method_input,
                save_lastframes_input, fmt_input,
            ],
            outputs=[final_video, output_message, gallery],
        )
        stop_btn.click(
            fn=request_stop, inputs=None, outputs=None,
            cancels=[generate_event],
        )
