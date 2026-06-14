"""動画バッチ生成タブ

3つのサブタブ:
- Seed バッチ: 同じプロンプトで N 個の seed バリエーション
- Variable Prompt: 変数テンプレートで全組み合わせ生成
- X/Y/Z Plot: 軸ごとにパラメータを変化させて格子状生成
"""
import gradio as gr

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT, VIDEO_DEFAULT_NEGATIVE,
    request_stop,
)
from pipelines.video_batch import (
    generate_video_seed_batch,
    generate_video_variable_prompt,
    generate_video_xyz_plot,
    VIDEO_PLOT_PARAMETERS,
    VIDEO_VARIABLE_PROMPT_EXAMPLES,
)
from .video_tab_common import (
    model_choices, model_status_text, ensure_model_ready, build_smooth_kwargs,
)


def create_video_batch_tab():
    """動画バッチ生成タブ（共通モデル・パラメータ + 3サブタブ）"""

    default_model = "wan21_1_3b"  # 最も品質の良いモデルをデフォルト

    with gr.Column():
        gr.Markdown("## 動画バッチ生成")
        gr.Markdown(
            "同じプロンプトでseedを変えたり、変数で組み合わせ生成、X/Y/Z軸でパラメータ走査ができます。"
            "Wan2.1 (1動画~6分) や LTX-Video (1動画~3分) は時間がかかるので、**まず少数で試す**のがおすすめ。"
        )

        # ===== 共通: モデル選択 + 解像度・ステップなどの基本パラメータ =====
        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Dropdown(
                    label="モデル",
                    choices=model_choices(),
                    value=default_model,
                )
                model_info = gr.Markdown(model_status_text(default_model, "oneline"))

                with gr.Row():
                    width_input = gr.Slider(
                        label="幅", minimum=256, maximum=1280,
                        value=VIDEO_MODELS[default_model]["default_width"], step=64,
                    )
                    height_input = gr.Slider(
                        label="高さ", minimum=256, maximum=720,
                        value=VIDEO_MODELS[default_model]["default_height"], step=64,
                    )
                with gr.Row():
                    num_frames_input = gr.Slider(
                        label="フレーム数", minimum=8,
                        maximum=VIDEO_MODELS[default_model]["max_frames"],
                        value=VIDEO_MODELS[default_model]["default_frames"], step=1,
                    )
                    fps_input = gr.Slider(
                        label="FPS", minimum=1, maximum=30,
                        value=VIDEO_MODELS[default_model]["default_fps"], step=1,
                    )
                with gr.Row():
                    steps_input = gr.Slider(
                        label="Steps", minimum=10, maximum=100,
                        value=VIDEO_MODELS[default_model]["default_steps"], step=1,
                    )
                    guidance_input = gr.Slider(
                        label="Guidance Scale", minimum=1.0, maximum=15.0,
                        value=VIDEO_MODELS[default_model]["default_guidance"], step=0.5,
                    )

                with gr.Row():
                    smooth_enable_input = gr.Checkbox(
                        label="動き補間で滑らかに",
                        value=VIDEO_MODELS[default_model].get("auto_smooth", False),
                    )
                    smooth_fps_input = gr.Slider(
                        label="目標FPS", minimum=12, maximum=60, value=24, step=1,
                    )
                    smooth_method_input = gr.Radio(
                        label="補間方法",
                        choices=[("RIFE (推奨)", "rife"), ("minterpolate", "minterpolate")],
                        value="rife",
                    )

                fmt_input = gr.Radio(
                    label="出力フォーマット",
                    choices=["mp4", "gif"], value=VIDEO_DEFAULT_FORMAT,
                )

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="生成された動画一覧（パスのみ表示）", columns=2, height="auto")
                output_message = gr.Textbox(label="ステータス", interactive=False, lines=8)

        # モデル切替時に解像度・ステップなどを自動更新
        def on_model_change(model_key):
            cfg = VIDEO_MODELS.get(model_key, {})
            return (
                model_status_text(model_key, "oneline"),
                gr.update(value=cfg.get("default_width", 512)),
                gr.update(value=cfg.get("default_height", 512)),
                gr.update(
                    value=cfg.get("default_frames", 16),
                    maximum=cfg.get("max_frames", 32),
                ),
                gr.update(value=cfg.get("default_fps", 8)),
                gr.update(value=cfg.get("default_steps", 25)),
                gr.update(value=cfg.get("default_guidance", 7.5)),
                gr.update(value=cfg.get("auto_smooth", False)),
                gr.update(value=cfg.get("default_smooth_target_fps", 24)),
            )

        model_input.change(
            fn=on_model_change, inputs=[model_input],
            outputs=[
                model_info, width_input, height_input,
                num_frames_input, fps_input, steps_input, guidance_input,
                smooth_enable_input, smooth_fps_input,
            ],
        )

        # ===== サブタブ =====
        with gr.Tabs():

            # --- 1. Seed バッチ ---
            with gr.Tab("Seed バッチ"):
                gr.Markdown(
                    "**同じプロンプトで N 個の seed 違い動画を生成。** "
                    "プロンプトは固定なので、ランダム性によるバリエーションを比較したいときに使う。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_prompt = gr.Textbox(
                            label="プロンプト", lines=3,
                            placeholder="a cat walking through a sunlit forest, cinematic",
                        )
                        seed_negative = gr.Textbox(
                            label="ネガティブプロンプト", lines=2,
                            value=VIDEO_DEFAULT_NEGATIVE,
                        )
                        seed_image = gr.Image(
                            label="入力画像（i2v モデルのみ）", type="pil",
                            sources=["upload", "clipboard"], visible=False,
                        )
                        with gr.Row():
                            seed_count = gr.Slider(
                                label="生成本数", minimum=1, maximum=50, value=4, step=1,
                            )
                            seed_base = gr.Number(
                                label="ベース Seed", value=-1,
                                info="-1=毎回ランダム / 整数=連番 (base, base+1, ...)",
                            )
                        with gr.Row():
                            seed_btn = gr.Button("Seed バッチ生成", variant="primary")
                            seed_stop_btn = gr.Button("停止", variant="stop")

            # --- 2. Variable Prompt ---
            with gr.Tab("Variable Prompt"):
                gr.Markdown(
                    "**プロンプトテンプレートと変数定義から全組み合わせ動画を生成。** "
                    "例: `a {animal} in {place}` + 変数 `animal: cat, dog` / `place: forest, beach` → 4 通り。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vp_example = gr.Dropdown(
                            label="プリセット例",
                            choices=list(VIDEO_VARIABLE_PROMPT_EXAMPLES.keys()),
                            value=None,
                            info="選ぶと固定/テンプレ/変数定義が自動で埋まる",
                        )
                        vp_fixed = gr.Textbox(
                            label="固定プロンプト（全動画共通）", lines=2,
                            placeholder="cinematic, masterpiece, high quality",
                        )
                        vp_template = gr.Textbox(
                            label="変数テンプレート（{変数名} を含める）", lines=2,
                            placeholder="a {animal} walking through {place}",
                        )
                        vp_variables = gr.Textbox(
                            label="変数定義（1行1変数：変数名: 値1, 値2, ...）", lines=4,
                            placeholder="animal: cat, dog\nplace: forest, beach",
                        )
                        vp_negative = gr.Textbox(
                            label="ネガティブプロンプト", lines=2,
                            value=VIDEO_DEFAULT_NEGATIVE,
                        )
                        vp_image = gr.Image(
                            label="入力画像（i2v モデルのみ・全組み合わせ共通）", type="pil",
                            sources=["upload", "clipboard"], visible=False,
                        )
                        with gr.Row():
                            vp_seed_count = gr.Slider(
                                label="各組み合わせのSeed数", minimum=1, maximum=10, value=1, step=1,
                            )
                            vp_base_seed = gr.Number(label="ベース Seed", value=42)
                        with gr.Row():
                            vp_btn = gr.Button("Variable Prompt 生成", variant="primary")
                            vp_stop_btn = gr.Button("停止", variant="stop")

                # プリセット選択時に各フィールドを埋める
                def apply_vp_example(name):
                    if not name or name not in VIDEO_VARIABLE_PROMPT_EXAMPLES:
                        return gr.update(), gr.update(), gr.update()
                    ex = VIDEO_VARIABLE_PROMPT_EXAMPLES[name]
                    return ex["fixed"], ex["template"], ex["variables"]

                vp_example.change(
                    fn=apply_vp_example, inputs=[vp_example],
                    outputs=[vp_fixed, vp_template, vp_variables],
                )

            # --- 3. X/Y/Z Plot ---
            with gr.Tab("X/Y/Z Plot"):
                gr.Markdown(
                    "**X/Y/Z 3軸でパラメータを変化させて格子状生成。** "
                    "値はカンマ区切り (例: `1, 2, 3`) または範囲 (例: `1-10:2`=1から10まで2刻み)。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        xyz_prompt = gr.Textbox(
                            label="プロンプト", lines=3,
                            placeholder="a cat walking through a sunlit forest, cinematic",
                        )
                        xyz_negative = gr.Textbox(
                            label="ネガティブプロンプト", lines=2,
                            value=VIDEO_DEFAULT_NEGATIVE,
                        )
                        xyz_image = gr.Image(
                            label="入力画像（i2v モデルのみ）", type="pil",
                            sources=["upload", "clipboard"], visible=False,
                        )
                        xyz_base_seed = gr.Number(label="ベース Seed", value=42)

                        param_choices = list(VIDEO_PLOT_PARAMETERS.keys())
                        with gr.Row():
                            x_param = gr.Dropdown(label="X軸", choices=param_choices, value="Seed")
                            x_values = gr.Textbox(label="X値（例: 1, 2, 3 or 1-10:2）", value="42, 100, 200")
                        with gr.Row():
                            y_param = gr.Dropdown(label="Y軸", choices=param_choices, value="Guidance Scale")
                            y_values = gr.Textbox(label="Y値", value="3.0, 5.0, 7.0")
                        with gr.Row():
                            z_param = gr.Dropdown(label="Z軸", choices=param_choices, value="なし")
                            z_values = gr.Textbox(label="Z値", value="")

                        with gr.Row():
                            xyz_btn = gr.Button("X/Y/Z Plot 生成", variant="primary")
                            xyz_stop_btn = gr.Button("停止", variant="stop")

        # ===== コールバック =====

        # Seed バッチ
        def run_seed_batch(model_key, prompt, negative, image, num_videos, base_seed,
                           width, height, num_frames, fps, steps, guidance,
                           smooth_enable, smooth_fps, smooth_method, fmt,
                           progress=gr.Progress(track_tqdm=True)):
            ok, dl_status = ensure_model_ready(model_key, progress)
            if not ok:
                return [], f"モデルDL失敗:\n{dl_status}"
            progress(0.1, desc=f"Seed バッチ生成中...")

            kwargs = build_smooth_kwargs(smooth_enable, smooth_fps, fps, fmt, smooth_method)

            paths, msg = generate_video_seed_batch(
                model_key=model_key, prompt=prompt, negative_prompt=negative,
                image=image,
                num_videos=int(num_videos), base_seed=int(base_seed),
                width=int(width), height=int(height),
                num_frames=int(num_frames), fps=int(fps),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                fmt=fmt, **kwargs,
            )
            return paths, msg

        seed_event = seed_btn.click(
            fn=run_seed_batch,
            inputs=[
                model_input, seed_prompt, seed_negative, seed_image, seed_count, seed_base,
                width_input, height_input, num_frames_input, fps_input,
                steps_input, guidance_input,
                smooth_enable_input, smooth_fps_input, smooth_method_input,
                fmt_input,
            ],
            outputs=[gallery, output_message],
        )
        seed_stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[seed_event])

        # Variable Prompt
        def run_variable_prompt(model_key, fixed, template, variables, negative, image,
                                seed_count_var, base_seed_var,
                                width, height, num_frames, fps, steps, guidance,
                                smooth_enable, smooth_fps, smooth_method, fmt,
                                progress=gr.Progress(track_tqdm=True)):
            ok, dl_status = ensure_model_ready(model_key, progress)
            if not ok:
                return [], f"モデルDL失敗:\n{dl_status}"
            progress(0.1, desc="Variable Prompt 生成中...")

            kwargs = build_smooth_kwargs(smooth_enable, smooth_fps, fps, fmt, smooth_method)

            paths, msg = generate_video_variable_prompt(
                model_key=model_key, fixed_prompt=fixed,
                variable_template=template, variable_definitions=variables,
                negative_prompt=negative, image=image,
                num_seed_variations=int(seed_count_var), base_seed=int(base_seed_var),
                width=int(width), height=int(height),
                num_frames=int(num_frames), fps=int(fps),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                fmt=fmt, **kwargs,
            )
            return paths, msg

        vp_event = vp_btn.click(
            fn=run_variable_prompt,
            inputs=[
                model_input, vp_fixed, vp_template, vp_variables, vp_negative, vp_image,
                vp_seed_count, vp_base_seed,
                width_input, height_input, num_frames_input, fps_input,
                steps_input, guidance_input,
                smooth_enable_input, smooth_fps_input, smooth_method_input,
                fmt_input,
            ],
            outputs=[gallery, output_message],
        )
        vp_stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[vp_event])

        # X/Y/Z Plot
        def run_xyz(model_key, prompt, negative, image, base_seed,
                    x_p, x_v, y_p, y_v, z_p, z_v,
                    width, height, num_frames, fps, steps, guidance,
                    smooth_enable, smooth_fps, smooth_method, fmt,
                    progress=gr.Progress(track_tqdm=True)):
            ok, dl_status = ensure_model_ready(model_key, progress)
            if not ok:
                return [], f"モデルDL失敗:\n{dl_status}"
            progress(0.1, desc="X/Y/Z Plot 生成中...")

            kwargs = build_smooth_kwargs(smooth_enable, smooth_fps, fps, fmt, smooth_method)

            paths, msg = generate_video_xyz_plot(
                model_key=model_key, prompt=prompt, negative_prompt=negative,
                image=image, base_seed=int(base_seed),
                base_width=int(width), base_height=int(height),
                base_num_frames=int(num_frames), base_fps=int(fps),
                base_num_inference_steps=int(steps),
                base_guidance_scale=float(guidance),
                x_param=x_p, x_values=x_v,
                y_param=y_p, y_values=y_v,
                z_param=z_p, z_values=z_v,
                fmt=fmt, **kwargs,
            )
            return paths, msg

        xyz_event = xyz_btn.click(
            fn=run_xyz,
            inputs=[
                model_input, xyz_prompt, xyz_negative, xyz_image, xyz_base_seed,
                x_param, x_values, y_param, y_values, z_param, z_values,
                width_input, height_input, num_frames_input, fps_input,
                steps_input, guidance_input,
                smooth_enable_input, smooth_fps_input, smooth_method_input,
                fmt_input,
            ],
            outputs=[gallery, output_message],
        )
        xyz_stop_btn.click(fn=request_stop, inputs=None, outputs=None, cancels=[xyz_event])

        # 入力画像欄: i2v モデルのときのみ表示
        def on_model_change_image(model_key):
            is_i2v = VIDEO_MODELS.get(model_key, {}).get("type") == "image-to-video"
            return gr.update(visible=is_i2v), gr.update(visible=is_i2v), gr.update(visible=is_i2v)

        model_input.change(
            fn=on_model_change_image, inputs=[model_input],
            outputs=[seed_image, vp_image, xyz_image],
        )
