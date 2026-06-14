"""音楽バッチ生成モジュール

MusicGen用の2種類のバッチ生成を提供:
- X/Y/Z Plot: パラメータ（Seed/Temperature/Guidance Scale/Top-K/Top-P/Duration）を軸として変化
- Variable Prompt: プロンプトテンプレート + 変数定義 で全組み合わせ × Seed
"""
import os
import gc
import re
import traceback
from typing import Optional

import numpy as np
import torch

from config import DEVICE, MUSIC_GEN_MODELS, clear_stop, is_stop_requested
from utils.audio import (
    create_audio_output_dir, normalize_audio, save_audio, add_fade, load_audio
)
from utils.batch_parsing import (
    parse_values, parse_variable_definitions, generate_combinations, save_metadata_csv
)
from utils.file import sanitize_filename
from .audio_manager import audio_pipeline_manager


# X/Y/Z Plot で変化させられるパラメータ
MUSIC_PLOT_PARAMETERS = {
    "なし": None,
    "Seed": "seed",
    "Temperature": "temperature",
    "Guidance Scale": "guidance_scale",
    "Top-K": "top_k",
    "Top-P": "top_p",
    "Duration": "duration",
}

# 整数として扱うパラメータ
_INT_PARAMS = {"seed", "top_k"}

# バッチ生成の上限
MUSIC_BATCH_LIMIT = 10000


# Variable Prompt 用のテンプレート例
# 各エントリは fixed / template / variables の3要素
MUSIC_VARIABLE_PROMPT_EXAMPLES = {
    "ジャンル × テンポ": {
        "fixed": "instrumental music",
        "template": "{genre}, {tempo} tempo",
        "variables": (
            "genre: jazz, rock, classical, electronic, ambient\n"
            "tempo: slow, medium, fast"
        ),
    },
    "楽器ソロ × ムード": {
        "fixed": "solo instrumental music",
        "template": "{instrument} solo, {mood}",
        "variables": (
            "instrument: piano, violin, guitar, saxophone, flute\n"
            "mood: happy, sad, mysterious, romantic"
        ),
    },
    "ゲームBGM (シーン別)": {
        "fixed": "video game soundtrack, orchestral",
        "template": "{scene} scene, {atmosphere} atmosphere",
        "variables": (
            "scene: battle, town, dungeon, forest, boss\n"
            "atmosphere: epic, mysterious, peaceful"
        ),
    },
    "Lo-Fi バリエーション": {
        "fixed": "lo-fi hip hop beats, vinyl crackling",
        "template": "{mood}, with {instrument}",
        "variables": (
            "mood: chill, melancholic, nostalgic, dreamy\n"
            "instrument: jazzy piano, soft saxophone, mellow guitar"
        ),
    },
    "EDM ビルド比較": {
        "fixed": "electronic dance music",
        "template": "{subgenre} with {synth} synth, {bpm} BPM",
        "variables": (
            "subgenre: house, trance, techno, dubstep\n"
            "synth: warm analog, bright digital, plucky\n"
            "bpm: 120, 128, 140"
        ),
    },
    "オーケストラ編成比較": {
        "fixed": "orchestral cinematic music",
        "template": "featuring {section}, {mood} mood",
        "variables": (
            "section: strings, brass, woodwinds, full orchestra\n"
            "mood: heroic, sad, mysterious, triumphant"
        ),
    },
    "和楽器コンビネーション": {
        "fixed": "traditional Japanese music",
        "template": "{instrument1} and {instrument2}, {mood}",
        "variables": (
            "instrument1: shakuhachi, koto, shamisen\n"
            "instrument2: taiko drums, biwa, fue flute\n"
            "mood: serene, festive, dramatic"
        ),
    },
    "カフェBGM (時間帯別)": {
        "fixed": "cafe background music, instrumental",
        "template": "{time_of_day}, {style} style",
        "variables": (
            "time_of_day: morning, afternoon, evening, late night\n"
            "style: jazz, bossa nova, acoustic, lo-fi"
        ),
    },
    "アンビエント (環境音)": {
        "fixed": "ambient atmospheric music",
        "template": "with {sound} sounds, {texture} texture",
        "variables": (
            "sound: rain, ocean waves, forest, wind\n"
            "texture: soft pads, crystalline bells, deep drone"
        ),
    },
    "テンポ × キー": {
        "fixed": "piano instrumental",
        "template": "{tempo} tempo, in {key}",
        "variables": (
            "tempo: slow, medium, fast\n"
            "key: C major, A minor, D minor, F major"
        ),
    },
}


def _generate_one(
    model,
    processor,
    prompt: str,
    duration: float,
    temperature: float,
    guidance_scale: float,
    top_k: int,
    top_p: float,
    seed: int,
    melody_audio: Optional[str] = None,
) -> tuple[np.ndarray, int, int]:
    """MusicGenで1回分生成して (audio_numpy, sample_rate, used_seed) を返す"""
    if seed == -1 or seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    melody = None
    if melody_audio and os.path.exists(melody_audio) and "melody" in (audio_pipeline_manager.current_music_model or ""):
        melody_data, _ = load_audio(melody_audio, target_sr=model.config.audio_encoder.sampling_rate)
        melody = torch.tensor(melody_data).unsqueeze(0).to(DEVICE)

    # Seedを設定（manual_seedはグローバルRNGに作用するため毎回呼ぶ）
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_new_tokens = int(duration * model.config.audio_encoder.frame_rate)

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": float(temperature),
        "guidance_scale": float(guidance_scale),
    }
    if top_k and top_k > 0:
        gen_config["top_k"] = int(top_k)
    if top_p and top_p > 0:
        gen_config["top_p"] = float(top_p)

    with torch.no_grad():
        if melody is not None:
            audio_values = model.generate(**inputs, audio=melody, **gen_config)
        else:
            audio_values = model.generate(**inputs, **gen_config)

    audio_numpy = audio_values.cpu().numpy().squeeze()
    sample_rate = model.config.audio_encoder.sampling_rate

    audio_numpy = normalize_audio(audio_numpy)
    audio_numpy = add_fade(audio_numpy, sample_rate, fade_in_ms=50, fade_out_ms=200)

    del audio_values
    return audio_numpy, sample_rate, seed


# =============================================================================
# X/Y/Z Plot
# =============================================================================

def generate_music_xyz_plot(
    prompt: str,
    model_name: str,
    base_duration: float,
    base_temperature: float,
    base_guidance_scale: float,
    base_top_k: int,
    base_top_p: float,
    base_seed: int,
    melody_audio: Optional[str],
    x_param: str, x_values: str,
    y_param: str, y_values: str,
    z_param: str, z_values: str,
) -> tuple[list[str], str]:
    """軸ごとにパラメータを変化させて音楽を生成

    Returns:
        (output_paths, status_message) - 生成された音声ファイルパスのリスト
    """
    if not prompt.strip():
        return [], "プロンプトを入力してください"

    x_key = MUSIC_PLOT_PARAMETERS.get(x_param)
    y_key = MUSIC_PLOT_PARAMETERS.get(y_param)
    z_key = MUSIC_PLOT_PARAMETERS.get(z_param)

    x_vals = parse_values(x_values, x_key, _INT_PARAMS) if x_key else [None]
    y_vals = parse_values(y_values, y_key, _INT_PARAMS) if y_key else [None]
    z_vals = parse_values(z_values, z_key, _INT_PARAMS) if z_key else [None]

    if x_vals == [None] and y_vals == [None] and z_vals == [None]:
        return [], "少なくとも1つの軸にパラメータと値を設定してください"

    total = len(x_vals) * len(y_vals) * len(z_vals)
    if total > MUSIC_BATCH_LIMIT:
        return [], f"生成数が多すぎます: {total}件 (上限: {MUSIC_BATCH_LIMIT}件)"

    print(f"Music X/Y/Z Plot: total={total}")
    print(f"  X: {x_param} = {x_vals}")
    print(f"  Y: {y_param} = {y_vals}")
    print(f"  Z: {z_param} = {z_vals}")

    output_dir = create_audio_output_dir("music_xyz", prompt[:30])

    try:
        model, processor = audio_pipeline_manager.load_musicgen(model_name)
    except Exception as e:
        return [], f"モデルロードエラー: {e}"

    csv_rows = []
    output_paths = []
    generated = 0
    stopped = False
    clear_stop()

    for z_idx, z_val in enumerate(z_vals):
        if is_stop_requested():
            stopped = True
            break
        for y_idx, y_val in enumerate(y_vals):
            if is_stop_requested():
                stopped = True
                break
            for x_idx, x_val in enumerate(x_vals):
                if is_stop_requested():
                    stopped = True
                    break

                params = {
                    "duration": float(base_duration),
                    "temperature": float(base_temperature),
                    "guidance_scale": float(base_guidance_scale),
                    "top_k": int(base_top_k),
                    "top_p": float(base_top_p),
                    "seed": int(base_seed),
                }
                if x_key and x_val is not None:
                    params[x_key] = x_val
                if y_key and y_val is not None:
                    params[y_key] = y_val
                if z_key and z_val is not None:
                    params[z_key] = z_val

                try:
                    audio_numpy, sample_rate, used_seed = _generate_one(
                        model, processor, prompt,
                        duration=params["duration"],
                        temperature=params["temperature"],
                        guidance_scale=params["guidance_scale"],
                        top_k=params["top_k"],
                        top_p=params["top_p"],
                        seed=params["seed"],
                        melody_audio=melody_audio,
                    )
                except Exception as e:
                    traceback.print_exc()
                    return output_paths, f"生成中にエラー: {e}\n（{generated}/{total}件完了）"

                filename = f"music_x{x_idx}_y{y_idx}_z{z_idx}_seed{used_seed}.wav"
                filepath = os.path.join(output_dir, filename)
                save_audio(audio_numpy, sample_rate, filepath, normalize=False)
                output_paths.append(filepath)

                csv_rows.append({
                    "filename": filename,
                    "prompt": prompt,
                    "model": model_name,
                    "duration": params["duration"],
                    "temperature": params["temperature"],
                    "guidance_scale": params["guidance_scale"],
                    "top_k": params["top_k"],
                    "top_p": params["top_p"],
                    "seed": used_seed,
                    "x_param": x_param, "x_value": x_val,
                    "y_param": y_param, "y_value": y_val,
                    "z_param": z_param, "z_value": z_val,
                })

                generated += 1
                print(f"Generated {generated}/{total}: x={x_val}, y={y_val}, z={z_val}, seed={used_seed}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # CSV保存
    csv_path = os.path.join(output_dir, "metadata.csv")
    if csv_rows:
        save_metadata_csv(csv_path, csv_rows, list(csv_rows[0].keys()))

    if stopped and generated == 0:
        return [], "生成が中止されました"

    head = "中止しました" if stopped else "生成完了"
    msg = (
        f"{head}（{generated}/{total}件）\n"
        f"X軸: {x_param} = {x_vals}\n"
        f"Y軸: {y_param} = {y_vals}\n"
        f"Z軸: {z_param} = {z_vals}\n"
        f"CSV: {csv_path}\n"
        f"保存先: {output_dir}"
    )
    return output_paths, msg


# =============================================================================
# Variable Prompt
# =============================================================================

def generate_music_variable_prompt(
    fixed_prompt: str,
    variable_template: str,
    variable_definitions: str,
    model_name: str,
    duration: float,
    temperature: float,
    guidance_scale: float,
    top_k: int,
    top_p: float,
    base_seed: int,
    num_seed_variations: int,
    melody_audio: Optional[str],
) -> tuple[list[str], str]:
    """変数プロンプトで音楽をバッチ生成"""
    if not fixed_prompt.strip() and not variable_template.strip():
        return [], "固定プロンプトまたは変数テンプレートを入力してください"

    variables = parse_variable_definitions(variable_definitions)
    if not variables:
        return [], "変数が定義されていません。形式: 変数名: 値1, 値2, 値3"

    template_vars = set(re.findall(r"\{(\w+)\}", variable_template))
    undefined = template_vars - set(variables.keys())
    if undefined:
        return [], f"未定義の変数があります: {', '.join(undefined)}"

    combinations = list(generate_combinations(variables))
    total_combos = len(combinations)
    num_seeds = max(1, int(num_seed_variations))
    total = total_combos * num_seeds

    if total > MUSIC_BATCH_LIMIT:
        return [], (
            f"生成数が多すぎます: {total}件 (上限: {MUSIC_BATCH_LIMIT}件)\n"
            f"組み合わせ {total_combos} × Seed {num_seeds}"
        )

    print(f"Music Variable Prompt: {total_combos} combos × {num_seeds} seeds = {total}")
    for k, v in variables.items():
        print(f"  {k}: {v}")

    output_dir = create_audio_output_dir("music_varprompt", fixed_prompt[:30] or variable_template[:30])

    try:
        model, processor = audio_pipeline_manager.load_musicgen(model_name)
    except Exception as e:
        return [], f"モデルロードエラー: {e}"

    csv_rows = []
    output_paths = []
    generated = 0
    stopped = False
    clear_stop()

    for combo_idx, combo in enumerate(combinations):
        if is_stop_requested():
            stopped = True
            break
        try:
            filled = variable_template.format(**combo) if variable_template.strip() else ""
        except KeyError as e:
            return [], f"テンプレートエラー: 変数 {e} が見つかりません"

        if fixed_prompt.strip() and filled.strip():
            full_prompt = fixed_prompt.strip() + ", " + filled.strip()
        elif fixed_prompt.strip():
            full_prompt = fixed_prompt.strip()
        else:
            full_prompt = filled.strip()

        for seed_offset in range(num_seeds):
            if is_stop_requested():
                stopped = True
                break

            current_seed = int(base_seed) + seed_offset

            try:
                audio_numpy, sample_rate, used_seed = _generate_one(
                    model, processor, full_prompt,
                    duration=float(duration),
                    temperature=float(temperature),
                    guidance_scale=float(guidance_scale),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    seed=current_seed,
                    melody_audio=melody_audio,
                )
            except Exception as e:
                traceback.print_exc()
                return output_paths, f"生成中にエラー: {e}\n（{generated}/{total}件完了）"

            combo_parts = []
            for k, v in combo.items():
                safe_k = sanitize_filename(str(k), 8)
                safe_v = sanitize_filename(str(v), 12)
                combo_parts.append(f"{safe_k}-{safe_v}")
            combo_str = "_".join(combo_parts)
            if len(combo_str) > 40:
                combo_str = combo_str[:40]

            filename = f"music_{combo_idx:04d}_{combo_str}_seed{used_seed}.wav"
            filepath = os.path.join(output_dir, filename)
            save_audio(audio_numpy, sample_rate, filepath, normalize=False)
            output_paths.append(filepath)

            csv_rows.append({
                "filename": filename,
                "full_prompt": full_prompt,
                "fixed_prompt": fixed_prompt,
                "variable_template": variable_template,
                **combo,
                "model": model_name,
                "duration": duration,
                "temperature": temperature,
                "guidance_scale": guidance_scale,
                "top_k": top_k,
                "top_p": top_p,
                "seed": used_seed,
            })

            generated += 1
            print(f"Generated {generated}/{total}: {combo_str}, seed={used_seed}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, "metadata.csv")
    if csv_rows:
        # 全行に共通のキーセットを作る
        all_keys = []
        seen = set()
        for row in csv_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        save_metadata_csv(csv_path, csv_rows, all_keys)

    if stopped and generated == 0:
        return [], "生成が中止されました"

    var_summary = ", ".join(f"{k}({len(v)})" for k, v in variables.items())
    head = "中止しました" if stopped else "生成完了"
    msg = (
        f"{head}（{generated}/{total}件）\n"
        f"変数: {var_summary}\n"
        f"組み合わせ: {total_combos}通り × Seed: {num_seeds}\n"
        f"CSV: {csv_path}\n"
        f"保存先: {output_dir}"
    )
    return output_paths, msg
