"""音楽生成モジュール"""
import os
import gc
from typing import Optional, Tuple

import torch

from config import DEVICE, MUSIC_GEN_MODELS, AUDIOLDM_MODELS
from utils.audio import (
    create_audio_output_dir, normalize_audio, save_audio, add_fade, load_audio
)
from .audio_manager import audio_pipeline_manager


def generate_music_musicgen(
    prompt: str,
    model_name: str = "musicgen-small",
    duration: float = 10.0,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    top_k: int = 250,
    top_p: float = 0.0,
    seed: int = -1,
    melody_audio: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """MusicGenモデルで音楽を生成

    Args:
        prompt: 音楽の説明（例: "upbeat jazz with piano"）
        model_name: モデル名（musicgen-small, musicgen-medium, musicgen-large, musicgen-melody）
        duration: 生成する音楽の長さ（秒）
        temperature: 温度パラメータ
        guidance_scale: ガイダンススケール
        top_k: top-kサンプリング
        top_p: top-pサンプリング（0=無効）
        seed: シード値（-1=ランダム）
        melody_audio: メロディー条件付け用音声（musicgen-melodyのみ）

    Returns:
        (output_path, status_message) タプル
    """
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    try:
        # MusicGenモデルをロード
        model, processor = audio_pipeline_manager.load_musicgen(model_name)

        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("music_musicgen", prompt[:30])

        # シード設定
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        print(f"Generating music with MusicGen ({model_name}): {prompt}")
        print(f"Duration: {duration}s, Seed: {seed}")

        # 入力を準備
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # メロディー条件付け（musicgen-melodyのみ）
        melody = None
        if melody_audio and os.path.exists(melody_audio) and "melody" in model_name:
            print(f"Loading melody audio: {melody_audio}")
            melody_data, _ = load_audio(melody_audio, target_sr=model.config.audio_encoder.sampling_rate)
            melody = torch.tensor(melody_data).unsqueeze(0).to(DEVICE)

        # 音楽生成
        with torch.no_grad():
            # サンプル数を計算
            max_new_tokens = int(duration * model.config.audio_encoder.frame_rate)

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "guidance_scale": guidance_scale,
            }

            if top_k > 0:
                generation_config["top_k"] = top_k
            if top_p > 0:
                generation_config["top_p"] = top_p

            if melody is not None:
                audio_values = model.generate(
                    **inputs,
                    audio=melody,
                    **generation_config
                )
            else:
                audio_values = model.generate(
                    **inputs,
                    **generation_config
                )

        # numpy配列に変換
        audio_numpy = audio_values.cpu().numpy().squeeze()

        # サンプルレート取得
        sample_rate = model.config.audio_encoder.sampling_rate

        # 正規化とフェード処理
        audio_numpy = normalize_audio(audio_numpy)
        audio_numpy = add_fade(audio_numpy, sample_rate, fade_in_ms=50, fade_out_ms=200)

        # ファイル保存
        output_filename = f"musicgen_{model_name}_{duration}s_seed{seed}.wav"
        output_path = os.path.join(output_dir, output_filename)
        save_audio(audio_numpy, sample_rate, output_path, normalize=False)

        # メモリ解放
        del audio_values
        gc.collect()
        audio_pipeline_manager.clear_cache()

        actual_duration = len(audio_numpy) / sample_rate
        model_desc = MUSIC_GEN_MODELS.get(model_name, {}).get('description', '')
        return output_path, f"音楽生成完了: {actual_duration:.1f}秒\nモデル: {model_name} ({model_desc})\nSeed: {seed}\n保存先: {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"MusicGen音楽生成エラー: {str(e)}"


def generate_music_audioldm(
    prompt: str,
    negative_prompt: str = "",
    model_name: str = "audioldm2-music",
    duration: float = 10.0,
    num_inference_steps: int = 200,
    guidance_scale: float = 3.5,
    seed: int = -1,
) -> Tuple[Optional[str], str]:
    """AudioLDM2モデルで音楽を生成

    Args:
        prompt: 音楽の説明
        negative_prompt: ネガティブプロンプト
        model_name: モデル名（audioldm2-music, audioldm2）
        duration: 生成する音楽の長さ（秒）
        num_inference_steps: 推論ステップ数
        guidance_scale: ガイダンススケール
        seed: シード値（-1=ランダム）

    Returns:
        (output_path, status_message) タプル
    """
    if not prompt.strip():
        return None, "プロンプトを入力してください"

    try:
        # AudioLDM2モデルをロード
        pipeline = audio_pipeline_manager.load_audioldm(model_name)

        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("music_audioldm", prompt[:30])

        # シード設定
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        print(f"Generating music with AudioLDM2 ({model_name}): {prompt}")
        print(f"Duration: {duration}s, Steps: {num_inference_steps}, Seed: {seed}")

        # 音楽生成
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            audio_length_in_s=duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        audio_numpy = result.audios[0]

        # サンプルレート（AudioLDM2は16kHz）
        sample_rate = 16000

        # 正規化とフェード処理
        audio_numpy = normalize_audio(audio_numpy)
        audio_numpy = add_fade(audio_numpy, sample_rate, fade_in_ms=50, fade_out_ms=200)

        # ファイル保存
        output_filename = f"audioldm_{model_name}_{duration}s_seed{seed}.wav"
        output_path = os.path.join(output_dir, output_filename)
        save_audio(audio_numpy, sample_rate, output_path, normalize=False)

        # メモリ解放
        del result
        gc.collect()
        audio_pipeline_manager.clear_cache()

        actual_duration = len(audio_numpy) / sample_rate
        model_desc = AUDIOLDM_MODELS.get(model_name, {}).get('description', '')
        return output_path, f"音楽生成完了: {actual_duration:.1f}秒\nモデル: {model_name} ({model_desc})\nSeed: {seed}\n保存先: {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_str = str(e)
        if "_get_initial_cache_position" in error_str or "language_model" in error_str:
            return None, "AudioLDM2は現在のdiffusers/transformersバージョンとの互換性問題があります。\nMusicGenを使用してください。"
        return None, f"AudioLDM2音楽生成エラー: {error_str}"


def generate_music(
    prompt: str,
    model_type: str = "musicgen",
    model_name: str = "musicgen-small",
    duration: float = 10.0,
    seed: int = -1,
    **kwargs
) -> Tuple[Optional[str], str]:
    """統合音楽生成関数

    Args:
        prompt: 音楽の説明
        model_type: モデルタイプ（musicgen, audioldm）
        model_name: モデル名
        duration: 生成する音楽の長さ（秒）
        seed: シード値
        **kwargs: モデル固有のパラメータ

    Returns:
        (output_path, status_message) タプル
    """
    if model_type == "musicgen":
        return generate_music_musicgen(
            prompt=prompt,
            model_name=model_name,
            duration=duration,
            temperature=kwargs.get('temperature', 1.0),
            guidance_scale=kwargs.get('guidance_scale', 3.0),
            top_k=kwargs.get('top_k', 250),
            top_p=kwargs.get('top_p', 0.0),
            seed=seed,
            melody_audio=kwargs.get('melody_audio'),
        )
    elif model_type == "audioldm":
        return generate_music_audioldm(
            prompt=prompt,
            negative_prompt=kwargs.get('negative_prompt', ''),
            model_name=model_name,
            duration=duration,
            num_inference_steps=kwargs.get('num_inference_steps', 200),
            guidance_scale=kwargs.get('guidance_scale', 3.5),
            seed=seed,
        )
    else:
        return None, f"未対応の音楽生成モデル: {model_type}"


# 音楽ジャンル・スタイルのプロンプト例（カテゴリで整理）
MUSIC_PROMPT_EXAMPLES = {
    # ===== ジャンル =====
    "[ジャンル] ジャズ": "smooth jazz with piano and saxophone, relaxing background music",
    "[ジャンル] スウィングジャズ": "upbeat swing jazz with big band, walking bass and brass section",
    "[ジャンル] ボサノヴァ": "bossa nova with nylon guitar and gentle samba rhythm, warm and laid-back",
    "[ジャンル] ロック": "energetic rock music with electric guitar and drums, powerful riffs",
    "[ジャンル] ハードロック": "heavy metal rock with distorted electric guitar, fast drums, aggressive",
    "[ジャンル] パンク": "fast punk rock with raw guitar and pounding drums, rebellious energy",
    "[ジャンル] ブルース": "slow blues with electric guitar and harmonica, soulful and emotional",
    "[ジャンル] ファンク": "funky groove with slap bass, wah guitar, tight drums",
    "[ジャンル] ソウル": "soulful R&B with rhodes piano, smooth bass, and warm horns",
    "[ジャンル] ヒップホップ": "boom bap hip hop with vinyl drums and jazzy sample",
    "[ジャンル] Lo-Fi": "lo-fi hip hop beats, chill and relaxing, vinyl crackling",
    "[ジャンル] レゲエ": "reggae music with offbeat guitar, dub bass, relaxed groove",
    "[ジャンル] カントリー": "country music with acoustic guitar, banjo, and steel guitar",
    "[ジャンル] フォーク": "acoustic folk music with fingerpicked guitar, warm and intimate",
    "[ジャンル] ポップ": "catchy pop music with upbeat tempo and cheerful melody",
    "[ジャンル] EDM": "energetic EDM with big synth lead, sidechain bass, and festival drop",
    "[ジャンル] ハウス": "deep house music with warm bassline and groovy four-on-the-floor drums",
    "[ジャンル] テクノ": "minimal techno with driving 4/4 beat, hypnotic synth lines",
    "[ジャンル] トランス": "uplifting trance with arpeggiated synths and soaring lead",
    "[ジャンル] ドラムンベース": "drum and bass with fast breakbeats and rolling sub bass",
    "[ジャンル] アンビエント": "calm ambient music with soft pads and nature sounds",
    "[ジャンル] チップチューン": "8-bit chiptune music with retro game console sounds",
    "[ジャンル] クラシック": "classical orchestral music with violin and piano",
    "[ジャンル] バロック": "baroque chamber music with harpsichord and strings, ornate",
    "[ジャンル] オーケストラ": "full symphonic orchestra, dramatic and majestic",
    "[ジャンル] 弦楽四重奏": "string quartet, elegant and refined chamber music",
    "[ジャンル] ピアノソロ": "beautiful solo piano music, emotional and expressive",
    "[ジャンル] アコースティック": "acoustic guitar folk music, warm and intimate",
    "[ジャンル] シネマティック": "epic cinematic orchestral music with dramatic strings",
    "[ジャンル] 和風": "traditional Japanese music with shakuhachi flute, koto, and taiko drums",
    "[ジャンル] ケルト": "celtic folk music with fiddle, tin whistle, and bodhrán drum",
    "[ジャンル] ラテン": "latin music with congas, brass section, and rhythmic guitar",
    "[ジャンル] フラメンコ": "flamenco guitar with passionate strumming and palmas claps",

    # ===== ムード・雰囲気 =====
    "[ムード] 切ない": "melancholic piano melody, sad and emotional, slow tempo",
    "[ムード] 明るい": "cheerful and uplifting music, bright and joyful",
    "[ムード] 神秘的": "mysterious atmospheric music with ethereal pads and reverb",
    "[ムード] 緊張感": "tense suspenseful music with dark strings and ominous percussion",
    "[ムード] 勇ましい": "heroic epic music with brass fanfare and pounding drums",
    "[ムード] 癒し": "healing relaxation music with gentle piano and soft nature sounds",
    "[ムード] 瞑想": "meditation music with singing bowls and ambient drones",
    "[ムード] ノスタルジック": "nostalgic warm music with vintage tape feel and soft melody",
    "[ムード] ドラマチック": "highly dramatic music with sweeping strings and powerful crescendo",
    "[ムード] ダーク": "dark brooding music with low strings and dissonant tones",
    "[ムード] ロマンチック": "romantic music with lush strings and tender piano melody",
    "[ムード] 楽しい": "fun playful music with bouncy rhythm and whimsical melody",

    # ===== 用途・シーン =====
    "[用途] 作業BGM": "calm background music for studying, instrumental, focus and concentration",
    "[用途] カフェ": "cafe background music, jazzy and relaxed, light percussion",
    "[用途] 朝の目覚め": "gentle morning music with acoustic guitar, fresh and uplifting",
    "[用途] 睡眠": "peaceful sleep music with soft pads, slow tempo, deeply relaxing",
    "[用途] 運動": "high energy workout music with strong beat and motivating rhythm",
    "[用途] ヨガ": "calm yoga music with flute, soft strings, and gentle bells",
    "[用途] 瞑想・座禅": "deep meditation music with binaural drone and tibetan bowls",
    "[用途] ゲーム戦闘": "intense battle game music with heavy drums and aggressive orchestra",
    "[用途] ゲーム冒険": "adventurous fantasy game music with sweeping melody and epic feel",
    "[用途] ゲームRPG街": "warm RPG town theme with flute, harp, and gentle melody",
    "[用途] 映画予告": "epic movie trailer music, building tension to powerful climax",
    "[用途] 動画OP": "energetic vlog opening music, upbeat and cheerful",
    "[用途] ホラー": "creepy horror music with dissonant strings, whispers and unsettling sounds",
    "[用途] ニュース": "professional news jingle, neutral, clean and informative",
    "[用途] CM": "catchy commercial jingle, short, memorable and upbeat",
    "[用途] 結婚式": "wedding music with elegant piano and uplifting strings",

    # ===== テンポ =====
    "[テンポ] スロー (60BPM)": "slow tempo instrumental music at 60 BPM, gentle and calm",
    "[テンポ] ミディアム (100BPM)": "medium tempo instrumental music at 100 BPM, steady groove",
    "[テンポ] アップテンポ (140BPM)": "fast tempo instrumental music at 140 BPM, energetic and driving",
    "[テンポ] 高速 (170BPM)": "high BPM instrumental music at 170 BPM, intense and fast",
}
