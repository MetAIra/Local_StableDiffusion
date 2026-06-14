"""Text-to-Speech生成モジュール"""
import os
import gc
from typing import Optional, Tuple

import torch

from config import DEVICE, BARK_VOICE_PRESETS, XTTS_LANGUAGES
from utils.audio import (
    create_audio_output_dir, normalize_audio, save_audio,
    split_text_for_tts, add_fade, concatenate_audio
)
from .audio_manager import audio_pipeline_manager


def generate_speech_bark(
    text: str,
    voice_preset: str = "ja_speaker_0",
    temperature: float = 0.7,
    semantic_temperature: float = 0.7,
) -> Tuple[Optional[str], str]:
    """Barkモデルで音声を生成

    Args:
        text: 生成するテキスト
        voice_preset: ボイスプリセット（例: ja_speaker_0）
        temperature: 温度パラメータ
        semantic_temperature: セマンティック温度パラメータ

    Returns:
        (output_path, status_message) タプル
    """
    if not text.strip():
        return None, "テキストを入力してください"

    try:
        # Barkモデルをロード
        model, processor = audio_pipeline_manager.load_bark()

        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("tts_bark", text[:30])

        # 長いテキストを分割
        text_chunks = split_text_for_tts(text, max_length=200)

        audio_segments = []
        sample_rate = model.generation_config.sample_rate

        print(f"Generating speech with Bark: {len(text_chunks)} chunks")

        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")

            # 入力を準備（Barkプロセッサは voice_preset を受け付ける）
            inputs = processor(chunk, voice_preset=voice_preset)

            # input_idsをデバイスに移動
            input_ids = inputs["input_ids"].to(DEVICE)

            # 音声生成
            with torch.no_grad():
                audio_array = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                )

            # numpy配列に変換
            audio_numpy = audio_array.cpu().numpy().squeeze()
            audio_segments.append(audio_numpy)

            # メモリ解放
            del audio_array, input_ids
            gc.collect()
            audio_pipeline_manager.clear_cache()

        # 音声を連結
        if len(audio_segments) > 1:
            # チャンク間に少し間を空ける（200ms）
            final_audio = concatenate_audio(audio_segments, sample_rate, gap_ms=200)
        else:
            final_audio = audio_segments[0]

        # 正規化とフェード処理
        final_audio = normalize_audio(final_audio)
        final_audio = add_fade(final_audio, sample_rate, fade_in_ms=10, fade_out_ms=50)

        # ファイル保存
        output_filename = f"bark_{voice_preset}_{len(text)}_chars.wav"
        output_path = os.path.join(output_dir, output_filename)
        save_audio(final_audio, sample_rate, output_path, normalize=False)

        # メモリ解放
        audio_pipeline_manager.clear_cache()
        gc.collect()

        duration = len(final_audio) / sample_rate
        voice_name = BARK_VOICE_PRESETS.get(voice_preset, voice_preset)
        return output_path, f"音声生成完了: {duration:.1f}秒\nボイス: {voice_name}\n保存先: {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Bark音声生成エラー: {str(e)}"


def generate_speech_xtts(
    text: str,
    language: str = "ja",
    reference_audio: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.85,
    repetition_penalty: float = 2.0,
) -> Tuple[Optional[str], str]:
    """XTTS v2モデルで音声を生成（ボイスクローン対応）

    Args:
        text: 生成するテキスト
        language: 言語コード（ja, en, zh等）
        reference_audio: 参照音声ファイルパス（ボイスクローン用）
        temperature: 温度パラメータ
        top_k: top-kサンプリング
        top_p: top-pサンプリング
        repetition_penalty: 繰り返しペナルティ

    Returns:
        (output_path, status_message) タプル
    """
    if not text.strip():
        return None, "テキストを入力してください"

    try:
        # XTTS v2モデルをロード
        tts = audio_pipeline_manager.load_xtts()

        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("tts_xtts", text[:30])

        # 出力ファイルパス
        clone_suffix = "_clone" if reference_audio else ""
        output_filename = f"xtts_{language}{clone_suffix}_{len(text)}_chars.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Generating speech with XTTS v2: language={language}")

        if reference_audio and os.path.exists(reference_audio):
            # ボイスクローンモード
            print(f"Using reference audio: {reference_audio}")
            tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            # デフォルトスピーカーモード
            tts.tts_to_file(
                text=text,
                language=language,
                file_path=output_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # メモリ解放
        audio_pipeline_manager.clear_cache()
        gc.collect()

        # 生成された音声の長さを取得
        from utils.audio import get_audio_duration
        duration = get_audio_duration(output_path)

        lang_name = XTTS_LANGUAGES.get(language, language)
        clone_info = "\nボイスクローン: 有効" if reference_audio else ""
        return output_path, f"音声生成完了: {duration:.1f}秒\n言語: {lang_name}{clone_info}\n保存先: {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"XTTS音声生成エラー: {str(e)}"


def generate_speech(
    text: str,
    model: str = "bark",
    voice_preset: str = "ja_speaker_0",
    language: str = "ja",
    reference_audio: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Tuple[Optional[str], str]:
    """統合TTS生成関数

    Args:
        text: 生成するテキスト
        model: 使用するモデル（bark, xtts_v2）
        voice_preset: Bark用ボイスプリセット
        language: 言語コード
        reference_audio: XTTS用参照音声
        temperature: 温度パラメータ
        **kwargs: モデル固有のパラメータ

    Returns:
        (output_path, status_message) タプル
    """
    if model == "bark":
        semantic_temp = kwargs.get('semantic_temperature', temperature)
        return generate_speech_bark(
            text=text,
            voice_preset=voice_preset,
            temperature=temperature,
            semantic_temperature=semantic_temp,
        )
    elif model == "xtts_v2":
        return generate_speech_xtts(
            text=text,
            language=language,
            reference_audio=reference_audio,
            temperature=temperature,
            top_k=kwargs.get('top_k', 50),
            top_p=kwargs.get('top_p', 0.85),
            repetition_penalty=kwargs.get('repetition_penalty', 2.0),
        )
    else:
        return None, f"未対応のTTSモデル: {model}"


def get_bark_voice_presets_for_language(language: str) -> dict:
    """指定した言語のBarkボイスプリセットを取得

    Args:
        language: 言語コード（en, ja, zh, ko等）

    Returns:
        プリセット名をキー、表示名を値とする辞書
    """
    presets = {}
    prefix = f"{language}_speaker_"

    for preset_id, display_name in BARK_VOICE_PRESETS.items():
        if preset_id.startswith(prefix):
            presets[preset_id] = display_name

    return presets if presets else {"en_speaker_0": "英語 男性1"}  # フォールバック
