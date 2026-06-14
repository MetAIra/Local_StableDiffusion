"""音声処理ユーティリティ"""
import os
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

import config


def create_audio_output_dir(prefix: str, prompt: str = "") -> str:
    """音声出力ディレクトリを作成して返す

    Args:
        prefix: ディレクトリ名プレフィックス（tts, music, voice_convなど）
        prompt: 生成プロンプト（オプション）

    Returns:
        作成した出力ディレクトリのパス
    """
    from utils.file import sanitize_filename

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = sanitize_filename(prompt) if prompt else ""

    if safe_prompt:
        dir_name = f"output_{prefix}_{timestamp}_{safe_prompt}"
    else:
        dir_name = f"output_{prefix}_{timestamp}"

    # ベースパスを絶対パスに変換
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_output_base = os.path.join(base_path, config.AUDIO_OUTPUT_DIR)

    output_dir = os.path.join(audio_output_base, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """音声を指定されたdBレベルに正規化

    Args:
        audio: 音声データ（numpy配列）
        target_db: ターゲットdBレベル（デフォルト: -3dB）

    Returns:
        正規化された音声データ
    """
    if len(audio) == 0:
        return audio

    # RMS計算
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    # ターゲットRMS計算
    target_rms = 10 ** (target_db / 20)

    # 正規化
    normalized = audio * (target_rms / rms)

    # クリッピング防止
    max_val = np.abs(normalized).max()
    if max_val > 1.0:
        normalized = normalized / max_val

    return normalized


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """音声をリサンプリング

    Args:
        audio: 音声データ（numpy配列）
        orig_sr: 元のサンプルレート
        target_sr: ターゲットサンプルレート

    Returns:
        リサンプリングされた音声データ
    """
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled
    except ImportError:
        # librosaがない場合は簡易リサンプリング
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length).astype(int)
        return audio[indices]


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """ステレオ音声をモノラルに変換

    Args:
        audio: 音声データ（numpy配列、shape: (samples,) or (channels, samples)）

    Returns:
        モノラル音声データ
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        if audio.shape[0] == 2:  # (channels, samples)
            return np.mean(audio, axis=0)
        elif audio.shape[1] == 2:  # (samples, channels)
            return np.mean(audio, axis=1)
    return audio


def add_fade(audio: np.ndarray, sample_rate: int,
             fade_in_ms: float = 10.0, fade_out_ms: float = 10.0) -> np.ndarray:
    """音声にフェードイン・フェードアウトを追加

    Args:
        audio: 音声データ（numpy配列）
        sample_rate: サンプルレート
        fade_in_ms: フェードイン長（ミリ秒）
        fade_out_ms: フェードアウト長（ミリ秒）

    Returns:
        フェード処理された音声データ
    """
    fade_in_samples = int(sample_rate * fade_in_ms / 1000)
    fade_out_samples = int(sample_rate * fade_out_ms / 1000)

    audio = audio.copy()

    # フェードイン
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in_curve

    # フェードアウト
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out_curve

    return audio


def save_audio(audio: np.ndarray, sample_rate: int, output_path: str,
               format: str = "wav", normalize: bool = True) -> str:
    """音声をファイルに保存

    Args:
        audio: 音声データ（numpy配列）
        sample_rate: サンプルレート
        output_path: 出力ファイルパス
        format: 出力フォーマット（wav, mp3, flac, ogg）
        normalize: 正規化するかどうか

    Returns:
        保存したファイルパス
    """
    # 1次元配列に変換（squeeze）
    audio = np.squeeze(audio)

    if normalize:
        audio = normalize_audio(audio)

    # float32に変換
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 拡張子を確認・修正
    base, ext = os.path.splitext(output_path)
    if ext.lower() != f".{format}":
        output_path = f"{base}.{format}"

    # 出力ディレクトリを確認
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存（soundfile優先、なければscipy使用）
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
    except ImportError:
        # soundfileがない場合はscipy.io.wavfileを使用
        from scipy.io import wavfile
        # scipy.io.wavfileはint16を期待するため変換
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)

    return output_path


def load_audio(file_path: str, target_sr: Optional[int] = None,
               mono: bool = True) -> Tuple[np.ndarray, int]:
    """音声ファイルを読み込み

    Args:
        file_path: 音声ファイルパス
        target_sr: ターゲットサンプルレート（Noneの場合は元のまま）
        mono: モノラルに変換するかどうか

    Returns:
        (音声データ, サンプルレート)
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(file_path)
        if audio.ndim > 1 and mono:
            audio = convert_to_mono(audio.T)
        if target_sr and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        return audio, sr


def get_audio_duration(file_path: str) -> float:
    """音声ファイルの長さを取得（秒）

    Args:
        file_path: 音声ファイルパス

    Returns:
        音声の長さ（秒）
    """
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except ImportError:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration


def concatenate_audio(audio_list: list, sample_rate: int,
                      gap_ms: float = 0.0) -> np.ndarray:
    """複数の音声を連結

    Args:
        audio_list: 音声データのリスト
        sample_rate: サンプルレート
        gap_ms: 音声間のギャップ（ミリ秒）

    Returns:
        連結された音声データ
    """
    if not audio_list:
        return np.array([])

    gap_samples = int(sample_rate * gap_ms / 1000)
    gap = np.zeros(gap_samples)

    result = []
    for i, audio in enumerate(audio_list):
        result.append(audio)
        if i < len(audio_list) - 1 and gap_samples > 0:
            result.append(gap)

    return np.concatenate(result)


def split_text_for_tts(text: str, max_length: int = 200) -> list:
    """長いテキストをTTS用に分割

    句読点や改行で分割し、各チャンクがmax_length以下になるようにする

    Args:
        text: 入力テキスト
        max_length: 最大文字数

    Returns:
        分割されたテキストのリスト
    """
    # 句読点や改行で分割
    import re
    delimiters = r'[。．.！!？?\n]'
    sentences = re.split(delimiters, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_length:
            # 文が長すぎる場合は読点で分割
            sub_sentences = re.split(r'[、，,]', sentence)
            for sub in sub_sentences:
                if len(current_chunk) + len(sub) < max_length:
                    current_chunk += sub + "、"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.rstrip("、"))
                    current_chunk = sub + "、"
        elif len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "。"
        else:
            if current_chunk:
                chunks.append(current_chunk.rstrip("。"))
            current_chunk = sentence + "。"

    if current_chunk:
        chunks.append(current_chunk.rstrip("。、"))

    return chunks if chunks else [text]
