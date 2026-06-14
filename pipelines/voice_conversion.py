"""Voice Conversion (RVC) モジュール"""
import os
import gc
from typing import Optional, Tuple
from pathlib import Path

import torch

# PyTorch 2.6+ 互換性: fairseqのhubert_base.ptがpickleを使用しているため
# torch.loadのデフォルト動作を変更
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """PyTorch 2.6+でweights_only=Falseをデフォルトに"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from config import (
    DEVICE, HUBERT_BASE_MODEL,
    RVC_INDEX_RATE, RVC_FILTER_RADIUS, RVC_RESAMPLE_SR, RVC_RMS_MIX_RATE,
    RVC_PROTECT, RVC_METHODS, get_available_rvc_models
)
from utils.audio import (
    create_audio_output_dir, normalize_audio, save_audio,
    load_audio, add_fade, get_audio_duration
)
from .audio_manager import audio_pipeline_manager


def get_rvc_models() -> dict:
    """利用可能なRVCモデルを取得

    Returns:
        モデル名をキー、パスを値とする辞書
    """
    return get_available_rvc_models()


def get_rvc_index_file(model_path: str) -> Optional[str]:
    """RVCモデルに対応するインデックスファイルを探す

    Args:
        model_path: RVCモデルファイルパス

    Returns:
        インデックスファイルパス、または None
    """
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # インデックスファイルの候補
    index_patterns = [
        f"{model_name}.index",
        f"{model_name}_v2.index",
        f"added_{model_name}.index",
        "*.index",  # 同じディレクトリ内の任意のindexファイル
    ]

    for pattern in index_patterns:
        if "*" in pattern:
            # ワイルドカード検索
            for file in Path(model_dir).glob(pattern):
                return str(file)
        else:
            index_path = os.path.join(model_dir, pattern)
            if os.path.exists(index_path):
                return index_path

    return None


def check_hubert_model() -> Tuple[bool, str]:
    """HuBERTベースモデルの存在を確認

    Returns:
        (存在するか, メッセージ) タプル
    """
    if os.path.exists(HUBERT_BASE_MODEL):
        return True, f"HuBERT model found: {HUBERT_BASE_MODEL}"
    return False, f"HuBERT model not found at: {HUBERT_BASE_MODEL}\nPlease download hubert_base.pt"


def convert_voice_rvc(
    input_audio: str,
    model_name: str,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = RVC_INDEX_RATE,
    filter_radius: int = RVC_FILTER_RADIUS,
    resample_sr: int = RVC_RESAMPLE_SR,
    rms_mix_rate: float = RVC_RMS_MIX_RATE,
    protect: float = RVC_PROTECT,
) -> Tuple[Optional[str], str]:
    """RVCで音声変換を実行

    Args:
        input_audio: 入力音声ファイルパス
        model_name: 使用するRVCモデル名
        pitch_shift: ピッチシフト量（半音単位、-12〜+12）
        f0_method: F0推定方式（rmvpe, harvest, crepe, pm）
        index_rate: インデックス率（0-1）
        filter_radius: フィルタ半径
        resample_sr: リサンプルレート（0=自動）
        rms_mix_rate: RMSミックス率
        protect: プロテクト値

    Returns:
        (output_path, status_message) タプル
    """
    if not input_audio or not os.path.exists(input_audio):
        return None, "入力音声ファイルを選択してください"

    # モデルパスを取得
    rvc_models = get_rvc_models()
    if model_name not in rvc_models:
        return None, f"RVCモデルが見つかりません: {model_name}\nモデルを models/audio/voice_conversion/rvc/ に配置してください"

    model_path = rvc_models[model_name]

    try:
        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("voice_conv", model_name)

        # インデックスファイルを探す
        index_path = get_rvc_index_file(model_path)
        if index_path:
            print(f"Found index file: {index_path}")
        else:
            print("No index file found, proceeding without index")

        print(f"Converting voice with RVC: {model_name}")
        print(f"Pitch shift: {pitch_shift}, F0 method: {f0_method}")

        # RVCライブラリの動的インポート
        try:
            from rvc_python.infer import RVCInference

            # RVCモデルをロード
            rvc = RVCInference(device=DEVICE)
            rvc.load_model(model_path, version="v2", index_path=index_path or "")

            # パラメータを設定
            rvc.set_params(
                f0up_key=pitch_shift,
                f0method=f0_method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

            # 音声変換
            output_filename = f"rvc_{model_name}_pitch{pitch_shift:+d}.wav"
            output_path = os.path.join(output_dir, output_filename)

            rvc.infer_file(
                input_path=input_audio,
                output_path=output_path,
            )

            # メモリ解放
            del rvc
            gc.collect()
            audio_pipeline_manager.clear_cache()

            # 生成された音声の長さを取得
            duration = get_audio_duration(output_path)
            method_name = RVC_METHODS.get(f0_method, f0_method)

            return output_path, f"音声変換完了: {duration:.1f}秒\nモデル: {model_name}\nピッチ: {pitch_shift:+d}半音\nF0方式: {method_name}\n保存先: {output_path}"

        except ImportError:
            # rvc-pythonがインストールされていない場合のフォールバック
            return None, "RVCライブラリがインストールされていません。\npip install rvc-python を実行してください。"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"RVC音声変換エラー: {str(e)}"


def convert_voice_simple_pitch(
    input_audio: str,
    pitch_shift: int = 0,
) -> Tuple[Optional[str], str]:
    """シンプルなピッチシフト（RVCなし）

    Args:
        input_audio: 入力音声ファイルパス
        pitch_shift: ピッチシフト量（半音単位）

    Returns:
        (output_path, status_message) タプル
    """
    if not input_audio or not os.path.exists(input_audio):
        return None, "入力音声ファイルを選択してください"

    try:
        # 出力ディレクトリを作成
        output_dir = create_audio_output_dir("pitch_shift", f"shift{pitch_shift:+d}")

        # 音声を読み込み
        audio, sr = load_audio(input_audio, mono=True)

        print(f"Pitch shifting: {pitch_shift} semitones")

        try:
            import librosa

            # ピッチシフト
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

            # 正規化
            shifted = normalize_audio(shifted)
            shifted = add_fade(shifted, sr, fade_in_ms=10, fade_out_ms=50)

            # 保存
            output_filename = f"pitch_shift_{pitch_shift:+d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            save_audio(shifted, sr, output_path, normalize=False)

            duration = len(shifted) / sr
            return output_path, f"ピッチシフト完了: {duration:.1f}秒\nシフト量: {pitch_shift:+d}半音\n保存先: {output_path}"

        except ImportError:
            return None, "librosaがインストールされていません。\npip install librosa を実行してください。"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"ピッチシフトエラー: {str(e)}"


def convert_voice(
    input_audio: str,
    model_name: Optional[str] = None,
    pitch_shift: int = 0,
    use_rvc: bool = True,
    **kwargs
) -> Tuple[Optional[str], str]:
    """統合音声変換関数

    Args:
        input_audio: 入力音声ファイルパス
        model_name: RVCモデル名（use_rvc=Trueの場合必須）
        pitch_shift: ピッチシフト量
        use_rvc: RVCを使用するかどうか
        **kwargs: RVC固有のパラメータ

    Returns:
        (output_path, status_message) タプル
    """
    if use_rvc and model_name:
        return convert_voice_rvc(
            input_audio=input_audio,
            model_name=model_name,
            pitch_shift=pitch_shift,
            f0_method=kwargs.get('f0_method', 'rmvpe'),
            index_rate=kwargs.get('index_rate', RVC_INDEX_RATE),
            filter_radius=kwargs.get('filter_radius', RVC_FILTER_RADIUS),
            resample_sr=kwargs.get('resample_sr', RVC_RESAMPLE_SR),
            rms_mix_rate=kwargs.get('rms_mix_rate', RVC_RMS_MIX_RATE),
            protect=kwargs.get('protect', RVC_PROTECT),
        )
    else:
        return convert_voice_simple_pitch(
            input_audio=input_audio,
            pitch_shift=pitch_shift,
        )


def check_rvc_installation() -> dict:
    """RVC関連のインストール状況を確認

    Returns:
        インストール状況の辞書
    """
    status = {
        "rvc_python": False,
        "librosa": False,
        "fairseq": False,
        "hubert_model": False,
        "models_found": 0,
    }

    try:
        import rvc_python
        status["rvc_python"] = True
    except ImportError:
        pass

    try:
        import librosa
        status["librosa"] = True
    except ImportError:
        pass

    try:
        import fairseq
        status["fairseq"] = True
    except ImportError:
        pass

    # HuBERTモデルの確認
    hubert_exists, _ = check_hubert_model()
    status["hubert_model"] = hubert_exists

    status["models_found"] = len(get_rvc_models())

    return status
