"""背景除去モジュール

依存パッケージ: pip install rembg[gpu] (GPU版) または pip install rembg (CPU版)
"""
import os
import gc
import traceback
import torch
import numpy as np
from PIL import Image

from config import DEVICE, REMBG_MODELS
from utils.file import create_output_dir


class BackgroundRemover:
    """背景除去クラス"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.session = None
        self.current_model = None
        self.device = DEVICE

    def load_model(self, model_name: str = "u2net"):
        """背景除去モデルをロード"""
        if self.session is not None and self.current_model == model_name:
            return self.session

        print(f"Loading background removal model: {model_name}...")

        try:
            from rembg import new_session

            self.session = new_session(model_name)
            self.current_model = model_name
            print(f"Model {model_name} loaded successfully")
            return self.session

        except ImportError as e:
            print(f"rembg not found: {e}")
            print("Please install: pip install rembg[gpu] (GPU) or pip install rembg (CPU)")
            raise

    def remove_background(
        self,
        image: Image.Image,
        model_name: str = "u2net",
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        only_mask: bool = False,
        post_process_mask: bool = False,
        bgcolor: tuple = None
    ) -> Image.Image:
        """背景を除去"""
        try:
            from rembg import remove

            session = self.load_model(model_name)

            # 背景除去実行
            result = remove(
                image,
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                only_mask=only_mask,
                post_process_mask=post_process_mask,
                bgcolor=bgcolor
            )

            return result

        except Exception as e:
            print(f"Background removal failed: {e}")
            raise

    def remove_background_with_color(
        self,
        image: Image.Image,
        model_name: str = "u2net",
        bg_color: str = "transparent",
        alpha_matting: bool = False
    ) -> Image.Image:
        """背景を指定色に置換"""
        # 背景色を設定
        if bg_color == "transparent":
            bgcolor = None
        elif bg_color == "white":
            bgcolor = (255, 255, 255, 255)
        elif bg_color == "black":
            bgcolor = (0, 0, 0, 255)
        elif bg_color == "green":
            bgcolor = (0, 255, 0, 255)
        elif bg_color == "blue":
            bgcolor = (0, 0, 255, 255)
        else:
            bgcolor = None

        result = self.remove_background(
            image,
            model_name=model_name,
            alpha_matting=alpha_matting,
            bgcolor=bgcolor
        )

        # 透明背景でない場合、RGBに変換
        if bgcolor is not None:
            result = result.convert("RGB")

        return result

    def get_mask(
        self,
        image: Image.Image,
        model_name: str = "u2net"
    ) -> Image.Image:
        """マスク画像のみを取得"""
        return self.remove_background(
            image,
            model_name=model_name,
            only_mask=True
        )

    def clear_cache(self):
        """メモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


# グローバルインスタンス
background_remover = BackgroundRemover()


# 背景色オプション
BG_COLOR_OPTIONS = {
    "透明": "transparent",
    "白": "white",
    "黒": "black",
    "緑（クロマキー）": "green",
    "青（クロマキー）": "blue",
}


def remove_background_image(
    input_image,
    model_name: str,
    bg_color: str,
    alpha_matting: bool,
    output_mask: bool
) -> tuple[list[Image.Image], str]:
    """画像の背景を除去"""
    try:
        if input_image is None:
            return None, "入力画像をアップロードしてください"

        # 入力画像をPIL Imageに変換
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")

        print(f"Removing background with model: {model_name}")

        # モデル名を内部値に変換
        model_key = model_name
        for key, desc in REMBG_MODELS.items():
            if desc == model_name:
                model_key = key
                break

        # 背景色を内部値に変換
        bg_color_value = bg_color
        for label, value in BG_COLOR_OPTIONS.items():
            if label == bg_color:
                bg_color_value = value
                break

        results = []

        # 背景除去
        result_image = background_remover.remove_background_with_color(
            image,
            model_name=model_key,
            bg_color=bg_color_value,
            alpha_matting=alpha_matting
        )
        results.append(result_image)

        # マスクも出力する場合
        if output_mask:
            mask_image = background_remover.get_mask(image, model_name=model_key)
            results.append(mask_image)

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "bg_remove", model_key)

        # 画像を保存
        ext = "png"
        filename = f"bg_removed_{model_key}.{ext}"
        filepath = os.path.join(output_dir, filename)
        result_image.save(filepath)

        if output_mask:
            mask_filepath = os.path.join(output_dir, f"mask_{model_key}.png")
            mask_image.save(mask_filepath)

        background_remover.clear_cache()

        mask_info = "\nマスク画像も出力しました" if output_mask else ""
        return results, f"背景除去完了\nモデル: {model_name}\n背景: {bg_color}{mask_info}\n保存先: {output_dir}"

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
