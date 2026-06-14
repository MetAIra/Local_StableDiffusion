"""Multi-view生成モジュール (Zero123++ / MVDream)

Zero123++: 1枚の画像から6視点の多方向画像を生成
MVDream: テキストプロンプトから4視点の多方向画像を生成

依存パッケージ: pip install diffusers transformers
"""
import os
import gc
import traceback
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    DEVICE,
    ZERO123_MODEL,
    ZERO123_PIPELINE,
    ZERO123_DEFAULT_STEPS,
    ZERO123_DEFAULT_CFG,
    ZERO123_MIN_INPUT_SIZE,
    ZERO123_OUTPUT_SIZE
)
from utils.file import create_output_dir


# 視点角度の定義（Zero123++ v1.1の仕様）
# 上段(0-2): 仰角30°（上から見下ろす）
# 下段(3-5): 仰角-20°（下から見上げる）
VIEW_ANGLES = [
    {"index": 1, "elevation": 30, "azimuth": 30, "label": "Front-Right (elev:30°, az:30°)"},
    {"index": 2, "elevation": 30, "azimuth": 90, "label": "Right (elev:30°, az:90°)"},
    {"index": 3, "elevation": 30, "azimuth": 150, "label": "Back-Right (elev:30°, az:150°)"},
    {"index": 4, "elevation": -20, "azimuth": 210, "label": "Back-Left (elev:-20°, az:210°)"},
    {"index": 5, "elevation": -20, "azimuth": 270, "label": "Left (elev:-20°, az:270°)"},
    {"index": 6, "elevation": -20, "azimuth": 330, "label": "Front-Left (elev:-20°, az:330°)"},
]


class Zero123PipelineManager:
    """Zero123++パイプラインを管理するシングルトンクラス"""

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

        self.pipeline = None
        self.device = DEVICE

    def load_pipeline(self):
        """Zero123++パイプラインをロード"""
        if self.pipeline is not None:
            return self.pipeline

        print("Loading Zero123++ pipeline...")
        print(f"Model: {ZERO123_MODEL}")
        print(f"Device: {self.device}")

        try:
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

            # Custom pipeline from sudo-ai
            self.pipeline = DiffusionPipeline.from_pretrained(
                ZERO123_MODEL,
                custom_pipeline=ZERO123_PIPELINE,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Set recommended scheduler
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config,
                timestep_spacing='trailing'
            )

            self.pipeline.to(self.device)

            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()

            print("Zero123++ pipeline loaded successfully")
            return self.pipeline

        except Exception as e:
            print(f"Failed to load Zero123++ pipeline: {e}")
            raise

    def generate_views(
        self,
        input_image: Image.Image,
        num_inference_steps: int = ZERO123_DEFAULT_STEPS,
        guidance_scale: float = ZERO123_DEFAULT_CFG,
        seed: int = 42
    ) -> list[Image.Image]:
        """6視点の画像を生成

        Args:
            input_image: 入力画像（PIL Image）
            num_inference_steps: 推論ステップ数
            guidance_scale: CFG Scale
            seed: シード値

        Returns:
            6視点の画像リスト
        """
        pipeline = self.load_pipeline()

        # 入力画像のサイズチェック・リサイズ
        w, h = input_image.size
        if w < ZERO123_MIN_INPUT_SIZE or h < ZERO123_MIN_INPUT_SIZE:
            scale = max(ZERO123_MIN_INPUT_SIZE / w, ZERO123_MIN_INPUT_SIZE / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"Resized input image from {w}x{h} to {new_w}x{new_h}")

        generator = torch.Generator(self.device).manual_seed(seed)

        # 6視点を一度に生成（768x768のグリッド出力）
        result = pipeline(
            input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # 768x768の出力を6つの256x256画像に分割（2行×3列）
        views = []
        view_size = ZERO123_OUTPUT_SIZE
        for row in range(2):
            for col in range(3):
                x = col * view_size
                y = row * view_size
                view = result.crop((x, y, x + view_size, y + view_size))
                views.append(view)

        return views

    def clear_cache(self):
        """メモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


# グローバルインスタンス
zero123_manager = Zero123PipelineManager()


def create_labeled_grid(views: list[Image.Image], input_image: Image.Image = None) -> Image.Image:
    """5視点（Right方向3視点 + 正面 + 背面）をラベル付きグリッド画像にまとめる

    Args:
        views: 6視点の画像リスト
        input_image: 入力画像（正面として使用）

    Returns:
        ラベル付きグリッド画像（十字配置）
    """
    view_size = ZERO123_OUTPUT_SIZE  # 256
    label_height = 30  # ラベルの高さ
    padding = 10  # 画像間のパディング

    # 3x3グリッド（十字配置）のサイズを計算
    cell_width = view_size
    cell_height = view_size + label_height
    grid_width = cell_width * 3 + padding * 4
    grid_height = cell_height * 3 + padding * 4

    # 白背景のグリッド画像を作成
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # フォントを取得（システムフォントにフォールバック）
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

    # 十字配置レイアウト（Right方向のみ使用）:
    #        ---        |  Back(150°)   |     ---
    #   Front-Right(30°)|  Front(Input) |  Right(90°)
    #        ---        |     ---       |     ---
    #
    # views[0]=Front-Right(30°), views[1]=Right(90°), views[2]=Back-Right(150°)

    grid_positions = [
        # (row, col, view_index, label)
        (0, 1, 2, "Back\n(150°)"),           # 上: Back-Rightを背面として使用
        (1, 0, 0, "Front-Right\n(30°)"),     # 左: Front-Right
        (1, 2, 1, "Right\n(90°)"),           # 右: Right
    ]

    # 各視点を配置
    for row, col, view_idx, label in grid_positions:
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)

        # 画像を配置
        grid.paste(views[view_idx], (x, y))

        # ラベルを描画
        label_y = y + view_size + 2
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x + (cell_width - text_width) // 2
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    # 中央に入力画像を正面として配置
    center_x = padding + 1 * (cell_width + padding)
    center_y = padding + 1 * (cell_height + padding)

    if input_image is not None:
        # 入力画像をview_sizeにリサイズ
        input_resized = input_image.copy()
        input_resized.thumbnail((view_size, view_size), Image.LANCZOS)
        # 中央に配置
        paste_x = center_x + (view_size - input_resized.width) // 2
        paste_y = center_y + (view_size - input_resized.height) // 2

        # RGBA画像の場合は白背景と合成
        if input_resized.mode == 'RGBA':
            bg = Image.new('RGB', input_resized.size, (255, 255, 255))
            bg.paste(input_resized, mask=input_resized.split()[3])
            grid.paste(bg, (paste_x, paste_y))
        else:
            grid.paste(input_resized, (paste_x, paste_y))

    # 正面ラベル
    label = "Front\n(Input)"
    label_y = center_y + view_size + 2
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    label_x = center_x + (cell_width - text_width) // 2
    draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    return grid


def generate_multiview(
    input_image,
    num_inference_steps: int = ZERO123_DEFAULT_STEPS,
    guidance_scale: float = ZERO123_DEFAULT_CFG,
    seed: int = 42
) -> tuple[list[Image.Image], str]:
    """Multi-view画像を生成

    Args:
        input_image: 入力画像（numpy array, PIL Image, or file path）
        num_inference_steps: 推論ステップ数
        guidance_scale: CFG Scale
        seed: シード値

    Returns:
        (画像リスト, ステータスメッセージ)
    """
    try:
        if input_image is None:
            return None, "入力画像をアップロードしてください"

        # 入力をPIL Imageに変換
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")

        print(f"Generating multi-view images from: {image.size}")
        print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")

        # 6視点を生成
        views = zero123_manager.generate_views(
            image,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed)
        )

        # 出力ディレクトリを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "multiview", f"seed{seed}")

        # 各視点を保存
        for i, view in enumerate(views):
            angle_info = VIEW_ANGLES[i]
            filename = f"view_{i+1:02d}_az{angle_info['azimuth']:03d}.png"
            filepath = os.path.join(output_dir, filename)
            view.save(filepath)
            print(f"Saved: {filename}")

        # ラベル付きグリッド画像を作成・保存
        grid_image = create_labeled_grid(views, image)
        grid_filename = "multiview_grid.png"
        grid_filepath = os.path.join(output_dir, grid_filename)
        grid_image.save(grid_filepath)
        print(f"Saved: {grid_filename}")

        zero123_manager.clear_cache()

        # ステータスメッセージを作成
        status_msg = (
            f"6視点画像を生成しました\n"
            f"Seed: {seed}\n"
            f"ステップ: {num_inference_steps}\n"
            f"CFG: {guidance_scale}\n\n"
            f"グリッド画像: {grid_filename}\n"
            f"保存先: {output_dir}"
        )

        # グリッド画像を最初に表示
        return [grid_image] + views, status_msg

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
