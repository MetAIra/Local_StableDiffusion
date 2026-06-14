"""画像処理ユーティリティ"""
import numpy as np
from PIL import Image, ImageFilter


def round_to_multiple(value: int, multiple: int = 64) -> int:
    """値を指定の倍数に丸める（Stable Diffusionの制約対応）"""
    return int((value + multiple // 2) // multiple * multiple)


def to_pil_rgb(image) -> Image.Image:
    """ファイルパス / numpy array / PIL Image を RGB の PIL Image に変換"""
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def create_outpaint_image_and_mask(
    image: Image.Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
    feather_size: int = 32
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """元画像を拡張し、マスク画像を作成（フェザリング対応）"""

    orig_width, orig_height = image.size

    # 拡張サイズを64の倍数に丸める
    left = round_to_multiple(left)
    right = round_to_multiple(right)
    top = round_to_multiple(top)
    bottom = round_to_multiple(bottom)

    # 新しいキャンバスサイズを計算し、64の倍数に調整
    new_width = round_to_multiple(orig_width + left + right)
    new_height = round_to_multiple(orig_height + top + bottom)

    # 元画像をnumpy配列に変換
    img_array = np.array(image)

    # 拡張画像を作成（端のピクセルを引き伸ばして初期化）
    expanded_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 元画像を配置
    expanded_array[top:top+orig_height, left:left+orig_width] = img_array

    # 上部を引き伸ばし
    if top > 0:
        for i in range(top):
            expanded_array[i, left:left+orig_width] = img_array[0, :]

    # 下部を引き伸ばし
    if bottom > 0:
        for i in range(top + orig_height, new_height):
            expanded_array[i, left:left+orig_width] = img_array[-1, :]

    # 左部を引き伸ばし
    if left > 0:
        for i in range(left):
            expanded_array[top:top+orig_height, i] = img_array[:, 0]

    # 右部を引き伸ばし
    if right > 0:
        for i in range(left + orig_width, new_width):
            expanded_array[top:top+orig_height, i] = img_array[:, -1]

    # 角を埋める（左上）
    if top > 0 and left > 0:
        expanded_array[0:top, 0:left] = img_array[0, 0]

    # 角を埋める（右上）
    if top > 0 and right > 0:
        expanded_array[0:top, left+orig_width:new_width] = img_array[0, -1]

    # 角を埋める（左下）
    if bottom > 0 and left > 0:
        expanded_array[top+orig_height:new_height, 0:left] = img_array[-1, 0]

    # 角を埋める（右下）
    if bottom > 0 and right > 0:
        expanded_array[top+orig_height:new_height, left+orig_width:new_width] = img_array[-1, -1]

    expanded_image = Image.fromarray(expanded_array)

    # ぼかしを適用して自然な遷移を作る
    expanded_image = expanded_image.filter(ImageFilter.GaussianBlur(radius=5))
    # 元画像部分を再度貼り付け
    expanded_image.paste(image, (left, top))

    # マスク画像を作成（白=生成する部分、黒=保持する部分）
    mask = Image.new("L", (new_width, new_height), 255)  # 全体を白で初期化

    # 元画像の部分を黒にする（フェザリング用に少し小さくする）
    inner_margin = min(feather_size // 2, 8)  # 境界部分のマージン
    mask_inner = Image.new("L", (orig_width - inner_margin * 2, orig_height - inner_margin * 2), 0)
    mask.paste(mask_inner, (left + inner_margin, top + inner_margin))

    # マスクをぼかしてフェザリング効果を追加
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_size))

    return expanded_image, mask, (new_width, new_height)
