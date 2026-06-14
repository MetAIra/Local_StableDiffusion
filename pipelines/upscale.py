"""Upscaler モジュール（Real-ESRGAN + SD Upscaler）"""
import os
import gc
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config import DEVICE, PROMPT_PREFIX, REALESRGAN_MODELS, SD_UPSCALER_MODEL
from utils.file import create_output_dir


# ========================================
# RRDBNet アーキテクチャ（basicsrに依存しない実装）
# ========================================

def make_layer(block, n_layers, **kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ========================================
# Upscaler クラス
# ========================================

class Upscaler:
    """Real-ESRGAN + SD Upscalerを管理するクラス"""

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

        self.realesrgan_model = None
        self.realesrgan_model_name = None
        self.sd_upscaler = None
        self.device = DEVICE

    def load_realesrgan(self, model_name: str = "RealESRGAN_x4plus_anime_6B"):
        """Real-ESRGANモデルをロード"""
        if self.realesrgan_model is not None and self.realesrgan_model_name == model_name:
            return self.realesrgan_model

        print(f"Loading Real-ESRGAN model: {model_name}")

        # モデルアーキテクチャを設定
        if model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4
            )
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        else:  # RealESRGAN_x4plus
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

        # モデルをダウンロード・ロード
        model_path = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location='cpu')

        # state_dictのキー調整
        if 'params_ema' in model_path:
            keyname = 'params_ema'
        elif 'params' in model_path:
            keyname = 'params'
        else:
            keyname = None

        if keyname:
            model.load_state_dict(model_path[keyname], strict=True)
        else:
            model.load_state_dict(model_path, strict=True)

        model.eval()
        model = model.to(self.device)

        if self.device == "cuda":
            model = model.half()

        self.realesrgan_model = model
        self.realesrgan_model_name = model_name
        print(f"Real-ESRGAN model loaded successfully: {model_name}")

        return self.realesrgan_model

    def load_sd_upscaler(self):
        """SD Upscalerパイプラインをロード"""
        if self.sd_upscaler is not None:
            return self.sd_upscaler

        print(f"Loading SD Upscaler: {SD_UPSCALER_MODEL}")

        try:
            from diffusers import StableDiffusionUpscalePipeline

            self.sd_upscaler = StableDiffusionUpscalePipeline.from_pretrained(
                SD_UPSCALER_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.sd_upscaler.to(self.device)

            if self.device == "cuda":
                self.sd_upscaler.enable_attention_slicing()

            print("SD Upscaler loaded successfully")

        except Exception as e:
            print(f"Error loading SD Upscaler: {e}")
            raise

        return self.sd_upscaler

    def upscale_with_realesrgan(self, image: Image.Image, model_name: str, tile_size: int = 512) -> Image.Image:
        """Real-ESRGANでアップスケール（タイル処理対応）"""
        model = self.load_realesrgan(model_name)

        w, h = image.size
        scale = 4

        # 小さい画像は直接処理
        if w <= tile_size and h <= tile_size:
            return self._esrgan_inference(model, image)

        # 大きい画像はタイル処理
        print(f"Using tile processing: {w}x{h} with tile_size={tile_size}")
        overlap = 32
        result = Image.new("RGB", (w * scale, h * scale))

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # タイル領域を計算
                x1 = x
                y1 = y
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)

                tile = image.crop((x1, y1, x2, y2))
                upscaled_tile = self._esrgan_inference(model, tile)

                # オーバーラップ部分を考慮して貼り付け
                paste_x = x1 * scale
                paste_y = y1 * scale

                # 最初のタイル以外はオーバーラップ部分をカット
                crop_left = (overlap * scale // 2) if x > 0 else 0
                crop_top = (overlap * scale // 2) if y > 0 else 0

                if crop_left > 0 or crop_top > 0:
                    tile_w, tile_h = upscaled_tile.size
                    upscaled_tile = upscaled_tile.crop((crop_left, crop_top, tile_w, tile_h))
                    paste_x += crop_left
                    paste_y += crop_top

                result.paste(upscaled_tile, (paste_x, paste_y))

                # メモリ解放
                self.clear_cache()

        return result

    def _esrgan_inference(self, model, image: Image.Image) -> Image.Image:
        """Real-ESRGAN推論（単一画像）"""
        # PIL Image -> tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        if self.device == "cuda":
            img_tensor = img_tensor.half()

        # 推論
        with torch.no_grad():
            output = model(img_tensor)

        # tensor -> PIL Image
        output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(output)

    def upscale_with_sd(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int
    ) -> Image.Image:
        """SD Upscalerでアップスケール"""
        pipeline = self.load_sd_upscaler()

        # 入力画像を最大128x128にリサイズ（SD Upscalerの制約）
        max_size = 128
        w, h = image.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            low_res = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            low_res = image

        generator = torch.Generator(self.device).manual_seed(seed)

        full_prompt = PROMPT_PREFIX + prompt if prompt else PROMPT_PREFIX
        full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

        result = pipeline(
            prompt=full_prompt,
            image=low_res,
            negative_prompt=full_negative,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator
        ).images[0]

        return result

    def refine_with_img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        tile_size: int = 512
    ) -> Image.Image:
        """img2imgで画像を仕上げ（タイル処理対応、サイズ維持）"""
        from .manager import pipeline_manager

        # img2imgパイプラインを取得
        pipeline = pipeline_manager.get_img2img_pipeline("CleanVAE")

        w, h = image.size

        # 小さい画像は直接処理
        if w <= tile_size and h <= tile_size:
            # 画像サイズを64の倍数に調整
            new_w = (w // 64) * 64
            new_h = (h // 64) * 64
            if new_w != w or new_h != h:
                image = image.resize((new_w, new_h), Image.LANCZOS)

            generator = torch.Generator(self.device).manual_seed(seed)
            full_prompt = PROMPT_PREFIX + prompt if prompt else PROMPT_PREFIX
            full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

            result = pipeline(
                prompt=full_prompt,
                image=image,
                negative_prompt=full_negative,
                strength=float(strength),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator
            ).images[0]
            return result

        # 大きい画像はタイル処理
        print(f"Img2img tile processing: {w}x{h}")
        overlap = 64
        result = Image.new("RGB", (w, h))

        full_prompt = PROMPT_PREFIX + prompt if prompt else PROMPT_PREFIX
        full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                x1 = x
                y1 = y
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)

                # タイルサイズを64の倍数に
                tile_w = ((x2 - x1) // 64) * 64
                tile_h = ((y2 - y1) // 64) * 64
                if tile_w < 64 or tile_h < 64:
                    continue

                tile = image.crop((x1, y1, x1 + tile_w, y1 + tile_h))

                generator = torch.Generator(self.device).manual_seed(seed + y * w + x)

                refined_tile = pipeline(
                    prompt=full_prompt,
                    image=tile,
                    negative_prompt=full_negative,
                    strength=float(strength),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator
                ).images[0]

                # オーバーラップ考慮して貼り付け
                paste_x = x1
                paste_y = y1
                crop_left = (overlap // 2) if x > 0 else 0
                crop_top = (overlap // 2) if y > 0 else 0

                if crop_left > 0 or crop_top > 0:
                    tw, th = refined_tile.size
                    refined_tile = refined_tile.crop((crop_left, crop_top, tw, th))
                    paste_x += crop_left
                    paste_y += crop_top

                result.paste(refined_tile, (paste_x, paste_y))
                self.clear_cache()

        return result

    def clear_cache(self):
        """メモリキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


# グローバルインスタンス
upscaler = Upscaler()


def upscale_image(
    input_image,
    mode: str,
    realesrgan_model: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    denoising_strength: float,
    seed: int
) -> tuple[list[Image.Image], str]:
    """画像をアップスケール"""
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

        orig_size = image.size
        print(f"Upscaling image: {orig_size} with mode: {mode}")

        # 入力画像サイズ制限（メモリ節約）
        max_input_size = 1024
        if orig_size[0] > max_input_size or orig_size[1] > max_input_size:
            ratio = min(max_input_size / orig_size[0], max_input_size / orig_size[1])
            new_w = int(orig_size[0] * ratio)
            new_h = int(orig_size[1] * ratio)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"Input image resized: {orig_size} -> {image.size}")

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "upscale", prompt or "upscale")

        result_image = None

        if mode == "realesrgan":
            # Real-ESRGANのみ
            result_image = upscaler.upscale_with_realesrgan(image, realesrgan_model)
            method_info = f"Real-ESRGAN ({realesrgan_model})"

        elif mode == "sd":
            # SD Upscalerのみ
            result_image = upscaler.upscale_with_sd(
                image, prompt, negative_prompt,
                num_inference_steps, guidance_scale, seed
            )
            method_info = "SD Upscaler"

        elif mode == "both":
            # Real-ESRGAN → SD img2img refinement
            print("Step 1: Real-ESRGAN upscaling...")
            esrgan_result = upscaler.upscale_with_realesrgan(image, realesrgan_model)
            upscaler.clear_cache()

            print("Step 2: SD img2img refinement...")
            # ESRGANの結果をimg2imgで仕上げ（サイズ維持）
            result_image = upscaler.refine_with_img2img(
                esrgan_result, prompt, negative_prompt,
                num_inference_steps, guidance_scale, denoising_strength, seed
            )
            method_info = f"Real-ESRGAN ({realesrgan_model}) → SD Refinement"

        else:
            return None, f"不明なモード: {mode}"

        # 画像を保存
        filename = f"upscale_seed{seed}_{mode}.png"
        filepath = os.path.join(output_dir, filename)
        result_image.save(filepath)

        upscaler.clear_cache()

        new_size = result_image.size
        return [result_image], f"アップスケール完了\n方式: {method_info}\n{orig_size[0]}x{orig_size[1]} → {new_size[0]}x{new_size[1]}\n保存先: {output_dir}"

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
