"""Inpainting生成モジュール - マスク領域の部分再生成"""
import os
import gc
import traceback
import torch
from PIL import Image
import numpy as np

from config import DEVICE, clear_stop, is_stop_requested
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.image import round_to_multiple, to_pil_rgb
from utils.long_prompt import encode_long_prompts_if_needed, build_long_prompt_kwargs
from .manager import pipeline_manager
from .sd_common import (
    build_full_prompts,
    get_pipeline_checked,
    get_face_restorer,
    apply_face_restore,
    save_partial_and_message,
)


def process_mask_image(mask_data) -> Image.Image:
    """マスクデータを処理してPIL Imageに変換

    Args:
        mask_data: Gradio ImageEditorからのマスクデータ
                   dict形式: {"background": ..., "layers": [...], "composite": ...}
                   または直接numpy array/PIL Image

    Returns:
        白黒のマスク画像（白=再生成する領域）
    """
    if mask_data is None:
        return None

    # ImageEditorからのdict形式
    if isinstance(mask_data, dict):
        # layersから描画されたマスクを取得
        layers = mask_data.get("layers", [])
        if layers and len(layers) > 0:
            # 最初のレイヤーを使用
            layer = layers[0]
            if isinstance(layer, np.ndarray):
                # RGBA画像の場合、アルファチャンネルまたは描画部分をマスクとして使用
                if layer.ndim == 3 and layer.shape[2] == 4:
                    # アルファチャンネルをマスクとして使用
                    alpha = layer[:, :, 3]
                    mask = Image.fromarray(alpha).convert("L")
                else:
                    mask = Image.fromarray(layer).convert("L")
                return mask
            elif isinstance(layer, Image.Image):
                return layer.convert("L")

        # composite画像から抽出を試みる
        composite = mask_data.get("composite")
        if composite is not None:
            if isinstance(composite, np.ndarray):
                if composite.ndim == 3 and composite.shape[2] == 4:
                    alpha = composite[:, :, 3]
                    mask = Image.fromarray(alpha).convert("L")
                    return mask
            elif isinstance(composite, Image.Image):
                if composite.mode == "RGBA":
                    return composite.split()[3]

        return None

    # numpy array形式
    if isinstance(mask_data, np.ndarray):
        if mask_data.ndim == 3 and mask_data.shape[2] == 4:
            alpha = mask_data[:, :, 3]
            mask = Image.fromarray(alpha).convert("L")
        else:
            mask = Image.fromarray(mask_data).convert("L")
        return mask

    # PIL Image形式
    if isinstance(mask_data, Image.Image):
        return mask_data.convert("L")

    return None


def generate_inpaint(
    input_image,
    mask_data,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    base_seed: int,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0,
    face_restore_enabled: bool = False,
    face_restore_method: str = "GFPGAN",
    face_restore_weight: float = 0.5,
    mask_blur: int = 4
) -> tuple[list[Image.Image], str]:
    """マスク領域のInpainting処理

    Args:
        input_image: 入力画像（ImageEditorからのdict、numpy array、またはPIL Image）
        mask_data: マスクデータ（ImageEditorからのdict、numpy array、またはPIL Image）
        prompt: 生成プロンプト
        negative_prompt: ネガティブプロンプト
        num_images: 生成枚数
        base_seed: ベースシード値
        strength: 変化の強度 (0.0-1.0)
        num_inference_steps: 推論ステップ数
        guidance_scale: CFGスケール
        vae_name: VAE名
        model_name: モデル名
        lora1-3: LoRA名
        weight1-3: LoRAの重み
        face_restore_enabled: 顔修正を有効にするか
        face_restore_method: 顔修正方式
        face_restore_weight: 顔修正の強度
        mask_blur: マスクのぼかし半径
    """
    try:
        # 入力画像の処理
        if input_image is None:
            return None, "入力画像をアップロードしてください"

        # ImageEditorからのdict形式を処理
        if isinstance(input_image, dict):
            # backgroundまたはcompositeから画像を取得
            bg = input_image.get("background")
            if bg is not None:
                if isinstance(bg, np.ndarray):
                    init_image = Image.fromarray(bg).convert("RGB")
                elif isinstance(bg, Image.Image):
                    init_image = bg.convert("RGB")
                else:
                    return None, "画像形式が不正です"
            else:
                composite = input_image.get("composite")
                if composite is not None:
                    if isinstance(composite, np.ndarray):
                        init_image = Image.fromarray(composite).convert("RGB")
                    elif isinstance(composite, Image.Image):
                        init_image = composite.convert("RGB")
                    else:
                        return None, "画像形式が不正です"
                else:
                    return None, "入力画像が見つかりません"
        elif isinstance(input_image, (str, np.ndarray, Image.Image)):
            init_image = to_pil_rgb(input_image)
        else:
            return None, f"未対応の画像形式です: {type(input_image)}"

        # マスクの処理
        mask = process_mask_image(mask_data)
        if mask is None:
            return None, "マスクを描画してください（再生成したい領域を白く塗ってください）"

        # マスクが全て黒（何も選択されていない）かチェック
        mask_array = np.array(mask)
        if mask_array.max() == 0:
            return None, "マスクを描画してください（再生成したい領域を白く塗ってください）"

        if not prompt.strip():
            return None, "プロンプトを入力してください"

        # マスクを画像サイズに合わせる
        if mask.size != init_image.size:
            mask = mask.resize(init_image.size, Image.LANCZOS)

        # マスクのぼかし処理
        if mask_blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # パイプラインを取得（複数LoRA適用、スケジューラ指定）+ LoRAエラーチェック
        pipeline, lora_error = get_pipeline_checked(
            "inpaint", vae_name, model_name, scheduler_name,
            lora1, weight1, lora2, weight2, lora3, weight3
        )
        if lora_error:
            return None, lora_error

        # 画像サイズを64の倍数に調整
        orig_w, orig_h = init_image.size
        new_w = round_to_multiple(orig_w)
        new_h = round_to_multiple(orig_h)
        if new_w != orig_w or new_h != orig_h:
            init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.LANCZOS)
            print(f"Resized image from {orig_w}x{orig_h} to {new_w}x{new_h}")

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "inpaint", prompt)

        generated_images = []
        csv_rows = []  # CSV用のパラメータリスト

        # プロンプト/ネガティブプロンプトを構築（SDXL判定込み）
        full_prompt, full_negative, is_sdxl = build_full_prompts(prompt, negative_prompt, model_name)

        # 顔修正が有効な場合、リストアラーを準備
        face_restorer = get_face_restorer(face_restore_enabled, face_restore_method, face_restore_weight)

        # Long Prompt対応
        (use_long_prompt, prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = encode_long_prompts_if_needed(
            pipeline_manager.pipe, full_prompt, full_negative, is_sdxl
        )

        clear_stop()

        for i in range(int(num_images)):
            if is_stop_requested():
                return save_partial_and_message(output_dir, csv_rows, generated_images, int(num_images))

            seed = int(base_seed) + i
            generator = torch.Generator(DEVICE).manual_seed(seed)

            if use_long_prompt:
                gen_kwargs = build_long_prompt_kwargs(
                    is_sdxl,
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    image=init_image,
                    mask_image=mask,
                    width=new_w,
                    height=new_h,
                    strength=float(strength),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                )
                image = pipeline(**gen_kwargs).images[0]
            else:
                image = pipeline(
                    prompt=full_prompt,
                    image=init_image,
                    mask_image=mask,
                    negative_prompt=full_negative,
                    width=new_w,
                    height=new_h,
                    strength=float(strength),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                ).images[0]

            # 顔修正を適用
            image = apply_face_restore(face_restorer, image, face_restore_method, face_restore_weight, i)

            # 画像を保存
            suffix = "_fr" if face_restorer is not None else ""
            filename = f"inpaint_{i+1:03d}_seed{seed}_str{strength:.2f}{suffix}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_safe(image, filepath)
            generated_images.append(image)

            # CSV用のパラメータを記録
            csv_rows.append({
                "filename": filename,
                "seed": seed,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "width": new_w,
                "height": new_h,
                "steps": num_inference_steps,
                "cfg_scale": guidance_scale,
                "mask_blur": mask_blur,
                "model": model_name,
                "vae": vae_name,
                "scheduler": scheduler_name,
                "lora1": lora1,
                "lora1_weight": weight1,
                "lora2": lora2,
                "lora2_weight": weight2,
                "lora3": lora3,
                "lora3_weight": weight3,
                "face_restore": face_restore_method if face_restore_enabled else "なし",
                "face_restore_weight": face_restore_weight if face_restore_enabled else 0
            })

            print(f"Generated inpaint {i+1}/{int(num_images)}: seed={seed}, strength={strength}")

            # メモリ解放
            pipeline_manager.clear_cache()
            gc.collect()

        # 生成パラメータをCSVに保存
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        print(f"Generation params saved to: {csv_path}")

        face_restore_info = f"\n顔修正: {face_restore_method}" if face_restore_enabled else ""
        return generated_images, f"{num_images}枚の画像を生成しました（Inpainting）{face_restore_info}\n保存先: {output_dir}"

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
