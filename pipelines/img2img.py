"""Image to Image生成モジュール"""
import os
import gc
import torch
from PIL import Image

from config import DEVICE, clear_stop, is_stop_requested
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.image import to_pil_rgb
from utils.long_prompt import encode_long_prompts_if_needed, build_long_prompt_kwargs
from .manager import pipeline_manager
from .sd_common import (
    build_full_prompts,
    get_pipeline_checked,
    get_face_restorer,
    apply_face_restore,
    save_partial_and_message,
)


def generate_images_img2img(
    input_image,
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
    face_restore_weight: float = 0.5
) -> tuple[list[Image.Image], str]:
    """複数の画像を生成（img2img）"""
    if input_image is None:
        return None, "入力画像をアップロードしてください"

    if not prompt.strip():
        return None, "プロンプトを入力してください"

    # パイプラインを取得（複数LoRA適用、スケジューラ指定）+ LoRAエラーチェック
    pipeline, lora_error = get_pipeline_checked(
        "img2img", vae_name, model_name, scheduler_name,
        lora1, weight1, lora2, weight2, lora3, weight3
    )
    if lora_error:
        return None, lora_error

    # 入力画像をPIL Imageに変換
    init_image = to_pil_rgb(input_image)

    # 出力フォルダを作成
    base_path = os.path.dirname(os.path.dirname(__file__))
    output_dir = create_output_dir(base_path, "i2i", prompt)

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
                negative_prompt=full_negative,
                strength=float(strength),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]

        # 顔修正を適用
        image = apply_face_restore(face_restorer, image, face_restore_method, face_restore_weight, i)

        # 画像を保存
        suffix = "_fr" if face_restorer is not None else ""
        filename = f"i2i_{i+1:03d}_seed{seed}_str{strength:.2f}{suffix}.png"
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
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
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

        print(f"Generated img2img {i+1}/{int(num_images)}: seed={seed}, strength={strength}")

        # メモリ解放
        pipeline_manager.clear_cache()
        gc.collect()

    # 生成パラメータをCSVに保存
    csv_path = save_batch_generation_params(output_dir, csv_rows)
    print(f"Generation params saved to: {csv_path}")

    face_restore_info = f"\n顔修正: {face_restore_method}" if face_restore_enabled else ""
    return generated_images, f"{num_images}枚の画像を生成しました（img2img）{face_restore_info}\n保存先: {output_dir}"
