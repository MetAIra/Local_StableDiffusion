"""Outpainting生成モジュール"""
import os
import gc
import traceback
import torch
from PIL import Image

from config import DEVICE, clear_stop, is_stop_requested
from utils.file import create_output_dir, save_batch_generation_params, save_image_safe
from utils.image import round_to_multiple, create_outpaint_image_and_mask, to_pil_rgb
from utils.long_prompt import encode_long_prompts_if_needed, build_long_prompt_kwargs
from .manager import pipeline_manager
from .sd_common import build_full_prompts, get_pipeline_checked, save_partial_and_message


def generate_outpaint(
    input_image,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    base_seed: int,
    left_expand: int,
    right_expand: int,
    top_expand: int,
    bottom_expand: int,
    feather_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    vae_name: str,
    model_name: str,
    scheduler_name: str = "DPM++ 2M Karras",
    lora1: str = "なし", weight1: float = 1.0,
    lora2: str = "なし", weight2: float = 1.0,
    lora3: str = "なし", weight3: float = 1.0
) -> tuple[list[Image.Image], str]:
    """Outpainting処理"""
    try:
        if input_image is None:
            return None, "入力画像をアップロードしてください"

        if not prompt.strip():
            return None, "プロンプトを入力してください"

        # 拡張サイズのチェック
        if left_expand == 0 and right_expand == 0 and top_expand == 0 and bottom_expand == 0:
            return None, "少なくとも1方向の拡張サイズを指定してください"

        # パイプラインを取得（複数LoRA適用、スケジューラ指定）+ LoRAエラーチェック
        pipeline, lora_error = get_pipeline_checked(
            "inpaint", vae_name, model_name, scheduler_name,
            lora1, weight1, lora2, weight2, lora3, weight3
        )
        if lora_error:
            return None, lora_error

        # 入力画像をPIL Imageに変換
        init_image = to_pil_rgb(input_image)

        # 元画像のサイズを64の倍数に調整
        orig_w, orig_h = init_image.size
        new_orig_w = round_to_multiple(orig_w)
        new_orig_h = round_to_multiple(orig_h)
        if new_orig_w != orig_w or new_orig_h != orig_h:
            init_image = init_image.resize((new_orig_w, new_orig_h), Image.LANCZOS)
            print(f"Resized input image from {orig_w}x{orig_h} to {new_orig_w}x{new_orig_h}")

        # 拡張画像とマスクを作成
        expanded_image, mask, (new_width, new_height) = create_outpaint_image_and_mask(
            init_image,
            int(left_expand),
            int(right_expand),
            int(top_expand),
            int(bottom_expand),
            int(feather_size)
        )

        print(f"Outpaint: original={init_image.size}, expanded={new_width}x{new_height}")

        # 出力フォルダを作成
        base_path = os.path.dirname(os.path.dirname(__file__))
        output_dir = create_output_dir(base_path, "outpaint", prompt)

        generated_images = []
        csv_rows = []  # CSV用のパラメータリスト

        # プロンプト/ネガティブプロンプトを構築（SDXL判定込み）
        full_prompt, full_negative, is_sdxl = build_full_prompts(prompt, negative_prompt, model_name)

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
                    image=expanded_image,
                    mask_image=mask,
                    width=new_width,
                    height=new_height,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                )
                image = pipeline(**gen_kwargs).images[0]
            else:
                image = pipeline(
                    prompt=full_prompt,
                    image=expanded_image,
                    mask_image=mask,
                    negative_prompt=full_negative,
                    width=new_width,
                    height=new_height,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                ).images[0]

            # 画像を保存
            filename = f"outpaint_{i+1:03d}_seed{seed}_L{int(left_expand)}R{int(right_expand)}T{int(top_expand)}B{int(bottom_expand)}.png"
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
                "width": new_width,
                "height": new_height,
                "left_expand": int(left_expand),
                "right_expand": int(right_expand),
                "top_expand": int(top_expand),
                "bottom_expand": int(bottom_expand),
                "feather_size": int(feather_size),
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
                "lora3_weight": weight3
            })

            print(f"Generated outpaint {i+1}/{int(num_images)}: seed={seed}")

            # メモリ解放
            pipeline_manager.clear_cache()
            gc.collect()

        # 生成パラメータをCSVに保存
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        print(f"Generation params saved to: {csv_path}")

        return generated_images, f"{num_images}枚の画像を生成しました（Outpainting）\n拡張: 左{int(left_expand)}px 右{int(right_expand)}px 上{int(top_expand)}px 下{int(bottom_expand)}px\n保存先: {output_dir}"

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
