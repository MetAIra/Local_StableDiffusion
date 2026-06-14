"""txt2img/img2img/inpaint/outpaint 共通の生成前処理ヘルパー"""
from config import PROMPT_PREFIX, PROMPT_PREFIX_SDXL, FACE_RESTORE_METHODS, is_sdxl_model
from utils.file import save_batch_generation_params
from .manager import pipeline_manager


def build_full_prompts(prompt: str, negative_prompt: str, model_name: str) -> tuple[str, str, bool]:
    """プロンプト/ネガティブプロンプトにプレフィックスを適用して返す

    Returns:
        (full_prompt, full_negative, is_sdxl)
    """
    # SDXLモデルかどうかを判定
    is_sdxl = is_sdxl_model(model_name)

    # Negative promptの設定（SDXLの場合はEasyNegativeを追加しない）
    if is_sdxl:
        full_negative = negative_prompt
    else:
        full_negative = negative_prompt
        if "EasyNegative" not in negative_prompt:
            full_negative = f"EasyNegative, {negative_prompt}" if negative_prompt else "EasyNegative"

    # プロンプトにプレフィックスを追加（SDXLの場合は別のプレフィックス）
    if is_sdxl:
        full_prompt = PROMPT_PREFIX_SDXL + prompt if prompt else PROMPT_PREFIX_SDXL
    else:
        full_prompt = PROMPT_PREFIX + prompt if prompt else PROMPT_PREFIX

    return full_prompt, full_negative, is_sdxl


def get_pipeline_checked(
    kind: str,
    vae_name: str,
    model_name: str,
    scheduler_name: str,
    lora1: str, weight1: float,
    lora2: str, weight2: float,
    lora3: str, weight3: float,
):
    """LoRA設定を組み立ててパイプラインを取得し、LoRAエラーをチェックする

    Args:
        kind: "txt2img" / "img2img" / "inpaint"

    Returns:
        (pipeline, error_message)。エラーがなければ error_message は None。
    """
    # LoRA設定を作成
    lora_configs = [
        (lora1, weight1),
        (lora2, weight2),
        (lora3, weight3)
    ]

    # パイプラインを取得（複数LoRA適用、スケジューラ指定）
    getter = getattr(pipeline_manager, f"get_{kind}_pipeline_with_loras")
    pipeline = getter(vae_name, model_name, lora_configs, scheduler_name)

    # LoRAエラーチェック
    if pipeline_manager.lora_error:
        return pipeline, f"⚠️ {pipeline_manager.lora_error}\nLoRAなしで生成を続行するか、別のLoRAを選択してください。"

    return pipeline, None


def get_face_restorer(enabled: bool, face_restore_method: str = "", face_restore_weight: float = 0.0):
    """顔修正が有効な場合、リストアラーを準備（失敗時はNone）"""
    face_restorer = None
    if enabled:
        try:
            from .face_restore import face_restorer as fr
            face_restorer = fr
            print(f"Face restore enabled: {face_restore_method} (weight={face_restore_weight})")
        except Exception as e:
            print(f"Warning: Failed to load face restorer: {e}")
            face_restorer = None
    return face_restorer


def apply_face_restore(restorer, image, face_restore_method: str, face_restore_weight: float, index: int):
    """顔修正を適用（restorerがNoneならそのまま返す）"""
    if restorer is None:
        return image
    try:
        print(f"Applying face restoration to image {index+1}...")
        method_value = FACE_RESTORE_METHODS.get(face_restore_method, "gfpgan")
        image = restorer.restore_face(
            image,
            method=method_value,
            gfpgan_version="1.4",
            gfpgan_weight=face_restore_weight,
            codeformer_fidelity=face_restore_weight
        )
        restorer.clear_cache()
    except Exception as e:
        print(f"Warning: Face restore failed: {e}")
    return image


def save_partial_and_message(output_dir: str, csv_rows: list, generated_images: list, total: int):
    """中止時の早期リターン用: 生成済み分を保存してメッセージを返す"""
    if generated_images:
        csv_path = save_batch_generation_params(output_dir, csv_rows)
        return generated_images, f"中止しました（{len(generated_images)}/{total}枚生成済み）\nCSV: {csv_path}\n保存先: {output_dir}"
    return None, "生成が中止されました"
