"""LoRAロード/アンロードの共有ヘルパー

PipelineManager (pipelines/manager.py) と ControlNetPipelineManager
(pipelines/controlnet.py) の load_loras / unload_all_loras から
そのまま抽出した共通ロジック。

NOTE: pipelines/ip_adapter.py の LoRA メソッドは意図的に移行していない。
ip_adapter.py 側の実装は簡略版で、'PEFT backend is required' のエラー分岐と
traceback 出力を持たないため、このヘルパーに統一するとユーザー向けの
エラーメッセージが変わってしまう（挙動非互換になる）。
"""
from config import LORA_FILES


def load_loras_into(pipe, lora_configs: list, current_loras: list, log_label: str = ""):
    """複数のLoRAをロード（fuse方式で確実に適用）

    Args:
        pipe: パイプライン
        lora_configs: [(lora_name, weight), ...] のリスト
        current_loras: 現在ロード済みのLoRAリスト [(name, weight), ...]
        log_label: ログ用ラベル（例: " for ControlNet"）

    Returns:
        (new_current_loras, error_message_or_None) のタプル
    """
    # 有効なLoRAのみフィルタリング
    valid_loras = [
        (name, weight) for name, weight in lora_configs
        if name and name != "なし" and LORA_FILES.get(name)
    ]

    # 現在のLoRAと同じ場合はスキップ
    if valid_loras == current_loras:
        return current_loras, None

    # 以前のLoRAをアンロード
    current_loras = unload_loras_from(pipe, current_loras)

    if not valid_loras:
        return current_loras, None

    # 複数LoRAをロード
    adapter_names = []
    adapter_weights = []
    errors = []

    for lora_name, lora_weight in valid_loras:
        lora_path = LORA_FILES[lora_name]
        try:
            print(f"Loading LoRA{log_label}: {lora_name} (weight: {lora_weight})")
            print(f"LoRA path: {lora_path}")

            # LoRAをロード
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            adapter_names.append(lora_name)
            adapter_weights.append(lora_weight)
            print(f"LoRA loaded: {lora_name}")
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Error loading LoRA {lora_name}: {error_msg}")
            print(traceback.format_exc())

            # ユーザー向けエラーメッセージを設定
            if "Target modules" in error_msg and "not found" in error_msg:
                errors.append(f"LoRA '{lora_name}' はこのモデルと互換性がありません")
            elif "PEFT backend is required" in error_msg:
                errors.append(f"LoRA '{lora_name}' のロードにはPEFTライブラリが必要です")
            else:
                errors.append(f"LoRA '{lora_name}' のロード失敗: {error_msg[:50]}")

    # 複数アダプターを設定してfuse
    if adapter_names:
        try:
            pipe.set_adapters(adapter_names, adapter_weights)
            pipe.fuse_lora(adapter_names=adapter_names)
            current_loras = [(name, weight) for name, weight in zip(adapter_names, adapter_weights)]
            print(f"LoRAs fused successfully: {adapter_names} with weights {adapter_weights}")
        except Exception as e:
            import traceback
            print(f"Error fusing LoRAs: {e}")
            print(traceback.format_exc())
            errors.append(f"LoRA融合エラー: {str(e)[:50]}")
            current_loras = []

    if errors:
        return current_loras, "\n".join(errors)
    return current_loras, None


def unload_loras_from(pipe, current_loras: list) -> list:
    """すべてのLoRAをアンロード

    Returns:
        新しい current_loras（成功時は []、失敗時は元のリストを維持）
    """
    if current_loras:
        try:
            print(f"Unloading LoRAs: {[name for name, _ in current_loras]}")
            # 融合を解除してからアンロード
            try:
                pipe.unfuse_lora()
            except Exception:
                pass  # 融合されていない場合は無視
            pipe.unload_lora_weights()
            current_loras = []
            print("LoRAs unloaded successfully")
        except Exception as e:
            import traceback
            print(f"Error unloading LoRAs: {e}")
            print(traceback.format_exc())
    return current_loras
