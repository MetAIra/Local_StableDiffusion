"""Model Catalog タブ

利用可能なモデル・VAE・LoRA・Embeddingの一覧と説明を表示する。
"""
import gradio as gr

from config import MODEL_CATALOG, MODEL_FILES, LORA_FILES, format_model_choice


def create_model_catalog_tab():
    """Model Catalog タブを作成"""

    gr.Markdown("""
    ### Model Catalog
    利用可能なモデル・VAE・LoRA・Embeddingの情報一覧です。
    """)

    # カテゴリごとに分類
    checkpoints = {}
    vaes = {}
    loras = {}
    embeddings = {}
    others = {}

    for name, info in MODEL_CATALOG.items():
        category = info.get('category', '')
        if category == 'Checkpoint':
            checkpoints[name] = info
        elif category == 'VAE':
            vaes[name] = info
        elif category == 'LoRA':
            loras[name] = info
        elif category == 'Embedding':
            embeddings[name] = info
        else:
            others[name] = info

    # Checkpoints
    if checkpoints:
        with gr.Accordion("Checkpoint モデル", open=True):
            # SD1.5 と SDXL系 に分けて表示
            sd15_models = {k: v for k, v in checkpoints.items() if v['base_model'] == 'SD 1.5'}
            sdxl_models = {k: v for k, v in checkpoints.items() if v['base_model'] != 'SD 1.5'}

            if sd15_models:
                gr.Markdown("#### SD 1.5 モデル")
                md = _build_model_table(sd15_models)
                gr.Markdown(md)

            if sdxl_models:
                gr.Markdown("#### SDXL / Illustrious / Pony モデル")
                md = _build_model_table(sdxl_models)
                gr.Markdown(md)

    # VAE
    if vaes:
        with gr.Accordion("VAE", open=False):
            md = _build_model_table(vaes)
            gr.Markdown(md)

    # LoRA
    if loras:
        with gr.Accordion("LoRA", open=False):
            md = _build_model_table(loras)
            gr.Markdown(md)

    # Embedding
    if embeddings:
        with gr.Accordion("Embedding", open=False):
            md = _build_model_table(embeddings)
            gr.Markdown(md)

    # 利用可能状況
    with gr.Accordion("現在利用可能なファイル", open=False):
        available_models = list(MODEL_FILES.keys())
        available_loras = [k for k in LORA_FILES.keys() if k != "なし"]

        # 商用OKマーカー付きで表示（commercial_safe_models.md 由来）
        marked_models = [format_model_choice(m) for m in available_models]

        gr.Markdown(f"**Checkpoint ({len(available_models)}):** {', '.join(marked_models) if marked_models else 'なし'}")
        gr.Markdown("*✓ = `commercial_safe_models.md` 掲載の商用利用OKモデル / P2推奨は推奨タグ付き*")
        gr.Markdown(f"**LoRA ({len(available_loras)}):** {', '.join(available_loras) if available_loras else 'なし'}")


def _build_model_table(models: dict) -> str:
    """モデル情報をMarkdownテーブルとして構築"""
    lines = []
    lines.append("| 名前 | ベース | 説明 | 推奨設定 |")
    lines.append("|------|--------|------|----------|")

    for name, info in models.items():
        desc = info.get('description', '-')
        base = info.get('base_model', '-')
        settings = info.get('settings', '-')
        url = info.get('url', '')

        # 名前にURLリンクを付ける
        if url and url != '-':
            display_name = f"[{name}]({url})"
        else:
            display_name = name

        # 長い説明は短縮
        if len(desc) > 80:
            desc = desc[:77] + "..."

        lines.append(f"| {display_name} | {base} | {desc} | {settings} |")

    return "\n".join(lines)
