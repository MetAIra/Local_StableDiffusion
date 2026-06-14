"""設定・定数モジュール"""
import os
import csv
import threading
from pathlib import Path

# =============================================================================
# Hugging Face キャッシュディレクトリの設定（他のimportより先に設定）
# =============================================================================
from dotenv import load_dotenv
# プロジェクトルート（このファイルと同じディレクトリ）の .env を読み込む
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)
if os.path.exists(_ENV_PATH):
    print(f"Loaded .env from: {_ENV_PATH}")

# ベースパス（環境変数 BASE_MODEL_DIR で上書き可能）
# - Windows 実機: G:\マイドライブ\...\models（既定）
# - Linux/Colab: BASE_MODEL_DIR を明示設定するのが推奨。未設定時は
#   /content/models（Colab ローカル）または ~/sd_models にフォールバックする。
def _default_base_model_dir() -> str:
    env = os.environ.get('BASE_MODEL_DIR')
    if env:
        return env
    if os.name == 'nt':
        return r'G:\マイドライブ\000_Metaverse\SD_dataset\image_data\models'
    # Linux / Google Colab
    if os.path.isdir('/content'):
        return '/content/models'
    return os.path.expanduser('~/sd_models')


BASE_MODEL_DIR = _default_base_model_dir()

# キャッシュディレクトリ（.envで上書き可能）
HF_CACHE_DIR = os.environ.get('HF_HOME', os.path.join(BASE_MODEL_DIR, '.cache'))

# CLIP Interrogator のキャッシュ（ViT-L-14_openai_*.safetensors）
# デフォルトはプロジェクト直下の "cache" だが、models/ 配下にまとめてSSD空きを集約
CLIP_CACHE_DIR = os.environ.get('CLIP_CACHE_DIR', os.path.join(BASE_MODEL_DIR, 'cache'))

# Hugging Face関連の環境変数を設定（diffusers/transformersのimport前に必要）
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Google Driveではsymlinkが使えないため警告を抑制

# torchはHF環境変数設定後にimport
import torch

MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'model')
VAE_DIR = os.path.join(BASE_MODEL_DIR, 'VAE')
NEGATIVE_DIR = os.path.join(BASE_MODEL_DIR, 'Negative')
LORA_DIR = os.path.join(BASE_MODEL_DIR, 'LoRA')
# 動画モデル: 環境変数 VIDEO_MODEL_DIR で上書き可能。
# Google Drive 上だと mmap や大容量ファイルの読み込みで OSError 22 が出るため、
# C: などのローカルドライブに置くことを推奨。
VIDEO_MODEL_DIR = os.environ.get('VIDEO_MODEL_DIR', os.path.join(BASE_MODEL_DIR, 'Video-model'))


def get_available_models() -> dict[str, str]:
    """モデルフォルダから利用可能なモデルファイルを取得"""
    model_files = {}
    model_path = Path(MODEL_DIR)

    if model_path.exists():
        for file in model_path.glob("*.safetensors"):
            name = file.stem
            model_files[name] = str(file)

    return model_files


# モデルファイル一覧（動的に取得）
MODEL_FILES = get_available_models()

# デフォルトモデル
DEFAULT_MODEL = list(MODEL_FILES.keys())[0] if MODEL_FILES else None


def get_available_loras() -> dict[str, str]:
    """LoRAフォルダから利用可能なLoRAファイルを取得"""
    lora_files = {"なし": None}
    lora_path = Path(LORA_DIR)

    if lora_path.exists():
        for file in lora_path.glob("*.safetensors"):
            # ファイル名から拡張子を除いた名前をキーにする
            name = file.stem
            lora_files[name] = str(file)

    return lora_files


# LoRAファイル一覧（動的に取得）
LORA_FILES = get_available_loras()


# =============================================================================
# モデルカタログ（CSVから読み込み）
# =============================================================================

def load_model_catalog() -> dict[str, dict]:
    """models_catalog.csv からモデル情報を読み込む

    Returns:
        {ファイル名(拡張子なし): {category, base_model, description, settings, url}, ...}
    """
    catalog = {}
    csv_path = os.path.join(BASE_MODEL_DIR, 'models_catalog.csv')

    if not os.path.exists(csv_path):
        print(f"Model catalog not found: {csv_path}")
        return catalog

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('ファイル名', '').strip()
                if not filename:
                    continue
                # 拡張子を除いたキー
                stem = Path(filename).stem
                catalog[stem] = {
                    'filename': filename,
                    'category': row.get('カテゴリ', '').strip(),
                    'base_model': row.get('ベースモデル', '').strip(),
                    'description': row.get('説明', '').strip(),
                    'settings': row.get('推奨設定', '').strip(),
                    'url': row.get('CivitaiURL', '').strip(),
                }
    except Exception as e:
        print(f"Error loading model catalog: {e}")

    return catalog


MODEL_CATALOG = load_model_catalog()


def get_catalog_info(name: str) -> dict | None:
    """モデル/LoRA/VAE名からカタログ情報を取得"""
    return MODEL_CATALOG.get(name)


def format_catalog_info(name: str) -> str:
    """モデル/LoRA/VAE名からフォーマット済みの説明文を返す

    商用利用可否のステータス（commercial_safe_models.md 由来）も先頭に表示する。
    """
    lines = []

    # 商用利用可否を先頭に
    safety = format_commercial_safety_info(name)
    if safety:
        lines.append(safety)

    info = get_catalog_info(name)
    if info:
        if info['base_model']:
            lines.append(f"**ベースモデル:** {info['base_model']}")
        if info['description']:
            lines.append(f"**説明:** {info['description']}")
        if info['settings'] and info['settings'] != '-':
            lines.append(f"**推奨設定:** {info['settings']}")
        if info['url'] and info['url'] != '-':
            lines.append(f"**Civitai:** {info['url']}")

    return "\n\n".join(lines)


# =============================================================================
# 商用利用可能モデル（commercial_safe_models.md, 2026-05-10判定 の13モデル）
# =============================================================================
# 大量生成→Booth有料データセット販売(MetAIra事業)で使用可能なモデル。
# `recommended=True` は P2推奨5モデル（Sampler×CFG挙動マップ用厳選セット）。
COMMERCIAL_SAFE_MODELS: dict[str, dict] = {
    "absolutereality_v181":         {"recommended": False, "lineage": "SD1.5 リアル(Lykon)",        "url": "https://civitai.com/models/81458"},
    "animagineXLV31_v31":           {"recommended": True,  "lineage": "SDXL アニメ(Cagliostro)",    "url": "https://civitai.com/models/260267"},
    "architectureUrbanSdlife_v60":  {"recommended": False, "lineage": "SD1.5 建築",                  "url": "https://civitai.com/models/128280"},
    "dreamshaper_8":                {"recommended": True,  "lineage": "SD1.5 汎用(Lykon)",          "url": "https://civitai.com/models/4384"},
    "epicrealism_naturalSinRC1VAE": {"recommended": True,  "lineage": "SD1.5 リアル",                "url": "https://civitai.com/models/25694"},
    "homeRoomsDecoration_v10":      {"recommended": False, "lineage": "SD1.5 インテリア",            "url": "https://civitai.com/models/137424"},
    "juggernautXL_ragnarokBy":      {"recommended": True,  "lineage": "SDXL リアル(RunDiffusion)",  "url": "https://civitai.com/models/133005"},
    "realbeautymix_v15":            {"recommended": False, "lineage": "SD1.5 リアル",                "url": "https://civitai.com/models/85156"},
    "realcartoon3d_v18":            {"recommended": True,  "lineage": "SD1.5 カートゥーン(RCNZ)",    "url": "https://civitai.com/models/94809"},
    "realcartoonAnime_v11":         {"recommended": False, "lineage": "SD1.5 アニメ風(RCNZ)",        "url": "https://civitai.com/models/96629"},
    "realcartoonPixar_v12":         {"recommended": False, "lineage": "SD1.5 Pixar風(RCNZ)",         "url": "https://civitai.com/models/107289"},
    "realisian_v60":                {"recommended": False, "lineage": "SD1.5 リアル",                "url": "https://civitai.com/models/47130"},
    "yayoiMix_v25":                 {"recommended": False, "lineage": "SD1.5 リアル(Kotajiro)",      "url": "https://civitai.com/models/83096"},
}


def format_model_choice(model_name: str) -> str:
    """モデル名を Dropdown 表示用ラベルに整形（商用OKマーカー付き）

    Gradio Dropdown は choices=[(label, value), ...] のタプル形式を受けるので、
    この関数の戻り値はラベルとしてのみ表示され、value は元の model_name のまま。
    """
    info = COMMERCIAL_SAFE_MODELS.get(model_name)
    if not info:
        return model_name
    if info["recommended"]:
        return f"✓ {model_name}  (P2推奨/商用OK)"
    return f"✓ {model_name}  (商用OK)"


def get_model_dropdown_choices() -> list[tuple[str, str]]:
    """ベースモデル Dropdown 用の (label, value) リストを返す

    商用OKモデルはラベルにマーカーが付き、value は元のキーのまま。
    """
    return [(format_model_choice(name), name) for name in MODEL_FILES.keys()]


def format_commercial_safety_info(model_name: str) -> str:
    """商用利用可否を Markdown 形式で返す（モデル情報パネル用）"""
    if not model_name:
        return ""
    info = COMMERCIAL_SAFE_MODELS.get(model_name)
    if info:
        marker = "**P2推奨モデル**" if info["recommended"] else "商用利用可"
        return (
            f"**✅ 商用利用OK** （{marker} / commercial_safe_models.md 掲載）  \n"
            f"系譜: {info['lineage']}  \n"
            f"出典: {info['url']}"
        )
    # MODEL_FILES に存在するが商用OKリスト外のもの
    if model_name in MODEL_FILES:
        return (
            "**⚠️ 商用利用は要確認** — `commercial_safe_models.md` の13モデル外。"
            "大量生成→販売用途では使用しないでください。"
        )
    return ""


# VAEファイルの一覧
VAE_FILES = {
    "CleanVAE": os.path.join(VAE_DIR, 'CleanVAE.safetensors'),
    "kl-f8-anime2": os.path.join(VAE_DIR, 'kl-f8-anime2.safetensors'),
    "なし（デフォルト）": None
}

# Negative Embedding
NEGATIVE_EMBEDDING = os.path.join(NEGATIVE_DIR, 'EasyNegativeV2.safetensors')

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# デフォルトのNegative prompt
DEFAULT_NEGATIVE = """((((mutated hands and fingers)))), deformed, blurry, bad anatomy, long neck, long_neck, long body, long_body, deformed mutated disfigured, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, ugly face, poorly drawn hands, poorly_drawn_hands, missing limb, missing_limb, blurry, floating limbs, floating_limbs, disconnected limbs, disconnected_limbs, malformed hands, malformed_hands, blur, out of focus, text, title, flat color, flat shading, bad fingers, liquid fingers, poorly drawn fingers, bad anatomy, missing fingers, signature, watermark, username, artist name, missing legs, extra legs, extra_legs, bad hands, mutated hands, missing arms, extra_arms, bad proportions, extra fingers, extra_fingers, extra digit, fewer digits"""

# デフォルトのプロンプトプレフィックス（SD1.5用）
PROMPT_PREFIX = "wdgoodprompt, (new, newest, best quality, extremely detailed, high resolution, anime:1.2), "

# SDXL用プロンプトプレフィックス
PROMPT_PREFIX_SDXL = "masterpiece, best quality, highly detailed, "

# SDXL用ネガティブプロンプト
DEFAULT_NEGATIVE_SDXL = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

# Pony Diffusion V6 XL用プロンプトプレフィックス（独自スコアタグシステム）
PROMPT_PREFIX_PONY = "score_9, score_8_up, score_7_up, "

# Pony Diffusion用ネガティブプロンプト（基本的に不要だが、アニメ寄りにしたい場合）
DEFAULT_NEGATIVE_PONY = "source_cartoon, source_furry, source_pony, sketch, painting, monochrome"

# リアル系モデル用プロンプトプレフィックス（epiCRealism, AbsoluteReality等）
PROMPT_PREFIX_REALISTIC = ""  # シンプルなプロンプト推奨

# リアル系モデル用ネガティブプロンプト
DEFAULT_NEGATIVE_REALISTIC = "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)"

# Real-ESRGANモデル設定
REALESRGAN_MODELS = {
    "RealESRGAN_x4plus_anime_6B": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "scale": 4,
        "description": "アニメ/イラスト向け"
    },
    "RealESRGAN_x4plus": {
        "name": "RealESRGAN_x4plus",
        "scale": 4,
        "description": "汎用/実写向け"
    }
}

# SD Upscalerモデル
SD_UPSCALER_MODEL = "stabilityai/stable-diffusion-x4-upscaler"

# Zero123++ Multi-View Model (Image to Multi-View)
ZERO123_MODEL = "sudo-ai/zero123plus-v1.1"
ZERO123_PIPELINE = "sudo-ai/zero123plus-pipeline"
ZERO123_DEFAULT_STEPS = 75
ZERO123_DEFAULT_CFG = 4.0
ZERO123_MIN_INPUT_SIZE = 320
ZERO123_OUTPUT_SIZE = 256  # 各視点の出力サイズ

# ControlNetモデル設定
CONTROLNET_MODELS = {
    "openpose": {
        "model_id": "lllyasviel/control_v11p_sd15_openpose",
        "description": "ポーズ制御（人物のポーズを維持）"
    },
    "canny": {
        "model_id": "lllyasviel/control_v11p_sd15_canny",
        "description": "エッジ検出（輪郭線を維持）"
    },
    "depth": {
        "model_id": "lllyasviel/control_v11f1p_sd15_depth",
        "description": "深度マップ（奥行き構造を維持）"
    },
    "lineart": {
        "model_id": "lllyasviel/control_v11p_sd15_lineart",
        "description": "線画（線画から生成）"
    },
    "scribble": {
        "model_id": "lllyasviel/control_v11p_sd15_scribble",
        "description": "落書き（ラフスケッチから生成）"
    },
    "tile": {
        "model_id": "lllyasviel/control_v11f1e_sd15_tile",
        "description": "タイル（ディテール維持アップスケール）"
    }
}

# Upscaleモード
UPSCALE_MODES = {
    "Real-ESRGAN Only": "realesrgan",
    "SD Upscaler Only": "sd",
    "Real-ESRGAN → SD": "both"
}

# 顔修正設定
FACE_RESTORE_METHODS = {
    "GFPGAN": "gfpgan",
    "CodeFormer": "codeformer"
}

GFPGAN_VERSIONS = {
    "v1.4 (推奨)": "1.4",
    "v1.3": "1.3"
}

# 背景除去モデル設定
REMBG_MODELS = {
    "u2net": "U2-Net（汎用・高品質）",
    "u2netp": "U2-Net Portrait（人物特化・軽量）",
    "u2net_human_seg": "U2-Net Human Seg（人物セグメンテーション）",
    "u2net_cloth_seg": "U2-Net Cloth Seg（衣服セグメンテーション）",
    "isnet-general-use": "IS-Net（汎用・高精度）",
    "isnet-anime": "IS-Net Anime（アニメ/イラスト特化）",
    "silueta": "Silueta（シルエット抽出）",
}

# スケジューラ設定
SCHEDULERS = {
    "DDIM": "ddim",
    "Euler": "euler",
    "Euler a": "euler_a",
    "DPM++ 2M": "dpm_2m",
    "DPM++ 2M Karras": "dpm_2m_karras",
    "DPM++ SDE": "dpm_sde",
    "DPM++ SDE Karras": "dpm_sde_karras",
    "UniPC": "unipc",
    "LMS": "lms",
    "PNDM": "pndm",
    "Heun": "heun",
}

DEFAULT_SCHEDULER = "DPM++ 2M Karras"


def is_flux_model(model_name: str) -> bool:
    """モデル名からFluxモデルかどうかを判定"""
    flux_keywords = ['flux']
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in flux_keywords)


def is_pony_model(model_name: str) -> bool:
    """モデル名からPony Diffusionモデルかどうかを判定"""
    pony_keywords = ['pony']
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in pony_keywords)


def is_realistic_model(model_name: str) -> bool:
    """モデル名からリアル系モデルかどうかを判定"""
    realistic_keywords = ['realistic', 'realism', 'realvis', 'absolutereality', 'epicrealism', 'juggernaut']
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in realistic_keywords)


def is_sdxl_model(model_name: str) -> bool:
    """モデル名からSDXLモデルかどうかを判定"""
    # Fluxモデルを除外
    if is_flux_model(model_name):
        return False
    sdxl_keywords = ['sdxl', 'xl', 'pony', 'illustrious', 'noob']
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in sdxl_keywords)


def get_model_type(model_name: str) -> str:
    """モデルタイプを取得（'flux', 'pony', 'sdxl', 'realistic', or 'sd15'）"""
    if is_flux_model(model_name):
        return "flux"
    if is_pony_model(model_name):
        return "pony"
    if is_sdxl_model(model_name):
        return "sdxl"
    if is_realistic_model(model_name):
        return "realistic"
    return "sd15"


def get_prompt_prefix(model_name: str) -> str:
    """モデルに適したプロンプトプレフィックスを取得"""
    model_type = get_model_type(model_name)
    if model_type == "pony":
        return PROMPT_PREFIX_PONY
    elif model_type == "sdxl":
        return PROMPT_PREFIX_SDXL
    elif model_type == "realistic":
        return PROMPT_PREFIX_REALISTIC
    else:
        return PROMPT_PREFIX


def get_negative_prompt(model_name: str) -> str:
    """モデルに適したネガティブプロンプトを取得"""
    model_type = get_model_type(model_name)
    if model_type == "pony":
        return DEFAULT_NEGATIVE_PONY
    elif model_type == "sdxl":
        return DEFAULT_NEGATIVE_SDXL
    elif model_type == "realistic":
        return DEFAULT_NEGATIVE_REALISTIC
    else:
        return DEFAULT_NEGATIVE


# SDXL用ControlNetモデル設定
CONTROLNET_MODELS_SDXL = {
    "canny": {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "description": "エッジ検出（SDXL用）"
    },
    "depth": {
        "model_id": "diffusers/controlnet-depth-sdxl-1.0",
        "description": "深度マップ（SDXL用）"
    },
}

# IP-Adapter設定
IP_ADAPTER_REPO = "h94/IP-Adapter"

# SD1.5用IP-Adapterモデル
IP_ADAPTER_MODELS_SD15 = {
    "ip-adapter_sd15": {
        "subfolder": "models",
        "weight_name": "ip-adapter_sd15.bin",
        "description": "汎用（スタイル転写）"
    },
    "ip-adapter-plus_sd15": {
        "subfolder": "models",
        "weight_name": "ip-adapter-plus_sd15.bin",
        "description": "高品質（スタイル転写強化）"
    },
    "ip-adapter-plus-face_sd15": {
        "subfolder": "models",
        "weight_name": "ip-adapter-plus-face_sd15.bin",
        "description": "顔特化（顔の特徴を保持）"
    },
    "ip-adapter-full-face_sd15": {
        "subfolder": "models",
        "weight_name": "ip-adapter-full-face_sd15.bin",
        "description": "顔全体（顔全体の特徴を強く保持）"
    },
}

# SDXL用IP-Adapterモデル
# 注: vit-hバリアントはmodels/image_encoderを使用、それ以外はsdxl_models/image_encoderを使用
IP_ADAPTER_MODELS_SDXL = {
    "ip-adapter_sdxl": {
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter_sdxl.bin",
        "description": "汎用（SDXL用スタイル転写）"
    },
    "ip-adapter-plus_sdxl_vit-h": {
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.bin",
        "description": "高品質（SDXL用スタイル転写強化）"
    },
    "ip-adapter-plus-face_sdxl_vit-h": {
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus-face_sdxl_vit-h.bin",
        "description": "顔特化（SDXL用顔の特徴を保持）"
    },
}

# IP-Adapterデフォルト設定
IP_ADAPTER_DEFAULT_SCALE = 0.7
IP_ADAPTER_DEFAULT_STEPS = 30


# =============================================================================
# Image to Text (Claude API + CLIP Interrogator) 設定
# =============================================================================

# Claude API設定
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 2048

# CLIP Interrogatorモデル設定
CLIP_MODELS = {
    "ViT-L-14/openai": "標準（バランス型）",
    "ViT-H-14/laion2b_s32b_b79k": "高精度（SDXL向け）",
}
DEFAULT_CLIP_MODEL = "ViT-L-14/openai"

# 出力スタイル設定
I2T_STYLES = {
    "auto": "自動検出",
    "anime": "アニメ・イラスト",
    "realistic": "写真・実写",
    "artistic": "アート・絵画",
}
DEFAULT_I2T_STYLE = "auto"

# Claude APIのシステムプロンプト
CLAUDE_SYSTEM_PROMPT = """あなたはStable Diffusionプロンプトの専門家です。
画像を分析し、この画像をSD/SDXLで再現するための最適なプロンプトを生成してください。

出力は以下のJSON形式で返してください（他のテキストは含めないでください）:
{
  "main_prompt": "英語のプロンプト、カンマ区切り",
  "negative_prompt": "避けるべき要素（英語）",
  "style": "anime/realistic/artistic のいずれか",
  "recommended_settings": {
    "steps": 25,
    "cfg_scale": 7.0,
    "width": 512,
    "height": 768,
    "sampler": "DPM++ 2M Karras"
  },
  "analysis": {
    "subject": "主要な被写体",
    "composition": "構図",
    "lighting": "照明",
    "mood": "雰囲気"
  }
}

プロンプトに含めるべき要素:
1. 品質タグ (masterpiece, best quality, highly detailed等)
2. 被写体の詳細 (人物なら: 髪色・髪型、服装、表情、ポーズ等)
3. 構図・アングル (full body, upper body, close-up, from above, from side等)
4. 背景・環境の詳細
5. 照明・時間帯 (soft lighting, dramatic lighting, golden hour等)
6. アートスタイル (anime style, photorealistic, oil painting等)

アニメ/イラスト画像の場合は以下のタグを優先:
- 1girl, 1boy等の人数タグ
- 髪色・髪型の詳細タグ
- アニメスタイルの品質タグ

実写/写真画像の場合は以下を優先:
- photorealistic, photograph等
- カメラ設定に関するタグ（bokeh, shallow depth of field等）
- 自然な照明の記述"""


# =============================================================================
# Text to Keywords (Claude API) 設定
# =============================================================================

# 出力スタイル設定
T2K_STYLES = {
    "auto": "自動（入力から判断）",
    "anime": "アニメ・イラスト向け",
    "realistic": "写真・実写向け",
    "artistic": "アート・絵画向け",
    "fantasy": "ファンタジー向け",
    "cyberpunk": "サイバーパンク向け",
}
DEFAULT_T2K_STYLE = "auto"

# 単語数プリセット
T2K_WORD_COUNTS = {
    "少なめ（10-15語）": 15,
    "標準（20-30語）": 25,
    "詳細（40-50語）": 45,
    "非常に詳細（60-80語）": 70,
}
DEFAULT_T2K_WORD_COUNT = "標準（20-30語）"

# 詳細度設定
T2K_DETAIL_LEVELS = {
    "簡潔": "simple",
    "標準": "normal",
    "詳細": "detailed",
}
DEFAULT_T2K_DETAIL_LEVEL = "標準"

# Claude APIのシステムプロンプト（Text to Keywords用）
CLAUDE_T2K_SYSTEM_PROMPT = """あなたはStable Diffusion用のプロンプト生成の専門家です。
ユーザーが日本語で説明した画像イメージを、Stable Diffusionで高品質な画像を生成するための英語キーワード（タグ）に変換してください。

出力ルール:
1. 英語のキーワードをカンマ区切りで出力
2. 品質タグを最初に含める（masterpiece, best quality, highly detailed等）
3. 重要な要素ほど先に配置
4. 括弧()で強調、数値で重み付け可能（例: (detailed eyes:1.2)）
5. 出力は指定された単語数を目安にする
6. 余計な説明は不要、キーワードのみ出力

スタイル別の注意点:
- anime: 1girl, 1boy, hair color, eye color, anime style等のタグを使用
- realistic: photorealistic, photograph, 8k, RAW photo等を使用
- artistic: oil painting, watercolor, digital art等のスタイルタグを使用
- fantasy: magical, ethereal, glowing, mystical等を含める
- cyberpunk: neon, futuristic, cyber, hologram等を含める

出力形式: キーワードをカンマ区切りで出力するだけ。他の説明は不要。"""


# =============================================================================
# Variable Prompt Generator (Claude API) 設定
# =============================================================================

# Claude APIのシステムプロンプト（Variable Prompt Generator用）
CLAUDE_VPG_SYSTEM_PROMPT = """あなたはStable Diffusion用のVariable Prompt生成の専門家です。
ユーザーが日本語で説明した「変数（バリエーション）のあるバッチ生成」の要件から、
Variable Promptシステム向けのプロンプト一式を生成してください。

Variable Promptシステムの仕組み:
- 固定プロンプト: 全画像に共通で適用されるプロンプト
- 変数テンプレート: {変数名} の形式で変数を埋め込むテンプレート
- 変数定義: 「変数名: 値1, 値2, 値3」の形式で各変数の値を列挙
- ネガティブプロンプト: 避けるべき要素

出力は必ず以下のフォーマットで返してください（他のテキストは含めないでください）:

[FIXED_PROMPT]
（全画像に適用される固定プロンプト。英語キーワード、カンマ区切り）

[VARIABLE_TEMPLATE]
（{変数名}を含むテンプレート。英語キーワード）

[VARIABLE_DEFINITIONS]
（変数名: 値1, 値2, 値3 の形式。1行1変数。値は全て英語）

[NEGATIVE_PROMPT]
（ネガティブプロンプト。英語キーワード、カンマ区切り）

重要なルール:
1. 固定プロンプトには品質タグ（masterpiece, best quality等）と全画像共通の要素を含める
2. 変数テンプレートでは {変数名} の形式で変数を参照する
3. 変数定義の値は全て英語で記述する（日本語の概念も英語に翻訳）
4. 変数名は英語のスネークケース（例: hair_color, outfit_style）を使用
5. ユーザーが具体的な数や種類を指定した場合は正確にその数だけ列挙する
6. 変数の値はStable Diffusionが理解しやすいキーワードにする
7. ネガティブプロンプトは画像品質を保つための標準的なものを含める"""

# Variable Prompt Generator カテゴリプリセット
VPG_CATEGORY_PRESETS = {
    "自由入力": "",
    "干支（12種類）": "十二支の動物（子丑寅卯辰巳午未申酉戌亥）をそれぞれキャラクターとして描く",
    "四季": "春夏秋冬の4つの季節をそれぞれ表現する",
    "12星座": "12の星座（牡羊座〜魚座）をそれぞれキャラクターとして描く",
    "感情表現": "喜怒哀楽などの様々な感情表現のバリエーションを描く",
    "ファンタジー職業": "ファンタジー世界の様々な職業（戦士、魔法使い、僧侶、盗賊など）を描く",
    "時間帯": "朝昼夕夜など1日の様々な時間帯のシーンを描く",
    "天候": "晴れ、曇り、雨、雪など様々な天候のシーンを描く",
}

# =============================================================================
# Video Generation 設定
# =============================================================================

# 動画モデル設定（HuggingFaceから事前ダウンロード可能なモデル）
# 8GB VRAMで動作させることを想定したモデル群。
# 各モデルは scripts/download_video_models.py によって VIDEO_MODEL_DIR 配下にDLされる。
VIDEO_MODELS = {
    "animatediff": {
        "label": "AnimateDiff (SD1.5)",
        "repo_id": "guoyww/animatediff-motion-adapter-v1-5-3",
        "local_subdir": "animatediff",
        # 旧 runwayml/stable-diffusion-v1-5 は2024年に削除済み。新公式ミラーを使用
        "base_model_repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_model_subdir": "stable-diffusion-v1-5",
        "type": "text-to-video",
        "vram": 8,
        "size_gb": 1.7,
        "description": "SD1.5ベースのMotion Adapter。既存SD1.5モデルに動きを付与（512x768 縦長推奨, 16フレーム）",
        # 512x768 縦長を既定にすると顔・手のピクセル数が増えてアニメ系で破綻が大幅に減る。
        # 16フレーム/steps=50/CFG=8 が dreamshaper_8 等の高品質 SD1.5 ベースとの組合せで
        # 最もバランス良く品質が出ることを確認済み。
        "default_frames": 16,
        "default_fps": 8,
        "default_steps": 50,
        "default_guidance": 8.0,
        "default_width": 512,
        "default_height": 768,
        "max_frames": 64,  # FreeNoise 有効時の上限（>16 は自動で FreeNoise 適用）
        # 16フレーム = motion adapter v1-5-3 のネイティブ訓練長。
        # これを超えると滑動窓 (FreeNoise) なしでは attention が訓練分布外で
        # 不安定化し、特に 32fr/512x768 のような訓練外形状でノイズ出力になる。
        "free_noise_threshold": 16,
        # 8fps native は低すぎてカクカクするので RIFE で 24fps 化を推奨
        "auto_smooth": True,
        "default_smooth_target_fps": 24,
    },
    "svd_xt": {
        "label": "Stable Video Diffusion XT (img→video)",
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "local_subdir": "svd_xt",
        "type": "image-to-video",
        "vram": 8,
        "size_gb": 9.5,
        "description": "画像から滑らかな動画を生成（25フレーム, 1024x576）。CPU offloadで8GB対応",
        "default_frames": 25,
        "default_fps": 7,
        "default_steps": 25,
        "default_guidance": 3.0,  # min_guidance / max_guidance
        "default_width": 1024,
        "default_height": 576,
        "max_frames": 25,           # 1チャンク当たりの上限（モデル固定）
        "max_chunks": 8,            # 連続生成: 末尾フレーム→次の入力で 25×N まで延長
        "default_chunks": 1,
        # 7fps native は低すぎてカクカクするので RIFE で 24fps 化を推奨
        "auto_smooth": True,
        "default_smooth_target_fps": 24,
    },
    "ltx_video": {
        "label": "LTX-Video 0.9 (T2V/I2V)",
        "repo_id": "Lightricks/LTX-Video",
        "local_subdir": "ltx_video",
        "type": "text-to-video",  # I2Vも対応
        "vram": 8,
        "size_gb": 7.5,
        "description": "高速・高品質なtext→video / image→video。bf16+VAE tilingで8GB可",
        "default_frames": 129,
        "default_fps": 24,
        "default_steps": 30,
        "default_guidance": 3.5,
        "default_width": 1024,
        "default_height": 576,
        "max_frames": 257,  # 公式上限（10.7秒 @ 24fps）。VRAM 許す限り増やせる
        # 24fps ネイティブで補間不要
        "auto_smooth": False,
        "default_smooth_target_fps": 24,
    },
    "cogvideox_2b": {
        "label": "CogVideoX-2B",
        "repo_id": "THUDM/CogVideoX-2b",
        "local_subdir": "cogvideox_2b",
        "type": "text-to-video",
        "vram": 8,
        "size_gb": 10.0,
        "description": "高品質なtext→video（720x480, 49フレーム）。sequential CPU offloadで8GB可・低速",
        "default_frames": 49,
        "default_fps": 8,
        "default_steps": 50,
        "default_guidance": 6.0,
        "default_width": 720,
        "default_height": 480,
        "max_frames": 49,
        # 8fps native なので補間推奨（ただし元から重いモデルなので任意）
        "auto_smooth": True,
        "default_smooth_target_fps": 24,
    },
    "wan21_1_3b": {
        "label": "Wan2.1 1.3B (T2V)",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "local_subdir": "wan21_1_3b",
        "type": "text-to-video",
        "vram": 8,
        "size_gb": 5.0,
        "description": "軽量で8GB余裕あり。text→video（832x480, 81フレーム）",
        "default_frames": 81,
        "default_fps": 16,
        "default_steps": 30,
        "default_guidance": 5.0,
        "default_width": 832,
        "default_height": 480,
        "max_frames": 121,  # 81 を超えると VRAM ギリギリ + 時間も伸びるが概ね動く
        # 16fps native で既に十分滑らか。RIFE 補間はかえって違和感を出すため OFF 推奨
        "auto_smooth": False,
        "default_smooth_target_fps": 24,
    },
}

# 動画出力デフォルトフォーマット
VIDEO_DEFAULT_FORMAT = "mp4"  # "mp4" or "gif"
VIDEO_OUTPUT_DIR_PREFIX = "video"

# 動画生成用デフォルトネガティブプロンプト
# 動画特有の劣化（チラつき、形状崩壊、ピンボケ、人体欠損、アーティファクト）を抑える
VIDEO_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, text, "
    "ugly, distorted, deformed, mutated, bad anatomy, bad proportions, "
    "extra limbs, missing limbs, malformed hands, bad hands, "
    "static, frozen, glitch, flicker, ghosting, motion blur, "
    "duplicate, error, cropped, out of frame, low resolution, lowres, "
    "oversaturated, washed out colors, grainy, noisy"
)

# AnimateDiff 専用ネガティブ。SD1.5 系は (token:weight) の重み付け構文を解釈するため、
# 顔と手の崩壊（AnimateDiff の最大の弱点）に対して強めの重みでネガを効かせる。
# 標準ネガに加えて顔・手・指の異常を重点的に抑える。
VIDEO_ANIMATEDIFF_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, text, "
    "ugly, distorted, deformed, mutated, bad anatomy, bad proportions, "
    "extra limbs, missing limbs, static, frozen, glitch, flicker, ghosting, "
    "duplicate, error, cropped, out of frame, low resolution, lowres, "
    "oversaturated, washed out colors, grainy, noisy, "
    "(bad face:1.4), (deformed face:1.4), (mutated face:1.4), "
    "(bad hands:1.4), (deformed hands:1.4), (malformed hands:1.4), "
    "(missing fingers:1.3), (extra fingers:1.3), (fused fingers:1.3), "
    "ugly face, asymmetric eyes, cross-eyed"
)

# Wan2.1 公式推奨ネガティブプロンプト（中国語版を英訳した公式推奨セット）。
# Wan は 832x480 + この長いネガティブで最も品質が安定する。標準ネガでは砂嵐になる場合あり。
VIDEO_WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# AnimateDiff 用の品質向上プロンプトプレフィックス（ポジ側に追記する）
# 動画では各フレームの一貫性が重要なので「stable, consistent」系のキーワードを推奨
VIDEO_QUALITY_PREFIX = (
    "masterpiece, best quality, ultra detailed, 8k, sharp focus, "
    "professional cinematography, beautiful lighting, "
    "consistent motion, smooth animation, fluid movement, cinematic"
)

# 動画モデルファイル存在チェック用（ダウンロード済み判定）
# 主要重みファイルが存在するサブフォルダを示すヒント。空なら自動推定
_VIDEO_WEIGHT_SUBFOLDERS = {
    "animatediff": [],  # 単体モデル: root直下のsafetensors
    "svd_xt": ["unet", "vae", "image_encoder"],
    "ltx_video": ["transformer", "vae", "text_encoder"],
    "cogvideox_2b": ["transformer", "vae", "text_encoder"],
    "wan21_1_3b": ["transformer", "vae", "text_encoder"],
}


def _has_significant_weight(folder: Path, min_mb: int = 50) -> bool:
    """指定フォルダ内に min_mb 以上の重みファイル(.safetensors/.bin)が1つでもあるか"""
    if not folder.exists():
        return False
    threshold = min_mb * 1024 * 1024
    try:
        for f in folder.iterdir():
            if not f.is_file():
                continue
            if f.suffix in (".safetensors", ".bin") and f.stat().st_size >= threshold:
                return True
    except Exception:
        return False
    return False


def is_video_model_downloaded(model_key: str) -> bool:
    """指定された動画モデルがローカルにダウンロード済みか判定

    - 単体モデル(motion adapter等): root直下に重みファイルがあればOK
    - diffusers folder構造: model_index.json + 主要サブフォルダに重みファイル
    """
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return False
    local_path = Path(VIDEO_MODEL_DIR) / cfg["local_subdir"]
    if not local_path.exists():
        return False

    has_diffusers_index = (local_path / "model_index.json").exists()
    has_root_weights = _has_significant_weight(local_path, min_mb=100)

    # 単体モデル（diffusersフォルダ無し）
    subfolders = _VIDEO_WEIGHT_SUBFOLDERS.get(model_key, [])
    if not subfolders:
        return has_root_weights

    # diffusers folder 構造: 主要サブフォルダに重みがあること
    if not has_diffusers_index:
        return False
    for sub in subfolders:
        if not _has_significant_weight(local_path / sub, min_mb=50):
            return False
    return True


def get_video_model_path(model_key: str) -> str | None:
    """指定された動画モデルのローカルパスを返す（ダウンロード済みの場合）"""
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return None
    local_path = Path(VIDEO_MODEL_DIR) / cfg["local_subdir"]
    return str(local_path) if local_path.exists() else None


def get_video_base_model_path(model_key: str) -> str | None:
    """AnimateDiff用のベースSD1.5モデルのローカルパスを返す"""
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg or "base_model_subdir" not in cfg:
        return None
    base_path = Path(VIDEO_MODEL_DIR) / cfg["base_model_subdir"]
    return str(base_path) if base_path.exists() else None


def is_video_base_model_downloaded(model_key: str) -> bool:
    """AnimateDiff用ベースSD1.5モデルがDL済みか判定（主要サブフォルダの重み有り）"""
    cfg = VIDEO_MODELS.get(model_key)
    if not cfg or "base_model_subdir" not in cfg:
        return True  # ベース不要モデル
    base_path = Path(VIDEO_MODEL_DIR) / cfg["base_model_subdir"]
    if not base_path.exists():
        return False
    if not (base_path / "model_index.json").exists():
        return False
    # SD1.5の主要重み: unet, vae, text_encoder
    for sub in ("unet", "vae", "text_encoder"):
        if not _has_significant_weight(base_path / sub, min_mb=50):
            return False
    return True


# =============================================================================
# Audio Generation 設定
# =============================================================================

# Audio モデルディレクトリ
AUDIO_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'audio')
TTS_MODEL_DIR = os.path.join(AUDIO_MODEL_DIR, 'tts')
MUSIC_MODEL_DIR = os.path.join(AUDIO_MODEL_DIR, 'music')
VOICE_CONV_MODEL_DIR = os.path.join(AUDIO_MODEL_DIR, 'voice_conversion')

# RVC モデルディレクトリ（新構造）
RVC_MODEL_DIR = os.path.join(AUDIO_MODEL_DIR, 'RVC')
HUBERT_BASE_MODEL = os.path.join(AUDIO_MODEL_DIR, 'base_model', 'hubert_base.pt')

# TTS モデル設定
TTS_MODELS = {
    "bark": {
        "repo_id": "suno/bark",
        "vram": 12,
        "description": "多言語対応（英語、日本語など）、感情表現可能"
    },
    "xtts_v2": {
        "repo_id": "coqui/XTTS-v2",
        "vram": 12,
        "description": "高品質ボイスクローン対応、16言語"
    },
}

# Bark ボイスプリセット
BARK_VOICE_PRESETS = {
    # English speakers
    "en_speaker_0": "英語 男性1",
    "en_speaker_1": "英語 男性2",
    "en_speaker_2": "英語 男性3",
    "en_speaker_3": "英語 男性4",
    "en_speaker_4": "英語 男性5",
    "en_speaker_5": "英語 女性1",
    "en_speaker_6": "英語 女性2",
    "en_speaker_7": "英語 女性3",
    "en_speaker_8": "英語 女性4",
    "en_speaker_9": "英語 女性5",
    # Japanese speakers
    "ja_speaker_0": "日本語 女性1",
    "ja_speaker_1": "日本語 女性2",
    "ja_speaker_2": "日本語 女性3",
    "ja_speaker_3": "日本語 男性1",
    "ja_speaker_4": "日本語 男性2",
    "ja_speaker_5": "日本語 男性3",
    "ja_speaker_6": "日本語 女性4",
    "ja_speaker_7": "日本語 女性5",
    "ja_speaker_8": "日本語 女性6",
    "ja_speaker_9": "日本語 男性4",
    # Chinese speakers
    "zh_speaker_0": "中国語 女性1",
    "zh_speaker_1": "中国語 男性1",
    "zh_speaker_2": "中国語 男性2",
    "zh_speaker_3": "中国語 女性2",
    "zh_speaker_4": "中国語 女性3",
    "zh_speaker_5": "中国語 男性3",
    "zh_speaker_6": "中国語 女性4",
    "zh_speaker_7": "中国語 女性5",
    "zh_speaker_8": "中国語 男性4",
    "zh_speaker_9": "中国語 女性6",
    # Korean speakers
    "ko_speaker_0": "韓国語 女性1",
    "ko_speaker_1": "韓国語 男性1",
    "ko_speaker_2": "韓国語 女性2",
    "ko_speaker_3": "韓国語 男性2",
    "ko_speaker_4": "韓国語 男性3",
    "ko_speaker_5": "韓国語 女性3",
    "ko_speaker_6": "韓国語 男性4",
    "ko_speaker_7": "韓国語 女性4",
    "ko_speaker_8": "韓国語 男性5",
    "ko_speaker_9": "韓国語 女性5",
}

# Bark 言語設定
BARK_LANGUAGES = {
    "en": "英語",
    "ja": "日本語",
    "zh": "中国語",
    "ko": "韓国語",
    "de": "ドイツ語",
    "es": "スペイン語",
    "fr": "フランス語",
    "it": "イタリア語",
    "pl": "ポーランド語",
    "pt": "ポルトガル語",
    "ru": "ロシア語",
    "tr": "トルコ語",
}

# XTTS 言語設定
XTTS_LANGUAGES = {
    "en": "英語",
    "es": "スペイン語",
    "fr": "フランス語",
    "de": "ドイツ語",
    "it": "イタリア語",
    "pt": "ポルトガル語",
    "pl": "ポーランド語",
    "tr": "トルコ語",
    "ru": "ロシア語",
    "nl": "オランダ語",
    "cs": "チェコ語",
    "ar": "アラビア語",
    "zh": "中国語",
    "ja": "日本語",
    "hu": "ハンガリー語",
    "ko": "韓国語",
}

# Music Generation モデル設定
MUSIC_GEN_MODELS = {
    "musicgen-small": {
        "repo_id": "facebook/musicgen-small",
        "vram": 8,
        "description": "軽量版（300Mパラメータ）"
    },
    "musicgen-medium": {
        "repo_id": "facebook/musicgen-medium",
        "vram": 12,
        "description": "標準版（1.5Bパラメータ）"
    },
    "musicgen-large": {
        "repo_id": "facebook/musicgen-large",
        "vram": 16,
        "description": "高品質版（3.3Bパラメータ）"
    },
    "musicgen-melody": {
        "repo_id": "facebook/musicgen-melody",
        "vram": 12,
        "description": "メロディー条件付け対応（1.5Bパラメータ）"
    },
}

# AudioLDM モデル設定
AUDIOLDM_MODELS = {
    "audioldm2-music": {
        "repo_id": "cvssp/audioldm2-music",
        "vram": 16,
        "description": "音楽生成特化"
    },
    "audioldm2": {
        "repo_id": "cvssp/audioldm2",
        "vram": 16,
        "description": "汎用音声生成"
    },
}

# オーディオサンプルレート設定
DEFAULT_SAMPLE_RATE = 32000

# 音声生成デフォルト設定
TTS_DEFAULT_SETTINGS = {
    "bark": {
        "temperature": 0.7,
        "semantic_temperature": 0.7,
    },
    "xtts_v2": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 2.0,
    },
}

# 音楽生成デフォルト設定
MUSIC_DEFAULT_SETTINGS = {
    "duration": 10,  # 秒
    "temperature": 1.0,
    "guidance_scale": 3.0,
    "top_k": 250,
    "top_p": 0.0,  # 0 = disabled
}

# Voice Conversion 設定
RVC_INDEX_RATE = 0.75
RVC_FILTER_RADIUS = 3
RVC_RESAMPLE_SR = 0
RVC_RMS_MIX_RATE = 0.25
RVC_PROTECT = 0.33

# RVC ピッチシフト範囲
RVC_PITCH_RANGE = {
    "min": -12,
    "max": 12,
    "default": 0,
}

# RVC メソッド
RVC_METHODS = {
    "rmvpe": "RMVPE（推奨・高品質）",
    "harvest": "Harvest（低速・高精度）",
    "crepe": "Crepe（GPU・高精度）",
    "pm": "PM（高速・低品質）",
}

# 出力ディレクトリ（output/ 配下に画像/音声/動画を集約）
OUTPUT_BASE_DIR = "output"
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "image_output")
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "audio_output")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "video_output")


def get_available_rvc_models() -> dict[str, str]:
    """RVCモデルフォルダから利用可能なモデルを取得

    新構造: models/audio/RVC/モデル名/model.pth
    旧構造: models/audio/voice_conversion/rvc/*.pth
    """
    rvc_models = {}

    # 新構造: models/audio/RVC/ (サブディレクトリ内を再帰検索)
    rvc_path = Path(RVC_MODEL_DIR)
    if rvc_path.exists():
        # サブディレクトリ内の .pth ファイルを検索
        for file in rvc_path.glob("**/*.pth"):
            # 親ディレクトリ名をモデル名として使用
            if file.parent != rvc_path:
                name = file.parent.name
            else:
                name = file.stem
            rvc_models[name] = str(file)

    # 旧構造: models/audio/voice_conversion/rvc/ (後方互換性)
    old_rvc_path = Path(os.path.join(VOICE_CONV_MODEL_DIR, 'rvc'))
    if old_rvc_path.exists():
        for file in old_rvc_path.glob("*.pth"):
            name = file.stem
            if name not in rvc_models:  # 重複を避ける
                rvc_models[name] = str(file)

    return rvc_models


# =============================================================================
# 生成停止フラグ（協調的キャンセル用）
# =============================================================================
_generation_stop_event = threading.Event()


def request_stop():
    """停止ボタンから呼ばれる: 停止フラグをセット"""
    _generation_stop_event.set()


def clear_stop():
    """生成開始時に呼ばれる: 停止フラグをクリア"""
    _generation_stop_event.clear()


def is_stop_requested():
    """ループ内で呼ばれる: 停止が要求されたかチェック"""
    return _generation_stop_event.is_set()


print(f"Using device: {DEVICE}")
print(f"HF cache dir: {HF_CACHE_DIR}")
