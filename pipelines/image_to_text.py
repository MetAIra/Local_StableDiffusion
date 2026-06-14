"""Image to Text パイプライン

Claude API + CLIP Interrogator を使用して
画像から Stable Diffusion 向けの高品質プロンプトを生成する。

主な機能:
- Claude API Vision: 深い画像理解と構造化プロンプト生成
- CLIP Interrogator: SD特化タグ（アーティスト名、スタイル等）の補完
- フォールバック: CLIP失敗時はClaude単体で続行
"""
import os
import json
import base64
import gc
from io import BytesIO
from PIL import Image
from typing import Optional

from config import (
    DEVICE,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_SYSTEM_PROMPT,
    CLIP_CACHE_DIR,
    DEFAULT_CLIP_MODEL,
)
from .claude_client import get_claude_client


class ClaudeImageAnalyzer:
    """Claude API を使用した画像分析クラス"""

    def __init__(self):
        self.client = None
        self._initialized = False

    def _ensure_initialized(self) -> tuple[bool, str]:
        """Claude API クライアントの初期化を確認"""
        if self._initialized and self.client is not None:
            return True, ""

        client, error_msg = get_claude_client()
        if client is None:
            return False, error_msg

        self.client = client
        self._initialized = True
        return True, ""

    def _image_to_base64(self, image: Image.Image) -> tuple[str, str]:
        """PIL Image を base64 エンコード"""
        # RGB に変換（RGBA の場合）
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # JPEG としてエンコード（サイズ効率が良い）
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        return image_data, "image/jpeg"

    def analyze(
        self,
        image: Image.Image,
        style_hint: str = "auto"
    ) -> tuple[Optional[dict], str]:
        """
        画像を分析してSD向けプロンプトを生成

        Args:
            image: 入力画像 (PIL Image)
            style_hint: スタイルヒント (auto/anime/realistic/artistic)

        Returns:
            (result_dict, status_message)
            result_dict: {
                "main_prompt": str,
                "negative_prompt": str,
                "style": str,
                "recommended_settings": dict,
                "analysis": dict
            }
        """
        # 初期化チェック
        ok, error_msg = self._ensure_initialized()
        if not ok:
            return None, error_msg

        # 画像をbase64エンコード
        image_data, media_type = self._image_to_base64(image)

        # スタイルヒントをプロンプトに追加
        style_instruction = ""
        if style_hint != "auto":
            style_labels = {
                "anime": "アニメ・イラスト",
                "realistic": "写真・実写",
                "artistic": "アート・絵画"
            }
            style_instruction = f"\n\nこの画像は「{style_labels.get(style_hint, style_hint)}」スタイルです。"

        user_prompt = f"この画像を分析し、Stable Diffusionで再現するための最適なプロンプトをJSON形式で生成してください。{style_instruction}"

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ],
                }],
            )

            # レスポンスからテキストを抽出
            response_text = response.content[0].text

            # JSONをパース
            # コードブロック内のJSONを抽出する試み
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            return result, "Claude API による分析が完了しました"

        except json.JSONDecodeError as e:
            # JSONパースに失敗した場合、テキストとして返す
            return {
                "main_prompt": response_text if 'response_text' in locals() else "",
                "negative_prompt": "",
                "style": style_hint if style_hint != "auto" else "unknown",
                "recommended_settings": {},
                "analysis": {"error": f"JSONパースエラー: {str(e)}"}
            }, f"警告: レスポンスのJSONパースに失敗しました: {str(e)}"

        except Exception as e:
            import traceback
            return None, f"Claude API エラー: {str(e)}\n{traceback.format_exc()}"


class CLIPInterrogatorWrapper:
    """CLIP Interrogator のラッパークラス（遅延ロード）"""

    def __init__(self):
        self.interrogator = None
        self.current_model = None
        self._available = None

    def is_available(self) -> bool:
        """CLIP Interrogator が利用可能か確認"""
        if self._available is not None:
            return self._available

        try:
            from clip_interrogator import Config, Interrogator
            self._available = True
        except ImportError:
            self._available = False
            print("Warning: clip-interrogator パッケージが見つかりません")

        return self._available

    def _ensure_loaded(self, model_name: str = None) -> tuple[bool, str]:
        """CLIP Interrogator のロードを確認"""
        if not self.is_available():
            return False, (
                "clip-interrogator パッケージがインストールされていません。\n"
                "インストール: pip install clip-interrogator"
            )

        model_name = model_name or DEFAULT_CLIP_MODEL

        # 既にロード済みで同じモデルなら何もしない
        if self.interrogator is not None and self.current_model == model_name:
            return True, ""

        try:
            from clip_interrogator import Config, Interrogator

            # 既存のモデルをアンロード
            if self.interrogator is not None:
                del self.interrogator
                self.interrogator = None
                gc.collect()
                if DEVICE == "cuda":
                    import torch
                    torch.cuda.empty_cache()

            # 設定（cache_path で CLIP 用 .safetensors 群の保存先を指定）
            os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
            config = Config(
                clip_model_name=model_name,
                caption_model_name=None,  # BLIPは使わない
                device=DEVICE,
                quiet=True,
                cache_path=CLIP_CACHE_DIR,
            )
            config.apply_low_vram_defaults()

            # ロード
            print(f"CLIP Interrogator をロード中: {model_name}")
            self.interrogator = Interrogator(config)
            self.current_model = model_name
            return True, ""

        except Exception as e:
            import traceback
            return False, f"CLIP Interrogator ロードエラー: {str(e)}\n{traceback.format_exc()}"

    def interrogate(
        self,
        image: Image.Image,
        model_name: str = None,
        mode: str = "fast"
    ) -> tuple[Optional[str], str]:
        """
        画像からSD向けタグを抽出

        Args:
            image: 入力画像 (PIL Image)
            model_name: CLIPモデル名
            mode: "fast" (高速) または "full" (詳細)

        Returns:
            (tags_string, status_message)
        """
        ok, error_msg = self._ensure_loaded(model_name)
        if not ok:
            return None, error_msg

        try:
            # RGB に変換
            if image.mode != "RGB":
                image = image.convert("RGB")

            if mode == "fast":
                # 高速モード: アーティスト・スタイルタグのみ
                result = self.interrogator.interrogate_fast(image)
            else:
                # 詳細モード: 全てのタグ
                result = self.interrogator.interrogate(image)

            return result, "CLIP Interrogator による分析が完了しました"

        except Exception as e:
            import traceback
            return None, f"CLIP Interrogator エラー: {str(e)}\n{traceback.format_exc()}"

    def unload(self):
        """モデルをアンロードしてメモリを解放"""
        if self.interrogator is not None:
            del self.interrogator
            self.interrogator = None
            self.current_model = None
            gc.collect()
            if DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()


# シングルトンインスタンス
claude_analyzer = ClaudeImageAnalyzer()
clip_interrogator = CLIPInterrogatorWrapper()


def merge_prompts(claude_prompt: str, clip_tags: str) -> str:
    """
    Claude と CLIP のプロンプトをマージ

    Args:
        claude_prompt: Claude が生成したメインプロンプト
        clip_tags: CLIP Interrogator が生成したタグ

    Returns:
        マージされたプロンプト
    """
    if not clip_tags:
        return claude_prompt

    if not claude_prompt:
        return clip_tags

    # Claude のプロンプトをベースに、CLIP のタグから重複しないものを追加
    claude_parts = set(p.strip().lower() for p in claude_prompt.split(","))
    clip_parts = [p.strip() for p in clip_tags.split(",")]

    # 重複しないタグを追加
    additional_tags = []
    for tag in clip_parts:
        if tag.lower() not in claude_parts:
            additional_tags.append(tag)

    if additional_tags:
        return f"{claude_prompt}, {', '.join(additional_tags)}"
    return claude_prompt


def generate_prompt_from_image(
    image: Image.Image,
    style: str = "auto",
    use_clip: bool = True,
    clip_model: str = None,
    clip_mode: str = "fast"
) -> tuple[str, str, dict, str]:
    """
    画像からSD向けプロンプトを生成

    Args:
        image: 入力画像 (PIL Image)
        style: スタイルヒント (auto/anime/realistic/artistic)
        use_clip: CLIP Interrogator を使用するか
        clip_model: CLIPモデル名
        clip_mode: CLIP モード (fast/full)

    Returns:
        (main_prompt, negative_prompt, settings_dict, status_message)
    """
    if image is None:
        return "", "", {}, "画像をアップロードしてください"

    status_messages = []
    main_prompt = ""
    negative_prompt = ""
    settings = {}
    analysis = {}

    # Step 1: Claude API で分析
    claude_result, claude_status = claude_analyzer.analyze(image, style)
    status_messages.append(f"Claude: {claude_status}")

    if claude_result:
        main_prompt = claude_result.get("main_prompt", "")
        negative_prompt = claude_result.get("negative_prompt", "")
        settings = claude_result.get("recommended_settings", {})
        analysis = claude_result.get("analysis", {})
        detected_style = claude_result.get("style", "unknown")
    else:
        return "", "", {}, f"エラー: {claude_status}"

    # Step 2: CLIP Interrogator で補完（オプション）
    clip_tags = None
    if use_clip:
        if clip_interrogator.is_available():
            clip_tags, clip_status = clip_interrogator.interrogate(
                image,
                model_name=clip_model,
                mode=clip_mode
            )
            if clip_tags:
                status_messages.append(f"CLIP: {clip_status}")
                # プロンプトをマージ
                main_prompt = merge_prompts(main_prompt, clip_tags)
            else:
                status_messages.append(f"CLIP: {clip_status} (フォールバック: Claude単体で続行)")
        else:
            status_messages.append("CLIP: 利用不可（Claude単体で続行）")

    # 結果を整形
    final_status = "\n".join(status_messages)

    # 分析結果を設定に追加
    if analysis:
        settings["analysis"] = analysis

    return main_prompt, negative_prompt, settings, final_status


def format_settings_display(settings: dict) -> str:
    """設定辞書を表示用文字列に変換"""
    if not settings:
        return "推奨設定なし"

    lines = []

    # 基本設定
    if "steps" in settings:
        lines.append(f"Steps: {settings['steps']}")
    if "cfg_scale" in settings:
        lines.append(f"CFG Scale: {settings['cfg_scale']}")
    if "width" in settings and "height" in settings:
        lines.append(f"Size: {settings['width']}x{settings['height']}")
    if "sampler" in settings:
        lines.append(f"Sampler: {settings['sampler']}")

    # 分析結果
    if "analysis" in settings:
        analysis = settings["analysis"]
        lines.append("")
        lines.append("--- 画像分析 ---")
        if "subject" in analysis:
            lines.append(f"被写体: {analysis['subject']}")
        if "composition" in analysis:
            lines.append(f"構図: {analysis['composition']}")
        if "lighting" in analysis:
            lines.append(f"照明: {analysis['lighting']}")
        if "mood" in analysis:
            lines.append(f"雰囲気: {analysis['mood']}")

    return "\n".join(lines) if lines else "推奨設定なし"
