"""Stable Diffusion 画像生成アプリケーション

機能:
- Text to Image (txt2img)
- Image to Image (img2img)
- Inpainting (マスク描画で部分再生成)
- Outpainting
- ControlNet (ポーズ・エッジ・深度マップ・線画・落書き・タイルで構図制御)
- Multi ControlNet (複数ControlNet同時使用)
- IP-Adapter (参考画像から顔・スタイルを保持して生成)
- Upscaler (Real-ESRGAN + SD Upscaler)
- Face Restore (GFPGAN / CodeFormer で顔を自動補正)
- Background Remove (rembg で背景除去・人物切り抜き)
- X/Y/Z Plot (同じseedでパラメータを変えながら比較画像を生成)
- Variable Prompt (変数プロンプトで大量の画像をバッチ生成)
- Model Comparison (複数モデルを最適パラメータで比較生成)
- Image to Text (Claude API + CLIP Interrogatorで画像からプロンプト生成)

音声機能:
- Text to Speech (Bark / XTTS v2)
- Music Generation (MusicGen / AudioLDM2)
- Voice Conversion (RVC)

動画機能:
- Video Generation (AnimateDiff / SVD-XT / LTX-Video / CogVideoX-2B / Wan2.1)

機能群の切り替え（環境変数 SD_FEATURE_SET）:
- 未設定 / "all": すべての機能を表示（ローカル実機の既定）
- "image" / "image-only": 画像生成コアのみ（Colab 画像専用配布の既定）
- カンマ区切り（例 "image,video"）: コア + 指定したオプション機能群
画像生成コア（画像生成 / 構図制御 / 画像編集 / バッチ生成 / モデル情報）は常に表示。
オプション機能群（プロンプト支援 / 動画生成 / 音声生成）は依存関係が無い環境では
自動的に案内表示へフォールバックし、アプリ全体は停止しない。
"""
import os

import gradio as gr

# --- 画像生成コア（常に利用可能。標準的な SD 依存のみで動く） ---
from ui.txt2img_tab import create_txt2img_tab
from ui.img2img_tab import create_img2img_tab
from ui.inpaint_tab import create_inpaint_tab
from ui.outpaint_tab import create_outpaint_tab
from ui.controlnet_tab import create_controlnet_tab, create_multi_controlnet_tab
from ui.ip_adapter_tab import create_ip_adapter_tab
from ui.upscale_tab import create_upscale_tab
from ui.face_restore_tab import create_face_restore_tab
from ui.background_remove_tab import create_background_remove_tab
from ui.xyz_plot_tab import create_xyz_plot_tab
from ui.multiview_tab import create_multiview_tab
from ui.variable_prompt_tab import create_variable_prompt_tab
from ui.model_comparison_tab import create_model_comparison_tab
from ui.model_catalog_tab import create_model_catalog_tab


def _enabled_optional_groups() -> set[str]:
    """環境変数 SD_FEATURE_SET から有効なオプション機能群を決定する。

    画像生成コアは常に有効。戻り値は {"prompt", "video", "audio"} の部分集合。
    """
    raw = os.environ.get("SD_FEATURE_SET", "all").strip().lower()
    optional = {"prompt", "video", "audio"}
    if raw in ("", "all", "full"):
        return optional
    if raw in ("image", "image_only", "image-only", "img", "core"):
        return set()
    tokens = {t.strip() for t in raw.split(",") if t.strip()}
    return tokens & optional


def _unavailable_notice(title: str, detail: str) -> None:
    """依存関係が不足してタブを構築できない場合の案内表示。"""
    gr.Markdown(
        f"### ⚠️ {title}は利用できません\n\n"
        "この機能に必要な依存関係がインストールされていません。\n"
        "（Colab の画像専用構成では既定で無効です）\n\n"
        f"```\n{detail}\n```"
    )


def _build_prompt_support_group() -> None:
    """プロンプト支援（Claude API 系）。anthropic 等が無ければ案内表示にフォールバック。"""
    with gr.TabItem("プロンプト支援"):
        try:
            from ui.image_to_text_tab import create_image_to_text_tab
            from ui.text_to_keywords_tab import create_text_to_keywords_tab
            from ui.variable_prompt_generator_tab import create_variable_prompt_generator_tab

            with gr.Tabs():
                with gr.TabItem("Image to Text"):
                    create_image_to_text_tab()

                with gr.TabItem("Text to Keywords"):
                    create_text_to_keywords_tab()

                with gr.TabItem("Variable Prompt Generator"):
                    create_variable_prompt_generator_tab()
        except Exception as e:  # noqa: BLE001 - 依存欠如時も他機能を止めない
            _unavailable_notice("プロンプト支援機能", str(e))


def _build_video_group() -> None:
    """動画生成。diffusers の動画パイプラインや imageio 等が無ければ案内表示にフォールバック。"""
    with gr.TabItem("動画生成"):
        try:
            from ui.video_gen_tab import create_video_gen_tab
            from ui.video_batch_tab import create_video_batch_tab
            from ui.video_chain_tab import create_video_chain_tab

            with gr.Tabs():
                with gr.TabItem("単発生成"):
                    create_video_gen_tab()
                with gr.TabItem("バッチ生成"):
                    create_video_batch_tab()
                with gr.TabItem("連鎖生成 (長尺)"):
                    create_video_chain_tab()
        except Exception as e:  # noqa: BLE001
            _unavailable_notice("動画生成機能", str(e))


def _build_audio_group() -> None:
    """音声生成。TTS / audiocraft / fairseq 等が無ければ案内表示にフォールバック。"""
    with gr.TabItem("音声生成"):
        try:
            from ui.tts_tab import create_tts_tab
            from ui.music_gen_tab import create_music_gen_tab
            from ui.music_batch_tab import create_music_batch_tab
            from ui.voice_conversion_tab import create_voice_conversion_tab

            with gr.Tabs():
                with gr.TabItem("Text to Speech"):
                    create_tts_tab()

                with gr.TabItem("Music Generation"):
                    create_music_gen_tab()

                with gr.TabItem("Music Batch"):
                    create_music_batch_tab()

                with gr.TabItem("Voice Conversion"):
                    create_voice_conversion_tab()
        except Exception as e:  # noqa: BLE001
            _unavailable_notice("音声生成機能", str(e))


def create_app():
    """Gradioアプリケーションを作成"""
    enabled = _enabled_optional_groups()

    with gr.Blocks(title="Stable Diffusion Image Generator") as demo:
        gr.Markdown("# Stable Diffusion 画像生成")
        gr.Markdown("bluepencil.py方式 - DDIMScheduler + 高品質設定")

        with gr.Tabs():
            # =============================================
            # 画像生成
            # =============================================
            with gr.TabItem("画像生成"):
                with gr.Tabs():
                    with gr.TabItem("Text to Image"):
                        create_txt2img_tab()

                    with gr.TabItem("Image to Image"):
                        create_img2img_tab()

                    with gr.TabItem("Multi-View Generation"):
                        create_multiview_tab()

            # =============================================
            # 構図制御
            # =============================================
            with gr.TabItem("構図制御"):
                with gr.Tabs():
                    with gr.TabItem("ControlNet"):
                        create_controlnet_tab()

                    with gr.TabItem("Multi ControlNet"):
                        create_multi_controlnet_tab()

                    with gr.TabItem("IP-Adapter"):
                        create_ip_adapter_tab()

            # =============================================
            # 画像編集
            # =============================================
            with gr.TabItem("画像編集"):
                with gr.Tabs():
                    with gr.TabItem("Inpainting"):
                        create_inpaint_tab()

                    with gr.TabItem("Outpainting"):
                        create_outpaint_tab()

                    with gr.TabItem("Upscale"):
                        create_upscale_tab()

                    with gr.TabItem("Face Restore"):
                        create_face_restore_tab()

                    with gr.TabItem("Background Remove"):
                        create_background_remove_tab()

            # =============================================
            # バッチ生成
            # =============================================
            with gr.TabItem("バッチ生成"):
                with gr.Tabs():
                    with gr.TabItem("X/Y/Z Plot"):
                        create_xyz_plot_tab()

                    with gr.TabItem("Variable Prompt"):
                        create_variable_prompt_tab()

                    with gr.TabItem("モデル比較"):
                        create_model_comparison_tab()

            # =============================================
            # プロンプト支援（オプション: Claude API 系）
            # =============================================
            if "prompt" in enabled:
                _build_prompt_support_group()

            # =============================================
            # モデル情報
            # =============================================
            with gr.TabItem("モデル情報"):
                create_model_catalog_tab()

            # =============================================
            # 動画生成（オプション）
            # =============================================
            if "video" in enabled:
                _build_video_group()

            # =============================================
            # 音声生成（オプション）
            # =============================================
            if "audio" in enabled:
                _build_audio_group()

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch()
