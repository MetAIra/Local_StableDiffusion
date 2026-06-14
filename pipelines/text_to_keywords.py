"""Text to Keywords パイプライン

Claude API を使用して日本語のテキスト入力から
Stable Diffusion 向けの英語キーワードを生成する。
"""
from config import (
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_T2K_SYSTEM_PROMPT,
    T2K_WORD_COUNTS,
    T2K_DETAIL_LEVELS,
)
from .claude_client import get_claude_client


class TextToKeywordsGenerator:
    """Claude API を使用したキーワード生成クラス"""

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

    def generate(
        self,
        text_input: str,
        style: str = "auto",
        word_count: int = 25,
        detail_level: str = "normal",
        include_negative: bool = True,
    ) -> tuple[str, str, str]:
        """
        テキストからSD向けキーワードを生成

        Args:
            text_input: 日本語の画像イメージ説明
            style: スタイル (auto/anime/realistic/artistic/fantasy/cyberpunk)
            word_count: 目標単語数
            detail_level: 詳細度 (simple/normal/detailed)
            include_negative: ネガティブプロンプトも生成するか

        Returns:
            (positive_keywords, negative_keywords, status_message)
        """
        if not text_input or not text_input.strip():
            return "", "", "テキストを入力してください"

        # 初期化チェック
        ok, error_msg = self._ensure_initialized()
        if not ok:
            return "", "", error_msg

        # スタイルの説明を取得
        style_hint = ""
        if style != "auto":
            style_labels = {
                "anime": "アニメ・イラストスタイル",
                "realistic": "写真・実写スタイル",
                "artistic": "アート・絵画スタイル",
                "fantasy": "ファンタジースタイル",
                "cyberpunk": "サイバーパンクスタイル",
            }
            style_hint = f"\nスタイル指定: {style_labels.get(style, style)}"

        # 詳細度の説明
        detail_hints = {
            "simple": "シンプルで簡潔なキーワード",
            "normal": "標準的な詳細度",
            "detailed": "非常に詳細で具体的なキーワード",
        }
        detail_hint = detail_hints.get(detail_level, "標準的な詳細度")

        # ユーザープロンプトを構築
        user_prompt = f"""以下の画像イメージを、Stable Diffusion用の英語キーワードに変換してください。

画像イメージ:
{text_input}

設定:
- 目標単語数: 約{word_count}語{style_hint}
- 詳細度: {detail_hint}
"""

        if include_negative:
            user_prompt += """
出力形式:
[POSITIVE]
（ポジティブプロンプトのキーワードをカンマ区切りで）

[NEGATIVE]
（ネガティブプロンプトのキーワードをカンマ区切りで）
"""
        else:
            user_prompt += """
キーワードをカンマ区切りで出力してください。他の説明は不要です。
"""

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=CLAUDE_T2K_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
            )

            response_text = response.content[0].text.strip()

            # レスポンスをパース
            positive_keywords = ""
            negative_keywords = ""

            if include_negative and "[POSITIVE]" in response_text:
                # フォーマット付きレスポンスをパース
                parts = response_text.split("[NEGATIVE]")
                if len(parts) >= 2:
                    positive_part = parts[0].replace("[POSITIVE]", "").strip()
                    negative_part = parts[1].strip()
                    positive_keywords = positive_part
                    negative_keywords = negative_part
                else:
                    positive_keywords = parts[0].replace("[POSITIVE]", "").strip()
            else:
                # 単純なキーワード出力
                positive_keywords = response_text

            # クリーンアップ
            positive_keywords = self._clean_keywords(positive_keywords)
            negative_keywords = self._clean_keywords(negative_keywords)

            # 単語数をカウント
            actual_count = len([k for k in positive_keywords.split(",") if k.strip()])

            status = f"キーワード生成完了（{actual_count}語）"
            return positive_keywords, negative_keywords, status

        except Exception as e:
            import traceback
            return "", "", f"Claude API エラー: {str(e)}\n{traceback.format_exc()}"

    def _clean_keywords(self, keywords: str) -> str:
        """キーワード文字列をクリーンアップ"""
        if not keywords:
            return ""

        # 改行をカンマに変換
        keywords = keywords.replace("\n", ", ")

        # 連続するカンマを1つに
        while ",," in keywords:
            keywords = keywords.replace(",,", ",")

        # 前後の空白とカンマを除去
        keywords = keywords.strip().strip(",").strip()

        # 各キーワードをトリム
        parts = [p.strip() for p in keywords.split(",") if p.strip()]

        return ", ".join(parts)


# シングルトンインスタンス
t2k_generator = TextToKeywordsGenerator()


def generate_keywords_from_text(
    text_input: str,
    style: str = "auto",
    word_count_preset: str = "標準（20-30語）",
    detail_level: str = "標準",
    include_negative: bool = True,
) -> tuple[str, str, str]:
    """
    テキストからSD向けキーワードを生成（UIから呼び出される関数）

    Args:
        text_input: 日本語の画像イメージ説明
        style: スタイル選択
        word_count_preset: 単語数プリセット
        detail_level: 詳細度
        include_negative: ネガティブプロンプトも生成

    Returns:
        (positive_keywords, negative_keywords, status_message)
    """
    # プリセットから実際の値を取得
    word_count = T2K_WORD_COUNTS.get(word_count_preset, 25)
    detail = T2K_DETAIL_LEVELS.get(detail_level, "normal")

    return t2k_generator.generate(
        text_input=text_input,
        style=style,
        word_count=word_count,
        detail_level=detail,
        include_negative=include_negative,
    )
