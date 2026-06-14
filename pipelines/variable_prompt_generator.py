"""Variable Prompt Generator パイプライン

Claude API を使用して日本語のテキスト入力から
Variable Prompt用のプロンプト一式（固定プロンプト、変数テンプレート、変数定義、ネガティブプロンプト）を生成する。
"""
from config import (
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_VPG_SYSTEM_PROMPT,
)
from .claude_client import get_claude_client


class VariablePromptGenerator:
    """Claude API を使用したVariable Prompt生成クラス"""

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
        description: str,
        style: str = "auto",
        additional_instructions: str = "",
    ) -> tuple[str, str, str, str, str]:
        """
        説明テキストからVariable Prompt一式を生成

        Args:
            description: 日本語のバッチ生成要件説明
            style: スタイル (auto/anime/realistic/artistic/fantasy/cyberpunk)
            additional_instructions: 追加の指示（任意）

        Returns:
            (fixed_prompt, variable_template, variable_definitions, negative_prompt, status_message)
        """
        if not description or not description.strip():
            return "", "", "", "", "説明テキストを入力してください"

        # 初期化チェック
        ok, error_msg = self._ensure_initialized()
        if not ok:
            return "", "", "", "", error_msg

        # スタイルヒント
        style_hint = ""
        if style != "auto":
            style_labels = {
                "anime": "アニメ・イラストスタイル（1girl, 1boy等のタグを使用）",
                "realistic": "写真・実写スタイル（photorealistic等を使用）",
                "artistic": "アート・絵画スタイル（oil painting等を使用）",
                "fantasy": "ファンタジースタイル（magical, ethereal等を使用）",
                "cyberpunk": "サイバーパンクスタイル（neon, futuristic等を使用）",
            }
            style_hint = f"\nスタイル指定: {style_labels.get(style, style)}"

        # ユーザープロンプトを構築
        user_prompt = f"""以下の要件に基づいて、Variable Prompt用のプロンプト一式を生成してください。

要件:
{description}
{style_hint}"""

        if additional_instructions.strip():
            user_prompt += f"""

追加指示:
{additional_instructions}"""

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=CLAUDE_VPG_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
            )

            response_text = response.content[0].text.strip()

            # レスポンスをパース
            fixed_prompt = ""
            variable_template = ""
            variable_definitions = ""
            negative_prompt = ""

            sections = {
                "[FIXED_PROMPT]": "fixed_prompt",
                "[VARIABLE_TEMPLATE]": "variable_template",
                "[VARIABLE_DEFINITIONS]": "variable_definitions",
                "[NEGATIVE_PROMPT]": "negative_prompt",
            }

            # セクションごとにパース
            current_section = None
            current_content = []

            for line in response_text.split("\n"):
                stripped = line.strip()
                if stripped in sections:
                    # 前のセクションを保存
                    if current_section:
                        content = "\n".join(current_content).strip()
                        if current_section == "fixed_prompt":
                            fixed_prompt = content
                        elif current_section == "variable_template":
                            variable_template = content
                        elif current_section == "variable_definitions":
                            variable_definitions = content
                        elif current_section == "negative_prompt":
                            negative_prompt = content
                    current_section = sections[stripped]
                    current_content = []
                elif current_section:
                    current_content.append(line)

            # 最後のセクションを保存
            if current_section:
                content = "\n".join(current_content).strip()
                if current_section == "fixed_prompt":
                    fixed_prompt = content
                elif current_section == "variable_template":
                    variable_template = content
                elif current_section == "variable_definitions":
                    variable_definitions = content
                elif current_section == "negative_prompt":
                    negative_prompt = content

            # 変数定義から組み合わせ数を計算
            num_combos = 1
            var_count = 0
            for line in variable_definitions.split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                _, values_str = line.split(":", 1)
                values = [v.strip() for v in values_str.split(",") if v.strip()]
                if values:
                    num_combos *= len(values)
                    var_count += 1

            status = (
                f"Variable Prompt 生成完了\n"
                f"変数数: {var_count}\n"
                f"組み合わせ数: {num_combos}通り"
            )

            return fixed_prompt, variable_template, variable_definitions, negative_prompt, status

        except Exception as e:
            import traceback
            return "", "", "", "", f"Claude API エラー: {str(e)}\n{traceback.format_exc()}"


# シングルトンインスタンス
vpg_generator = VariablePromptGenerator()


def generate_variable_prompt_template(
    description: str,
    style: str = "auto",
    additional_instructions: str = "",
) -> tuple[str, str, str, str, str]:
    """
    Variable Prompt用テンプレートを生成（UIから呼び出される関数）

    Args:
        description: 日本語のバッチ生成要件説明
        style: スタイル選択
        additional_instructions: 追加の指示

    Returns:
        (fixed_prompt, variable_template, variable_definitions, negative_prompt, status_message)
    """
    return vpg_generator.generate(
        description=description,
        style=style,
        additional_instructions=additional_instructions,
    )
