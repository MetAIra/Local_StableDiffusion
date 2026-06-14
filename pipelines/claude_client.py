"""Claude API クライアント共有モジュール

image_to_text / text_to_keywords / variable_prompt_generator で重複していた
Claude API クライアント初期化処理を一元化する。
"""
import os


def get_claude_client() -> "tuple[anthropic.Anthropic | None, str]":
    """Claude API クライアントを生成する

    Returns:
        (client, error_message):
            成功時は (anthropic.Anthropic, "")、失敗時は (None, エラーメッセージ)
    """
    # 環境変数からAPIキーを取得
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, (
            "ANTHROPIC_API_KEY が設定されていません。\n\n"
            "設定方法:\n"
            "1. .env ファイルを作成（推奨）:\n"
            "   .env.example を .env にコピーして編集\n"
            "   ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "2. または環境変数で設定:\n"
            "   Windows: $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "   Linux/Mac: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key), ""
    except ImportError:
        return None, (
            "anthropic パッケージがインストールされていません。\n"
            "インストール: pip install anthropic"
        )
    except Exception as e:
        return None, f"Claude API クライアント初期化エラー: {str(e)}"
