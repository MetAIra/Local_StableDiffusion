# Local Stable Diffusion WebUI

Stable Diffusion画像生成アプリケーション（Gradio WebUI）

## 機能一覧

| 機能 | 説明 |
|------|------|
| Text to Image | テキストから画像生成 |
| Image to Image | 画像を参考に新しい画像生成 |
| Inpainting | マスクで指定した部分を再生成 |
| Outpainting | 画像の外側を拡張 |
| ControlNet | ポーズ・エッジ・深度マップで構図制御 |
| Multi ControlNet | 複数のControlNetを同時使用 |
| IP-Adapter | 参照画像から顔・スタイルを保持 |
| Upscale | Real-ESRGAN + SD Upscalerで高解像度化 |
| Face Restore | GFPGAN/CodeFormerで顔を補正 |
| Background Remove | 背景除去・人物切り抜き |
| X/Y/Z Plot | パラメータ比較グリッド生成 |
| Variable Prompt | 変数テンプレートで大量バッチ生成 |
| Image to Text | Claude API + CLIPで画像からプロンプト生成 |
| Multi-View | Zero123++で多視点画像生成 |

## 対応モデル

- **SD 1.5系**: 全機能対応
- **SDXL系**: 主要機能対応（Pony, Illustrious, NoobAI等）

---

## Google Colabでの実行方法

### 必要なもの

- Googleアカウント
- Google Drive（モデルファイル保存用）
- Stable Diffusionモデルファイル（.safetensors形式）

### 手順

#### 1. モデルを用意（各自で）

このアプリは **モデル（.safetensors）を同梱しません**。各自で用意してください。方法は2つ:

- **A. ノートブック内で URL からダウンロード**（Drive 不要・最も手軽）
- **B. 自分の Google Drive に置いてコピー** — その場合は以下のフォルダ構造で配置：

```
MyDrive/
└── SD_models/          # 任意の名前でOK
    ├── model/          # ベースモデル (.safetensors)
    ├── VAE/            # VAEファイル（オプション）
    ├── LoRA/           # LoRAファイル（オプション）
    └── Negative/       # Negative Embedding（オプション）
```

#### 2. Colabノートブックを開く

以下のリンクからノートブックを開く：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MetAIra/Local_StableDiffusion/blob/master/colab/StableDiffusion_Colab.ipynb)

> ※ リポジトリを **public** にしないと、このバッジからの Colab 起動（clone）は失敗します（後述の「配布する人向け」を参照）。
> `colab/StableDiffusion_Colab.ipynb` をダウンロードして Colab にアップロードしても使えます。

#### 3. GPUランタイムを設定

1. メニューから **ランタイム** → **ランタイムのタイプを変更** を選択
2. **ハードウェア アクセラレータ** を **GPU** に変更
3. **保存** をクリック

#### 4. セルを順番に実行

| セル | 内容 |
|------|------|
| 1. GPU確認 | GPUが認識されているか確認 |
| 2. Driveマウント | （任意）モデルのコピーや出力保存に使う |
| 3. コード取得 | 公開リポジトリから `git clone`（`REPO_URL` を編集） |
| 4. 依存インストール | 画像生成に必要なライブラリのみ（数分） |
| 5. モデル用意 | `.safetensors` を URL DL か Drive コピーで配置 |
| 6. 出力保存 | （任意）生成物を Drive に保存 |
| 7. アプリ起動 | WebUIを起動（公開 URL が出る） |

#### 5. モデルを用意（重要）

モデルは Colab のローカル領域 `/content/models/model/` に置きます（高速・Drive の mmap エラー回避のため）。
セル 5 でフォルダが作られるので、次のどちらかで `.safetensors` を入れます:

```python
# A. URL からダウンロード（セル 5-A）
MODEL_URL = "https://huggingface.co/.../model.safetensors"
FILE_NAME = "my_model.safetensors"

# B. 自分の Drive からコピー（セル 5-B。先に Drive をマウント）
DRIVE_MODEL_DIR = "/content/drive/MyDrive/SD_models/model"
```

#### 6. アプリにアクセス

最後のセルを実行すると、以下のようなURLが表示されます：

```
Running on public URL: https://xxxxx.gradio.live
```

このURLをクリックしてWebUIにアクセス。

---

## トラブルシューティング

### モデルが見つからない

```
✗ Model directory not found
```

→ `MODEL_BASE_DIR` のパスが正しいか確認。Google Driveをマウント後、以下で確認：

```python
!ls "/content/drive/MyDrive/"
```

### CUDA out of memory

→ SDXLモデルはVRAMを多く使用。以下を試す：
- 画像サイズを小さくする（512x512など）
- バッチサイズを1にする
- SD1.5系モデルを使用する

### Gradio公開URLが生成されない

→ Colabのセッションを再起動して再実行。

### 依存関係エラー

→ 以下でキャッシュをクリア後、再インストール：

```python
!pip cache purge
```

---

## 配布する人向け（公開リポジトリ化の手順）

このプロジェクトは GitHub の `MetAIra/Local_StableDiffusion`（origin 設定済み）に紐づいています。
他の人（GPU が無い人）が Colab で使えるようにするには、**リポジトリを public にして最新コードを push** します。

```bash
# 1) リポジトリを public に変更
gh repo edit MetAIra/Local_StableDiffusion --visibility public
#   （または GitHub の Settings → General → Danger Zone → Change repository visibility）

# 2) 変更をコミットして push（.gitignore により .env / models/ / output/ は自動除外）
git add .
git commit -m "Colab image-only distribution"
git push
```

push 後、利用者は README 冒頭の「Open in Colab」バッジを押すだけで起動でき、各自のモデルを入れて使えます。
（`colab/StableDiffusion_Colab.ipynb` の `REPO_URL` と冒頭バッジは既に `MetAIra/Local_StableDiffusion` を指しています。）

### ⚠️ セキュリティ（重要）

- `.env`（API キー入り）は **`.gitignore` で除外済み**。公開リポジトリに絶対含めないこと。
- これまでに `.env` を共有・コミットしたことがある場合は、**Anthropic API キーを再発行**してください（[console.anthropic.com](https://console.anthropic.com/)）。
- 利用者が Claude API 機能（プロンプト支援）を使う場合は、各自で `ANTHROPIC_API_KEY` を設定してもらいます（画像生成のみなら不要）。

---

## ローカル実行

### 必要環境

- Python 3.10+
- CUDA対応GPU（VRAM 8GB以上推奨）
- CUDA 11.8 / 12.x

### インストール

```bash
git clone https://github.com/MetAIra/Local_StableDiffusion.git
cd Local_StableDiffusion
pip install -r requirements.txt
```

### 設定

1. `config.py` の `BASE_MODEL_DIR` を自分の環境に合わせて編集
2. または環境変数で設定：

```bash
export BASE_MODEL_DIR="/path/to/your/models"
```

### 起動

```bash
python app.py
```

ブラウザで http://localhost:7860 にアクセス。

---

## ライセンス

MIT License
