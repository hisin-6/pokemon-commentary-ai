# Ollama セットアップガイド

## Ollamaとは

ローカルPCでLLM（大規模言語モデル）を動かすためのツール。
このプロジェクトでは **Phi-3 mini 4bit量子化** をOllama経由で実行し、ポケモン対戦の実況文を生成する。

- 公式サイト: https://ollama.com
- ライセンス: MIT
- 対応OS: Windows / macOS / Linux

---

## インストール

### Windowsの場合

1. https://ollama.com/download/windows からインストーラーをダウンロード
2. インストーラーを実行
3. PowerShellを**新しく開き直して**から確認：

```powershell
ollama --version
# ollama version is 0.x.x が出ればOK
```

> インストール直後は古いPowerShellにPATHが反映されていないので、必ず再起動すること。

---

## Phi-3 mini のダウンロード

```powershell
ollama pull phi3:mini
```

- サイズ: 約2GB
- VRAM使用量: 約2〜3GB（RTX 3080で動作確認済み）

ダウンロード確認：

```powershell
ollama list
# phi3:mini が表示されればOK
```

---

## 動作確認

```powershell
# 対話モードで起動（Ctrl+D で終了）
ollama run phi3:mini

# 一発で出力確認
ollama run phi3:mini "ポケモン対戦を1文で実況してください"
```

---

## APIサーバーとして使う（Python連携）

Ollamaはローカルに HTTPサーバーを立てる。起動後は以下のエンドポイントが使える：

| エンドポイント | 説明 |
|-------------|------|
| `POST http://localhost:11434/api/generate` | テキスト生成 |
| `GET http://localhost:11434/api/tags` | インストール済みモデル一覧 |

Pythonからの呼び出し例（`src/commentary/phi3_client.py` 参照）：

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3:mini",
        "prompt": "ガブリアスのじしんがサーフゴーに命中！実況してください。",
        "stream": False,
    }
)
print(response.json()["response"])
```

---

## VRAM配分（RTX 3080 / 10GB）

| コンポーネント | VRAM |
|-------------|------|
| Phi-3 mini 4bit | 2〜3 GB |
| VTube Studio（Sprint 3以降） | 4〜5 GB |
| YOLO + EasyOCR | 0.5〜1.5 GB |
| **合計（最大）** | **9.5 GB以内** |

---

## トラブルシューティング

| 症状 | 対処 |
|-----|------|
| `ollama` が認識されない | PowerShellを開き直す |
| モデルのダウンロードが遅い | 2GB弱あるので数分かかる。そのまま待つ |
| VRAM不足でOOM | `ollama run phi3:mini --num-ctx 1024` でコンテキスト長を短縮 |
| Ollamaサーバーが起動しない | タスクトレイにOllamaのアイコンがあるか確認 |
