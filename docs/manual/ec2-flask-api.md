# EC2 Flask API 運用マニュアル

## 構成概要

```
ローカルPC → port 5000 → nginx → gunicorn (127.0.0.1:8000) → Flask (server.py)
```

| 項目 | 値 |
|------|-----|
| サービス名 | `pokemon-api` |
| アプリディレクトリ | `/home/app_admin/pokemon-api/` |
| gunicorn ポート | `127.0.0.1:8000`（内部のみ） |
| nginx ポート | `5000`（外部公開） |
| nginx 設定ファイル | `/etc/nginx/conf.d/pokemon-api.conf` |
| systemd サービス | `/etc/systemd/system/pokemon-api.service` |
| 認証 | EC2 IAMロール経由（アクセスキー不要） |

---

## 死活確認

```bash
# ヘルスチェック（ローカルから）
curl http://{EC2のElasticIP}:5000/health
# → {"status": "ok", "timestamp": "..."}

# サービスの状態確認（EC2で）
sudo systemctl status pokemon-api

# nginx の状態確認（EC2で）
sudo systemctl status nginx
```

---

## サーバーが落ちていた場合

### 1. どのプロセスが落ちているか確認

```bash
# gunicorn（Flask）の状態
sudo systemctl status pokemon-api

# nginx の状態
sudo systemctl status nginx
```

### 2. Flask（gunicorn）を再起動

```bash
sudo systemctl restart pokemon-api

# 起動したか確認
sudo systemctl status pokemon-api
```

### 3. nginx を再起動

```bash
sudo systemctl restart nginx

# 起動したか確認
sudo systemctl status nginx
```

### 4. 両方まとめて再起動

```bash
sudo systemctl restart pokemon-api nginx
```

---

## ログ確認

```bash
# Flask アプリのログ（直近50行）
sudo journalctl -u pokemon-api -n 50

# リアルタイムでログを流す
sudo journalctl -u pokemon-api -f

# nginx のエラーログ
sudo tail -50 /var/log/nginx/error.log
```

---

## server.py を更新した場合

```bash
# ローカル（WSL2）から転送
scp /mnt/c/Users/rotat/AITuberProject/src/api/server.py \
  app_admin@{EC2のElasticIP}:/home/app_admin/pokemon-api/

# EC2 でサービスを再起動
sudo systemctl restart pokemon-api
```

---

## 設定ファイルの内容確認

```bash
# systemd サービス定義
cat /etc/systemd/system/pokemon-api.service

# nginx リバースプロキシ設定
cat /etc/nginx/conf.d/pokemon-api.conf

# nginx 設定の文法チェック
sudo nginx -t
```

---

## 環境変数の変更（S3バケット名など）

`/etc/systemd/system/pokemon-api.service` を編集して再起動する。

```bash
sudo vi /etc/systemd/system/pokemon-api.service
# Environment="S3_BUCKET=新しいバケット名" を編集

sudo systemctl daemon-reload
sudo systemctl restart pokemon-api
```

---

## 自動起動の確認

EC2 再起動後も自動で立ち上がるよう設定済み。確認するには：

```bash
sudo systemctl is-enabled pokemon-api
# → enabled が返ればOK
```
