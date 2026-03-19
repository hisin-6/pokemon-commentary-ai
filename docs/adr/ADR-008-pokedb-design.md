# ADR-008: PokeDB 設計方針（Sprint 6）

## ステータス
承認済み（2026-03-19）

## コンテキスト

Sprint 6 でポケモン・技・特性・アイテムのデータベース（PokeDB）を構築し、
OCR テキストの分類精度向上と Bedrock プロンプトへの RAG 組み込みを行う。

以下の要件を満たす必要がある：
- OCR で読み取ったテキストを「ポケモン名 / 技名 / 特性名 / アイテム名 / 不明」に分類する
- 日本語・英語・中国語の名前を収録する（英語・中国語表記のポケモン名が対戦に出るため）
- EasyOCR の誤読（例: 「エルフー」→「エルフーン」）を曖昧マッチングで補正する
- 確定したポケモン名から タイプ・特性・代表技を取得して Bedrock プロンプトに渡す（RAG）
- 将来的に対戦ログの保存も行う予定がある

## 決定内容

### DB エンジン: SQLite（現時点）

- データソース: PokeAPI（無料 JSON API）
- 対象世代: **第9世代（SV）のみ**（近日リリース予定の対戦特化タイトルに合わせて随時更新）
- 曖昧マッチング: **rapidfuzz**（Python ライブラリ）で実装
- DB ファイル構成:

```
data/
├── pokedb.sqlite       # PokeAPI 由来の静的データ（読み取り専用）
└── battle_log.sqlite   # 対戦ログ（将来的に分離・移行予定）
```

### テーブル構成

```sql
-- ポケモン本体
CREATE TABLE pokemon (
    id             INTEGER PRIMARY KEY,
    name_ja        TEXT,
    name_en        TEXT,
    name_zh        TEXT,
    type1          TEXT,
    type2          TEXT,   -- NULL の場合は単タイプ
    ability1       TEXT,
    ability2       TEXT,
    ability_hidden TEXT
);

-- 技
CREATE TABLE moves (
    id       INTEGER PRIMARY KEY,
    name_ja  TEXT,
    name_en  TEXT,
    type     TEXT,
    category TEXT,   -- 物理 / 特殊 / 変化
    power    INTEGER,
    accuracy INTEGER
);

-- 特性
CREATE TABLE abilities (
    id      INTEGER PRIMARY KEY,
    name_ja TEXT,
    name_en TEXT
);

-- どうぐ
CREATE TABLE items (
    id      INTEGER PRIMARY KEY,
    name_ja TEXT,
    name_en TEXT
);

-- ポケモンが覚える技（RAG 用）
CREATE TABLE pokemon_moves (
    pokemon_id INTEGER,
    move_id    INTEGER,
    PRIMARY KEY (pokemon_id, move_id)
);
```

### 曖昧マッチング閾値

| スコア | 扱い |
|---|---|
| 95 以上 | 確定採用 |
| 80〜94 | 候補として採用（低信頼度マーク） |
| 80 未満 | 不明扱い（除外） |

## 検討した代替案

### PostgreSQL

**メリット**:
- pg_trgm 拡張による強力なサーバーサイド曖昧検索
- 複数プロセスからの同時書き込みに強い
- Web ダッシュボードや複雑な集計クエリに適している

**デメリット**:
- DB サーバーのインストール・起動管理が必要
- 第9世代のみ・1プロセス書き込みの現スコープではオーバースペック

## 将来の移行方針

**以下の条件が揃った場合は PostgreSQL への移行を推奨する:**

- 対戦ログを Web ブラウザでグラフ・統計として可視化したい
- 複数のスクリプト・ツールが同時にログを書き込む構成になった
- 「このポケモンの勝率」「この構築への対戦回数」など複雑な分析クエリが必要になった

**移行コストを最小化するため、`BattleLogger` クラスを薄いラッパーとして設計し、
バックエンド（SQLite / PostgreSQL）を差し替えられるインターフェースにしておくこと。**

## 影響範囲

- `scripts/build_pokedb.py`: DB ビルドスクリプト（新規）
- `src/pokedb/classifier.py`: OCR テキスト分類モジュール（新規）
- `src/pipeline.py`: `_extract_structured_info` を DB 分類に置き換え
- `src/api/server.py`: RAG セクションをプロンプトに追加
