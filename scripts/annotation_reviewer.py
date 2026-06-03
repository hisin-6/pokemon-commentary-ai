#!/usr/bin/env python3
"""
アノテーションレビューツール
使い方: venv/bin/python scripts/annotation_reviewer.py
ブラウザで http://localhost:5001 を開く

キーボードショートカット:
  →  / L    次へ（確認済みにして進む）
  ←  / H    前へ
  D         削除フラグ（LMDB作成時にスキップ）
  Enter     ラベルを保存して次へ
  Escape    編集キャンセル
"""

import json
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)

CROPS_DIR = Path(__file__).parent.parent / "data/ocr_crops/text_crops"
ANNO_FILE = Path(__file__).parent.parent / "data/ocr_crops/annotations.json"

annotations: list = []


def load_annotations():
    global annotations
    with open(ANNO_FILE, encoding="utf-8") as f:
        annotations = json.load(f)


def save_annotations():
    with open(ANNO_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


load_annotations()

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>アノテーションレビュー</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0f0f1a;
  color: #e0e0f0;
  font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ---- ヘッダー ---- */
header {
  background: #1a1a30;
  border-bottom: 1px solid #333;
  padding: 10px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
header h1 { font-size: 15px; color: #9090cc; white-space: nowrap; }

.filter-group { display: flex; gap: 6px; }
.filter-btn {
  padding: 5px 12px;
  border: 1px solid #444;
  border-radius: 20px;
  background: #222238;
  color: #aaa;
  cursor: pointer;
  font-size: 12px;
  transition: all .15s;
}
.filter-btn:hover { border-color: #7070cc; color: #ddd; }
.filter-btn.active { background: #3a3a7a; border-color: #7070cc; color: #fff; }

.progress-wrap { flex: 1; min-width: 120px; }
.progress-bar {
  height: 6px;
  background: #333;
  border-radius: 3px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #5050cc, #9050cc);
  transition: width .3s;
}
.progress-text { font-size: 11px; color: #888; margin-top: 3px; }

/* ---- メイン ---- */
main {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  gap: 40px;
  overflow: hidden;
}

/* ---- 画像パネル ---- */
.img-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}
.img-wrap {
  background: #1a1a30;
  border: 2px solid #333;
  border-radius: 8px;
  padding: 16px;
  min-width: 200px;
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
}
#crop-img {
  image-rendering: pixelated;
  image-rendering: -moz-crisp-edges;
  max-width: 100%;
}
.img-filename { font-size: 10px; color: #555; max-width: 280px; text-align: center; word-break: break-all; }

/* ---- 情報パネル ---- */
.info-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 320px;
  max-width: 400px;
}

.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
}
.badge-corrected { background: #4a2060; color: #cc88ff; border: 1px solid #7040a0; }
.badge-low  { background: #60200a; color: #ff9060; border: 1px solid #a04020; }
.badge-mid  { background: #503a00; color: #ffcc60; border: 1px solid #907020; }
.badge-high { background: #0a4020; color: #60ff90; border: 1px solid #209040; }
.badge-reviewed { background: #1a3050; color: #60aaff; border: 1px solid #2060a0; }
.badge-deleted  { background: #501a1a; color: #ff6060; border: 1px solid #a02020; }

.label-section { display: flex; flex-direction: column; gap: 8px; }
.label-section label { font-size: 12px; color: #888; }
#label-input {
  background: #1a1a30;
  border: 2px solid #4a4a80;
  border-radius: 6px;
  color: #fff;
  font-size: 22px;
  padding: 10px 14px;
  outline: none;
  width: 100%;
  transition: border-color .15s;
}
#label-input:focus { border-color: #7070cc; }
#label-input.changed { border-color: #cc8830; }

.conf-row { font-size: 12px; color: #666; }
.conf-val { color: #aaa; font-weight: bold; }

/* ---- ボタン群 ---- */
.btn-row { display: flex; gap: 10px; }
.btn {
  flex: 1;
  padding: 10px;
  border: 1px solid #444;
  border-radius: 6px;
  background: #222238;
  color: #ccc;
  cursor: pointer;
  font-size: 13px;
  transition: all .15s;
}
.btn:hover { background: #2a2a50; border-color: #666; }
.btn-save  { background: #1a3a1a; border-color: #3a7a3a; color: #80ff80; }
.btn-save:hover  { background: #1e4a1e; }
.btn-delete { background: #3a1a1a; border-color: #7a3a3a; color: #ff8080; }
.btn-delete:hover { background: #4a2020; }
.btn-prev, .btn-next { background: #1a1a3a; border-color: #3a3a7a; color: #8080ff; }
.btn-prev:hover, .btn-next:hover { background: #222260; }

/* ---- ショートカットヒント ---- */
.shortcuts {
  font-size: 11px;
  color: #444;
  line-height: 1.8;
}
.shortcuts kbd {
  background: #2a2a40;
  border: 1px solid #555;
  border-radius: 3px;
  padding: 1px 5px;
  font-family: monospace;
  color: #aaa;
  font-size: 10px;
}

/* ---- 空状態 ---- */
#empty-msg {
  display: none;
  text-align: center;
  color: #556;
  font-size: 18px;
}
</style>
</head>
<body>

<header>
  <h1>🔍 アノテーションレビュー</h1>
  <div class="filter-group">
    <button class="filter-btn active" data-filter="corrected">自動修正 <span id="cnt-corrected"></span></button>
    <button class="filter-btn" data-filter="low_conf">低信頼 <span id="cnt-low"></span></button>
    <button class="filter-btn" data-filter="mid_conf">中信頼 <span id="cnt-mid"></span></button>
    <button class="filter-btn" data-filter="all">全件 <span id="cnt-all"></span></button>
  </div>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;color:#aaa;white-space:nowrap;">
    <input type="checkbox" id="chk-unreviewed" style="accent-color:#7070cc;width:14px;height:14px;">
    未確認のみ
  </label>
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="progress-text" id="prog-text">-</div>
  </div>
  <span id="save-indicator" style="font-size:12px;color:#666;"></span>
</header>

<main>
  <div id="empty-msg">このフィルターに該当するアイテムがありません 🎉</div>

  <div class="img-panel" id="img-panel">
    <div class="img-wrap">
      <img id="crop-img" src="" alt="crop">
    </div>
    <div class="img-filename" id="img-filename"></div>
  </div>

  <div class="info-panel" id="info-panel">
    <div class="badge-row" id="badge-row"></div>

    <div class="label-section">
      <label>ラベル（編集可）</label>
      <input id="label-input" type="text" autocomplete="off" spellcheck="false">
    </div>

    <div class="conf-row">
      信頼度: <span class="conf-val" id="conf-val">-</span>
    </div>

    <div class="btn-row">
      <button class="btn btn-prev" id="btn-prev">← 戻る</button>
      <button class="btn btn-save" id="btn-save">✓ 保存して次へ</button>
      <button class="btn btn-next" id="btn-next">次へ →</button>
    </div>
    <div class="btn-row">
      <button class="btn btn-delete" id="btn-delete">🗑 削除フラグ</button>
    </div>

    <div class="shortcuts">
      <kbd>→</kbd> 次へ（確認済み） &nbsp;
      <kbd>←</kbd> 戻る &nbsp;
      <kbd>Enter</kbd> 保存して次へ &nbsp;
      <kbd>D</kbd> 削除フラグ
    </div>
  </div>
</main>

<script>
let items = [];
let cursor = 0;
let filter = 'corrected';
let originalLabel = '';

async function fetchCounts() {
  const filters = ['corrected','low_conf','mid_conf','all'];
  const ids = ['cnt-corrected','cnt-low','cnt-mid','cnt-all'];
  for (let i = 0; i < filters.length; i++) {
    const r = await fetch(`/api/items?filter=${filters[i]}`);
    const d = await r.json();
    const unreviewedCount = d.filter(x => !x.reviewed && !x.deleted).length;
    const totalCount = d.length;
    document.getElementById(ids[i]).textContent =
      unreviewedCount < totalCount
        ? `(${unreviewedCount}/${totalCount})`
        : `(${totalCount})`;
  }
}

async function loadItems() {
  const unreviewed = document.getElementById('chk-unreviewed').checked ? '1' : '0';
  const r = await fetch(`/api/items?filter=${filter}&unreviewed=${unreviewed}`);
  items = await r.json();
  cursor = 0;
  // 未レビューの最初のものにジャンプ
  const firstUnreviewed = items.findIndex(x => !x.reviewed && !x.deleted);
  if (firstUnreviewed >= 0) cursor = firstUnreviewed;
  render();
}

function render() {
  const emptyMsg = document.getElementById('empty-msg');
  const imgPanel = document.getElementById('img-panel');
  const infoPanel = document.getElementById('info-panel');

  if (items.length === 0) {
    emptyMsg.style.display = 'block';
    imgPanel.style.display = 'none';
    infoPanel.style.display = 'none';
    document.getElementById('prog-text').textContent = '0件';
    return;
  }

  emptyMsg.style.display = 'none';
  imgPanel.style.display = 'flex';
  infoPanel.style.display = 'flex';

  const item = items[cursor];

  // 画像
  const img = document.getElementById('crop-img');
  img.src = `/image/${encodeURIComponent(item.file)}`;
  img.style.width = '512px';  // 128px × 4倍
  img.style.height = 'auto';
  document.getElementById('img-filename').textContent = item.file;

  // ラベル
  const input = document.getElementById('label-input');
  input.value = item.deleted ? '[削除済み]' : (item.ocr_text || '');
  input.disabled = !!item.deleted;
  originalLabel = input.value;
  input.classList.remove('changed');

  // conf
  const conf = item.conf ?? null;
  const confEl = document.getElementById('conf-val');
  if (conf !== null) {
    confEl.textContent = conf.toFixed(3);
    confEl.style.color = conf < 0.5 ? '#ff8060' : conf < 0.8 ? '#ffcc60' : '#60ff90';
  } else {
    confEl.textContent = '-';
  }

  // バッジ
  const badgeRow = document.getElementById('badge-row');
  badgeRow.innerHTML = '';
  if (item.corrected) addBadge(badgeRow, '自動修正', 'badge-corrected');
  if (conf !== null) {
    if (conf < 0.5) addBadge(badgeRow, `conf低(${conf.toFixed(2)})`, 'badge-low');
    else if (conf < 0.8) addBadge(badgeRow, `conf中(${conf.toFixed(2)})`, 'badge-mid');
    else addBadge(badgeRow, `conf高(${conf.toFixed(2)})`, 'badge-high');
  }
  if (item.reviewed) addBadge(badgeRow, '確認済み', 'badge-reviewed');
  if (item.deleted)  addBadge(badgeRow, '削除フラグ', 'badge-deleted');

  // 進捗
  const reviewed = items.filter(x => x.reviewed || x.deleted).length;
  const pct = items.length > 0 ? (reviewed / items.length * 100) : 0;
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-text').textContent =
    `${cursor + 1} / ${items.length}件（確認済み ${reviewed}件）`;
}

function addBadge(parent, text, cls) {
  const b = document.createElement('span');
  b.className = `badge ${cls}`;
  b.textContent = text;
  parent.appendChild(b);
}

async function saveAndNext() {
  const input = document.getElementById('label-input');
  const item = items[cursor];
  if (item.deleted) { next(); return; }

  await fetch('/api/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ index: item.index, text: input.value })
  });
  item.ocr_text = input.value;
  item.reviewed = true;
  showSaved();
  fetchCounts();
  next();
}

async function markReviewed() {
  const item = items[cursor];
  if (!item.deleted) {
    const input = document.getElementById('label-input');
    const currentText = input.value;
    await fetch('/api/update', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ index: item.index, text: currentText })
    });
    item.ocr_text = currentText;
    item.reviewed = true;
    fetchCounts();
  }
  next();
}

async function deleteItem() {
  const item = items[cursor];
  if (item.deleted) { next(); return; }
  if (!confirm(`「${item.ocr_text}」に削除フラグを付けますか？`)) return;
  await fetch('/api/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ index: item.index })
  });
  item.deleted = true;
  item.reviewed = true;
  showSaved('🗑 削除フラグ設定');
  fetchCounts();
  render();
}

function next() {
  if (cursor < items.length - 1) cursor++;
  render();
}
function prev() {
  if (cursor > 0) cursor--;
  render();
}

function showSaved(msg = '✓ 保存') {
  const el = document.getElementById('save-indicator');
  el.textContent = msg;
  el.style.color = '#60ff90';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.textContent = ''; }, 1500);
}

// イベント
document.getElementById('btn-save').addEventListener('click', saveAndNext);
document.getElementById('btn-next').addEventListener('click', markReviewed);
document.getElementById('btn-prev').addEventListener('click', prev);
document.getElementById('btn-delete').addEventListener('click', deleteItem);

document.getElementById('label-input').addEventListener('input', () => {
  const input = document.getElementById('label-input');
  input.classList.toggle('changed', input.value !== originalLabel);
});

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filter = btn.dataset.filter;
    loadItems();
  });
});

document.getElementById('chk-unreviewed').addEventListener('change', loadItems);

document.addEventListener('keydown', e => {
  // テキスト入力中はEnter以外のショートカットを無効化
  const focused = document.activeElement === document.getElementById('label-input');
  if (e.key === 'Enter' && focused) { e.preventDefault(); saveAndNext(); return; }
  if (e.key === 'Escape' && focused) {
    document.getElementById('label-input').blur();
    return;
  }
  if (focused) return;

  if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') markReviewed();
  else if (e.key === 'ArrowLeft' || e.key === 'h' || e.key === 'H') prev();
  else if (e.key === 'd' || e.key === 'D') deleteItem();
  else if (e.key === 'Enter') saveAndNext();
});

// 初期化
fetchCounts();
loadItems();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/items")
def get_items():
    filter_mode = request.args.get("filter", "corrected")
    unreviewed_only = request.args.get("unreviewed", "0") == "1"

    if filter_mode == "corrected":
        result = [(i, d) for i, d in enumerate(annotations) if d.get("corrected")]
    elif filter_mode == "low_conf":
        result = [(i, d) for i, d in enumerate(annotations) if d.get("conf", 1.0) < 0.5]
    elif filter_mode == "mid_conf":
        result = [(i, d) for i, d in enumerate(annotations) if 0.5 <= d.get("conf", 1.0) < 0.8]
    else:
        result = [(i, d) for i, d in enumerate(annotations)]

    if unreviewed_only:
        result = [(i, d) for i, d in result if not d.get("reviewed") and not d.get("deleted")]

    return jsonify([{"index": i, **d} for i, d in result])


@app.route("/image/<path:filename>")
def get_image(filename):
    path = CROPS_DIR / filename
    if path.exists():
        return send_file(path)
    return "Not found", 404


@app.route("/api/update", methods=["POST"])
def update():
    data = request.get_json()
    idx = data["index"]
    annotations[idx]["ocr_text"] = data["text"]
    annotations[idx]["reviewed"] = True
    save_annotations()
    return jsonify({"ok": True})


@app.route("/api/delete", methods=["POST"])
def delete():
    data = request.get_json()
    idx = data["index"]
    annotations[idx]["deleted"] = True
    annotations[idx]["reviewed"] = True
    save_annotations()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 50)
    print("アノテーションレビューツール")
    print("ブラウザで http://localhost:5001 を開いてください")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False)
