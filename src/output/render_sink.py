"""実況動画レンダリング用の素材出力（ADR-009 パス1）。

``--render-out`` 指定時に、生成した実況音声のWAVファイルと、
「動画内のどの時刻の実況か」を記録したマニフェスト（JSONL・1行=1実況）を
出力ディレクトリへ書き出す。パス2（合成スクリプト）がこれを読んで
元動画に実況トラックを合成する。

出力構成:
    <out_dir>/
    ├── render_info.json   # 元動画パス等のメタ情報
    ├── manifest.jsonl     # 1行=1実況 {seq, event_time, event_type, commentary, wav, duration}
    └── wav/
        ├── 0001_battle_start.wav
        └── ...
"""

from __future__ import annotations

import io
import json
import logging
import threading
import wave
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class RenderSink:
    """実況WAVの保存とマニフェストJSONLへの追記を行う。

    音声合成スレッド（Pipeline._speak_async の内部スレッド）から呼ばれるため、
    連番の採番とマニフェスト追記はロックで保護する。

    同じ出力先での再実行時は前回の素材（manifest.jsonl・wav・fillers.jsonl）を
    クリアしてから始める。追記のままだと新旧エントリが混在し、連番WAVの
    同名衝突で旧音声が上書きされる（2026-07-14に実際に発生）。
    """

    def __init__(self, out_dir):
        self._dir = Path(out_dir)
        self._wav_dir = self._dir / "wav"
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "manifest.jsonl"
        self._timeline_path = self._dir / "timeline.jsonl"
        self._states_path = self._dir / "states.jsonl"
        self._clear_previous_run()
        self._lock = threading.Lock()
        self._seq = 0
        self._last_state_key: str | None = None

    def _clear_previous_run(self) -> None:
        """前回実行の素材を削除する（fillers.jsonl も旧manifest由来のため無効化）。"""
        removed = 0
        for path in [self._manifest_path, self._timeline_path,
                     self._states_path, self._dir / "fillers.jsonl"]:
            if path.exists():
                path.unlink()
                removed += 1
        for wav in self._wav_dir.glob("*.wav"):
            wav.unlink()
            removed += 1
        if removed:
            log.warning("[レンダ] 前回実行の素材 %d ファイルをクリアしました: %s",
                        removed, self._dir)

    @property
    def out_dir(self) -> Path:
        return self._dir

    @property
    def count(self) -> int:
        """これまでに保存した実況素材の件数。"""
        with self._lock:
            return self._seq

    @staticmethod
    def wav_duration(wav_bytes: bytes) -> float:
        """WAVバイト列の再生時間（秒）を返す。"""
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else 0.0

    def add_moment(self, time: float, kind: str, text: str,
                   side: str = None) -> None:
        """画面上の出来事の「瞬間ログ」を timeline.jsonl に追記する。

        技検出など、実況イベントにならなかった出来事も動画内時刻付きで
        記録しておき、台本パス（ギャップフィラー生成）がライブ実況風の
        時刻アンカーとして使う。

        Args:
            time: 検出時点の動画内時刻（秒）
            kind: 出来事の種別（move 等）
            text: 出来事の内容（例: "T3:ガブリアスのドラゴンクロー"）
            side: 出来事の主体の陣営（"自分"/"相手"・不明ならNone）。
                text の形式は既存パーサー互換のため変えず、別フィールドで持つ。
        """
        record = {"time": round(float(time), 3), "kind": kind, "text": text}
        if side is not None:
            record["side"] = side
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._timeline_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def add_state(self, time: float, state: dict) -> None:
        """戦況パネル用スナップショットを states.jsonl に追記する（v2b）。

        直前に記録した状態と同一内容ならスキップする（定期OCRサイクル毎に
        呼ばれるため、変化がない限りファイルは増えない）。

        Args:
            time:  スナップショット時点の動画内時刻（秒）
            state: ``BattleStateTracker.to_panel_state()`` の戻り値
        """
        key = json.dumps(state, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if key == self._last_state_key:
                return
            self._last_state_key = key
            entry = {"time": round(float(time), 3)}
            entry.update(state)
            with self._states_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def write_info(self, info: dict) -> None:
        """元動画パス等のメタ情報を render_info.json に保存する（上書き）。"""
        payload = dict(info)
        payload.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        with (self._dir / "render_info.json").open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def add(self, event_time: float, event_type: str, commentary: str,
            wav_bytes: bytes, context: dict = None) -> dict:
        """実況1件を保存し、マニフェストに書いたエントリを返す。

        Args:
            event_time: イベント検知時点の動画内時刻（秒・``Pipeline._now()``）
            event_type: 発火イベント種別（battle_start / move_used / faint 等）
            commentary: 実況テキスト
            wav_bytes:  VOICEVOX が生成した WAV バイト列
            context:    イベント時点の戦況サマリー（台本パスのフィラー生成用・任意）
        """
        with self._lock:
            self._seq += 1
            seq = self._seq
        wav_name = f"{seq:04d}_{event_type}.wav"
        (self._wav_dir / wav_name).write_bytes(wav_bytes)
        entry = {
            "seq": seq,
            "event_time": round(float(event_time), 3),
            "event_type": event_type,
            "commentary": commentary,
            "wav": f"wav/{wav_name}",
            "duration": round(self.wav_duration(wav_bytes), 3),
        }
        if context:
            entry["context"] = context
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self._manifest_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        return entry
