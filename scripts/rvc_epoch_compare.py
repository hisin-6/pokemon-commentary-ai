"""RVCエポック聴き比べ用の音声を一括生成する。

VOICEVOX（四国めたん）で実況っぽいセリフを読み上げ、RVCの各エポックモデル
（kurepi_e25〜e250）でまとめて声質変換して、聴き比べ用のwavを並べて保存する。

⚠️ RVC-WebUI の venv（Windows側Python）で実行すること:
    C:\\Users\\rotat\\Retrieval-based-Voice-Conversion-WebUI-main\\Retrieval-based-Voice-Conversion-WebUI-main\\venv\\Scripts\\python.exe ^
        scripts\\rvc_epoch_compare.py

前提:
- VOICEVOXが起動していること（localhost:50021）
- VRAMはRVC推論で1〜2GB程度。実況パイプラインと同時に動かさないこと

ピッチ（--pitch）は既定で auto。録音コーパス（data/voices/）とVOICEVOX出力の
F0中央値の差から半音数を自動計算する。数値を渡せば固定値になる。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RVC_ROOT = Path(
    r"C:\Users\rotat\Retrieval-based-Voice-Conversion-WebUI-main"
    r"\Retrieval-based-Voice-Conversion-WebUI-main"
)
DEFAULT_INDEX = "logs/kurepi/added_IVF1717_Flat_nprobe_1_kurepi_v2.index"
DEFAULT_OUT = PROJECT_ROOT / "renders" / "rvc_compare"
VOICES_DIR = PROJECT_ROOT / "data" / "voices"

SENTENCES = [
    "さあ、いよいよ対戦スタートです！今日もよろしくお願いします！",
    "おっと、ここでガブリアスに交代！読み合いが熱いですね。",
    "急所に当たった！これは大きいダメージです！",
    "やった〜！勝ちました！みんな応援ありがとう！",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rvc-root", type=Path, default=DEFAULT_RVC_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--voicevox-url", default="http://localhost:50021")
    p.add_argument("--source-dir", type=Path, default=None,
                   help="VOICEVOXを使わず、このフォルダのwavを変換元にする（人の声での検証用）")
    p.add_argument("--speaker", type=int, default=2, help="VOICEVOX話者ID（2=四国めたん ノーマル）")
    p.add_argument("--vv-pitch-scale", type=float, default=0.0,
                   help="VOICEVOXの音高（0=標準・-0.1で約1.7半音下がる）")
    p.add_argument("--vv-intonation", type=float, default=1.0,
                   help="VOICEVOXの抑揚（1=標準・下げると平坦になる）")
    p.add_argument("--epochs", default="all", help="'all' または '25,100,250' のようなカンマ区切り")
    p.add_argument("--pitch", default="auto", help="半音数（例: -3）または auto")
    p.add_argument("--index-rate", type=float, default=0.75)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--f0-method", default="rmvpe")
    return p.parse_args()


# ---------------------------------------------------------------- VOICEVOX
def synthesize(
    url: str, speaker: int, text: str, out_path: Path,
    pitch_scale: float = 0.0, intonation_scale: float = 1.0,
) -> None:
    q = urllib.parse.urlencode({"text": text, "speaker": speaker})
    with urllib.request.urlopen(urllib.request.Request(f"{url}/audio_query?{q}", method="POST")) as r:
        query_obj = json.loads(r.read())
    query_obj["pitchScale"] = pitch_scale
    query_obj["intonationScale"] = intonation_scale
    query = json.dumps(query_obj).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/synthesis?speaker={speaker}",
        data=query,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        out_path.write_bytes(r.read())


# ---------------------------------------------------------------- ピッチ推定
def median_f0(paths: list[Path]) -> float:
    import numpy as np
    import pyworld
    import soundfile as sf

    voiced = []
    for path in paths:
        x, sr = sf.read(str(path), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0, t = pyworld.dio(x, sr)
        f0 = pyworld.stonemask(x, f0, t, sr)
        voiced.append(f0[f0 > 0])
    allf0 = np.concatenate(voiced)
    if allf0.size == 0:
        raise RuntimeError("有声区間が見つからなかった")
    return float(np.median(allf0))


def estimate_pitch_shift(source_wavs: list[Path]) -> tuple[int, float, float]:
    corpus = sorted(VOICES_DIR.glob("*.wav"))
    if not corpus:
        raise RuntimeError(f"録音が見つからない: {VOICES_DIR}（--pitch に数値を指定して）")
    target = median_f0(corpus[:8])
    source = median_f0(source_wavs)
    shift = round(12 * math.log2(target / source))
    return shift, source, target


# ---------------------------------------------------------------- RVC
def list_models(weights_dir: Path, epochs_arg: str) -> list[tuple[int, str]]:
    models = []
    for p in weights_dir.glob("kurepi_e*_s*.pth"):
        ep = int(p.stem.split("_")[1][1:])
        models.append((ep, p.name))
    models.sort()
    if epochs_arg != "all":
        wanted = {int(e) for e in epochs_arg.split(",")}
        models = [m for m in models if m[0] in wanted]
    if not models:
        raise RuntimeError(f"対象モデルが見つからない: {weights_dir}")
    return models


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    src_dir = out / "00_source_voicevox"
    out.mkdir(parents=True, exist_ok=True)
    if not args.source_dir:
        src_dir.mkdir(exist_ok=True)

    # 1) 変換元音声（--source-dir指定時は手持ちwav、なければVOICEVOXで生成）
    if args.source_dir:
        sources = sorted(args.source_dir.resolve().glob("*.wav"))
        if not sources:
            raise RuntimeError(f"wavが見つからない: {args.source_dir}")
        print(f"[source] {len(sources)}ファイル ← {args.source_dir}")
    else:
        sources = []
        for i, text in enumerate(SENTENCES, 1):
            path = src_dir / f"s{i}.wav"
            synthesize(args.voicevox_url, args.speaker, text, path,
                       args.vv_pitch_scale, args.vv_intonation)
            sources.append(path)
        print(f"[VOICEVOX] {len(sources)}文生成 → {src_dir}")

    # 2) ピッチ
    if args.pitch == "auto":
        pitch, src_hz, tgt_hz = estimate_pitch_shift(sources)
        print(f"[pitch] 変換元中央値 {src_hz:.0f}Hz / 録音中央値 {tgt_hz:.0f}Hz → {pitch:+d}半音")
    else:
        pitch = int(args.pitch)
        print(f"[pitch] 固定 {pitch:+d}半音")

    # 3) RVC（RVCのルートで動かす前提のコードなのでcwdとsys.pathを合わせる）
    rvc_root = args.rvc_root.resolve()
    os.chdir(rvc_root)
    sys.path.insert(0, str(rvc_root))
    sys.argv = sys.argv[:1]  # RVCのConfigがargparseするので空にする
    from dotenv import load_dotenv
    from scipy.io import wavfile

    # 引数なしだと呼び出し元スクリプトの場所から.envを探してしまうので明示する
    load_dotenv(rvc_root / ".env")
    from configs.config import Config
    from infer.modules.vc.modules import VC

    vc = VC(Config())
    index_path = str(rvc_root / DEFAULT_INDEX)
    models = list_models(rvc_root / "assets" / "weights", args.epochs)

    for ep, name in models:
        vc.get_vc(name)
        for i, src in enumerate(sources, 1):
            info, (sr, audio) = vc.vc_single(
                0, str(src), pitch, None, args.f0_method, index_path, None,
                args.index_rate, 3, 0, 0.25, args.protect,
            )
            if audio is None:
                print(f"[NG] e{ep} s{i}\n{info}")
                continue
            wavfile.write(str(out / f"{src.stem}_e{ep:03d}.wav"), sr, audio)
        print(f"[RVC] e{ep:03d} 完了")

    meta = {
        "sentences": None if args.source_dir else SENTENCES,
        "source_dir": str(args.source_dir) if args.source_dir else None,
        "speaker": args.speaker,
        "vv_pitch_scale": args.vv_pitch_scale,
        "vv_intonation": args.vv_intonation,
        "pitch": pitch,
        "index_rate": args.index_rate,
        "protect": args.protect,
        "f0_method": args.f0_method,
        "epochs": [ep for ep, _ in models],
    }
    (out / "settings.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完了 → {out}")


if __name__ == "__main__":
    main()
