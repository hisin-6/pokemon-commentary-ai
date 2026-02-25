# ADR-006: 3Dモデル描画にVRoid Studio + VTube Studioを使用する

## ステータス
承認済み

## 日付
2026-02-25

## 文脈

実況AIには視聴者に見せる3Dモデル（VTuberアバター）が必要。3Dモデルの描画はVRAMを4〜5GB消費する最大のコンポーネントであり、他コンポーネントとのリソース競合を避けながら口パク（リップシンク）と音声を同期させる仕組みが必要。

要件：
- VRAMを4〜5GB以内に収めること
- VOICEVOXの音声出力と口パクをリアルタイム同期できること
- Pythonから制御可能またはプロトコルで連携できること
- 無料または低コストで利用できること

## 決定

以下の構成で3Dモデルを実装する。

| 役割 | ツール |
|------|--------|
| 3Dモデル作成 | VRoid Studio（無料） |
| 3Dモデル表示・リップシンク | VTube Studio |
| 音声・モーション連携プロトコル | VMC Protocol（Virtual Motion Capture Protocol） |

## アーキテクチャ

```
Pythonアプリ
　↓ VOICEVOXで音声生成（WAVファイル）
音声再生
　↓ VMC Protocol（UDP）
VTube Studio
　↓
3Dモデル（VRMファイル）が口パク・表情変化
```

## 理由

### VRoid Studio
- 無料で高品質なVRMモデルが作れる
- VRM形式は各種ツールで広くサポートされている
- 日本語圏での事例が豊富でドキュメントが充実している

### VTube Studio
- VRMモデルの表示・リップシンクに特化したアプリ
- VMC Protocolに対応しており外部から制御可能
- Steam版が無料で利用できる（ウォーターマークあり、有料解除可）

### VMC Protocol
- UDP通信でモーション・表情データを送受信できる標準プロトコル
- PythonライブラリでVMC Protocolメッセージを送信可能
- VTube Studio・バーチャルモーションキャプチャー等が受信に対応

## VRAM消費

| コンポーネント | 消費VRAM |
|--------------|---------|
| VTube Studio（3Dモデル描画） | 約4〜5 GB |

VTube Studioはシステムのディスプレイ（GPU）リソースを使用するため、他のMLワークロード（PyTorch/YOLO）と同じGPUを共有する点に注意。

## 却下した選択肢

| 選択肢 | 却下理由 |
|--------|---------|
| Unity（カスタム実装） | 開発工数が大きい・Sprint 3では過剰 |
| Unreal Engine | 同上・VRAM消費がより大きい |
| nizima LIVE | 2Dモデル向け・3Dモデルの品質が限られる |
| バーチャルモーションキャプチャー | 動作確認中・VTube Studioの方が情報が多い |

## 連携実装方針

```python
# VMC Protocol送信サンプルイメージ
import socket
import struct

def send_lipsync(volume: float):
    # VMCPでVTube Studioにリップシンク値を送信
    ...
```

Pythonライブラリ `python-osc` を使ってOSC/UDP経由でVMC Protocolメッセージを送信する。

## 注意事項

- VTube StudioはWindowsアプリとして動作するため、WSL2からの制御はUDP経由で行う
- 音声再生のタイミングとVMC Protocol送信のタイミングを合わせる同期処理が必要
- VRAMが逼迫した場合はVTube Studioの解像度・品質設定を下げて対応する

## 結果

無料ツールの組み合わせでVTuberアバターのリップシンク動作を実現できる。Sprint 3で実装し、VOICEVOXとの音声同期まで完成させる。
