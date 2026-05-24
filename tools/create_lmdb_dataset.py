"""LMDB データセット作成スクリプト（deep-text-recognition-benchmark 互換）"""
import os
import lmdb
import cv2
import numpy as np
import fire


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return img.shape[0] * img.shape[1] > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Args:
        inputPath:  テキストクロップ画像のディレクトリ
        gtFile:     タブ区切りのGTファイル（ファイル名\tラベル）
        outputPath: LMDB出力先ディレクトリ
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    skipped = 0

    with open(gtFile, encoding='utf-8') as f:
        datalist = f.readlines()

    nSamples = len(datalist)
    print(f"{nSamples} 件を処理中 ...")

    for i, line in enumerate(datalist):
        line = line.strip('\n')
        if '\t' not in line:
            continue
        imagePath, label = line.split('\t', 1)
        fullPath = os.path.join(inputPath, imagePath)

        if not os.path.exists(fullPath):
            print(f"  [SKIP] 存在しない: {fullPath}")
            skipped += 1
            continue

        with open(fullPath, 'rb') as f:
            imageBin = f.read()

        if checkValid and not checkImageIsValid(imageBin):
            print(f"  [SKIP] 無効画像: {fullPath}")
            skipped += 1
            continue

        imageKey = f'image-{cnt:09d}'.encode()
        labelKey = f'label-{cnt:09d}'.encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f"  {cnt}/{nSamples} ...")

        cnt += 1

    nSamples = cnt - 1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print(f"完了: {nSamples} 件保存（スキップ {skipped} 件）→ {outputPath}")


if __name__ == '__main__':
    fire.Fire(createDataset)
