""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os

import cv2
import fasttext
import lmdb
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDatasetWithEmbedding(input_lmdb, output_lmdb, checkValid=True):
    """
    Create LMDB dataset with embedding of FastText.
    """
    env_input = lmdb.open(input_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    
    os.makedirs(output_lmdb, exist_ok=True)
    env_output = lmdb.open(output_lmdb, map_size=1099511627776)
    cache = {}
    cnt = 1

    with env_input.begin(write=False) as txn_input:
        length = int(txn_input.get('num-samples'.encode()))
        for i in tqdm(range(length)):
            imageKey, labelKey = f"image-{cnt:09d}", f"label-{cnt:09d}"
            embedKey = f"embed-{cnt:09d}"
            label = str(txn_input.get(labelKey.encode()), 'utf-8')  # label
            imageBin = txn_input.get(imageKey.encode())  # image
            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % imagePath)
                        continue
                except:
                    print('error occured', i)
                    with open(output_lmdb + '/error_image_log.txt', 'a') as log:
                        log.write('%s-th image data occured error\n' % str(i))
                    continue
            embed_vec = fasttext_model[label]
            cache[imageKey.encode()] = imageBin
            cache[labelKey.encode()] = label.encode()
            cache[embedKey.encode()] = ' '.join(str(v) for v in embed_vec.tolist()).encode()
            if cnt % 1000 == 0:
                writeCache(env_output, cache)
                cache = {}
            cnt += 1
        nSamples = cnt - 1
        cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(env_output, cache)
        print('Created dataset with %d samples' % nSamples)


def test_image_bin():
    with open("../figs/test/CANDY.png", "rb") as f:
        imageBin = f.read()
    print(type(imageBin))


if __name__ == '__main__':
    test_image_bin()
    input_data_dir = "../data/training"
    output_data_dir = "/home/chenlei/all/datasets/ABINet/data_with_embedding/training"
    lmdb_name = ["MJ/MJ_test", "MJ/MJ_valid", "MJ/MJ_train", "ST"]
    for i in lmdb_name:
        input_lmdb_path = os.path.join(input_data_dir, i)
        if not os.path.exists(input_lmdb_path):
            print(f"{input_lmdb_path} is not exist!")
            exit(0)
    fasttext_model = fasttext.load_model('/home/chenlei/data1/SEED/cc.en.300.bin')
    for i in lmdb_name:
        input_lmdb_path = os.path.join(input_data_dir, i)
        output_lmdb_path = os.path.join(output_data_dir, i)
        createDatasetWithEmbedding(input_lmdb_path, output_lmdb_path, checkValid=True)
