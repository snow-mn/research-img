from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    #　パーサー作成
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    # 引数追加(help=引数の説明、dest=parse.args()が返すオブジェクトの属性名）
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    # int型、デフォルト1
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    # 引数の解析
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        # ファイルを読み込んで1行毎に分割
        filenames = f.read().split('\n')
        for name in filenames:
            # ファイル名がないなら以降スキップ
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                # 文を1行毎に分割
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    # 国際符号化文字集合を空白に置き換え
                    sent = sent.replace("\ufffd\ufffd", " ")
                    # 文全体を小文字にしてから【英単語】で分割
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    # 分割された語の要素数が0ならsentを出力して以降スキップ
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        # t（恐らく単語？なのでstr）のlenは長さ、配列なら要素数が>0なら
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            # cap_lens内の要素の最大値を取り出す
            max_len = np.max(cap_lens)

            # np.argsort()は値ではなく並び替えたインデックス（元のndarrayでの位置 = 0始まりの順番）のndarrayを返す
            # この場合最後の軸に沿って降順（stepが-1なので）
            sorted_indices = np.argsort(cap_lens)[::-1]
            # np.asarray()はコピーを作る場合同一のものとして扱われる
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            # len(captions)×max_lenの0の配列生成
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    # args.cfg_fileが空でなければ、それを読み込んでデフォルトの設定とマージする（詳しくはconfig）
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # GPUIDが-1以外ならば、それを代入、-1ならばCUDAはFalse
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    # データディレクトリが空でなければ、cfgのそれに代入
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    #　リストの要素ごとに改行して見やすく表示
    pprint.pprint(cfg)

    # トレーニングフラグがTrueでないなら
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    # トレーニングフラグがTrueなら
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    #　現在時刻を取得（ローカルのタイムゾーン）
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    # これは画像であるPIL image または ndarrayのdata「Height×Width×Channel」をTensor型のdata「Channel×Height×Width」に変換するというもので,
    # transという変数がその機能を持つことを意味する.
    # torchvision.transforms.Composeは引数で渡されたlist型の[~~,~~,...]というのを先頭から順に実行していくもの
    # こうすることでimage_transformという変数はTensor変換と正規化を一気にしてくれるハイブリッドな変数になった
    image_transform = transforms.Compose([
        # 画像を拡大して
        transforms.Resize(int(imsize * 76 / 64)),
        # 元のサイズでランダムな位置から切り取る
        transforms.RandomCrop(imsize),
        # 指定された確率（デフォルト0.5）で指定された画像をランダムに水平方向に反転する。
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
