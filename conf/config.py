# -*- coding: utf-8 -*-

import os
from os.path import join

pro_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_path_dir = join(pro_dir, "data")
model_path_dir = join(pro_dir, "model")

stop_file = join(data_path_dir, "stopwords_ch.txt")
dict_file = join(data_path_dir, "short_video_dict_ch.txt")
pattern_dict_ch_path = join(model_path_dir, "pattern_dict_ch.json")
pattern_dict_en_path = join(model_path_dir, "pattern_dict_en.json")
fasttext_ch_path = join(model_path_dir, "fasttext_ch.bin")
fasttext_en_path = join(model_path_dir, "fasttext_en.bin")
token2id_ch_path = join(data_path_dir, "word-token2id-ch")
token2id_en_path = join(data_path_dir, "word-token2id-en")
textcnn_ch_path = join(model_path_dir, "textcnn-ch-word-focal-0531")
textcnn_en_path = join(model_path_dir, "textcnn-en")


class PretrainingConfig:
    seed = 42
    max_sen_len = 64
    train_batch_size = 8
    val_batch_size = 8
    bert_pretrained_name = model_path_dir + '/chinese-roberta-wwm-ext'
    pre_model_type = None
    multi_gpu = False
    params_path = './'
    n_classes = 19
    cuda_device = 0

    trained_model_path = './checkpoints/roberta_0531.pth'
    # 训练模型参数
    num_workers = 0
    n_epoch = 8
    min_store_epoch = 2
    scheduler_type = 'get_linear_schedule_with_warmup'  # get_linear_schedule_with_warmup
    # trick 参数
    attack_func = None  # fgm  pgd
    pgd_k = 3
    is_use_rdrop = True
    alpha = 0.25  # ghmloss
    is_use_swa = False
    ema_decay = 0.99  # 0.995
    rdrop_ismean = False
