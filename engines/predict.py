# -*- coding: utf-8 -*-
import jieba
import numpy as np
from nltk import word_tokenize
# jieba.load_userdict("../data/short_video_dict_ch.txt")
# stop_words = [line.strip() for line in open("../data/stopwords.txt", encoding='utf-8').readlines()]


def load_vocab(token_file):
    word_token2id, id2word_token = {}, {}
    with open(token_file, 'r', encoding='utf-8') as infile:
        for row in infile:
            row = row.strip()
            word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
            word_token2id[word_token] = word_token_id
            id2word_token[word_token_id] = word_token
    vocab_size = len(word_token2id)
    return vocab_size, word_token2id, id2word_token


def processing_sentence(x, stop_words):
    cut_word = jieba.cut(str(x).strip())
    if stop_words:
        words = [word for word in cut_word if word not in stop_words and word != ' ']
    else:
        words = list(cut_word)
        words = [word for word in words if word != ' ']
    return words


def processing_english_sentence(x, stop_words):
    x = str(x)
    cut_word = list(word_tokenize(x.lower().replace("\t", " ").replace("\n", " ").strip()))
    if stop_words:
        words = [word for word in cut_word if word not in stop_words and word != ' ']
    else:
        words = list(cut_word)
        words = [word for word in words if word != ' ']
    return words


def padding(sentence, max_sequence_length):
    """
    长度不足max_sequence_length则补齐
    :param sentence:
    :return:
    """
    if len(sentence) < max_sequence_length:
        sentence += ['[PAD]' for _ in range(max_sequence_length - len(sentence))]
    else:
        sentence = sentence[:max_sequence_length]
    return sentence


def prepare_single_sentence(sentence, token2id, stop_words, max_sequence_length, token_level="word"):
    """
    把预测的句子转成矩阵和向量
    :param sentence:
    :return:
    """
    if token_level == 'word':
        sentence = processing_sentence(sentence, stop_words)
    elif token_level == 'char':
        sentence = list(sentence)
        if stop_words:
            sentence = [char for char in sentence if char not in stop_words and char != ' ']
    elif token_level == 'English':
        sentence = processing_english_sentence(sentence, stop_words)
    sentence = padding(sentence, max_sequence_length)
    tokens = []
    for word in sentence:
        if word in token2id:
            tokens.append(token2id[word])
        else:
            tokens.append(token2id["[UNK]"])
    return np.array([tokens])