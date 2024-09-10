# -*- coding: utf-8 -*-
import jieba


# 检验字符串是否为中文
def check_chinese(sentence):
    cn_flag = False
    for ch in sentence:
        if '\u4e00' <= ch <= '\u9fa5':
            cn_flag = True
    return cn_flag