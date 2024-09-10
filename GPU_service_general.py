# -*- coding: utf-8 -*-

import torch
from torch import nn
import tensorflow as tf
from nltk import word_tokenize
import zhconv

from sanic import Sanic
from sanic.request import Request
from sanic import response
from sanic_cors import CORS

from engines.models.TextCNN import TextCNN
from engines.predict import load_vocab, prepare_single_sentence
from conf.config import PretrainingConfig
from conf.config import stop_file, dict_file
from conf.config import token2id_ch_path, token2id_en_path
from conf.config import fasttext_ch_path, fasttext_en_path, textcnn_en_path, textcnn_ch_path

from dd_nlp_arsenal.processor.tokenizer.nezha import SentenceTokenizer
from dd_nlp_arsenal.factory.task.cls_task.sentence_cls_task import SentenceCLSTask, logging
from dd_nlp_arsenal.model.text_cls.bert_model import BertClsModel, BertAttClsModel

from dd_nlp_arsenal.factory.untils.tools import seed_torch
from dd_nlp_arsenal.factory.untils.opt import get_default_bert_optimizer

# 加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pretraining_config = PretrainingConfig()
pretraining_config.pre_model_type = pretraining_config.bert_pretrained_name.split('/')[-1]
seed_torch(pretraining_config.seed)

tokenizer = SentenceTokenizer(pretraining_config.bert_pretrained_name, pretraining_config.max_sen_len)
model = BertClsModel(pretraining_config)
model.load(pretraining_config.trained_model_path, device)
optimizer = get_default_bert_optimizer(model, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task = SentenceCLSTask(model, optimizer, loss_func, pretraining_config)

categories = ['保险', '储蓄', '其它', '创业', '基金', '外汇', '房产', '数字货币', '泛财经', '留学', '移民', '税务', '美女',
              '股票-其他', '股票-基本面分析', '股票-技术面分析', '股票-行业分析', '财经段子', '鸡汤']
id2cat = dict(zip(range(len(categories)), categories))

# 加载结巴自定义词典
jieba.load_userdict(dict_file)
# 加载停用词
stopwords = [line.strip() for line in open(stop_file, encoding='utf-8').readlines()]
# 加载中英文fasttext模型，常驻内存
fasttext_ch = fasttext.load_model(fasttext_ch_path)
fasttext_en = fasttext.load_model(fasttext_en_path)
# 加载英文textCNN模型，常驻内存
vocab_size_en, token2id_en, id2token_en = load_vocab(token2id_en_path)
model_en = TextCNN(2, 300, vocab_size_en, None)
textcnn_en = tf.train.Checkpoint(model=model_en)
textcnn_en.restore(tf.train.latest_checkpoint(textcnn_en_path))
# 加载中文textCNN模型，常驻内存
vocab_size_ch, token2id_ch, id2token_ch = load_vocab(token2id_ch_path)
model_ch = TextCNN(2, 300, vocab_size_ch, None)
textcnn_ch = tf.train.Checkpoint(model=model_ch)
textcnn_ch.restore(tf.train.latest_checkpoint(textcnn_ch_path))
# 中文textCNN标签
labels_ch = ['泛财经', '创业', '美女', '鸡汤', '数字货币', '储蓄', '基金', '保险', '财经段子', '移民', '留学', '股票-技术面分析', '税务',
             '股票-其他', '房产', '股票-行业分析', '外汇', '股票-基本面分析', '其它']
class_id_ch = {cls: index for index, cls in enumerate(labels_ch)}
reverse_classes_ch = {class_id: class_name for class_name, class_id in class_id_ch.items()}
# 英文textCNN标签
labels_en = ['泛财经', '创业', '美女', '鸡汤', '数字货币', '储蓄', '基金', '保险', '股票-技术面分析', '税务',
             '股票-其他', '房产', '股票-行业分析', '股票-基本面分析', '其它']
class_id_en = {cls: index for index, cls in enumerate(labels_en)}
reverse_classes_en = {class_id: class_name for class_name, class_id in class_id_en.items()}


class Worker():
    def __init__(self, video_language=1):
        self.video_language = video_language

    def predict_single_sentence_fasttext(self, sentence):
        if self.video_language == 2:
            sentence = zhconv.convert(sentence, 'zh-cn')
        sentence = sentence.lower()
        sentence = sentence.replace("\t", " ").replace("\n", " ").strip()
        # 英文处理预测流程
        if self.video_language == 3:
            tokens = list(word_tokenize(sentence))
            tokens = [item for item in tokens if item not in stopwords and item != ' ']
            predict_result = fasttext_en.predict(" ".join(tokens), k=3)
        elif self.video_language == 2 or self.video_language == 1:
            words = jieba.cut(sentence)
            words = [item for item in words if item not in stopwords and item != ' ']
            predict_result = fasttext_ch.predict(" ".join(words), k=3)
        predict_label = predict_result[0][0].replace("__label__", "")
        predict_score = round(predict_result[1][0], 3)
        return predict_label, predict_score

    def predict_single_sentence_textCNN(self, sentence):
        if self.video_language == 2:
            sentence = zhconv.convert(sentence, 'zh-cn')
        # 英文处理预测流程
        sentence = sentence.lower()
        sentence = sentence.replace("\t", " ").replace("\n", " ").strip()
        if self.video_language == 3:
            vector = prepare_single_sentence(sentence, token2id_en, stopwords, 300, token_level="English")
            logits = model_en(inputs=vector)
            predict_label = tf.argmax(logits, axis=-1)
            predict_label = int(predict_label.numpy()[0])
            predict_score = tf.reduce_max(logits, axis=-1).numpy()[0]
            predict_score = round(float(predict_score), 3)
            predict_label = reverse_classes_en[predict_label]
        elif self.video_language == 2 or self.video_language == 1:
            vector = prepare_single_sentence(sentence, token2id_ch, stopwords, 300, token_level="word")
            logits = model_ch(inputs=vector)
            predict_label = tf.argmax(logits, axis=-1)
            predict_label = int(predict_label.numpy()[0])
            predict_score = tf.reduce_max(logits, axis=-1).numpy()[0]
            predict_score = round(float(predict_score), 3)
            predict_label = reverse_classes_ch[predict_label]
        return predict_label, predict_score

    def predict_single_sentence_Roberta(self, sentence):
        if self.video_language == 2:
            sentence = zhconv.convert(sentence, 'zh-cn')
        # 英文处理预测流程
        if self.video_language == 3:
            predict_score = 0
            predict_label = "暂无"
        elif self.video_language == 2 or self.video_language == 1:
            sentence = sentence.lower().strip()
            encoding = tokenizer.sequence_to_ids(sentence)
            feature = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'ori_text': sentence
            }
            input_ids = feature["input_ids"].to(task.device)
            attention_mask = feature["attention_mask"].to(task.device)
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.softmax(outputs, dim=0)
            predict_score, pred = torch.max(outputs, dim=0)
            predict_label = id2cat[int(pred)]
            predict_score = round(float(predict_score), 3)
        return predict_label, predict_score


class ServiceConfig:
    service_host = "0.0.0.0"  # 顶层外部agent_server的host.
    service_port = 8086  # 顶层外部agent_server的port.
    bot_agent_server_url = "http://{}:{}".format(service_host, service_port)


class ModelService:
    def __init__(self, app_name="ShortVideoClassifier"):
        self.service_config = ServiceConfig()
        print(self.service_config.service_host, self.service_config.service_port)
        self.app = Sanic(app_name)
        CORS(self.app)

    def start_service(self):
        self.add_routes()
        self.app.run(self.service_config.service_host, self.service_config.service_port, workers=1)

    def add_routes(self):
        self.app.add_route(self.textcnn_worker, "textcnn", methods=["POST"])
        self.app.add_route(self.fasttext_worker, "fasttext", methods=["POST"])
        self.app.add_route(self.roberta_worker, "roberta", methods=["POST"])

    @staticmethod
    async def roberta_worker(request: Request):
        req_json = request.json
        video_language = req_json["video_language"]
        video_title = req_json['video_title'].strip()
        video_desc = req_json["video_desc"].strip()
        if video_title != video_desc:
            video_text = video_title + video_desc
        else:
            video_text = video_title
        clf_worker = Worker(video_language=video_language)
        result_label, result_score = clf_worker.predict_single_sentence_Roberta(video_text)
        print(result_label)
        print(result_score)
        resp_body = {
            "code": 200,
            "msg": "success",
            "label": result_label,
            "score": result_score
        }
        resp_json = response.json(resp_body, status=200)
        return resp_json

    @staticmethod
    async def textcnn_worker(request: Request):
        req_json = request.json
        video_language = req_json["video_language"]
        video_title = req_json['video_title'].strip()
        video_desc = req_json["video_desc"].strip()
        if video_title != video_desc:
            video_text = video_title + video_desc
        else:
            video_text = video_title
        clf_worker = Worker(video_language=video_language)
        result_label, result_score = clf_worker.predict_single_sentence_textCNN(video_text)
        print(result_label)
        print(result_score)
        resp_body = {
            "code": 200,
            "msg": "success",
            "label": result_label,
            "score": result_score
        }
        resp_json = response.json(resp_body, status=200)
        return resp_json

    @staticmethod
    async def fasttext_worker(request: Request):
        req_json = request.json
        video_language = req_json["video_language"]
        video_title = req_json['video_title'].strip()
        video_desc = req_json["video_desc"].strip()
        if video_title != video_desc:
            video_text = video_title + video_desc
        else:
            video_text = video_title
        clf_worker = Worker(video_language=video_language)
        result_label, result_score = clf_worker.predict_single_sentence_fasttext(video_text)
        print(result_label)
        print(result_score)
        resp_body = {
            "code": 200,
            "msg": "success",
            "label": result_label,
            "score": result_score
        }
        resp_json = response.json(resp_body, status=200)
        return resp_json


if __name__ == "__main__":
    server = ModelService()
    server.start_service()
