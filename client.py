# -*- coding: utf-8 -*-
import requests

# server_ip = '10.210.24.255'
server_ip = '10.60.6.29'
server_port = '8085'
# server_port = '8086'
# video_desc = "黃國英：我不相信nft ！"
# video_desc = "10個財務比率，算出公司內在價值。"
# video_title = "1分鐘系列 - 看懂損益表"
video_title = "How to form a good habit in 21 days"
video_desc = ""
# 1：简体中文 2：繁体中文 3:英文
video_language = 3
# textcnn/fasttext/roberta
# model = "textcnn"
# model = "fasttext"
model = "roberta"

s = requests.post(f'http://{server_ip}:{server_port}/{model}', json={"video_title": video_title,
                                                                     "video_desc": video_desc,
                                                                     "video_language": video_language})

print(s.json())
