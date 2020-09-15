#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/23 16:49
# @Author : jj.wang

import os
import cv2
import time
import flask
import numpy as np
from utils.utils import cv2ImgAddText
from flask import render_template, request, url_for, jsonify

app = flask.Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    htmlFileName = 'lpr.html'
    return render_template(htmlFileName)


@app.route("/predict", methods=['POST'])
def predict():
    time_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time()))
    print(time_str)
    if flask.request.method == 'POST':
        start = time.time()
        received_file = request.files['input_image']
        imageFileName = received_file.filename
        if received_file:
            # 保存接收的图片到指定文件夹
            received_dirPath = 'static/images'
            if not os.path.isdir(received_dirPath):
                os.makedirs(received_dirPath)
            imageFilePath = os.path.join(received_dirPath, time_str + '_' + imageFileName)
            received_file.save(imageFilePath)
            print('receive image and save: %s' % imageFilePath)
            usedTime = time.time() - start
            print('receive image and save cost time: %f' % usedTime)
            results, draw_img = model(imageFilePath)
            drawed_imageFileName = time_str + '_draw_' + os.path.splitext(imageFileName)[0] + '.jpg'
            drawed_imageFilePath = os.path.join('static', drawed_imageFileName)
            print(f'draw image save: {drawed_imageFilePath}')
            cv2.imwrite(drawed_imageFilePath, draw_img)
            image_source_url = url_for('static', filename=drawed_imageFileName)
            return jsonify(src=image_source_url, count=f'{results}')


if __name__ == '__main__':
    from algo_interface import model
    import argparse

    parser = argparse.ArgumentParser(description='tianrang-ocr')
    parser.add_argument('--host', type=str, default='0.0.0.0', help=' ')
    parser.add_argument('--port', type=int, default=8080, help=' ')
    args = parser.parse_args()
    PATH = os.path.abspath(os.path.dirname(__file__))
    os.chdir(PATH)
    app.run(host=args.host, port=args.port)
