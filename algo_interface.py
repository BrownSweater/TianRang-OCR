#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/7 16:12
# @Author : jj.wang

import os
import cv2
import json
import torch
import pprint
import platform
import numpy as np
from utils.utils import draw_bbox, cv2ImgAddText
from easydict import EasyDict as edict

from flask_utils.log import Logger
from tools.det_predict import DetModel
from tools.rec_predict import RecModel
from data_loader.modules.random_crop_data import CropWordBox


class AlgoInterface(object):
    def __init__(self, json_config_path):
        self.PATH = os.path.abspath(os.path.dirname(__file__))
        self.config = self.init_args(json_config_path)
        self.logger = self.init_loger('algo_interface')
        self.logger.info(pprint.pformat(self.config))
        self.logger.info('========================================================================')
        self._select_device()
        self.init_model()
        self.vis = self.config.vis

    @staticmethod
    def init_args(config_path):
        '''
        :param config_path: json format config
        :return: edict object
        '''

        with open(config_path, 'r') as f:
            config = edict(json.load(f))
        return config

    def init_loger(self, name):
        logger = Logger(os.path.join(self.PATH, self.config.log_dir), name).logger()
        return logger

    def init_model(self):
        '''
        init det model and rec model
        :return:
        '''
        self.det_model = DetModel(self.config.det_model_path, self.config.det_box_thresh, self.config.det_pos_thresh,
                                  gpu_id=self.config.device_id)
        self.logger.info('init det model')
        self.logger.info('========================================================================')
        if self.config.use_hyperlpr:
            from hyperlpr import PR
            self.rec_model = PR
            self.logger.info('use hyperlpr as rec model')
        else:
            self.rec_model = RecModel(self.config.rec_model_path, gpu_id=self.config.device_id)
        self.logger.info('init rec model')
        self.logger.info('========================================================================')

    def process_input(self, input):
        '''

        :param input: BGR ndarray or img path
        :return:
        '''
        if isinstance(input, str):
            assert os.path.exists(input), 'file is not exists'
            img = cv2.imread(input)
        elif type(input) is np.ndarray:
            img = input
        else:
            raise ValueError
        return img

    def _select_device(self):
        '''
        use gpu or cpu and set cpu threads
        :return:
        '''
        torch.set_num_threads(self.config.cpu_num_thread)
        if self.config.device_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            self.logger.info('Use CPU')

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device_id)
            self.logger.info(f'Use GPU: {self.config.device_id}')
            self.config.device_id = 0

    @staticmethod
    def _vis(img, box, text, prob, offset=(0, 0)):
        im_h, im_w = img.shape[:2]
        text_scale = max(1, im_h / 1600.)  # 1600.
        text_thickness = 1
        x1, y1 = int(box[0][0]), int(box[0][1])
        x1 += offset[0]
        y1 += offset[1]
        rec_text = text
        prob_text = f'{prob}'
        t_size = cv2.getTextSize(rec_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)[0]
        img = cv2ImgAddText(img, text, (x1, y1 - int(t_size[1] * 2)), textColor=(225, 255, 255),
                            textSize=t_size[1] * 2)
        img = cv2ImgAddText(img, prob_text, (x1, y1 - int(t_size[1] * 4)), textColor=(225, 255, 255),
                            textSize=t_size[1] * 2)
        return img

    def __call__(self, img, *args, **kwargs):
        '''

        :param img: BGR ndarray or img path
        :param args:
        :param kwargs:
        :return:
        '''
        img = self.process_input(img)
        preds, boxes_list, score_list, det_time = self.det_model.predict(img, is_output_polygon=False,
                                                                         short_size=self.config.det_short_size)
        if self.vis:
            draw_img = draw_bbox(img, boxes_list)
        else:
            draw_img = None
        results = []
        rec_time = 0
        for i, box in enumerate(boxes_list):
            tmp_result = None
            for r in self.config.rec_crop_ratio:
                # rec_img = CropWordBox.crop_image_by_bbox(img, box, self.config.rec_crop_ratio)
                rec_img = CropWordBox.crop_image_by_bbox(img, box, r)
                text, prob, t = self.rec_model.predict(rec_img)
                rec_time += t
                prob = round(prob, 3)
                det_score = round(score_list[i].astype(float), 3)
                if tmp_result is None or prob > tmp_result['prob']:
                    tmp_result = {'box': box.tolist(), 'recognition': text, 'prob': prob}
            results.append(tmp_result)
        if self.vis:
            for result in results:
                draw_img = self._vis(draw_img, result['box'], result['recognition'], result['prob'])
        self.logger.info(f'========================================================================\n'
                         f'det preprocess time: {det_time[0] * 1000: .1f}ms \n'
                         f'det inference time: {det_time[1] * 1000: .1f}ms \n'
                         f'det postprocess time: {det_time[2] * 1000: .1f}ms \n'
                         f'det total time: {det_time[3] * 1000: .1f}ms \n'
                         f'rec total time: {rec_time * 1000: .1f}ms \n'
                         f'========================================================================\n')
        return results, draw_img


if __name__ == '__main__':
    # macos
    if platform.system() == 'Darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    PATH = os.path.abspath(os.path.dirname(__file__))
    model = AlgoInterface(json_config_path=os.path.join(PATH, 'interface_config.json'))
    results, draw_img = model('苏A5RH08.jpg')
    print(results)
    cv2.imshow('test', draw_img)
    cv2.waitKey(0)

if __name__ != '__main__':
    if platform.system() == 'Darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    PATH = os.path.abspath(os.path.dirname(__file__))
    model = AlgoInterface(json_config_path=os.path.join(PATH, 'interface_config.json'))
