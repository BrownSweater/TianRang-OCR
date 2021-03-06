#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/17 14:35
# @Author : jj.wang

import os
import sys

project = 'tianrang-ocr'  # 工作项目根目录
sys.path.append(os.getcwd().split(project)[0] + project)
import time
import math
import cv2
import torch
import numpy as np
from data_loader import get_transforms
from model import get_model, get_post_processing


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class DetModel:
    def __init__(self, model_path, box_thresh=0.7, pos_thresh=0.3, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['args']['pretrained'] = False
        self.model = get_model(config['arch'])
        # config['post_processing']['args']['unclip_ratio'] = 3
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = box_thresh
        self.post_process.thresh = pos_thresh
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(config)

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, data, is_output_polygon=False, short_size: int = 736):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param data: 图片地址，或者RGB的ndarray
        :return:
        '''
        t0 = time.time()
        if isinstance(data, str):
            assert os.path.exists(data), 'file is not exists'
            img = cv2.imread(data, 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif type(data) is np.ndarray:
            img = data.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            t1 = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            t2 = time.time()
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t3 = time.time()
            total_time = t3 - t0
            pre_time = t1 - t0
            inf_time = t2 - t1
            post_time = t3 - t2
            t = [pre_time, inf_time, post_time, total_time]
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default='model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.7, help='the thresh of post_processing', type=float)
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.utils import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)
    # 初始化网络
    model = DetModel(args.model_path, post_p_thre=args.thre, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)
