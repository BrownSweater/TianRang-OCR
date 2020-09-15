#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 16:14
# @Author : jj.wang

import cv2
import torch
import numpy as np


class PSEPostProcess():
    def __init__(self, thresh=1, kernel_num=7, scale=1, min_kernel_area=400, min_area=5, min_score=0.93):
        self.thresh = thresh
        self.kernel_num = kernel_num
        self.scale = scale
        self.min_kernel_area = min_kernel_area
        self.min_area = min_area
        self.min_score = min_score
        self.pse = self.load_pse()

    def __call__(self, pred, batch, *args, **kwargs):
        boxes_batch = []
        scores_batch = []
        scores = torch.sigmoid(pred[:, 0, :, :])
        outputs = self.binarize(pred)
        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:self.kernel_num, :, :] * text
        for batch_index in range(pred.size(0)):
            height, width = batch['shape'][batch_index]
            score = scores.data.cpu().numpy()[batch_index].astype(np.float32)
            kernel = kernels.data.cpu().numpy()[batch_index].astype(np.uint8)
            bboxes, s = self.get_boxes(score, kernel, height, width)
            boxes_batch.append(bboxes)
            scores_batch.append(s)
        return boxes_batch, scores_batch

    def get_boxes(self, score, kernel, height, width):
        pred = self.pse(kernel, self.min_kernel_area / (self.scale * self.scale))
        label_num = np.max(pred) + 1
        scale = (width * 1.0 / pred.shape[1], height * 1.0 / pred.shape[0])
        bboxes = []
        scores = []
        for i in range(1, label_num):
            points = np.array(np.where(pred == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < self.min_area / (self.scale * self.scale):
                continue
            score_i = np.mean(score[pred == i])
            if score_i < self.min_score:
                continue
            rect = cv2.minAreaRect(points)

            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox)
            scores.append(score_i)
        return bboxes, scores

    def load_pse(self):
        try:
            from psenet.pse import pse
        except:
            from pypse import pse
        return pse

    def binarize(self, pred):
        return (torch.sign(pred - self.thresh) + 1) / 2