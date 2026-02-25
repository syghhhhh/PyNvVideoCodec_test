# -*- coding: utf-8 -*-
# @Time    : 2023/9/6 15:33
# @Author  : 施昀谷
# @File    : pre_picture_merge.py

# import os
# import gc
# import sys
# import json
# import math
# import glob
# import time
# import shutil
# import random
# import argparse
# import threading
# import subprocess
# import traceback
import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from shutil import rmtree
from os.path import join, exists, basename, dirname, splitext
from os import listdir, makedirs, remove, walk
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torch.nn.functional as F
# import asyncio
# import websockets
# import websockets_routes
import cv2
# import soundfile as sf
# import moviepy.editor as mp

from config import parameters
from utils.file_transfer import upload_oss


class CustomError(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_correct_coordinates(x, y, add_width, add_height, background_width, background_height):
    """
    返回素材和背景图片叠加的坐标
    :param x: 素材左上角x坐标
    :param y: 素材左上角y坐标
    :param add_width: 素材宽度
    :param add_height: 素材高度
    :param background_width: 背景图片宽度
    :param background_height: 背景图片高度
    :return:
    """
    # x坐标小于0,素材从-x开始,背景图片从0开始; x坐标大于0,素材从0开始,背景图片从x开始
    # 如果素材全部都在左边或者右边,则add_x0=
    # if x < 0:
    #     add_x0 = -x
    #     bg_x0 = 0
    # else:
    #     add_x0 = 0
    #     bg_x0 = x
    add_x0, bg_x0 = max(0, -x), max(0, x)

    # if y < 0:
    #     add_y0 = -y
    #     bg_y0 = 0
    # else:
    #     add_y0 = 0
    #     bg_y0 = y
    add_y0, bg_y0 = max(0, -y), max(0, y)

    # if x + add_width > background_width:
    #     add_x1 = background_width - x
    #     bg_x1 = background_width
    # else:
    #     add_x1 = add_width
    #     bg_x1 = x + add_width
    add_x1, bg_x1 = min(add_width, background_width - x), min(background_width, x + add_width)

    # if y + add_height > background_height:
    #     add_y1 = background_height - y
    #     bg_y1 = background_height
    # else:
    #     add_y1 = add_height
    #     bg_y1 = y + add_height
    add_y1, bg_y1 = min(add_height, background_height - y), min(background_height, y + add_height)

    return [add_x0, add_x1, add_y0, add_y1], [bg_x0, bg_x1, bg_y0, bg_y1]


def picture_merge(workspace, human, background, add_name, width, height, style, nodes):
    # 处理背景图片
    background_picture = cv2.imread(background)
    background_picture = cv2.resize(background_picture, (int(float(width)), int(float(height))))

    # 处理额外添加的素材
    # nodes: [{"type":1,"url":"https:\/\/bsddata.oss-cn-hangzhou.aliyuncs.com\/virtual_live\/virtual_video_cover\/1687164513hNGMW5d2dr.mp4","style":{"x":112,"y":173,"width":1022,"height":574}}]
    if len(nodes) > 0:
        node = nodes[0]
        # 读取图片或视频
        if node['type'] == 1:
            # 视频素材
            video_path = join(workspace, add_name)
            # 读取视频的第一帧
            video = cv2.VideoCapture(video_path)
            success, frame = video.read()
            if success:
                add_picture = frame.copy()
            else:
                raise CustomError('video read error')
            del video
        elif node['type'] == 2:
            # 图片素材
            add_picture = cv2.imread(join(workspace, add_name))
        else:
            raise CustomError('nodes type value error')

        # 处理图片
        add_style = node['style']  # {"x":112,"y":173,"width":1022,"height":574}
        add_picture = cv2.resize(add_picture, (int(add_style['width']), int(add_style['height'])))

        # 获取素材和背景图片叠加的坐标
        [add_x0, add_x1, add_y0, add_y1], [bg_x0, bg_x1, bg_y0, bg_y1] = (
            get_correct_coordinates(int(float(add_style['x'])), int(float(add_style['y'])),
                                    add_picture.shape[1], add_picture.shape[0],
                                    background_picture.shape[1], background_picture.shape[0])
        )

        # 合成图片
        for c in range(0, 3):
            # background_picture[y_add:y_add + add_picture.shape[0], x_add:x_add + add_picture.shape[1], c] = add_picture[:, :, c]
            background_picture[bg_y0:bg_y1, bg_x0:bg_x1, c] = add_picture[add_y0:add_y1, add_x0:add_x1, c]

    # 处理虚拟人图片
    human_picture = cv2.imread(human, cv2.IMREAD_UNCHANGED)
    height_human = int(float(style['height']))
    scale = height_human / human_picture.shape[0]
    width_human = int(human_picture.shape[1] * scale)
    human_picture = cv2.resize(human_picture, (width_human, height_human))

    # 获取素材和背景图片叠加的坐标
    [add_x0, add_x1, add_y0, add_y1], [bg_x0, bg_x1, bg_y0, bg_y1] = (
        get_correct_coordinates(int(float(style['x'])), int(float(style['y'])),
                                human_picture.shape[1], human_picture.shape[0],
                                background_picture.shape[1], background_picture.shape[0])
    )

    # 合成虚拟人图片
    for c in range(0, 3):
        # part1 = human_picture[:, :, c] * (human_picture[:, :, 3] / 255.0)
        # part21 = background_picture[y:y + human_picture.shape[0], x:x + human_picture.shape[1], c]
        # part22 = 1.0 - human_picture[:, :, 3] / 255.0
        # part2 = part21 * part22
        # background_picture[y:y + human_picture.shape[0], x:x + human_picture.shape[1], c] = part1 + part2
        part1 = human_picture[add_y0:add_y1, add_x0:add_x1, c] * (human_picture[add_y0:add_y1, add_x0:add_x1, 3] / 255.0)
        part21 = background_picture[bg_y0:bg_y1, bg_x0:bg_x1, c]
        part22 = 1.0 - human_picture[add_y0:add_y1, add_x0:add_x1, 3] / 255.0
        part2 = part21 * part22
        background_picture[bg_y0:bg_y1, bg_x0:bg_x1, c] = part1 + part2

    background_picture = background_picture.astype(np.uint8)
    # 保存图片
    save_path = join(workspace, 'pre_merged.png')
    cv2.imwrite(save_path, background_picture)
    # 图片上传oss
    merge_id = basename(workspace)
    upload_oss(save_path, parameters, f'{merge_id}.png')
    # 返回图片地址
    url_preview = f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{merge_id}.png'
    return url_preview


if __name__ == '__main__':
    picture_merge(
        workspace=r'D:\project\Generating_offline_2D_lip-sync_videos\workspace\230906140149Wav2Lip0001',
        human=r'D:\project\Generating_offline_2D_lip-sync_videos\pre_human\0001.png',
        background=r'D:\project\Generating_offline_2D_lip-sync_videos\workspace\230906140149Wav2Lip0001\background.png',
        width=1920,
        height=1080,
        style={'x': 1440, 'y': 107, 'height': 800},
        nodes=[{"type": 1,
                "url": "https:\/\/bsddata.oss-cn-hangzhou.aliyuncs.com\/virtual_live\/virtual_video_cover\/1687164513hNGMW5d2dr.mp4",
                "style": {"x": 50, "y": 50, "width": 1920, "height": 1080}}],
        add_name='add_video.mp4'
    )
