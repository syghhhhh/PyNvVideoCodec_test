# -*- coding: utf-8 -*-
# @Time    : 2024/3/14 16:31
# @Author  : 施昀谷
# @File    : inf_prepare.py

import csv
import random
from moviepy.editor import VideoFileClip
from os.path import join, exists, basename, dirname, splitext
from os import makedirs, remove, listdir
import numpy as np
import cv2
import subprocess
# import os
from tqdm import tqdm
from glob import glob
from shutil import rmtree, copyfile


def get_face(video_file_path, landmark_openface_path, result_folder):
    def load_landmark_openface(csv_path):
        '''
        load openface landmark from .csv file
        '''

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            data_all = [row for row in reader]
        x_list = []
        y_list = []
        for row_index, row in enumerate(data_all[1:]):
            frame_num = float(row[0])
            if int(frame_num) != row_index + 1:
                return None
            x_list.append([float(x) for x in row[5:5 + 68]])
            y_list.append([float(y) for y in row[5 + 68:5 + 68 + 68]])
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        landmark_array = np.stack([x_array, y_array], 2)  # [frame_num, 68, 2]
        return landmark_array

    def compute_crop_radius(video_size, landmark_data_clip, random_scale=None):
        """
        判断是否需要裁剪人脸，计算裁剪半径
        :param video_size:
        :param landmark_data_clip:
        :param random_scale:
        :return:
        """
        video_w, video_h = video_size[0], video_size[1]
        landmark_max_clip = np.max(landmark_data_clip, axis=1)
        if random_scale is None:
            random_scale = random.random() / 10 + 1.05
        else:
            random_scale = random_scale
        radius_h = (landmark_max_clip[:, 1] - landmark_data_clip[:, 29, 1]) * random_scale
        radius_w = (landmark_data_clip[:, 54, 0] - landmark_data_clip[:, 48, 0]) * random_scale
        radius_clip = np.max(np.stack([radius_h, radius_w], 1), 1) // 2
        radius_max = np.max(radius_clip)
        radius_max = (np.int(radius_max / 4) + 1) * 4
        radius_max_1_4 = radius_max // 4
        clip_min_h = landmark_data_clip[:, 29, 1] - radius_max
        clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max_1_4
        clip_min_w = landmark_data_clip[:, 33, 0] - radius_max - radius_max_1_4
        clip_max_w = landmark_data_clip[:, 33, 0] + radius_max + radius_max_1_4
        if min(clip_min_h.tolist() + clip_min_w.tolist()) < 0:
            return False, f'min(clip_min_h.tolist() + clip_min_w.tolist()) = {min(clip_min_h.tolist() + clip_min_w.tolist())} < 0'
        elif max(clip_max_h.tolist()) > video_h:
            return False, f'max(clip_max_h.tolist()) = {max(clip_max_h.tolist())} > video_h = {video_h}'
        elif max(clip_max_w.tolist()) > video_w:
            return False, f'max(clip_max_w.tolist()) = {max(clip_max_w.tolist())} > video_w = {video_w}'
        elif max(radius_clip) > min(radius_clip) * 1.5:
            return False, f'max(radius_clip) = {max(radius_clip)} > min(radius_clip) * 1.5 = {min(radius_clip) * 1.5}'
        else:
            return True, radius_max

    def mask_gen(path):
        mask_clip = VideoFileClip(path, has_mask=True)
        for mask in mask_clip.mask.iter_frames():
            yield mask
        mask_clip.close()

    face_folder = join(result_folder, 'face')
    body_folder = join(result_folder, 'body')
    makedirs(face_folder, exist_ok=True)
    makedirs(body_folder, exist_ok=True)
    # mask_clip = VideoFileClip(video_file_path, has_mask=True)
    # mask_list = []
    # for mask in mask_clip.mask.iter_frames():
    #     mask_list.append(mask)
    # mask_clip.close()
    mask_gen = mask_gen(video_file_path)
    landmark_openface_data = load_landmark_openface(landmark_openface_path).astype(np.int)
    videoCapture = cv2.VideoCapture(video_file_path)
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_length = min(frames, landmark_openface_data.shape[0])
    video_h, video_w = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    list_xy = []
    for i in range(frame_length):
        if i < 2:
            offset = 2 - i
        elif i < frame_length - 2:
            offset = 0
        else:
            offset = frame_length - 3 - i
        crop_flag, radius_clip = compute_crop_radius((video_w, video_h), landmark_openface_data[(i - 2 + offset):(i + 3 + offset), :, :], random_scale=1.05)
        radius_clip = int(radius_clip * 1.25)
        radius_clip_1_4 = radius_clip // 4
        frame_index = i
        ret, source_frame_data = videoCapture.read()
        if not ret:
            break
        alpha = next(mask_gen) * 255
        alpha = np.expand_dims(alpha, axis=2)
        png_new = np.concatenate([source_frame_data, alpha], axis=2)
        frame_landmark = landmark_openface_data[frame_index, :, :]
        x1 = frame_landmark[33, 0] - radius_clip - radius_clip_1_4
        x2 = frame_landmark[33, 0] + radius_clip + radius_clip_1_4
        y1 = frame_landmark[29, 1] - radius_clip
        y2 = frame_landmark[29, 1] + radius_clip * 2 + radius_clip_1_4
        # crop_face_data = source_frame_data[y1:y2, x1:x2, :].copy()
        # res_crop_face_frame_path = join(face_folder, str(frame_index).zfill(6) + '.jpg')
        # if exists(res_crop_face_frame_path):
        #     remove(res_crop_face_frame_path)
        # cv2.imwrite(res_crop_face_frame_path, crop_face_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        crop_face_data = png_new[y1:y2, x1:x2, :].copy()
        res_crop_face_frame_path = join(face_folder, str(frame_index).zfill(6) + '.png')
        if exists(res_crop_face_frame_path):
            remove(res_crop_face_frame_path)
        cv2.imwrite(res_crop_face_frame_path, crop_face_data)
        list_xy.append([x1, x2, y1, y2])
        cv2.imwrite(join(body_folder, str(frame_index).zfill(6) + '.png'), png_new)
    videoCapture.release()
    # 保存为npy文件
    list_landmark = np.array(list_xy)
    np.save(join(result_folder, 'xy.npy'), list_landmark)


def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        if not ret:
            break
        result_path = join(save_dir, str(i).zfill(6) + '.jpg')
        # cv2.imwrite(result_path, frame)
        cv2.imwrite(result_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return (int(frame_width), int(frame_height))


if __name__ == '__main__':
    t = -2
    if t == -2:
        id_name_list = ['0006', '0008', '0009', '0010', '0011', '0012', '0013', '0016', '0017', '0018']
        for id_name in id_name_list[1:]:
            # 测试图片提取
            video_path = rf'E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\video.mp4'
            save_dir = rf'E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\video'
            makedirs(save_dir, exist_ok=True)
            extract_frames_from_video(video_path, save_dir)
            # 手动调用paddle去图片绿幕
    elif t == -1:
        id_name_list = ['0006', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018']
        for id_name in id_name_list:
            # 去完绿幕的图片生成mov
            cmd = rf'ffmpeg -i E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\body\%06d_rgba.png -vcodec qtrle E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\video.mov'
            subprocess.run(cmd, shell=True)
            # 生成最后的素材
            get_face(rf'E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\video.mov',
                     rf'E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}\DINet\video.csv',
                     rf'E:\Generating_offline_2D_lip-sync_videos\inf_data\{id_name}')
    elif t == 0:
        # 测试单个效果
        id = '6077'
        inf_folder = r'D:\Generating_offline_2D_lip-sync_videos\inf_data'
        mov_path = join(inf_folder, id, 'DINet2', 'video.mov')
        csv_path = join(inf_folder, id, 'DINet2', 'video.csv')
        result_folder = join(inf_folder, id)
        get_face(mov_path, csv_path, result_folder)
    elif t == 1:
        # 批量处理图片
        inf_folder = r'E:\Generating_offline_2D_lip-sync_videos\inf_data'
        for id in tqdm(listdir(inf_folder)):
            if exists(join(inf_folder, id, 'xy.npy')):
                remove(join(inf_folder, id, 'xy.npy'))
            if exists(join(inf_folder, id, 'face')):
                rmtree(join(inf_folder, id, 'face'))
            if exists(join(inf_folder, id, 'body')):
                rmtree(join(inf_folder, id, 'body'))
            mov_path = join(inf_folder, id, 'DINet2', 'video.mov')
            csv_path = join(inf_folder, id, 'DINet2', 'video.csv')
            if not exists(mov_path) or not exists(csv_path):
                continue
            get_face(mov_path, csv_path, join(inf_folder, id))
    elif t == 2:
        # 模型文件复制,参考图片复制
        inf_folder = r'D:\Generating_offline_2D_lip-sync_videos\inf_data'
        for id in tqdm(listdir(inf_folder)):
            for model in glob(join(inf_folder, id, 'DINet2', '*.pth')):
                if not exists(join(inf_folder, id, basename(model))):
                    copyfile(model, join(inf_folder, id, basename(model)))
            if exists(join(inf_folder, id, 'DINet2', 'reference_images')):
                # 复制整个文件夹
                makedirs(join(inf_folder, id, 'reference_images'), exist_ok=True)
                for img in listdir(join(inf_folder, id, 'DINet2', 'reference_images')):
                    if not exists(join(inf_folder, id, 'reference_images', img)):
                        copyfile(join(inf_folder, id, 'DINet2', 'reference_images', img), join(inf_folder, id, 'reference_images', img))
