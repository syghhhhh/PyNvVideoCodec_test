# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 10:13
# @Author  : 施昀谷
# @File    : data_prepare.py

import os
import re
import cv2
import math
import random
from os import listdir, makedirs, remove, rename
from os.path import join, exists, basename, dirname, isfile
import json
import shutil
import datetime
import traceback
import numpy as np
from glob import glob
from moviepy.editor import VideoFileClip
from PIL import ImageFont, ImageDraw, Image
import librosa
import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from config import parameters
from utils.callback import callback_merge, callback_base_task
from utils.file_transfer import download_requests
from utils.pre_picture_merge import get_correct_coordinates
from model.DINet_master.models.DINetV3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chinese_hubert_large_model_path = join(parameters['model_folder'], 'chinese-hubert-large')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(chinese_hubert_large_model_path)
model_hubert = HubertModel.from_pretrained(chinese_hubert_large_model_path)
model_hubert = model_hubert.to(device)
model_hubert.eval()
mask_png_path = join(parameters['path_base'], 'mask.png')
mask_tensor = torch.tensor(cv2.imread(mask_png_path, cv2.IMREAD_UNCHANGED) / 255.0, dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)


def error_handling(item, merge_id, status_path, log_path, error_reason, base_task=False):
    # 修改状态文件为error
    with open(status_path, 'w') as f:
        f.write(json.dumps({'status': 'error'}))
    # 添加错误原因到日志文件
    with open(log_path, 'a') as f:
        f.write('\n' + datetime.datetime.now().strftime("[%H:%M:%S]") + 'error reason: ' + error_reason)
    if not base_task:
        # 回调错误信息
        callback_merge(
            callbackUrl=item.callbackUrl,
            merge_id=merge_id,
            duration=0,
            videoUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{merge_id}.mp4',
            videoName=f'{merge_id}.mp4',
            result='fail',
            localPath=join(parameters['workspace'], merge_id, 'result.mp4'),
            failReason=error_reason,
            horizontal=item.width,
            vertical=item.height,
            coverUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{merge_id}.png',
        )
    else:
        # 回调错误信息
        callback_base_task(
            callbackUrl=item.callbackUrl,
            merge_id=merge_id,
            videoUrl='',
            MaskUrl='',
            result='fail',
            failReason=error_reason,
        )
    ret = {
        'code': 0,
        'meg': error_reason,
        'id_merge': merge_id,
    }
    return ret




def log_title_write(log_path, title):
    """
    在日志文件中添加标题和当前时间
    :param log_path: 日志文件
    :param title: 标题内容
    :return:
    """
    with open(log_path, 'a') as f:
        f.write('\n' + datetime.datetime.now().strftime("[%H:%M:%S]") + '------------------------------------' + title + '------------------------------------')


def log_content_write(log_path, content):
    """
    在日志文件中添加内容和当前时间
    :param log_path: 日志文件
    :param content: 需要写入的内容
    :return:
    """
    with open(log_path, 'a') as f:
        f.write('\n' + datetime.datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "]" + content)


def os_system(cmd, log_file):
    """
    执行cmd命令,并把cmd命令写入日志文件
    :param cmd: 需要执行的命令
    :param log_file: 日志文件
    :return:
    """
    log_content_write(log_file, cmd)
    os.system(cmd)


def check_existence(file_path, log_file):
    """
    检查文件是否存在,不存在就写入日志文件
    :param file_path: 文件路径
    :param log_file: 日志文件
    :return:
    """
    if not exists(file_path):
        log_content_write(log_file, f'{file_path} does not exist')
        return False
    return True


def get_video_fps(video_path):
    """
    使用cv2读取视频获取视频帧率
    :param video_path: 视频路径
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查是否成功打开视频
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 释放视频捕捉对象
    cap.release()
    return fps


def get_video_duration(video_path):
    """
    使用cv2读取视频获取视频时长
    :param video_path: 视频路径
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查是否成功打开视频
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 计算视频的时长
    duration = frame_count / fps
    # 释放视频文件
    cap.release()
    # 返回视频的时长
    return duration


def transcode_video(path_video, log_file):
    """
    检测视频帧率,如果不是25帧/s,就用ffmpeg转换为25帧/s
    :param path_video: 视频路径
    :param log_file: 日志文件
    :return:
    """
    fps = get_video_fps(path_video)
    if fps != 25:
        # 如果视频的帧率不是25，就转换为25帧
        len1 = len(path_video.split('.')[-1]) + 1
        new_path_video = path_video[:-len1] + '_25fps' + path_video[-len1:]
        os_system(f'ffmpeg -y -i {path_video} -r 25 {new_path_video} -loglevel quiet', log_file)
        # 删除原来的视频并把25帧率的重命名为原来的名字
        remove(path_video)
        os.rename(new_path_video, path_video)


def read_srt_file(file_path):
    """
    读取srt字幕文件,返回字幕列表
    :param file_path: srt字幕文件路径
    :return: [[frame_num_start, frame_num_end, text], ...]
    """

    def time_to_frame(time):
        # 把字幕文件中的时间戳转化为相应的25帧率视频的帧数
        # '00:00:02,350' 转成帧数, 25帧/s
        time = time.replace(',', ':')
        h, m, s, f = re.split(':', time)
        frame = int(h) * 3600 * 25 + int(m) * 60 * 25 + int(s) * 25 + int(f) // 40
        return frame

    subtitles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                i += 1
                timestamp = lines[i].strip()
                start_time, end_time = timestamp.split(" --> ")
                i += 1
                text = lines[i].strip()
                subtitles.append((start_time, end_time, text))
            i += 1
    list1 = []
    for subtitle in subtitles:
        list1.append([time_to_frame(subtitle[0]), time_to_frame(subtitle[1]), subtitle[2]])
    return list1


def mask_gen(path):
    mask_clip = VideoFileClip(path, has_mask=True)
    for mask in mask_clip.mask.iter_frames():
        yield mask
    mask_clip.close()


def download_gif(add_name, merge_folder, log_file):
    """
    下载gif,转换为对应的25帧率的png序列帧,返回png序列帧的文件夹路径
    :param add_name: 额外素材的文件名
    :param merge_folder: 合成任务的文件夹路径
    :param log_file: 日志文件路径
    :return: png序列帧的文件夹路径
    """
    gif_path = join(merge_folder, add_name)
    gif_mp4_path = join(merge_folder, add_name[:-3] + "mp4")
    os_system(f'ffmpeg -y -i {gif_path} {gif_mp4_path} -loglevel quiet', log_file)
    # 读取gif
    mask_gen_1 = mask_gen(gif_path)
    mask_clip = VideoFileClip(gif_path, has_mask=True)
    fps = mask_clip.fps
    duration = mask_clip.duration
    # 释放mask_clip
    mask_clip.close()
    log_content_write(log_file, f'gif {gif_path} fps: {fps}')
    mask = next(mask_gen_1)
    png_folder = join(merge_folder, add_name[:-4])
    makedirs(png_folder, exist_ok=True)
    # 把mp4转成png序列帧
    os_system(f'ffmpeg -y -i {gif_mp4_path} {png_folder}/%06d.png -loglevel quiet', log_file)
    # 判断原gif是否带alpha信息
    if np.min(mask) == 1.0:
        # 如果原gif不带alpha信息,就不做处理
        log_content_write(log_file, f'gif {gif_mp4_path} has no alpha')
    else:
        # 如果原gif带alpha信息,就把alpha信息保存到png的alpha通道中
        log_content_write(log_file, f'gif {gif_mp4_path} has alpha')
        png_frames = len(listdir(png_folder))
        total_frames = int(fps * duration)
        # 确保png序列帧的数量和gif的帧数相同
        if png_frames != total_frames:
            if png_frames > total_frames:
                # 删去后面多余的png
                for i in range(total_frames, png_frames):
                    if exists(join(png_folder, f'{i + 1:06d}.png')):
                        os.remove(join(png_folder, f'{i + 1:06d}.png'))
            else:
                # 复制最后一帧使得png序列帧的数量和gif的帧数相同
                for i in range(png_frames, total_frames):
                    shutil.copy(join(png_folder, f'{png_frames:06d}.png'), join(png_folder, f'{i + 1:06d}.png'))
        # 重新生成透明png
        mask_gen_1 = mask_gen(gif_path)
        for png_path in glob(join(png_folder, '*.png')):
            frame = cv2.imread(png_path)
            # 新增alpha通道
            frame = np.concatenate([frame, np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)], axis=2)
            mask = next(mask_gen_1)
            frame[:, :, 3] = mask * 255
            cv2.imwrite(png_path, frame)
    if fps != 25:
        # 把png序列帧转成mov
        gif_mp4_path_new = join(merge_folder, add_name[:-4] + "_new.mov")
        os_system(f'ffmpeg -y -framerate {fps} -i {png_folder}/%06d.png -c:v qtrle -pix_fmt rgba {gif_mp4_path_new} -loglevel quiet', log_file)
        # mov转25帧率
        gif_mp4_path_new_25 = join(merge_folder, add_name[:-4] + "_new_25.mov")
        os_system(f'ffmpeg -y -i {gif_mp4_path_new} -c:v prores_ks -profile:v 4444 -r 25 -crf 0 -preset slow {gif_mp4_path_new_25} -loglevel quiet', log_file)
        # 删除原来png_folder中的png,并把mov转成png序列帧保存到png_folder中,带alpha通道
        shutil.rmtree(png_folder)
        makedirs(png_folder, exist_ok=True)
        os_system(f'ffmpeg -y -i {gif_mp4_path_new_25} -pix_fmt rgba {png_folder}/%06d.png -loglevel quiet', log_file)
    return png_folder


def download_background(item, merge_folder, config_dict, log_file):
    """
    下载背景, 如果背景为视频, 进行帧率检测和转换
    :param item: 任务信息
    :param merge_folder: 合成任务的文件夹路径
    :param config_dict: 配置信息
    :param log_file: 日志文件路径
    :return: 背景素材路径, 是否成功, 错误原因
    """
    def download(item, merge_folder, config_dict, log_file):
        background_file_name = 'background.' + item.backgroundUrl.split('.')[-1]
        path_bg = join(merge_folder, background_file_name)
        status = download_requests(item.backgroundUrl, merge_folder, background_file_name, log_file)
        if not status:
            return {}, False, 'download background error'
        # 若背景为视频,进行帧率检测和转换
        if config_dict['background_type'] == 'video':
            transcode_video(path_bg, log_file)
        return path_bg, True, ''

    if (not config_dict['child']) or (item.childId == 1):
        # 如果为单一任务,或者为复合任务的第一个子任务,下载背景
        path_background, st, error_reason = download(item, merge_folder, config_dict, log_file)
        if not st:
            return {}, False, error_reason
        # 如果为复合任务,且这个任务是第一个子任务,处理完的背景素材路径写入cfg_dict
        if config_dict['child']:
            cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
            with open(cfg_json, 'r') as f:
                cfg_dict_list = json.load(f)
            # 处理完的背景素材路径写入cfg_dict
            update_cfg_dict(cfg_dict_list, 1, 'backgroundPath', path_background, cfg_json)
    else:
        # 如果为复合任务,且这个任务不是第一个子任务,读取cfg_json
        cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
        with open(cfg_json, 'r') as f:
            cfg_dict_list = json.load(f)
        # 找到childId为item.childId - 1的子任务的背景
        dict_last = [x for x in cfg_dict_list if (int(x['childId']) == item.childId - 1)]
        if len(dict_last) != 1:
            return {}, False, 'cfg_json error'
        dict_last = dict_last[0]
        if dict_last['backgroundUrl'] == item.backgroundUrl:
            # 如果这个任务的背景和上一个子任务的背景相同,就不下载背景,path_background为上一个子任务的背景
            path_background = dict_last['backgroundPath']
        else:
            # 如果这个任务的背景和上一个子任务的背景不同,就下载背景
            path_background, st, error_reason = download(item, merge_folder, config_dict, log_file)
            if not st:
                return {}, False, error_reason
        # 处理完的背景素材路径写入cfg_dict
        update_cfg_dict(cfg_dict_list, item.childId, 'backgroundPath', path_background, cfg_json)
    # 下载完之后,判断:如果背景为视频,则提取背景视频的音频并调整音量
    if config_dict['background_type'] == 'video':
        volume = item.backgroundVolume * 0.02
        save_path = join(merge_folder, 'background_audio.wav')
        os_system(f'ffmpeg -i {path_background} -filter:a \"volume={volume}\" {save_path} -loglevel quiet', log_file)
    return path_background, True, ''


def download_add_materials(item, config_dict, merge_folder, audio_duration, log_file):
    """
    下载额外素材,如果为视频,进行帧率检测和转换;如果为gif,转换为png序列帧
    :param item: 任务信息
    :param config_dict: 配置信息
    :param merge_folder: 合成任务的文件夹路径
    :param audio_duration: 音频的最大时长
    :param log_file: 日志文件路径
    :return: [额外素材的文件名列表, 路径字典], 是否成功, 错误原因
    """
    def download_add(node, merge_folder, add_name, log_file):
        status = download_requests(node['url'], merge_folder, add_name, log_file)
        if not status:
            return {}, False, 'download add_materials error'
        if node['type'] == 1:
            # 若为视频,进行帧率检测和转换
            transcode_video(join(merge_folder, add_name), log_file)
            return join(merge_folder, add_name), True, ''
        elif node['type'] == 2:
            # 若为图片,就不做处理
            return join(merge_folder, add_name), True, ''
        elif node['type'] == 3:
            # 若为gif,就转换为png序列帧
            return download_gif(add_name, merge_folder, log_file), True, ''
        else:
            return {}, False, 'node["type"] out of range'

    add_num = len(item.nodes)  # 额外素材的数量
    add_name_list = []
    add_path_dict = {}
    if add_num > 0:
        for i in range(add_num):
            node = item.nodes[i]
            add_name = f'add_{node["level"]}.' + node['url'].split('.')[-1]  # add_1.mp4, add_2.png,...
            add_name_list.append(add_name)
            if (not config_dict['child']) or (item.childId == 1):
                # 如果为单一任务,或者为复合任务的第一个子任务,下载额外素材
                ret, st, error_reason = download_add(node, merge_folder, add_name, log_file)
                if not st:
                    return ret, st, error_reason
                add_path_dict[add_name] = ret
            else:
                # 如果为复合任务,且这个任务不是第一个子任务,读取cfg_json
                cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
                with open(cfg_json, 'r') as f:
                    cfg_dict_list = json.load(f)
                # 搜索之前的子任务的额外素材,是否存在nodes中的url和新的子任务的url相同的
                find = False
                for cfg_dict in cfg_dict_list:
                    if (cfg_dict['childId'] < item.childId) and ('nodes' in cfg_dict):
                        for node_dict in cfg_dict['nodes']:
                            if node_dict['url'] == node['url']:
                                # 如果存在,就把之前的子任务的额外素材路径写入add_path_dict
                                find = True
                                add_path_dict[add_name] = cfg_dict['add_path_dict'][add_name]
                                break
                        if find:
                            break
                if not find:
                    # 如果不存在,就下载额外素材
                    ret, st, error_reason = download_add(node, merge_folder, add_name, log_file)
                    if not st:
                        return ret, st, error_reason
                    add_path_dict[add_name] = ret
            # 如果为视频素材,提取音频并调整音量保存到merge_wav文件夹中
            if node['type'] == 1:
                merge_wav = join(merge_folder, 'merge_wav')
                makedirs(merge_wav, exist_ok=True)
                os_system(f'ffmpeg -i {join(merge_folder, add_name)} -t {audio_duration} -filter:a \"volume={int(node["volume"]) * 0.02}\" {join(merge_wav, add_name[:-3] + "wav")} -loglevel quiet', log_file)
        # 如果为复合任务,处理完的额外素材路径写入cfg_dict
        if config_dict['child']:
            cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
            with open(cfg_json, 'r') as f:
                cfg_dict_list = json.load(f)
            update_cfg_dict(cfg_dict_list, item.childId, 'add_path_dict', add_path_dict, cfg_json)
        add_name_list.sort(key=lambda num: int(num.split('.')[0].split('_')[1]))
    return [add_name_list, add_path_dict], True, ''


def download_audio(item, merge_folder, log_file):
    """
    下载音频,转换为16k采样率的wav文件,调整音量保存到merge_wav文件夹中
    :param item: 任务信息
    :param merge_folder: 合成任务的文件夹路径
    :param log_file: 日志文件路径
    :return: 音频的帧数, 是否成功, 错误原因
    """
    url_audio = item.audioUrl
    merge_wav_folder = join(merge_folder, 'merge_wav')
    makedirs(merge_wav_folder, exist_ok=True)
    st = download_requests(url_audio, merge_wav_folder, 'audio.wav', log_file)
    if not st:
        return {}, False, 'download audio error'
    path_wav = join(merge_wav_folder, 'audio.wav')
    path_wav_16k = join(merge_wav_folder, 'audio_16k.wav')

    # 检测audio.wav的采样率
    audio_sample_rate = int(get_audio_sample_rate(path_wav))
    if audio_sample_rate != 16000:
        # 音频转换为16k采样率的wav文件
        log_title_write(log_file, 'start transcoding audio')
        os_system(f'ffmpeg -y -i {path_wav} -ar 16000 {path_wav_16k} -loglevel quiet', log_file)
        # 删除原来的audio.wav
        remove(path_wav)
    else:
        # 重命名音频文件
        rename(path_wav, path_wav_16k)

    if int(item.volume) != 50:
        # 调整音频音量保存到merge_wav文件夹中
        log_title_write(log_file, 'start adjust audio volume')
        os_system(f'ffmpeg -i {path_wav_16k} -filter:a \"volume={item.volume * 0.02}\" {path_wav} -loglevel quiet', log_file)
        remove(path_wav_16k)
        rename(path_wav, path_wav_16k)

    # 读取audio.wav的时长并计算对应的帧数
    audio_duration = get_audio_duration(path_wav_16k)

    return audio_duration, True, ''


def update_cfg_dict(dict_list, childId, key, value, cfg_json):
    """
    更新cfg_json中的dict_list中的childId的key的值为value
    :param dict_list: 复合任务的cfg_json的内容
    :param childId: 子任务的childId
    :param key: 更新的key
    :param value: 更新的value
    :param cfg_json: json文件路径
    :return:
    """
    dict_update = [x for x in dict_list if (x['childId'] == childId)]
    if len(dict_update) != 1:
        return {}, False, 'cfg_json error'
    dict_update = dict_update[0]
    dict_update[key] = value
    with open(cfg_json, 'w') as f:
        json.dump(dict_list, f)


def download_materials(item, config_dict, merge_folder, log_file, base_task=False):
    """
    下载背景,额外素材,字幕,背景音乐,驱动口型视频的音频,返回字典
    :param item: 任务信息
    :param config_dict: 配置信息
    :param merge_folder: 合成任务的文件夹路径
    :param log_file: 日志文件路径
    :return: {path_background, add_name_list, add_path_dict, subtitle_list, music_path, frame_num}, 是否成功, 错误原因
    """
    # 下载驱动口型视频的音频, 保存为merge_folder中的audio.wav
    log_title_write(log_file, 'start download audio')
    try:
        audio_duration, st, error_reason = download_audio(item, merge_folder, log_file)
        frame_num = int(audio_duration * 25)
        if not st:
            return {}, False, error_reason
    except BaseException as e:
        error_traceback = traceback.format_exc()
        error_reason = 'download audio error: ' + '\n' + error_traceback
        return {}, False, error_reason

    if base_task:
        ret_dict = {
            'frame_num': frame_num,
        }
        # ret_dict写进log
        log_content_write(log_file, 'ret_dict: ' + str(ret_dict))
        return ret_dict, True, ''
    else:
        # 下载背景
        log_content_write(log_file, 'start download background')
        try:
            path_background, st, error_reason = download_background(item, merge_folder, config_dict, log_file)
            if not st:
                return {}, False, error_reason
        except BaseException as e:
            error_traceback = traceback.format_exc()
            error_reason = 'download background image error: ' + '\n' + error_traceback
            return {}, False, error_reason

        # 下载额外素材
        add_name_list = []
        add_path_dict = {}
        if len(item.nodes) > 0:
            log_content_write(log_file, 'start download add_materials')
            try:
                [add_name_list, add_path_dict], st, error_reason = download_add_materials(item, config_dict, merge_folder, audio_duration, log_file)
                if not st:
                    return {}, False, error_reason
            except BaseException as e:
                error_traceback = traceback.format_exc()
                error_reason = 'download add_materials error: ' + '\n' + error_traceback
                return {}, False, error_reason

        # 下载置顶字幕图片
        floatTexts_path = ''
        if config_dict['floatTexts']:
            log_content_write(log_file, 'start download floatTexts')
            try:
                st = download_requests(item.floatTexts[0]['pngUrl'], merge_folder, 'floatTexts.png', log_file)
                if not st:
                    return {}, False, 'download floatTexts error'
                floatTexts_path = join(merge_folder, 'floatTexts.png')
            except BaseException as e:
                error_reason = 'download floatTexts error: ' + str(e).replace('\n', ' ')
                return {}, False, error_reason

        # 下载并处理字幕文件
        subtitle_list = []
        if config_dict['subtitle']:
            log_content_write(log_file, 'start download and process subtitle')
            try:
                subtitle_type = item.subtitle[0]['url'].split('.')[-1]
                st = download_requests(item.subtitle[0]['url'], merge_folder, f'subtitle.{subtitle_type}', log_file)
                if not st:
                    return {}, False, 'download subtitle error'
                subtitle_list = [join(merge_folder, f'subtitle.{subtitle_type}')]
            except BaseException as e:
                error_reason = 'download subtitle error: ' + str(e).replace('\n', ' ')
                return {}, False, error_reason

        # 下载并处理字幕背景图片
        subtitle_bg_path = ''
        if config_dict['subtitle_style']:
            log_content_write(log_file, 'start download subtitle background')
            try:
                st = download_requests(item.subtitle_style[0]['img_url'], merge_folder, 'subtitle_background.png', log_file)
                if not st:
                    return {}, False, 'download subtitle background error'
                subtitle_bg_path = join(merge_folder, 'subtitle_background.png')
            except BaseException as e:
                error_reason = 'download subtitle background error: ' + str(e).replace('\n', ' ')
                return {}, False, error_reason

        # 下载背景音乐
        music_path = ''
        if config_dict['music']:
            log_content_write(log_file, 'start download background music')
            # 若存在背景音乐,且为单一任务,或者为复合任务的最后一个子任务,下载背景音乐
            if (not config_dict['child']) or (config_dict['last_child']):
                try:
                    music_name = 'music.' + item.music[0]['music_url'].split('.')[-1]
                    st = download_requests(item.music[0]['music_url'], merge_folder, music_name, log_file)
                    if not st:
                        return {}, False, 'download background music error'
                    # 调整音量
                    os_system(f'ffmpeg -i {join(merge_folder, music_name)} -filter:a \"volume={item.music[0]["music_volume"] * 0.02}\" {join(merge_folder, "bg_music.wav")} -loglevel quiet', log_file)
                    music_path = join(merge_folder, 'bg_music.wav')
                except BaseException as e:
                    error_reason = 'download background music error: ' + str(e).replace('\n', ' ')
                    return {}, False, error_reason

        ret_dict = {
            'path_background': path_background,
            'add_name_list': add_name_list,
            'add_path_dict': add_path_dict,
            'subtitle_list': subtitle_list,
            'subtitle_bg_path': subtitle_bg_path,
            'music_path': music_path,
            'frame_num': frame_num,
            'floatTexts_path': floatTexts_path,
        }
        # ret_dict写进log
        log_content_write(log_file, 'ret_dict: ' + str(ret_dict))
        return ret_dict, True, ''


def get_audio_duration(audio_path):
    """
    获取音频时长
    :param audio_path: 音频文件路径
    :return: 音频时长,float
    """
    cmd = f'ffprobe -i {audio_path} -show_entries format=duration -v quiet -of csv="p=0"'
    audio_duration = float(os.popen(cmd).read().strip())
    return audio_duration


def get_audio_sample_rate(audio_path):
    """
    获取音频采样率
    :param audio_path: 音频文件路径
    :return: 音频采样率,int
    """
    cmd = f'ffprobe -i {audio_path} -show_entries stream=sample_rate -v quiet -of csv="p=0"'
    audio_sample_rate = int(os.popen(cmd).read().strip())
    return audio_sample_rate


def merge_audio(id_merge, item, merge_folder, config_dict, log_file):
    """
    合并音频,生成最终视频
    :param id_merge:
    :param item:
    :param merge_folder:
    :param log_file:
    :param config_dict:
    :return:
    """
    merge_wav_folder = join(merge_folder, 'merge_wav')
    merge_wav_list = [join(merge_wav_folder, x) for x in listdir(merge_wav_folder)]
    # 把所有音频文件合并成一个音频文件
    log_content_write(log_file, 'start merge audio')
    wav_path = join(merge_folder, 'result.wav')
    # 混合音频
    assert len(merge_wav_list) > 0
    if len(merge_wav_list) > 1:
        inputs = " ".join([f'-i {file}' for file in merge_wav_list])
        filter_complex = f'{"".join(["[" + str(i) + ":a]" for i in range(len(merge_wav_list))])}amix=inputs={len(merge_wav_list)}:duration=longest'
        os_system(f'ffmpeg -y {inputs} -filter_complex "{filter_complex}" {wav_path} -loglevel quiet', log_file)
    else:
        # 复制音频文件
        shutil.copy(merge_wav_list[0], wav_path)

    # 如果背景为视频,且视频有音频流,背景视频的音频循环到和合成的音频一样长,再混合两个音频
    if config_dict.get('background_type', '') == 'video':
        background_audio_path = join(merge_folder, 'background_audio.wav')
        # 如果背景视频有音频流
        if exists(background_audio_path):
            # 计算应该循环几次
            background_audio_duration = get_audio_duration(background_audio_path)
            result_audio_duration = get_audio_duration(wav_path)
            loop_times = math.ceil(result_audio_duration / background_audio_duration) - 1
            # 循环背景音频
            background_audio_path_loop = join(merge_folder, 'background_audio_loop.wav')
            os_system(f'ffmpeg -y -stream_loop {loop_times} -i {background_audio_path} -c copy {background_audio_path_loop} -loglevel quiet', log_file)
            # 把wav_path重命名为wav_path_old
            wav_path_old = join(merge_folder, 'result_old.wav')
            os.rename(wav_path, wav_path_old)
            # 混合音频
            os_system(f'ffmpeg -y -i {wav_path_old} -i {background_audio_path_loop} -filter_complex amix=inputs=2:duration=shortest:dropout_transition=3 {wav_path} -loglevel quiet', log_file)

    # 推理生成的视频和合并的音频合成为最终视频
    mp4_path = join(merge_folder, 'speaker_25fps_16k_merged.mp4')
    result_path = join(merge_folder, 'result.mp4')
    os_system(f'ffmpeg -y -i {mp4_path} -i {wav_path} -r {item.fps} -b:v {item.bitRate} -pix_fmt yuv420p {result_path} -loglevel quiet', log_file)


def merge_bg_music(log_file, path_bg_music, path_result, path_result_oss):
    """
    把result.mp4和背景音乐合并成result_oss.mp4, 背景音乐循环播放
    :param log_file:
    :param path_bg_music:
    :param path_result:
    :param path_result_oss:
    :return:
    """
    # 计算应该循环几次
    result_audio_duration = get_video_duration(path_result)
    background_audio_duration = get_audio_duration(path_bg_music)
    log_content_write(log_file, f'result_audio_duration:{result_audio_duration}, background_audio_duration:{background_audio_duration}')
    loop_times = math.ceil(result_audio_duration / background_audio_duration) - 1
    # 融合音频
    os_system(f'ffmpeg -y -stream_loop {loop_times} -i {path_bg_music} -i {path_result} -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=3" -c:v copy {path_result_oss} -loglevel quiet', log_file)


def get_font_path(font_name, fonts_name_list, fonts_folder, log_file):
    """
    获取字幕文件路径
    :param font_name:
    :param fonts_name_list:
    :param fonts_folder:
    :param log_file:
    :return:
    """
    font_name = [x for x in fonts_name_list if x.startswith(font_name)]
    if len(font_name) == 0:
        log_content_write(log_file, 'font_name not found')
        return '', False, 'font_name not found'
    font_path = join(fonts_folder, font_name[0])
    return font_path, True, ''


def adjust_font_size(subtitle_list, image, base_font_size, font_path, log_file):
    """
    确保字幕不会过大, 返回合适的字体大小
    :param subtitle_list:
    :param image:
    :param base_font_size:
    :param font_path:
    :param log_file:
    :return: font,
    """
    log_content_write(log_file, 'get font size')
    longest_subtitle = max(subtitle_list, key=lambda x: len(x[2]))
    text = longest_subtitle[2]
    font = ImageFont.truetype(font_path, base_font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    text_w, text_h = draw.textsize(text, font)
    # 减小字体大小
    while text_w > image.shape[1]:
        base_font_size = int(base_font_size * image.shape[1] / text_w)
        font = ImageFont.truetype(font_path, base_font_size)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        text_w, text_h = draw.textsize(text, font)
    return font


def get_text_position(image, text, font):
    """
    获取字幕位置
    :param image:
    :param text:
    :param font:
    :return:
    """
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    text_w, text_h = draw.textsize(text, font)
    text_position = ((image.shape[1] - text_w) // 2, int(image.shape[0] * 0.95) - text_h)
    return text_position


def add_text_to_image(image, text, font):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    # 文字居中, 贴合图片底部
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    text_position = ((image.shape[1] - text_w) // 2, int(image.shape[0] * 0.95) - text_h)
    # 添加黑色边框
    draw.text(text_position, text, font=font, fill=(255, 255, 255, 0), stroke_width=2, stroke_fill="black")
    image = np.array(img_pil)
    return image


def get_frame_tensor(frame, width, height):
    """
    将frame转成tensor,并resize到指定的宽高
    :param frame: (H, W, C), C=3 or 4
    :param width:
    :param height:
    :return:
    """
    frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255  # (1, 3, H, W) or (1, 4, H, W)
    frame_tensor = F.interpolate(frame_tensor, (height, width), mode='bilinear', align_corners=False)
    if frame_tensor.shape[1] == 3:
        # 如果是3通道, 加上alpha通道
        frame_tensor = torch.cat([frame_tensor, torch.ones(1, 1, height, width).cuda()], dim=1)
    return frame_tensor


def one_add_gen(item, node, ret_dict, log_file):
    """
    迭代器, 生成一个额外素材的帧
    :param item:
    :param node:
    :param ret_dict:
    :return:
    """
    width_bg = int(item.width)
    height_bg = int(item.height)
    level = int(node['level'])
    merge_mode = node['type']
    add_name = f'add_{level}.{node["url"].split(".")[-1]}'  # add_1.mp4, add_2.png,...
    width, height, x, y = int(node['style']['width']), int(node['style']['height']), int(node['style']['x']), int(node['style']['y'])
    path_add = ret_dict['add_path_dict'][add_name]
    if merge_mode == 1:
        # 视频素材
        video_capture = cv2.VideoCapture(path_add)
        width_video = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_video = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 处理宽高为非正数的情况,按比例缩放
        if width <= 0:
            width = int(width_video * height / height_video)
        elif height <= 0:
            height = int(height_video * width / width_video)
        add_coordinates_list, bg_coordinates_list = get_correct_coordinates(x, y, width, height, width_bg, height_bg)
        # log_content_write(log_file, f'add_name: {add_name}, x: {x}, y: {y}, width: {width}, height: {height}, add_coordinates_list: {add_coordinates_list}, bg_coordinates_list: {bg_coordinates_list}')
        # 视频帧逐帧返回
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_tensor = get_frame_tensor(frame, width, height)
            yield frame_tensor, add_coordinates_list, bg_coordinates_list
        video_capture.release()
        # 如果视频帧读取完毕,则一直返回最后一帧
        # 检查最后一个frame,若不为空,得到最后一个frame的tensor,否则直接使用之前的frame_tensor
        if frame is not None:
            frame_tensor = get_frame_tensor(frame, width, height)
        while True:
            yield frame_tensor, add_coordinates_list, bg_coordinates_list
    elif merge_mode == 2:
        # 图片素材
        frame = cv2.imread(path_add, cv2.IMREAD_UNCHANGED)
        # 处理宽高为非正数的情况,按比例缩放
        if width <= 0:
            width = int(frame.shape[1] * height / frame.shape[0])
        elif height <= 0:
            height = int(frame.shape[0] * width / frame.shape[1])
        add_coordinates_list, bg_coordinates_list = get_correct_coordinates(x, y, width, height, width_bg, height_bg)
        log_content_write(log_file, f'add_name: {add_name}, x: {x}, y: {y}, width: {width}, height: {height}, add_coordinates_list: {add_coordinates_list}, bg_coordinates_list: {bg_coordinates_list}')
        frame_tensor = get_frame_tensor(frame, width, height)
        while True:
            yield frame_tensor, add_coordinates_list, bg_coordinates_list
    elif merge_mode == 3:
        # gif素材 已经处理成25帧率的png序列
        frame_path_list = glob(join(path_add, '*.png'))
        frame_path_list_cycle = frame_path_list + frame_path_list[::-1]
        # 处理宽高为非正数的情况,按比例缩放
        if width <= 0:
            frame = cv2.imread(frame_path_list[0], cv2.IMREAD_UNCHANGED)
            width = int(frame.shape[1] * height / frame.shape[0])
        elif height <= 0:
            frame = cv2.imread(frame_path_list[0], cv2.IMREAD_UNCHANGED)
            height = int(frame.shape[0] * width / frame.shape[1])
        add_coordinates_list, bg_coordinates_list = get_correct_coordinates(x, y, width, height, width_bg, height_bg)
        log_content_write(log_file, f'add_name: {add_name}, x: {x}, y: {y}, width: {width}, height: {height}, add_coordinates_list: {add_coordinates_list}, bg_coordinates_list: {bg_coordinates_list}')
        while True:
            for frame_path in frame_path_list_cycle:
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                frame_tensor = get_frame_tensor(frame, width, height)
                yield frame_tensor, add_coordinates_list, bg_coordinates_list


def proxy_one_add_gen(item, node, ret_dict, log_file):
    """
    one_add_gen的代理函数, 用于处理异常
    :param item:
    :param node:
    :param ret_dict:
    :return:
    """
    yield from one_add_gen(item, node, ret_dict, log_file)


def merge_bg_add(bg_tensor, add_tensor, add_coordinates_list, bg_coordinates_list):
    """
    在背景帧上覆盖额外素材帧,根据坐标
    :param bg_tensor: 背景帧的tensor, (1, 4, H1, W1)
    :param add_tensor: 添加素材帧的tensor, (1, 4, H2, W2)
    :param add_coordinates_list: 添加素材的坐标列表[x0, x1, y0, y1]
    :param bg_coordinates_list: 背景的坐标列表[x0, x1, y0, y1]
    :return:
    """
    add_x0, add_x1, add_y0, add_y1 = add_coordinates_list
    bg_x0, bg_x1, bg_y0, bg_y1 = bg_coordinates_list
    if add_x0 > add_x1 or add_y0 > add_y1 or bg_x0 > bg_x1 or bg_y0 > bg_y1:
        # 此时为添加的素材超出背景的范围,不做处理
        return bg_tensor
    if add_tensor.size(1) == 4:
        bg_tensor[0, :3, bg_y0:bg_y1, bg_x0:bg_x1] = (bg_tensor[0, :3, bg_y0:bg_y1, bg_x0:bg_x1] * (1 - add_tensor[0, 3, add_y0:add_y1, add_x0:add_x1]) +
                                                      add_tensor[0, :3, add_y0:add_y1, add_x0:add_x1] * add_tensor[0, 3, add_y0:add_y1, add_x0:add_x1])
    elif add_tensor.size(1) == 3:
        bg_tensor[0, :3, bg_y0:bg_y1, bg_x0:bg_x1] = add_tensor[0, :3, add_y0:add_y1, add_x0:add_x1]
    return bg_tensor


class RGBPro(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGBPro, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

    def rgb_to_yuv(self, img):
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        y = y.unsqueeze(1)
        u = u.unsqueeze(1)
        v = v.unsqueeze(1)
        yuv = torch.cat([y, u, v], dim=1)
        return yuv

    def yuv_to_rgb(self, yuv):
        y, u, v = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

    def yuv_saturation(self, yuv, saturation):
        yuv[:, 1, :, :] = torch.clamp(yuv[:, 1, :, :] * saturation, 0, 1)
        yuv[:, 2, :, :] = torch.clamp(yuv[:, 2, :, :] * saturation, 0, 1)
        return yuv

    def img_contrast(self, img, contrast):
        img_contrast = torch.clamp((img - 0.5) * contrast + 0.5, 0, 1)
        return img_contrast

    def img_brightness(self, img, brightness):
        img_brightness = torch.clamp(img + brightness, 0, 1)
        return img_brightness

    def img_filter(self, img, videoFilter):
        """
        图像滤镜
        :param img: 输入图像 (1, 3, H, W), RGB
        :param videoFilter: {"rgba": "221,102,112,255", "brightness": "0.06", "contrast": "1.5", "saturation": "1.5"}
        :return:
        """
        # 调整亮度
        brightness = float(videoFilter['brightness']) / 4
        img = self.img_brightness(img, brightness)
        # 调整对比度
        contrast = float(videoFilter['contrast'])
        img = self.img_contrast(img, contrast)
        # RGB转YUV
        yuv = self.rgb_to_yuv(img)
        # 调整饱和度
        saturation = float(videoFilter['saturation'])
        yuv = self.yuv_saturation(yuv, saturation)
        # YUV转RGB
        img = self.yuv_to_rgb(yuv)
        if videoFilter['rgba'] != '':
            # RGB颜色参数
            # r = round(float(videoFilter['rgba'].split(',')[0]) / 255, 2)
            # g = round(float(videoFilter['rgba'].split(',')[1]) / 255, 2)
            # b = round(float(videoFilter['rgba'].split(',')[2]) / 255, 2)
            # r = math.sqrt(float(videoFilter['rgba'].split(',')[0]) / 255)
            # g = math.sqrt(float(videoFilter['rgba'].split(',')[1]) / 255)
            # b = math.sqrt(float(videoFilter['rgba'].split(',')[2]) / 255)
            r = math.sqrt((float(videoFilter['rgba'].split(',')[0]) + 255) / 255 / 2)
            g = math.sqrt((float(videoFilter['rgba'].split(',')[1]) + 255) / 255 / 2)
            b = math.sqrt((float(videoFilter['rgba'].split(',')[2]) + 255) / 255 / 2)
            # 调整颜色
            img[0, 0, :, :] = torch.clamp(img[0, 0, :, :] * r, 0, 1)
            img[0, 1, :, :] = torch.clamp(img[0, 1, :, :] * g, 0, 1)
            img[0, 2, :, :] = torch.clamp(img[0, 2, :, :] * b, 0, 1)
        return img


rgb_pro = RGBPro()


def image_filter(input_image, videoFilter):
    """
    图像滤镜
    :param input_image: 输入图像 (1, 4, H, W)
    :param videoFilter: {"rgba": "221,102,112,255", "brightness": "0.06", "contrast": "1.5", "saturation": "1.5"}
    :return: 输出图像 (1, 4, H, W)
    """
    # 分离alpha通道
    alpha = input_image[:, 3, :, :]
    input_image = input_image[:, :3, :, :]
    # BGR转RGB
    input_image = input_image[:, [2, 1, 0], :, :]
    # 调用RGBPro,进行图像滤镜
    input_image = rgb_pro.img_filter(input_image, videoFilter)
    # RGB转BGR
    input_image = input_image[:, [2, 1, 0], :, :]
    # 合并alpha通道
    input_image = torch.cat([input_image, alpha.unsqueeze(1)], dim=1)
    return input_image


def audio_feature_extraction_gen(audio_path, slient_add_num):
    # chinese_hubert_large_model_path = r'E:\model\chinese-hubert-large'
    # hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(chinese_hubert_large_model_path)
    # hubert_model = HubertModel.from_pretrained(chinese_hubert_large_model_path)
    # hubert_model = hubert_model.to(device)
    # hubert_model.eval()

    # 生成音频特征
    wav, sr = librosa.load(audio_path, sr=16000)
    if len(wav.shape) > 1:
        assert wav.shape[1] == 2
        wav = (wav[:, 0] + wav[:, 1]) / 2
    # 前后各补几帧静音音频
    slient_len = int(slient_add_num * 640)
    wav = np.concatenate([np.zeros(slient_len), wav, np.zeros(slient_len)], 0)
    wav = librosa.util.normalize(wav)
    input_values_all = feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_values
    input_values_all = input_values_all.to(device)
    kernel = 400  # 核大小
    stride = 320  # 连续窗口之间的步长
    clip_length = stride * 1000  # 切片长度
    num_iter = input_values_all.shape[1] // clip_length  # 处理整个音频波形的迭代次数
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride  # 提取的非重叠窗口的数量
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        with torch.no_grad():
            hidden_states = model_hubert(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        # res_lst.append(hidden_states.cpu().numpy()[0])
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
        with torch.no_grad():
            hidden_states = model_hubert(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        # res_lst.append(hidden_states.cpu().numpy()[0])
        res_lst.append(hidden_states[0])
    # ret = np.concatenate(res_lst, axis=0)
    ret = torch.cat(res_lst, dim=0)
    assert abs(ret.shape[0] - expected_T) <= 1, f"expected_T {expected_T} != ret.shape[0] {ret.shape[0]}"
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    if ret.shape[0] % 2 != 0:
        # ret = ret[:-1]
        # 复制最后一个
        ret = torch.cat([ret, ret[-1:]], 0)
    output = ret.reshape(-1, 2048).permute(1, 0).unsqueeze(0)  # (1, 2048, 100)
    for j in range(output.shape[2]):
        yield output[:, :, j]  # (1, 2048)
    while True:
        yield output[:, :, -1]  # (1, 2048)


def audio_feature_extraction_gen_unused(audio_path, slient_add_num):
    # 读取音频
    wav, sr = sf.read(audio_path)
    if len(wav.shape) > 1:
        wav = (wav[:, 0] + wav[:, 1]) / 2
    # 前后各补两帧静音音频
    slient_len = int(slient_add_num * 640)
    wav = np.concatenate([np.zeros(slient_len), wav, np.zeros(slient_len)], 0)
    # 按照2s切分
    slice_len = 2 * 16000
    # 向上取整
    slice_num = math.ceil(len(wav) / slice_len)
    for i in range(slice_num):
        wav_slice = wav[i * slice_len: (i + 1) * slice_len + 80]
        if len(wav_slice) < 400:
            break
        input_wav = librosa.util.normalize(wav_slice)
        input_values = feature_extractor(input_wav, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model_hubert(input_values)
            last_hidden_state = outputs.last_hidden_state
        output = last_hidden_state[0]  # (100, 1024)
        if output.shape[0] % 2 != 0:
            # 复制最后一个
            output = torch.cat([output, output[-1:]], 0)
        output = output.reshape(-1, 2048).permute(1, 0).unsqueeze(0)  # (1, 2048, 100)
        for j in range(output.shape[2]):
            yield output[:, :, j]  # (1, 2048)


def proxy_audio_feature_extraction_gen(audio_path, slient_add_num, use_new_audio_process):
    """
    audio_feature_extraction_gen的代理函数, 用于处理异常
    :param audio_path:
    :param slient_add_num:
    :return:
    """
    if use_new_audio_process:
        yield from audio_feature_extraction_gen(audio_path, slient_add_num)
    else:
        yield from audio_feature_extraction_gen_unused(audio_path, slient_add_num)


def audio_feature_concat_gen(audio_path, inf_len, use_new_audio_process):
    # test_tensor_path = r'G:\Generating_offline_2D_lip-sync_videos\test_1_silence.pt'
    # test_tensor = torch.load(test_tensor_path)
    slient_add_num = inf_len // 2
    feature_list = []
    gen = proxy_audio_feature_extraction_gen(audio_path, slient_add_num, use_new_audio_process)
    for i in range(inf_len):
        feature_list.append(next(gen))  # (1, 2048)
        # feature_list.append(test_tensor)
    while True:
        # 返回inf_len个特征在维度2上拼接的结果
        if inf_len != 16:
            yield torch.stack(feature_list, dim=2)  # (1, 2048, inf_len)
        else:
            yield torch.stack(feature_list, dim=2).resize(1, 32, 32, 32)  # (1, 2048, 16) -> (1, 32, 32, 32)
        # 滑动窗口
        feature_list.pop(0)
        feature_list.append(next(gen))


def proxy_audio_feature_concat_gen(audio_path, inf_len, use_new_audio_process):
    """
    audio_feature_concat_gen的代理函数, 用于处理异常
    :param audio_path:
    :param inf_len:
    :return:
    """
    yield from audio_feature_concat_gen(audio_path, inf_len, use_new_audio_process)


def face_gen(face_folder, inf_folder, start_human_frame, human_frame_count, mouth_region_size, model_version, square):
    # 确定face使用的是jpg还是png素材
    face_type = listdir(face_folder)[0].split('.')[-1]  # jpg or png
    # 坐标计算
    resize_w = int(mouth_region_size + mouth_region_size // 4)
    if square:
        resize_h = int(mouth_region_size + mouth_region_size // 4)
        y0_mouth = int(mouth_region_size // 16)
    else:
        resize_h = int((mouth_region_size // 2) * 3 + mouth_region_size // 8)
        y0_mouth = int(mouth_region_size // 2)
    y1_mouth = int(mouth_region_size // 2 + mouth_region_size + mouth_region_size // 16)
    x0_mouth = int(mouth_region_size // 16)
    x1_mouth = int(mouth_region_size // 8 + mouth_region_size + mouth_region_size // 16)
    # 参考图片获取
    # 如果inf_folder下有reference_images文件夹,则使用里面的图片,否则使用face文件夹下的图片
    if exists(join(inf_folder, 'reference_images')):
        ref_img_path_list = glob(join(inf_folder, 'reference_images', f'*.{face_type}'))
        assert len(ref_img_path_list) >= 5
        ref_img_path_list.sort()
    else:
        ref_img_path_list = glob(join(face_folder, f'*.{face_type}'))
        assert len(ref_img_path_list) >= 5
        # 随机选择5张图片
        ref_img_path_list = random.sample(ref_img_path_list, 5)

    ref_img_list = []
    for ref_img_path in ref_img_path_list:
        if face_type == 'jpg':
            ref_img = cv2.imread(ref_img_path)
            ref_img = torch.tensor(ref_img, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :] / 255  # (1, 3, H, W)
        else:
            # png
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)
            ref_img = torch.tensor(ref_img, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [3, 2, 1, 0], :, :] / 255  # (1, 4, H, W)
        ref_img = F.interpolate(ref_img, (resize_h, resize_w), mode='bilinear', align_corners=False)
        ref_img_list.append(ref_img)
    ref_img_tensor = torch.cat(ref_img_list, dim=1)  # (1, 15, H, W)

    # crop_frame_tensor
    face_path_list = glob(join(face_folder, f'*.{face_type}'))
    assert len(face_path_list) == human_frame_count
    face_path_list_cycle = face_path_list + face_path_list[::-1]
    frame_id = start_human_frame - 1
    while True:
        frame_id += 1
        frame_id = frame_id % (human_frame_count * 2)
        if face_type == 'jpg':
            frame = cv2.imread(face_path_list_cycle[frame_id])
            frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :] / 255  # (1, 3, H, W)
        else:
            # png
            frame = cv2.imread(face_path_list_cycle[frame_id], cv2.IMREAD_UNCHANGED)
            frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [3, 2, 1, 0], :, :] / 255  # (1, 4, H, W)
        frame_tensor = F.interpolate(frame_tensor, (resize_h, resize_w), mode='bilinear', align_corners=False)
        # 嘴巴区域为全黑
        if model_version == '4p4':
            ref_img_tensor = frame_tensor.clone()
        frame_tensor[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth] = 0
        yield frame_tensor, ref_img_tensor, frame_id


def proxy_face_gen(face_folder, inf_folder, start_human_frame, human_frame_count, mouth_region_size, model_version, square):
    """
    face_gen的代理函数, 用于处理异常
    :param face_folder:
    :param inf_folder:
    :param start_human_frame:
    :param human_frame_count:
    :param mouth_region_size:
    :param model_version:
    :return:
    """
    yield from face_gen(face_folder, inf_folder, start_human_frame, human_frame_count, mouth_region_size, model_version, square)


def human_gen(item, merge_folder, start_human_frame, human_frame_count, log_path, base_task=False):
    # 找到每个文件和文件夹路径
    # 判断是否使用新的音频处理方式
    if 100 < int(item.sceneId) < 6000:
        use_new_audio_process = True
    else:
        use_new_audio_process = False
    inf_folder = join(parameters['inf_data_folder'], item.sceneId)
    face_folder = join(inf_folder, 'face')
    # 确定face使用的是jpg还是png素材
    # face_type = listdir(face_folder)[0].split('.')[-1]  # jpg or png
    body_path_list = glob(join(inf_folder, 'body', '*'))
    body_path_list_cycle = body_path_list + body_path_list[::-1]
    xy_npy = np.load(join(inf_folder, 'xy.npy'))
    xy_npy_cycle = np.concatenate([xy_npy, xy_npy[::-1]], 0)
    audio_path = join(merge_folder, 'merge_wav', 'audio_16k.wav')
    assert check_existence(audio_path, log_path)

    if not base_task:
        # 计算人物的坐标和宽高
        human_x = int(item.style['x'])
        human_y = int(item.style['y'])
        human_h = int(item.style['height'])
        # 等比例计算人物的宽度
        frame = cv2.imread(body_path_list[0])
        human_w = int(human_h * frame.shape[1] / frame.shape[0])
        log_content_write(log_path, f'human_x: {human_x}, human_y: {human_y}, human_w: {human_w}, human_h: {human_h}, bg_w: {item.width}, bg_h: {item.height}')
        human_add_coordinates_list, human_bg_coordinates_list = get_correct_coordinates(human_x, human_y, human_w, human_h, int(item.width), int(item.height))
    else:
        human_h = item.height
        human_w = item.width

    # 加载模型
    model_path = join(parameters['model_folder'], 'DINet_master', 'asserts', 'clip_training_DINet_128v3p1.pth')
    model_version = '3p1'
    mouth_region_size = 128
    inf_len = 5
    for file in listdir(inf_folder):
        # if file.endswith('.pth'):  # DINetV3p3_256_7.pth
        if file.startswith('DINet') and file.endswith('.pth'):
            model_path = join(inf_folder, file)
            model_version = file.split('_')[0].split('DINetV')[-1]  # 3p3
            mouth_region_size = int(file.split('_')[1])  # 256
            inf_len = int(file.split('_')[-1].split('.')[0])  # 7
            break
    if model_version.endswith('s'):
        model_version = model_version[:-1]
        square = True
    else:
        square = False
    if model_version.endswith('m'):
        model_version = model_version[:-1]
        mask_process = True
    else:
        mask_process = False
        
    if exists(join(inf_folder, 'reference_images')):
        ref_img_path_list = glob(join(inf_folder, 'reference_images', '*.png'))
        ref_num = len(ref_img_path_list)
        assert ref_num >= 5
    else:
        ref_num = 5
    
    if model_version == '3p1':
        model = DINetV3p1(3, 3 * ref_num, 2048).cuda()
    elif model_version == '3p3':
        model = DINetV3p3(3, 3 * ref_num, 2048).cuda()
    elif model_version == '3p4':
        model = DINetV3p4(3, 3 * ref_num, 2048, inf_len).cuda()
    elif model_version == '3p5':
        model = DINetV3p5(3, 3 * ref_num, 2048, inf_len).cuda()
    elif model_version == '4p1':
        model = DINetV4p1(4, 4 * ref_num, 2048).cuda()
    elif model_version == '4p2':
        model = DINetV4p2(4, 4 * ref_num, 2048, inf_len).cuda()
    elif model_version == '4p3':
        model = DINetV4p3(4, 4 * ref_num, 2048, inf_len).cuda()
        square = True
    elif model_version == '4p4':
        model = DINetV4p4().cuda()
        square = True
    else:
        log_content_write(log_path, f'error model_version: {model_version}')
        raise Exception(f'error model_version: {model_version}')
    checkpoint = torch.load(model_path)
    log_content_write(log_path, f'load model: {model_path}')
    model.load_state_dict(checkpoint['state_dict']['net_g'])
    model.eval()
    if square == True:
        y0_mouth = int(mouth_region_size // 16)
    else:
        y0_mouth = int(mouth_region_size // 2)
    y1_mouth = int(mouth_region_size // 2 + mouth_region_size + mouth_region_size // 16)
    x0_mouth = int(mouth_region_size // 16)
    x1_mouth = int(mouth_region_size // 8 + mouth_region_size + mouth_region_size // 16)
    # 加载mask生成模型
    use_mask_model = exists(join(inf_folder, 'mask_model.pth'))
    if use_mask_model:
        mask_model = MaskNet().cuda()
        mask_model.load_state_dict(torch.load(join(inf_folder, 'mask_model.pth')))
        mask_model.eval()
    # 音频特征迭代器
    log_content_write(log_path, f'start audio_feature_concat_gen')
    audio_gen = proxy_audio_feature_concat_gen(audio_path, inf_len, use_new_audio_process)
    # 人脸特征迭代器
    log_content_write(log_path, f'start face_gen')
    face_gen = proxy_face_gen(face_folder, inf_folder, start_human_frame, human_frame_count, mouth_region_size, model_version, square)
    while True:
        # 获取音频特征,人脸特征
        audio_feature_tensor = next(audio_gen)
        crop_frame_tensor, ref_img_tensor, frame_id = next(face_gen)
        if int(model_version[0]) == 3 and crop_frame_tensor.shape[1] == 4:
            # 如果为不带alpha的模型,则去掉alpha通道
            crop_frame_tensor = crop_frame_tensor[:, 1:, :, :]
            # ref_img_tensor去掉0,4,8,12,16的alpha通道
            # ref_img_tensor = ref_img_tensor[:, 1:, :, :]
            # ref_img_tensor = ref_img_tensor[:, [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :, :]
            ref_img_tensor = ref_img_tensor[:, [i for i in range(ref_img_tensor.shape[1]) if i % 4 != 0], :, :]
        # log_content_write(log_path, f'crop_frame_tensor: {crop_frame_tensor.shape}, ref_img_tensor: {ref_img_tensor.shape}, audio_feature_tensor: {audio_feature_tensor.shape}')
        with torch.no_grad():
            output = model(crop_frame_tensor, ref_img_tensor, audio_feature_tensor)  # (1, 3, H, W)
        # face_pre = output.detach()[:, [2, 1, 0], :, :]
        output = output.detach()
        face_pre = crop_frame_tensor.clone()
        face_pre[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth] = output[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth]
        if int(model_version[0]) == 3:
            face_pre = face_pre[:, [2, 1, 0], :, :]
        elif int(model_version[0]) == 4:
            face_pre = face_pre[:, [3, 2, 1, 0], :, :]
        # 推理出来的face重新贴回body中的face部分
        body_frame = cv2.imread(body_path_list_cycle[frame_id], cv2.IMREAD_UNCHANGED)
        body_frame_tensor = torch.tensor(body_frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255  # (1, 4, Hb, Wb)
        x1, x2, y1, y2 = xy_npy_cycle[frame_id]
        # 使用模型推理得到alpha通道
        if use_mask_model:
            # 得到body中的face部分
            face_mask = body_frame_tensor[:, :, y1:y2, x1:x2]  # (1, 4, Hf, Wf)
            face_mask = F.interpolate(face_mask, (crop_frame_tensor.shape[2], crop_frame_tensor.shape[3]), mode='bilinear', align_corners=False)  # (1, 4, H, W)
            # 把face_mask和face_pre合并
            face_mask = torch.cat((face_mask, face_pre), 1)  # (1, 7, H, W)
            with torch.no_grad():
                mask = mask_model(face_mask)  # (1, 1, H, W)
            face_pre = torch.cat((face_pre, mask), 1)  # (1, 4, H, W)
            face_pre = F.interpolate(face_pre, (y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
            body_frame_tensor[:, :, y1:y2, x1:x2] = face_pre
        # 不使用mask模型,直接贴回
        else:
            face_pre = F.interpolate(face_pre, (y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
            if int(model_version[0]) == 3:
                body_frame_tensor[0, :3, y1:y2, x1:x2] = face_pre[0, :3]
            elif int(model_version[0]) == 4:
                if mask_process:
                    mask_tensor_reshape = F.interpolate(mask_tensor, (y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                    body_frame_tensor[:, :, y1:y2, x1:x2] = body_frame_tensor[:, :, y1:y2, x1:x2] * (1 - mask_tensor_reshape) + face_pre * mask_tensor_reshape
                else:
                    body_frame_tensor[0, :, y1:y2, x1:x2] = face_pre[0, :]
        # 改变大小成human_w, human_h
        body_frame_tensor = F.interpolate(body_frame_tensor, (human_h, human_w), mode='bilinear', align_corners=False)
        if base_task:
            # 输出base_task的结果
            yield body_frame_tensor
        else:
            yield body_frame_tensor, human_add_coordinates_list, human_bg_coordinates_list


def proxy_human_gen(item, merge_folder, start_human_frame, human_frame_count, log_path, base_task=False):
    """
    human_gen的代理函数, 用于处理异常
    :param item:
    :param merge_folder:
    :param start_human_frame:
    :param human_frame_count:
    :param log_path:
    :param base_task:
    :return:
    """
    yield from human_gen(item, merge_folder, start_human_frame, human_frame_count, log_path, base_task)


if __name__ == '__main__':
    pass
