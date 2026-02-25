# -*- coding: utf-8 -*-
# @Time    : 2023/9/5 17:30
# @Author  : 施昀谷
# @File    : callback.py


# import cv2
import json
import time
import requests
import subprocess
from os.path import join
# from config import parameters
# from utils.file_transfer import upload_oss


def callback_test(result1, result2):
    # 执行合成结果的回调操作
    # 这里可以根据需要进行合成结果的处理和返回

    # 例如，将结果作为JSON数据发送到指定URL
    url = "http://wa.gstai.com/api/sound/test1/callback"
    headers = {"Content-Type": "application/json"}
    data = {"xxxx": result1, "yyyy": result2}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("合成结果已成功发送")
    else:
        print("发送合成结果时出错：", response.text)


def callback_merge_once(callbackUrl, merge_id, duration, videoUrl, videoName, result, localPath, failReason, horizontal, vertical, coverUrl):
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "taskType": "video-training",
            "trainingId": merge_id,
            "duration": duration,
            "videoUrl": videoUrl,
            "videoName": videoName,
            "result": result,
            "localPath": localPath,
            "failReason": failReason,
            "horizontal": horizontal,
            "vertical": vertical,
            "coverUrl": coverUrl
        }

        response = requests.post(callbackUrl, json=data, headers=headers)
        response_json = response.json()
        if response_json['code'] == 0:
            print("合成结果已成功发送")
            return True
        else:
            print("发送合成结果时出错：", response.text)
            return False
    except BaseException as e:
        return False


def callback_merge(callbackUrl, merge_id, duration, videoUrl, videoName, result, localPath, failReason, horizontal, vertical, coverUrl):
    """
    回调合成结果，如果失败则每隔一分钟重试一次，最多重试10次
    :param merge_id:
    :param duration:
    :param videoUrl:
    :param videoName:
    :param result:
    :param failReason:
    :param horizontal:
    :param vertical:
    :param coverUrl:
    :return:
    """
    for callback_time in range(10):
        if callback_merge_once(callbackUrl, merge_id, duration, videoUrl, videoName, result, localPath, failReason, horizontal, vertical, coverUrl):
            return True
        else:
            time.sleep(10)
    return False


def callback_base_task_once(callbackUrl, merge_id, videoUrl,  result,  failReason, MaskUrl):
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "merge_id": merge_id,
            "videoUrl": videoUrl,
            "MaskUrl": MaskUrl,
            "result": result,
            "failReason": failReason,
        }

        response = requests.post(callbackUrl, json=data, headers=headers)
        response_json = response.json()
        if response_json['code'] == 0:
            print("合成结果已成功发送")
            return True
        else:
            print("发送合成结果时出错：", response.text)
            return False
    except BaseException as e:
        return False


def callback_base_task(callbackUrl, merge_id, videoUrl, result, failReason, MaskUrl):
    """
    回调合成结果，如果失败则每隔一分钟重试一次，最多重试10次
    :param callbackUrl:
    :param merge_id:
    :param videoUrl:
    :param result:
    :param failReason:
    :param MaskUrl:
    :return:
    """
    for callback_time in range(10):
        if callback_base_task_once(callbackUrl, merge_id, videoUrl, result, failReason, MaskUrl):
            return True
        else:
            time.sleep(10)
    return False


def callback_train_once(anchorIdentity, name, gender, id, width, height, imgUrl, avatayUrl, videoUrl, fileUrl, code, failReason, callbackUrl):
    headers = {"Content-Type": "application/json"}
    data = {
        "taskType": "video-synthesis",
        "anchorIdentity": anchorIdentity,
        "name": name,
        "gender": gender,
        "id": id,
        "width": width,
        "height": height,
        "imgUrl": imgUrl,
        "avatayUrl": avatayUrl,
        "videoUrl": videoUrl,
        "fileUrl": fileUrl,
        "code": code,
        "failReason": failReason
    }

    response = requests.post(callbackUrl, json=data, headers=headers)
    response_json = response.json()
    if response_json['code'] == 0:
        print("训练结果已成功发送")
        return True
    else:
        print("发送训练结果时出错：", response.text)
        return False


def callback_train(anchorIdentity, name, gender, id, width, height, imgUrl, avatayUrl, videoUrl, fileUrl, code, failReason, callbackUrl):
    for callback_time in range(10):
        if callback_train_once(anchorIdentity, name, gender, id, width, height, imgUrl, avatayUrl, videoUrl, fileUrl, code, failReason, callbackUrl):
            return True
        else:
            time.sleep(60)
    return False


def get_video_resolution(video_file):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = cmd.split()
    args.append(video_file)
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    ffprobe_output = json.loads(ffprobe_output)

    # 找到视频流
    for stream in ffprobe_output['streams']:
        if stream['codec_type'] == 'video':
            width = stream['width']
            height = stream['height']
            return width, height


# def callback_train_id(anchorIdentity):
#     json_path = join(parameters['inf_data_folder'], anchorIdentity, 'Wav2Lip', 'log.json')
#     with open(json_path, 'r') as f:
#         log = json.load(f)
#     # print(log)
#     # exit(0)

#     # # oss上传train.mp4
#     # upload_oss(join(parameters['inf_data_folder'], anchorIdentity, 'Wav2Lip', 'train.mp4'), parameters, f'{anchorIdentity}_15s.mp4')

#     # 用ffprobe读取视频的分辨率, 横向分辨率和纵向分辨率
#     width, height = get_video_resolution(join(parameters['inf_data_folder'], anchorIdentity, 'Wav2Lip', 'train.mp4'))

#     callback_train(
#         anchorIdentity=anchorIdentity,
#         name=log['name'],
#         gender=log['gender'],
#         id=log['id'],
#         width=width,
#         height=height,
#         imgUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{anchorIdentity}_whole_body.png',
#         avatayUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{anchorIdentity}_face.png',
#         videoUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{anchorIdentity}_15s.mp4',
#         fileUrl=join(parameters['inf_data_folder'], anchorIdentity, 'Wav2Lip'),
#         code=10000,
#         failReason='',
#         callbackUrl=log['callbackUrl']
#     )


if __name__ == '__main__':
    # callback_test("abgenjdwe", "euibrgegbroabf")

    callback_merge(
        callbackUrl='http://metawa.cn/api/callback/qingbo-notify', 
        merge_id='250807155826059186DINet20093', 
        duration=14.28, 
        videoUrl='https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/250807155826059186DN20093.mp4', 
        videoName='250807155826059186DN20093.mp4', 
        result='success', 
        localPath=r'E:\Generating_offline_2D_lip-sync_videos\workspace\250807155826059186DINet20093\result_oss.mp4', 
        failReason='', 
        horizontal=2560, 
        vertical=1440, 
        coverUrl='https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/250807155826059186DN20093.jpg'
    )
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-id', '--anchorIdentity', type=str, default='sunfsklfks')
    # arg = parser.parse_args()

    # callback_train_id(arg.anchorIdentity)

