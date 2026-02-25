# -*- coding: utf-8 -*-
# @Time    : 2023/9/6 11:20
# @Author  : 施昀谷
# @File    : file_transfer.py

import oss2
import requests
import traceback
import os
import shutil
from config import parameters
import time
from requests.exceptions import (
    ConnectTimeout,
    ReadTimeout,
    ConnectionError,
    HTTPError,
    RequestException
)

local_service = parameters['local_service']


def download_requests(url, save_folder, save_name, log_file, max_retries=3, retry_delay=2, timeout=(10, 30)):
    """
    使用 requests 下载文件，支持重试机制和超时控制。
    
    :param url: 要下载的文件 URL
    :param save_folder: 保存的文件夹路径
    :param save_name: 保存的文件名
    :param log_file: 日志文件路径
    :param max_retries: 最大重试次数（默认 3）
    :param retry_delay: 重试前等待秒数（默认 2 秒）
    :param timeout: (连接超时, 读取超时) 秒，默认 (10, 30)
    :return: bool，成功返回 True，失败返回 False
    """
    # 确保保存目录存在
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)

    for attempt in range(max_retries + 1):
        try:
            if not local_service:
                # 处理普通 URL
                clean_url = url.replace('\\', '')
                response = requests.get(clean_url, stream=True, timeout=timeout)
                response.raise_for_status()  # 抛出非 2xx 状态码异常

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # 过滤掉 keep-alive 的空块
                            f.write(chunk)

                # 验证文件是否写入
                if os.path.exists(save_path):
                    return True
                else:
                    raise IOError("File not created after write")

            else:
                # 本地服务：url 是本地路径
                if not os.path.exists(url):
                    with open(log_file, 'a', encoding='utf-8') as lf:
                        lf.write(f'\n[本地文件不存在]: {url}\n')
                    return False
                shutil.copy(url, save_path)
                return True

        except (ConnectTimeout, ReadTimeout, ConnectionError, HTTPError, RequestException, IOError) as e:
            error_msg = f'\n[第 {attempt + 1} 次尝试失败] URL: {url}\n错误: {type(e).__name__}: {e}\n'
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write(error_msg)

            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                # 最终失败
                with open(log_file, 'a', encoding='utf-8') as lf:
                    lf.write(f'\n[下载彻底失败] 经过 {max_retries + 1} 次尝试后仍无法下载: {url}\n')
                return False

        except Exception as e:
            # 捕获其他未预期异常（如权限、磁盘满等）
            error_traceback = traceback.format_exc()
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write(f'\n[未预期异常] URL: {url}\n{error_traceback}\n')
            return False

    return False


def get_bucket(endpoint, accesskey_id, accesskey_secret, bucket_name):
    """
    :param endpoint:访问域名
    :param accesskey_id:访问秘钥编号
    :param accesskey_secret:访问秘钥密码
    :param bucket_name:存储空间名
    :return:bucket存储空间
    """
    auth = oss2.Auth(access_key_id=accesskey_id, access_key_secret=accesskey_secret)
    bucket = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name=bucket_name, connect_timeout=3600)
    return bucket


def upload_oss(upload_file, config, upload_file_name):
    bucket = get_bucket(config['endpoint'], config['accesskey_id'], config['accesskey_secret'], config['bucket_name'])
    bucket.put_object_from_file('virtual_live/AIface/' + upload_file_name, upload_file)


def send_email(title, content):
    # smtplib 用于邮件的发信动作
    import smtplib
    # email 用于构建邮件内容
    from email.mime.text import MIMEText
    # 构建邮件头
    from email.header import Header

    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = '704736806@qq.com'
    password = 'ncypkfjjoljnbfie'
    # 收信方邮箱
    to_addr = '704736806@qq.com'
    # 发信服务器
    smtp_server = 'smtp.qq.com'

    # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
    msg = MIMEText(content, 'plain', 'utf-8')
    # 邮件头信息
    msg['From'] = Header('704736806@qq.com')  # 发送者
    msg['To'] = Header('704736806@qq.com')  # 接收者
    subject = title
    msg['Subject'] = Header(subject, 'utf-8')  # 邮件主题

    try:
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        # 建立连接--qq邮箱服务和端口号（可百度查询）
        smtpobj.connect(smtp_server, 465)
        print("连接成功")
        # 登录--发送者账号和口令
        smtpobj.login(from_addr, password)
        print("登录成功")
        # 发送邮件
        smtpobj.sendmail(from_addr, to_addr, msg.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException:
        print("无法发送邮件")
    finally:
        # 关闭服务器
        smtpobj.quit()


if __name__ == '__main__':
    # # 下载背景图片
    # download_requests(
    #     'https:\/\/bsddata.oss-cn-hangzhou.aliyuncs.com\/virtual_live\/common_202308\/202308221759275158.png',
    #     r'D:\project\Generating_offline_2D_lip-sync_videos\workspace\230906140149Wav2Lip0001',
    #     'background.png',
    #     r'D:\project\Generating_offline_2D_lip-sync_videos\log\230906140149Wav2Lip0001.txt'
    # )

    # 下载视频
    download_requests(
        'https:\/\/bsddata.oss-cn-hangzhou.aliyuncs.com\/virtual_live\/virtual_video_cover\/1687164513hNGMW5d2dr.mp4',
        r'D:\project\Generating_offline_2D_lip-sync_videos\workspace\230906140149Wav2Lip0001',
        'add_video.mp4',
        r'D:\project\Generating_offline_2D_lip-sync_videos\log\230906140149Wav2Lip0001.txt'
    )

