#!/usr/bin/env python3
"""
真实场景视频合成速度对比测试
=============================
使用真实的人物素材和背景图，对比两种编码方式：
1. cv2.VideoWriter (当前 video_merge.py 使用的方式)
2. PyNvVideoCodec GPU Buffer 模式 (优化方案)

素材：
- 背景图: bg.png
- 音频: yuanwa.wav
- 人物素材: inf_data/0203/

用法:
    python benchmark_real_merge.py
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from os.path import join, exists
from glob import glob
from os import listdir

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入模型
from model.DINet_master.models.DINetV3 import DINetV3p1, DINetV3p3, DINetV3p4, DINetV3p5, DINetV4p1, DINetV4p2, DINetV4p3, DINetV4p4


# 导入工具函数
from utils.data_prepare import (
    get_correct_coordinates,
    proxy_audio_feature_concat_gen,
    proxy_face_gen
)

# 测试参数
WIDTH = 1440
HEIGHT = 2560
FPS = 25
SCENE_ID = '0203'
AUDIO_PATH = join(os.path.dirname(os.path.abspath(__file__)), 'yuanwa.wav')
BG_PATH = join(os.path.dirname(os.path.abspath(__file__)), 'bg.png')
INF_FOLDER = join(os.path.dirname(os.path.abspath(__file__)), 'inf_data', SCENE_ID)

# 人物位置参数
HUMAN_X = 0
HUMAN_Y = 0
HUMAN_H = 2560

print("=" * 60)
print("   真实场景视频合成速度对比测试")
print("=" * 60)
print(f"  分辨率: {WIDTH}x{HEIGHT}")
print(f"  帧率: {FPS} fps")
print(f"  场景ID: {SCENE_ID}")
print(f"  背景图: {BG_PATH}")
print(f"  音频: {AUDIO_PATH}")
print("=" * 60)


def get_audio_duration(audio_path):
    """获取音频时长"""
    cmd = f'ffprobe -i {audio_path} -show_entries format=duration -v quiet -of csv="p=0"'
    return float(os.popen(cmd).read().strip())


def prepare_audio(audio_path, output_folder):
    """准备音频文件"""
    merge_wav_folder = join(output_folder, 'merge_wav')
    os.makedirs(merge_wav_folder, exist_ok=True)

    # 转换为16k采样率
    audio_16k_path = join(merge_wav_folder, 'audio_16k.wav')
    if not exists(audio_16k_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000', audio_16k_path, '-loglevel', 'quiet'
        ], check=True)

    return audio_16k_path


def load_model(inf_folder):
    """加载DINet模型"""
    model_path = None
    model_version = '3p1'
    mouth_region_size = 128
    inf_len = 5

    for file in listdir(inf_folder):
        if file.startswith('DINet') and file.endswith('.pth'):
            model_path = join(inf_folder, file)
            model_version = file.split('_')[0].split('DINetV')[-1]
            mouth_region_size = int(file.split('_')[1])
            inf_len = int(file.split('_')[-1].split('.')[0])
            break

    if model_path is None:
        raise FileNotFoundError(f"未找到模型文件在 {inf_folder}")

    print(f"  加载模型: {model_path}")
    print(f"  模型版本: {model_version}, mouth_region_size: {mouth_region_size}, inf_len: {inf_len}")

    square = model_version.endswith('s') or model_version.endswith('m')
    if model_version.endswith('s') or model_version.endswith('m'):
        model_version = model_version[:-1]

    # 检查是否有reference_images
    if exists(join(inf_folder, 'reference_images')):
        ref_img_path_list = glob(join(inf_folder, 'reference_images', '*.png'))
        ref_num = len(ref_img_path_list)
    else:
        ref_num = 5

    # 根据版本创建模型
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
        raise ValueError(f"不支持的模型版本: {model_version}")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']['net_g'])
    model.eval()

    return model, model_version, mouth_region_size, inf_len, square


def face_gen_simple(face_folder, inf_folder, human_frame_count, mouth_region_size, model_version, square):
    """简化版人脸特征生成器"""
    face_type = listdir(face_folder)[0].split('.')[-1]

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

    # 获取参考图片
    if exists(join(inf_folder, 'reference_images')):
        ref_img_path_list = sorted(glob(join(inf_folder, 'reference_images', f'*.{face_type}')))
    else:
        ref_img_path_list = glob(join(face_folder, f'*.{face_type}'))
        import random
        ref_img_path_list = random.sample(ref_img_path_list, min(5, len(ref_img_path_list)))

    ref_img_list = []
    for ref_img_path in ref_img_path_list:
        if face_type == 'jpg':
            ref_img = cv2.imread(ref_img_path)
            ref_img = torch.tensor(ref_img, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :] / 255
        else:
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)
            ref_img = torch.tensor(ref_img, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [3, 2, 1, 0], :, :] / 255
        ref_img = F.interpolate(ref_img, (resize_h, resize_w), mode='bilinear', align_corners=False)
        ref_img_list.append(ref_img)
    ref_img_tensor = torch.cat(ref_img_list, dim=1)

    # 人脸帧列表
    face_path_list = sorted(glob(join(face_folder, f'*.{face_type}')))
    face_path_list_cycle = face_path_list + face_path_list[::-1]

    frame_id = 0
    while True:
        frame_id = frame_id % (human_frame_count * 2)
        if face_type == 'jpg':
            frame = cv2.imread(face_path_list_cycle[frame_id])
            frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :] / 255
        else:
            frame = cv2.imread(face_path_list_cycle[frame_id], cv2.IMREAD_UNCHANGED)
            frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0)[:, [3, 2, 1, 0], :, :] / 255

        frame_tensor = F.interpolate(frame_tensor, (resize_h, resize_w), mode='bilinear', align_corners=False)

        if model_version == '4p4':
            ref_img_tensor = frame_tensor.clone()

        frame_tensor[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth] = 0

        yield frame_tensor, ref_img_tensor, frame_id
        frame_id += 1


def human_frame_generator(inf_folder, audio_16k_path, human_w, human_h, human_x, human_y, width, height):
    """生成人物帧的生成器"""

    # 加载模型
    model, model_version, mouth_region_size, inf_len, square = load_model(inf_folder)

    # 加载数据
    face_folder = join(inf_folder, 'face')
    body_path_list = sorted(glob(join(inf_folder, 'body', '*')))
    body_path_list_cycle = body_path_list + body_path_list[::-1]
    xy_npy = np.load(join(inf_folder, 'xy.npy'))
    xy_npy_cycle = np.concatenate([xy_npy, xy_npy[::-1]], 0)

    # 计算坐标
    human_add_coordinates_list, human_bg_coordinates_list = get_correct_coordinates(
        human_x, human_y, human_w, human_h, width, height
    )

    # 计算mask位置
    if square:
        y0_mouth = int(mouth_region_size // 16)
    else:
        y0_mouth = int(mouth_region_size // 2)
    y1_mouth = int(mouth_region_size // 2 + mouth_region_size + mouth_region_size // 16)
    x0_mouth = int(mouth_region_size // 16)
    x1_mouth = int(mouth_region_size // 8 + mouth_region_size + mouth_region_size // 16)

    # 创建迭代器
    use_new_audio_process = True
    audio_gen = proxy_audio_feature_concat_gen(audio_16k_path, inf_len, use_new_audio_process)
    human_frame_count = len(body_path_list)
    face_gen = proxy_face_gen(face_folder, inf_folder, 0, human_frame_count, mouth_region_size, model_version, square)

    frame_idx = 0
    while True:
        # 获取音频特征和人脸特征
        audio_feature_tensor = next(audio_gen)
        crop_frame_tensor, ref_img_tensor, frame_id = next(face_gen)

        # 处理通道
        if int(model_version[0]) == 3 and crop_frame_tensor.shape[1] == 4:
            crop_frame_tensor = crop_frame_tensor[:, 1:, :, :]
            ref_img_tensor = ref_img_tensor[:, [i for i in range(ref_img_tensor.shape[1]) if i % 4 != 0], :, :]

        # 模型推理
        with torch.no_grad():
            output = model(crop_frame_tensor, ref_img_tensor, audio_feature_tensor)

        output = output.detach()
        face_pre = crop_frame_tensor.clone()
        face_pre[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth] = output[:, :, y0_mouth:y1_mouth, x0_mouth:x1_mouth]

        if int(model_version[0]) == 3:
            face_pre = face_pre[:, [2, 1, 0], :, :]
        elif int(model_version[0]) == 4:
            face_pre = face_pre[:, [3, 2, 1, 0], :, :]

        # 贴回body
        body_frame = cv2.imread(body_path_list_cycle[frame_id], cv2.IMREAD_UNCHANGED)
        body_frame_tensor = torch.tensor(body_frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255

        x1, x2, y1, y2 = xy_npy_cycle[frame_id]
        face_pre = F.interpolate(face_pre, (y2 - y1, x2 - x1), mode='bilinear', align_corners=False)

        if int(model_version[0]) == 3:
            body_frame_tensor[0, :3, y1:y2, x1:x2] = face_pre[0, :3]
        elif int(model_version[0]) == 4:
            body_frame_tensor[0, :, y1:y2, x1:x2] = face_pre[0, :]

        # 调整大小
        body_frame_tensor = F.interpolate(body_frame_tensor, (human_h, human_w), mode='bilinear', align_corners=False)

        yield body_frame_tensor, human_add_coordinates_list, human_bg_coordinates_list
        frame_idx += 1


def merge_bg_add(bg_tensor, add_tensor, add_coordinates_list, bg_coordinates_list):
    """在背景帧上覆盖素材帧"""
    add_x0, add_x1, add_y0, add_y1 = add_coordinates_list
    bg_x0, bg_x1, bg_y0, bg_y1 = bg_coordinates_list

    if add_x0 > add_x1 or add_y0 > add_y1 or bg_x0 > bg_x1 or bg_y0 > bg_y1:
        return bg_tensor

    if add_tensor.size(1) == 4:
        bg_tensor[0, :3, bg_y0:bg_y1, bg_x0:bg_x1] = (
            bg_tensor[0, :3, bg_y0:bg_y1, bg_x0:bg_x1] * (1 - add_tensor[0, 3, add_y0:add_y1, add_x0:add_x1]) +
            add_tensor[0, :3, add_y0:add_y1, add_x0:add_x1] * add_tensor[0, 3, add_y0:add_y1, add_x0:add_x1]
        )
    return bg_tensor


def get_frame_tensor(frame, width, height):
    """将帧转换为tensor"""
    frame_tensor = torch.tensor(frame, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255
    frame_tensor = F.interpolate(frame_tensor, (height, width), mode='bilinear', align_corners=False)
    if frame_tensor.shape[1] == 3:
        frame_tensor = torch.cat([frame_tensor, torch.ones(1, 1, height, width).cuda()], dim=1)
    return frame_tensor


# ============================================================================
# GPU NV12 帧池 (用于 PyNvVideoCodec GPU Buffer 模式)
# ============================================================================
class GpuNV12FramePool:
    """预分配 GPU 内存池"""

    def __init__(self, width, height):
        import cupy as cp
        self.width = width
        self.height = height
        y_size = height * width
        uv_size = (height // 2) * width
        self._buf = cp.empty(y_size + uv_size, dtype=cp.uint8)
        self.y_plane = self._buf[:y_size].reshape(height, width, 1)
        self.uv_plane = self._buf[y_size:].reshape(height // 2, width // 2, 2)

    def update_from_rgba(self, rgba_tensor):
        """从 RGBA tensor 更新 NV12 数据 (RGB 输入)"""
        import cupy as cp

        height, width = self.height, self.width

        # 输入是 (1, 4, H, W) RGBA, float32 [0, 1]
        r = (rgba_tensor[0, 0, :, :] * 255).float()
        g = (rgba_tensor[0, 1, :, :] * 255).float()
        b = (rgba_tensor[0, 2, :, :] * 255).float()

        # BT.601 Full Range RGB -> YUV
        y = ( 0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)
        u = (-0.169 * r - 0.331 * g + 0.500 * b + 128).clamp(0, 255).to(torch.uint8)
        v = ( 0.500 * r - 0.419 * g - 0.081 * b + 128).clamp(0, 255).to(torch.uint8)

        y_cp = cp.asarray(y)
        u_cp = cp.asarray(u)
        v_cp = cp.asarray(v)

        # Y 平面
        y_size = height * width
        self._buf[:y_size] = y_cp.flatten()

        # UV 交织平面
        # 尝试交换 U 和 V 的位置 (NV12 vs NV21)
        uv = self._buf[y_size:].reshape(height // 2, width)
        uv[:, 0::2] = v_cp[::2, ::2]  # V 在偶数位置
        uv[:, 1::2] = u_cp[::2, ::2]  # U 在奇数位置

        return self

    def cuda(self):
        return [self.y_plane, self.uv_plane]


# ============================================================================
# 方案 1: cv2.VideoWriter (当前方式)
# ============================================================================
def test_cv2_videowriter(output_folder, frame_num, human_gen, background_tensor):
    """测试 cv2.VideoWriter 编码"""
    print("\n" + "=" * 60)
    print("   [方案 1] cv2.VideoWriter (CPU 软件编码)")
    print("=" * 60)

    mp4_path = join(output_folder, "result_cv2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (WIDTH, HEIGHT))

    t_start = time.time()

    for i in range(frame_num):
        # 获取人物帧
        body_frame_tensor, human_add_coords, human_bg_coords = next(human_gen)

        # 合成
        frame = background_tensor.clone()
        frame = merge_bg_add(frame, body_frame_tensor, human_add_coords, human_bg_coords)

        # 转换为 CPU numpy (GPU->CPU 传输)
        frame_np = frame.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        frame_np = frame_np.astype(np.uint8)[..., :3]

        # 写入
        writer.write(frame_np)

        if (i + 1) % 50 == 0:
            print(f"  编码进度: {i+1}/{frame_num}")

    writer.release()
    elapsed = time.time() - t_start
    fps_speed = frame_num / elapsed

    print(f"\n  [结果] 完成!")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {fps_speed:.1f} fps")
    print(f"  输出: {mp4_path}")

    return {'method': 'cv2.VideoWriter', 'elapsed': elapsed, 'fps': fps_speed}


# ============================================================================
# 方案 2: PyNvVideoCodec GPU Buffer 模式
# ============================================================================
def test_pynvcodec_gpu(output_folder, frame_num, human_gen, background_tensor, bitrate="15M"):
    """
    测试 PyNvVideoCodec GPU Buffer 模式编码
    
    Args:
        output_folder: 输出文件夹
        frame_num: 帧数
        human_gen: 人物帧生成器
        background_tensor: 背景张量
        bitrate: 码率，例如 "10M", "20M", "50M"
    """
    import PyNvVideoCodec as nvc

    print("\n" + "=" * 60)
    print("   [方案 2] PyNvVideoCodec GPU Buffer 模式")
    print("=" * 60)

    h264_path = join(output_folder, "result_nvenc_gpu.h264")
    mp4_path = join(output_folder, "result_nvenc_gpu.mp4")

    # 解析码率字符串为数字
    bitrate_value = bitrate.upper()
    if bitrate_value.endswith('M'):
        bitrate_num = int(float(bitrate_value[:-1]) * 1_000_000)
    elif bitrate_value.endswith('K'):
        bitrate_num = int(float(bitrate_value[:-1]) * 1_000)
    else:
        bitrate_num = int(bitrate_value)

    # 创建编码器，添加高码率和质量参数
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=False,
        gpu_id=0,
        codec="h264",
        fps=FPS,
        # 码率控制参数
        bitrate=bitrate_num,
        maxbitrate=int(bitrate_num * 1.5),
        # 质量相关参数
        preset="P1",  # P1 最高质量，P7 最快速度
        tuning_info="high_quality",
        profile="high",
        rc="vbr",  # 可变码率
        gop=FPS * 2,  # GOP 大小
    )

    # 预分配内存池 - 注意：需要修改为支持 RGB 输入
    frame_pool = GpuNV12FramePool(WIDTH, HEIGHT)

    t_start = time.time()

    with open(h264_path, "wb") as f:
        for i in range(frame_num):
            # 获取人物帧
            body_frame_tensor, human_add_coords, human_bg_coords = next(human_gen)

            # 合成 (全在 GPU 上)
            frame = background_tensor.clone()
            frame = merge_bg_add(frame, body_frame_tensor, human_add_coords, human_bg_coords)

            # GPU 上转换为 NV12 (无 GPU->CPU 传输)
            gpu_nv12_frame = frame_pool.update_from_rgba(frame)

            # 编码
            bitstream = nvenc.Encode(gpu_nv12_frame)
            if bitstream:
                f.write(bytearray(bitstream))

            if (i + 1) % 50 == 0:
                print(f"  编码进度: {i+1}/{frame_num}")

        # 刷新
        bitstream = nvenc.EndEncode()
        if bitstream:
            f.write(bytearray(bitstream))

    # 封装 MP4
    print("  正在封装 MP4...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", h264_path, "-c:v", "copy", "-movflags", "+faststart",
        mp4_path, "-loglevel", "quiet"
    ], check=True)
    # os.remove(h264_path)

    elapsed = time.time() - t_start
    fps_speed = frame_num / elapsed

    print(f"\n  [结果] 完成!")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {fps_speed:.1f} fps")
    print(f"  码率: {bitrate}")
    print(f"  输出: {mp4_path}")

    return {'method': 'PyNvVideoCodec GPU', 'elapsed': elapsed, 'fps': fps_speed}


# ============================================================================
# 主函数
# ============================================================================
def main():
    results = []

    # 创建输出目录
    output_dir = join(os.path.dirname(os.path.abspath(__file__)), "benchmark_real_output")
    os.makedirs(output_dir, exist_ok=True)

    # 检查素材
    if not exists(BG_PATH):
        print(f"[错误] 背景图不存在: {BG_PATH}")
        return
    if not exists(AUDIO_PATH):
        print(f"[错误] 音频不存在: {AUDIO_PATH}")
        return
    if not exists(INF_FOLDER):
        print(f"[错误] 人物素材不存在: {INF_FOLDER}")
        return

    # 准备音频
    print("\n[准备] 转换音频...")
    audio_16k_path = prepare_audio(AUDIO_PATH, output_dir)

    # 计算帧数
    audio_duration = get_audio_duration(AUDIO_PATH)
    frame_num = int(audio_duration * FPS)
    print(f"  音频时长: {audio_duration:.2f} 秒")
    print(f"  总帧数: {frame_num}")

    # 计算人物尺寸
    body_sample = cv2.imread(glob(join(INF_FOLDER, 'body', '*'))[0])
    human_w = int(HUMAN_H * body_sample.shape[1] / body_sample.shape[0])
    print(f"  人物尺寸: {human_w}x{HUMAN_H}")

    # 加载背景图
    print("\n[准备] 加载背景图...")
    background = cv2.imread(BG_PATH)
    background_tensor = get_frame_tensor(background, WIDTH, HEIGHT)
    print(f"  背景尺寸: {background.shape}")

    # ---------------------------
    # 方案 1: cv2.VideoWriter
    # ---------------------------
    print("\n" + "-" * 60)
    print("  测试方案 1: cv2.VideoWriter")
    print("-" * 60)

    human_gen1 = human_frame_generator(
        INF_FOLDER, audio_16k_path, human_w, HUMAN_H, HUMAN_X, HUMAN_Y, WIDTH, HEIGHT
    )
    result1 = test_cv2_videowriter(output_dir, frame_num, human_gen1, background_tensor)
    results.append(result1)

    # 释放 GPU 内存
    torch.cuda.empty_cache()
    time.sleep(5)

    # ---------------------------
    # 方案 2: PyNvVideoCodec GPU
    # ---------------------------
    print("\n" + "-" * 60)
    print("  测试方案 2: PyNvVideoCodec GPU Buffer")
    print("-" * 60)

    human_gen2 = human_frame_generator(
        INF_FOLDER, audio_16k_path, human_w, HUMAN_H, HUMAN_X, HUMAN_Y, WIDTH, HEIGHT
    )
    result2 = test_pynvcodec_gpu(output_dir, frame_num, human_gen2, background_tensor)
    results.append(result2)

    # ---------------------------
    # 汇总结果
    # ---------------------------
    print("\n")
    print("=" * 60)
    print("   性能测试结果汇总")
    print("=" * 60)
    print(f"  测试参数: {WIDTH}x{HEIGHT}, {FPS}fps, {frame_num}帧")
    print("-" * 60)
    print(f"  {'编码方式':<25} {'耗时(秒)':<12} {'速度(fps)':<12}")
    print("-" * 60)

    for r in results:
        if r.get('elapsed') is not None:
            speedup = r['fps'] / results[0]['fps']
            print(f"  {r['method']:<25} {r['elapsed']:<12.2f} {r['fps']:<12.1f}  ({speedup:.2f}x)")

    print("=" * 60)

    if len(results) >= 2:
        speedup = results[1]['fps'] / results[0]['fps']
        print(f"\n  GPU Buffer 模式 vs cv2.VideoWriter: {speedup:.2f}x 加速")

    print("\n  测试完成!")


if __name__ == "__main__":
    main()