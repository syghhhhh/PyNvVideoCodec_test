#!/usr/bin/env python3
"""
色彩空间转换对比测试
生成4种方案的视频并拼合对比
"""

import os
import subprocess
import numpy as np
import cv2
import torch

# 测试参数
WIDTH = 640
HEIGHT = 480
FPS = 25
DURATION = 1  # 秒
FRAME_NUM = FPS * DURATION
OUTPUT_DIR = "color_test_output"


def create_gradient_frame():
    """创建渐变颜色测试帧 (BGR格式)"""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # 水平渐变: 红 -> 绿 -> 蓝
    third = WIDTH // 3
    for x in range(WIDTH):
        if x < third:
            # 红到黄
            ratio = x / third if third > 0 else 0
            r = 255
            g = int(255 * ratio)
            b = 0
        elif x < 2 * third:
            # 黄到青
            ratio = (x - third) / third if third > 0 else 0
            r = int(255 * (1 - ratio))
            g = 255
            b = int(255 * ratio)
        else:
            # 青到蓝到紫
            ratio = (x - 2 * third) / (WIDTH - 2 * third) if (WIDTH - 2 * third) > 0 else 0
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            b = 255
        
        # 确保值在有效范围内
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        frame[:, x] = [b, g, r]  # BGR格式
    
    # 垂直渐变: 叠加亮度变化
    for y in range(HEIGHT):
        brightness = 0.3 + 0.7 * (y / HEIGHT)  # 从30%到100%亮度
        frame[y, :] = np.clip(frame[y, :] * brightness, 0, 255).astype(np.uint8)
    
    # 添加一些纯色块用于对比
    block_h = HEIGHT // 6
    block_w = WIDTH // 8
    margin = 10
    
    # 第一行色块
    colors = [
        ([255, 255, 255], "White"),   # 纯白
        ([0, 0, 0], "Black"),          # 纯黑
        ([0, 0, 255], "Red"),          # 纯红 (BGR)
        ([0, 255, 0], "Green"),        # 纯绿
        ([255, 0, 0], "Blue"),         # 纯蓝
        ([180, 200, 230], "Skin"),     # 肤色
        ([128, 128, 128], "Gray"),     # 灰色
    ]
    
    for i, (color, name) in enumerate(colors):
        x_start = margin + i * (block_w + margin)
        x_end = x_start + block_w
        if x_end <= WIDTH:
            frame[margin:margin+block_h, x_start:x_end] = color
    
    return frame


def method_cv2_baseline(frame_bgr, output_path):
    """方案0: CV2 基础方案 (作为对照)"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    for _ in range(FRAME_NUM):
        out.write(frame_bgr)
    
    out.release()
    
    # 转换为H264
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    os.rename(output_path, temp_path)
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_path,
        "-loglevel", "quiet"
    ], check=True)
    os.remove(temp_path)


class GpuNV12FramePool:
    """GPU NV12 帧池"""
    
    def __init__(self, width, height):
        import cupy as cp
        self.width = width
        self.height = height
        y_size = height * width
        uv_size = (height // 2) * width
        self._buf = cp.empty(y_size + uv_size, dtype=cp.uint8)
        self.y_plane = self._buf[:y_size].reshape(height, width, 1)
        self.uv_plane = self._buf[y_size:].reshape(height // 2, width // 2, 2)
    
    def update_bt709_limited(self, bgra_tensor):
        """方案1: BT.709 Limited Range (原始方案)"""
        import cupy as cp
        
        height, width = self.height, self.width
        
        # 输入是 BGRA 格式
        b = (bgra_tensor[0, 0, :, :] * 255).float()
        g = (bgra_tensor[0, 1, :, :] * 255).float()
        r = (bgra_tensor[0, 2, :, :] * 255).float()
        
        # BT.709 Limited Range
        y = (0.183 * r + 0.614 * g + 0.062 * b + 16).clamp(16, 235).to(torch.uint8)
        u = (-0.101 * r - 0.339 * g + 0.439 * b + 128).clamp(16, 240).to(torch.uint8)
        v = (0.439 * r - 0.399 * g - 0.040 * b + 128).clamp(16, 240).to(torch.uint8)
        
        self._fill_nv12(y, u, v, cp, height, width)
        return self
    
    def update_bt709_full(self, bgra_tensor):
        """方案2: BT.709 Full Range"""
        import cupy as cp
        
        height, width = self.height, self.width
        
        # 输入是 BGRA 格式
        b = (bgra_tensor[0, 0, :, :] * 255).float()
        g = (bgra_tensor[0, 1, :, :] * 255).float()
        r = (bgra_tensor[0, 2, :, :] * 255).float()
        
        # BT.709 Full Range
        y = (0.2126 * r + 0.7152 * g + 0.0722 * b).round().clamp(0, 255).to(torch.uint8)
        u = (-0.1146 * r - 0.3854 * g + 0.5000 * b + 128).round().clamp(0, 255).to(torch.uint8)
        v = (0.5000 * r - 0.4542 * g - 0.0458 * b + 128).round().clamp(0, 255).to(torch.uint8)
        
        self._fill_nv12(y, u, v, cp, height, width)
        return self
    
    def update_bt709_limited_precise(self, bgra_tensor):
        """方案3: BT.709 Limited Range (精确整数公式)"""
        import cupy as cp
        
        height, width = self.height, self.width
        
        # 输入是 BGRA 格式
        b = (bgra_tensor[0, 0, :, :] * 255).float()
        g = (bgra_tensor[0, 1, :, :] * 255).float()
        r = (bgra_tensor[0, 2, :, :] * 255).float()
        
        # BT.709 Limited Range (精确整数公式)
        y = (((47 * r + 157 * g + 16 * b + 128) / 256) + 16).round().clamp(16, 235).to(torch.uint8)
        u = (((-26 * r - 87 * g + 112 * b + 128) / 256) + 128).round().clamp(16, 240).to(torch.uint8)
        v = (((112 * r - 102 * g - 10 * b + 128) / 256) + 128).round().clamp(16, 240).to(torch.uint8)
        
        self._fill_nv12(y, u, v, cp, height, width)
        return self
    
    def _fill_nv12(self, y, u, v, cp, height, width):
        """填充NV12数据 - NV12格式是 UVUV... 交织"""
        y_cp = cp.asarray(y)
        u_cp = cp.asarray(u)
        v_cp = cp.asarray(v)
        
        y_size = height * width
        self._buf[:y_size] = y_cp.flatten()
        
        # NV12 UV平面: U在偶数位置, V在奇数位置 (UVUVUV...)
        uv = self._buf[y_size:].reshape(height // 2, width)
        uv[:, 0::2] = u_cp[::2, ::2]  # U 在偶数位置
        uv[:, 1::2] = v_cp[::2, ::2]  # V 在奇数位置
    
    def cuda(self):
        return [self.y_plane, self.uv_plane]


def method_nvenc(frame_tensor, output_path, method_name, color_range="tv"):
    """使用 PyNvVideoCodec 编码"""
    import PyNvVideoCodec as nvc
    
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=False,
        gpu_id=0,
        codec="h264",
        fps=FPS,
        bitrate=8_000_000,
        maxbitrate=12_000_000,
        preset="P1",
        tuning_info="high_quality",
        profile="high",
        rc="vbr",
        gop=50,
    )
    
    frame_pool = GpuNV12FramePool(WIDTH, HEIGHT)
    h264_path = output_path.replace('.mp4', '.h264')
    
    with open(h264_path, "wb") as h264_file:
        for _ in range(FRAME_NUM):
            if method_name == "bt709_limited":
                frame_pool.update_bt709_limited(frame_tensor)
            elif method_name == "bt709_full":
                frame_pool.update_bt709_full(frame_tensor)
            elif method_name == "bt709_limited_precise":
                frame_pool.update_bt709_limited_precise(frame_tensor)
            
            bitstream = nvenc.Encode(frame_pool)
            if bitstream:
                h264_file.write(bytearray(bitstream))
        
        bitstream = nvenc.EndEncode()
        if bitstream:
            h264_file.write(bytearray(bitstream))
    
    # 封装MP4
    color_range_flag = "pc" if color_range == "pc" else "tv"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", h264_path,
        "-c:v", "copy",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", color_range_flag,
        "-movflags", "+faststart",
        output_path,
        "-loglevel", "quiet"
    ], check=True)
    
    os.remove(h264_path)


def add_label_to_video(input_path, output_path, label):
    """给视频添加标签"""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"drawtext=text='{label}':fontsize=24:fontcolor=white:borderw=2:bordercolor=black:x=10:y=H-40",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_path,
        "-loglevel", "quiet"
    ], check=True)


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建测试帧
    print("创建渐变测试帧...")
    frame_bgr = create_gradient_frame()
    
    # 保存测试帧图片
    cv2.imwrite(os.path.join(OUTPUT_DIR, "test_frame.png"), frame_bgr)
    print(f"测试帧已保存: {OUTPUT_DIR}/test_frame.png")
    
    # 转换为BGRA tensor (保持BGR顺序)
    frame_bgra = np.concatenate([frame_bgr, np.full((HEIGHT, WIDTH, 1), 255, dtype=np.uint8)], axis=2)
    frame_tensor = torch.tensor(frame_bgra, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255
    
    # 方案0: CV2基础方案
    print("生成方案0: CV2基础方案...")
    cv2_path = os.path.join(OUTPUT_DIR, "0_cv2_baseline.mp4")
    method_cv2_baseline(frame_bgr, cv2_path)
    
    # 方案1: BT.709 Limited Range (原始方案)
    print("生成方案1: BT.709 Limited Range...")
    limited_path = os.path.join(OUTPUT_DIR, "1_bt709_limited.mp4")
    method_nvenc(frame_tensor, limited_path, "bt709_limited", "tv")
    
    # 方案2: BT.709 Full Range
    print("生成方案2: BT.709 Full Range...")
    full_path = os.path.join(OUTPUT_DIR, "2_bt709_full.mp4")
    method_nvenc(frame_tensor, full_path, "bt709_full", "pc")
    
    # 方案3: BT.709 Limited Range (精确公式)
    print("生成方案3: BT.709 Limited Range (精确公式)...")
    precise_path = os.path.join(OUTPUT_DIR, "3_bt709_limited_precise.mp4")
    method_nvenc(frame_tensor, precise_path, "bt709_limited_precise", "tv")
    
    # 添加标签
    print("添加视频标签...")
    videos_with_labels = []
    labels = [
        ("0_cv2_baseline.mp4", "CV2 Baseline"),
        ("1_bt709_limited.mp4", "BT709 Limited"),
        ("2_bt709_full.mp4", "BT709 Full"),
        ("3_bt709_limited_precise.mp4", "BT709 Limited Precise"),
    ]
    
    for filename, label in labels:
        input_path = os.path.join(OUTPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"labeled_{filename}")
        add_label_to_video(input_path, output_path, label)
        videos_with_labels.append(output_path)
    
    # 拼合成田字格视频
    print("拼合成对比视频...")
    final_output = os.path.join(OUTPUT_DIR, "comparison_grid.mp4")
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", videos_with_labels[0],
        "-i", videos_with_labels[1],
        "-i", videos_with_labels[2],
        "-i", videos_with_labels[3],
        "-filter_complex",
        "[0:v]scale=640:480[v0];"
        "[1:v]scale=640:480[v1];"
        "[2:v]scale=640:480[v2];"
        "[3:v]scale=640:480[v3];"
        "[v0][v1]hstack[top];"
        "[v2][v3]hstack[bottom];"
        "[top][bottom]vstack[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        final_output,
        "-loglevel", "warning"
    ], check=True)
    
    # 清理临时文件
    for path in videos_with_labels:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"\n完成! 对比视频已保存: {final_output}")
    print(f"各方案单独视频保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()