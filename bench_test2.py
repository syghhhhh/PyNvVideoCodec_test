import os
import subprocess
import numpy as np
import cv2
import torch
import time
import PyNvVideoCodec as nvc

# 测试参数
WIDTH = 1920
HEIGHT = 1080
FPS = 25
DURATION = 30  # 秒
FRAME_NUM = FPS * DURATION
OUTPUT_DIR = "color_test_output"


def create_gradient_frame():
    """创建渐变颜色测试帧 (BGR格式)"""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # 水平渐变: 红 -> 绿 -> 蓝
    third = WIDTH // 3
    for x in range(WIDTH):
        if x < third:
            ratio = x / third if third > 0 else 0
            r, g, b = 255, int(255 * ratio), 0
        elif x < 2 * third:
            ratio = (x - third) / third if third > 0 else 0
            r, g, b = int(255 * (1 - ratio)), 255, int(255 * ratio)
        else:
            ratio = (x - 2 * third) / (WIDTH - 2 * third) if (WIDTH - 2 * third) > 0 else 0
            r, g, b = int(255 * ratio), int(255 * (1 - ratio)), 255
        
        r, g, b = [max(0, min(255, v)) for v in [r, g, b]]
        frame[:, x] = [b, g, r]  # BGR格式
    
    # 垂直亮度渐变
    for y in range(HEIGHT):
        brightness = 0.3 + 0.7 * (y / HEIGHT)
        frame[y, :] = np.clip(frame[y, :] * brightness, 0, 255).astype(np.uint8)
    
    # 添加纯色块
    block_h, block_w, margin = HEIGHT // 6, WIDTH // 8, 20
    colors = [
        [255, 255, 255], [0, 0, 0], [0, 0, 255],
        [0, 255, 0], [255, 0, 0], [180, 200, 230], [128, 128, 128]
    ]
    for i, color in enumerate(colors):
        x_start = margin + i * (block_w + margin)
        if x_start + block_w <= WIDTH:
            frame[margin:margin+block_h, x_start:x_start+block_w] = color
    
    return frame


def bgr_to_nv12(bgr_frame):
    """将BGR帧转换为NV12格式"""
    yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr_frame.shape[:2]
    y_plane = yuv[:h, :]
    u_plane = yuv[h:h + h//4, :].reshape(-1)
    v_plane = yuv[h + h//4:, :].reshape(-1)
    uv_plane = np.empty((h * w // 4 * 2,), dtype=np.uint8)
    uv_plane[0::2], uv_plane[1::2] = u_plane, v_plane
    frame_nv12 = np.vstack([y_plane, uv_plane.reshape(h//2, w)])
    return frame_nv12


def method_cv2(base_frame, output_path):
    """CV2 编码"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    for _ in range(FRAME_NUM):
        out.write(base_frame)
    out.release()
    
    # 转H264
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    os.rename(output_path, temp_path)
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_path, "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", output_path, "-loglevel", "quiet"
    ], check=True)
    os.remove(temp_path)


def method_pynvc_cpu(base_frame, output_path):
    """PyNvVideoCodec CPU模式 (NV12)"""
    h264_path = output_path.replace('.mp4', '.h264')
    
    # 预先转换NV12
    frame_nv12 = bgr_to_nv12(base_frame)
    
    encoder = nvc.CreateEncoder(
        width=WIDTH, height=HEIGHT, fmt="NV12",
        usecpuinputbuffer=True, gpu_id=0, codec="h264",
        fps=FPS, bitrate=8000000, preset="P4",
        tuning_info="high_quality", gop=FPS
    )
    
    with open(h264_path, "wb") as f:
        for _ in range(FRAME_NUM):
            bs = encoder.Encode(frame_nv12)
            if bs: f.write(bytearray(bs))
        bs = encoder.EndEncode()
        if bs: f.write(bytearray(bs))
    
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", h264_path,
        "-c:v", "copy", output_path, "-loglevel", "quiet"
    ], check=True)
    os.remove(h264_path)


def method_pynvc_gpu(base_frame, output_path):
    """PyNvVideoCodec GPU模式 (NV12)"""
    h264_path = output_path.replace('.mp4', '.h264')
    
    # 预先转换NV12并上传GPU
    frame_nv12 = bgr_to_nv12(base_frame)
    frame_tensor = torch.from_numpy(frame_nv12).cuda().contiguous()
    
    encoder = nvc.CreateEncoder(
        width=WIDTH, height=HEIGHT, fmt="NV12",
        usecpuinputbuffer=False, gpu_id=0, codec="h264",
        fps=FPS, bitrate=8000000, preset="P4",
        tuning_info="high_quality", gop=FPS
    )
    
    with open(h264_path, "wb") as f:
        for _ in range(FRAME_NUM):
            bs = encoder.Encode(frame_tensor)
            if bs: f.write(bytearray(bs))
        bs = encoder.EndEncode()
        if bs: f.write(bytearray(bs))
    
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", h264_path,
        "-c:v", "copy", output_path, "-loglevel", "quiet"
    ], check=True)
    os.remove(h264_path)


def combine_videos_vertical(video_paths, labels, output_path):
    """纵向拼接视频"""
    n = len(video_paths)
    inputs = " ".join([f"-i {p}" for p in video_paths])
    
    filter_parts = [
        f"[{i}:v]drawtext=text='{labels[i]}':fontsize=36:fontcolor=white:"
        f"x=20:y=20:box=1:boxcolor=black@0.7:boxborderw=8[v{i}]"
        for i in range(n)
    ]
    filter_parts.append(f"{''.join([f'[v{i}]' for i in range(n)])}vstack=inputs={n}[out]")
    
    cmd = f'ffmpeg -y {inputs} -filter_complex "{";".join(filter_parts)}" -map "[out]" -c:v libx264 -crf 18 {output_path} -loglevel quiet'
    subprocess.run(cmd, shell=True, check=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 65)
    print("视频编码性能对比测试")
    print(f"分辨率: {WIDTH}x{HEIGHT}, 帧率: {FPS}, 时长: {DURATION}秒, 总帧数: {FRAME_NUM}")
    print("=" * 65)
    
    # 创建测试帧
    print("\n创建测试帧...")
    base_frame = create_gradient_frame()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "test_frame.png"), base_frame)
    
    # 测试方法
    methods = [
        ("CV2 (libx264)", method_cv2, "cv2.mp4"),
        ("PyNVC CPU (NV12)", method_pynvc_cpu, "pynvc_cpu.mp4"),
        ("PyNVC GPU (NV12)", method_pynvc_gpu, "pynvc_gpu.mp4"),
    ]
    
    timings = {}
    outputs = []
    labels = []
    
    for name, func, filename in methods:
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"\n[{name}] 编码中...")
        
        start = time.perf_counter()
        func(base_frame, output_path)
        elapsed = time.perf_counter() - start
        
        timings[name] = elapsed
        outputs.append(output_path)
        labels.append(name.replace(" ", "_"))
        print(f"[{name}] 完成: {elapsed:.3f}秒")
    
    # 打印统计
    print("\n" + "=" * 65)
    print(f"{'方案':<25} {'耗时(秒)':<12} {'FPS':<12} {'相对速度':<12}")
    print("-" * 65)
    
    baseline = timings["CV2 (libx264)"]
    for name, elapsed in timings.items():
        fps = FRAME_NUM / elapsed
        speedup = baseline / elapsed
        print(f"{name:<25} {elapsed:<12.3f} {fps:<12.1f} {speedup:<12.2f}x")
    
    print("=" * 65)
    
    # 拼接对比视频
    print("\n生成对比视频...")
    combine_videos_vertical(outputs, labels, os.path.join(OUTPUT_DIR, "comparison.mp4"))
    print(f"完成: {OUTPUT_DIR}/comparison.mp4")


if __name__ == "__main__":
    main()