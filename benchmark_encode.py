#!/usr/bin/env python3
"""
视频编码速度对比测试
======================
生成 10 秒 25fps 的 2560x1440 随机画面数据，对比三种编码方式的性能：
1. cv2.VideoWriter (CPU 软件编码，当前 video_merge.py 使用的方式)
2. PyNvVideoCodec CPU Buffer 模式
3. PyNvVideoCodec GPU Buffer 模式 (目标优化方案)

依赖:
    pip install PyNvVideoCodec numpy cupy-cuda12x opencv-python torch

用法:
    python benchmark_encode.py
"""

import os
import time
import subprocess
import numpy as np
import torch

# 测试参数
WIDTH = 2560
HEIGHT = 1440
FPS = 25
DURATION = 5  # 秒
NUM_FRAMES = FPS * DURATION  # 250 帧

print("=" * 60)
print("   视频编码速度对比测试")
print("=" * 60)
print(f"  分辨率: {WIDTH}x{HEIGHT}")
print(f"  帧率: {FPS} fps")
print(f"  时长: {DURATION} 秒")
print(f"  总帧数: {NUM_FRAMES} 帧")
print("=" * 60)


def _generate_random_frames_gpu(num_frames, width, height):
    """
    生成随机帧数据（GPU tensor）
    统一的数据源格式
    返回: list of torch tensors (1, 4, H, W) RGBA, float32 [0, 1], 位于 GPU
    """
    print("\n[生成测试数据] 生成随机帧数据 (GPU tensor)...")
    frames = []
    for i in range(num_frames):
        # 生成随机 RGBA 帧
        frame = torch.rand(1, 4, height, width, dtype=torch.float32, device='cuda')
        frames.append(frame)
        if (i + 1) % 50 == 0:
            print(f"  生成进度: {i+1}/{num_frames}")
    return frames


def generate_random_frames_gpu(num_frames, width, height):
    """
    生成彩虹色渐变测试帧数据（GPU tensor）
    颜色平滑过渡: 红 -> 橙 -> 黄 -> 绿 -> 青 -> 蓝 -> 紫 -> 红
    返回: list of torch tensors (1, 4, H, W) RGBA, float32 [0, 1], 位于 GPU
    """
    print("\n[生成测试数据] 生成彩虹色渐变测试帧 (GPU tensor)...")
    
    # 定义彩虹色关键帧 (RGB 格式, 0-1 范围)
    rainbow_colors = [
        [1.0, 0.0, 0.0],  # 红
        [1.0, 0.5, 0.0],  # 橙
        [1.0, 1.0, 0.0],  # 黄
        [0.0, 1.0, 0.0],  # 绿
        [0.0, 1.0, 1.0],  # 青
        [0.0, 0.0, 1.0],  # 蓝
        [0.5, 0.0, 1.0],  # 紫
        [1.0, 0.0, 0.0],  # 红 (回到起点，形成循环)
    ]
    
    frames = []
    for i in range(num_frames):
        # 计算在彩虹中的位置 (0.0 - 7.0)
        t = (i / num_frames) * (len(rainbow_colors) - 1)
        
        # 找到相邻的两个颜色进行插值
        idx1 = int(t)
        idx2 = min(idx1 + 1, len(rainbow_colors) - 1)
        frac = t - idx1  # 插值系数
        
        # 线性插值
        color = [
            rainbow_colors[idx1][c] * (1 - frac) + rainbow_colors[idx2][c] * frac
            for c in range(3)
        ]
        
        # 创建纯色帧 (1, 4, H, W) RGBA
        frame = torch.zeros(1, 4, height, width, dtype=torch.float32, device='cuda')
        frame[0, 0, :, :] = color[0]  # R
        frame[0, 1, :, :] = color[1]  # G
        frame[0, 2, :, :] = color[2]  # B
        frame[0, 3, :, :] = 1.0       # A
        
        frames.append(frame)
        
        if (i + 1) % 50 == 0:
            print(f"  生成进度: {i+1}/{num_frames}")
    
    print("  颜色: 彩虹渐变 (红->橙->黄->绿->青->蓝->紫->红)")
    return frames


def gpu_tensor_to_bgr_numpy(frame_tensor):
    """
    将 GPU tensor 转换为 BGR numpy array (用于 cv2.VideoWriter)
    :param frame_tensor: (1, 4, H, W) RGBA tensor，值范围 [0, 1]，位于 GPU
    :return: (H, W, 3) BGR numpy array, uint8
    """
    # 提取 RGB 通道并转换为 [0, 255] 范围
    rgb = frame_tensor[0, :3, :, :]  # (3, H, W)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    
    # 调整维度顺序: (3, H, W) -> (H, W, 3)
    rgb = rgb.permute(1, 2, 0)
    
    # GPU -> CPU，转换为 numpy
    rgb_np = rgb.cpu().numpy()
    
    # RGB -> BGR
    bgr_np = rgb_np[:, :, ::-1].copy()
    
    return bgr_np


def gpu_tensor_to_nv12_cpu(frame_tensor, width, height):
    """
    将 GPU tensor 转换为 NV12 numpy array (用于 PyNvVideoCodec CPU Buffer 模式)
    :param frame_tensor: (1, 4, H, W) RGBA tensor，值范围 [0, 1]，位于 GPU
    :return: NV12 numpy array
    """
    # 提取 RGB 通道，转换为 float 以便计算
    r = (frame_tensor[0, 0, :, :] * 255).float()
    g = (frame_tensor[0, 1, :, :] * 255).float()
    b = (frame_tensor[0, 2, :, :] * 255).float()

    # RGB -> YUV (BT.601) 在 GPU 上计算
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)
    u = (-0.14713 * r - 0.28886 * g + 0.436 * b + 128).clamp(0, 255).to(torch.uint8)
    v = (0.615 * r - 0.51499 * g - 0.10001 * b + 128).clamp(0, 255).to(torch.uint8)

    # 转换到 CPU
    y_np = y.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()

    # Y 平面
    y_plane = y_np.flatten()

    # UV 平面 (U/V 交错，2x2 下采样)
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u_np[::2, ::2]  # U
    uv_plane[:, 1::2] = v_np[::2, ::2]  # V

    return np.concatenate([y_plane, uv_plane.flatten()])


# ============================================================================
# 方案 1: cv2.VideoWriter (当前 video_merge.py 使用的方式)
# ============================================================================
def test_cv2_videowriter(frames_gpu, output_path):
    """
    测试 cv2.VideoWriter 编码速度
    模拟当前 video_merge.py 中的编码流程
    输入为 GPU tensor，每帧转换为 CPU BGR numpy array
    """
    import cv2

    print("\n" + "=" * 60)
    print("   [方案 1] cv2.VideoWriter (CPU 软件编码)")
    print("=" * 60)

    mp4_path = output_path.replace('.h264', '_cv2.mp4')

    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (WIDTH, HEIGHT))

    # 计时开始
    t_start = time.time()

    # 逐帧写入
    for i, frame_tensor in enumerate(frames_gpu):
        # GPU tensor -> BGR numpy (包含 GPU->CPU 传输)
        frame_bgr = gpu_tensor_to_bgr_numpy(frame_tensor)
        writer.write(frame_bgr)
        if (i + 1) % 50 == 0:
            print(f"  编码进度: {i+1}/{NUM_FRAMES}")

    writer.release()

    # 计时结束
    elapsed = time.time() - t_start
    fps_speed = NUM_FRAMES / elapsed

    # 获取文件大小
    file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB

    print(f"\n  [结果] 编码完成!")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {fps_speed:.1f} fps")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  输出文件: {mp4_path}")

    return {
        'method': 'cv2.VideoWriter',
        'elapsed': elapsed,
        'fps': fps_speed,
        'file_size_mb': file_size,
        'output_path': mp4_path
    }


# ============================================================================
# 方案 2: PyNvVideoCodec CPU Buffer 模式
# ============================================================================
def test_pynvcodec_cpu(frames_gpu, output_path):
    """
    测试 PyNvVideoCodec CPU Buffer 模式编码速度
    输入为 GPU tensor，每帧转换为 CPU NV12 numpy array
    """
    import PyNvVideoCodec as nvc

    print("\n" + "=" * 60)
    print("   [方案 2] PyNvVideoCodec CPU Buffer 模式")
    print("=" * 60)

    h264_path = output_path.replace('.h264', '_nvenc_cpu.h264')
    mp4_path = output_path.replace('.h264', '_nvenc_cpu.mp4')

    # 创建编码器
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=True,  # CPU Buffer 模式
        gpu_id=0,
        codec="h264",
        fps=FPS,  # 添加帧率设置
    )

    encoded_bytes = 0

    # 计时开始
    t_start = time.time()

    with open(h264_path, "wb") as f:
        for i, frame_tensor in enumerate(frames_gpu):
            # GPU tensor -> NV12 numpy (包含 GPU->CPU 传输)
            nv12_frame = gpu_tensor_to_nv12_cpu(frame_tensor, WIDTH, HEIGHT)

            # 编码
            bitstream = nvenc.Encode(nv12_frame)
            if bitstream:
                data = bytearray(bitstream)
                f.write(data)
                encoded_bytes += len(data)

            if (i + 1) % 50 == 0:
                print(f"  编码进度: {i+1}/{NUM_FRAMES}")

        # 刷新编码器
        bitstream = nvenc.EndEncode()
        if bitstream:
            data = bytearray(bitstream)
            f.write(data)
            encoded_bytes += len(data)

    # 计时结束
    elapsed = time.time() - t_start
    fps_speed = NUM_FRAMES / elapsed

    # 使用 ffmpeg 封装为 MP4
    print("  正在封装 MP4...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", h264_path, "-c:v", "copy", "-movflags", "+faststart",
        mp4_path, "-loglevel", "quiet"
    ], check=True)
    os.remove(h264_path)

    # 获取文件大小
    file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB

    print(f"\n  [结果] 编码完成!")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {fps_speed:.1f} fps")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  输出文件: {mp4_path}")

    return {
        'method': 'PyNvVideoCodec CPU',
        'elapsed': elapsed,
        'fps': fps_speed,
        'file_size_mb': file_size,
        'output_path': mp4_path
    }


# ============================================================================
# 方案 3: PyNvVideoCodec GPU Buffer 模式 (目标优化方案)
# ============================================================================

class GpuNV12FramePool:
    """
    GPU NV12 帧池 - 预分配 GPU 内存，避免每帧分配开销
    这是优化 GPU Buffer 模式的关键
    """
    def __init__(self, width, height):
        import cupy as cp
        self.width = width
        self.height = height
        y_size = height * width
        uv_size = (height // 2) * width
        # 预分配 GPU 内存（只需一次）
        self._buf = cp.empty(y_size + uv_size, dtype=cp.uint8)
        # 预创建视图
        self.y_plane = self._buf[:y_size].reshape(height, width, 1)
        self.uv_plane = self._buf[y_size:].reshape(height // 2, width // 2, 2)

    def update_from_rgba(self, rgba_tensor):
        """
        从 RGBA tensor 更新 NV12 数据（复用预分配内存）
        :param rgba_tensor: (1, 4, H, W) RGBA tensor，值范围 [0, 1]，位于 GPU
        """
        import cupy as cp

        height, width = self.height, self.width

        # 提取 RGB 通道，转换为 float
        r = (rgba_tensor[0, 0, :, :] * 255).float()
        g = (rgba_tensor[0, 1, :, :] * 255).float()
        b = (rgba_tensor[0, 2, :, :] * 255).float()

        # RGB -> YUV (BT.601)
        y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)
        u = (-0.14713 * r - 0.28886 * g + 0.436 * b + 128).clamp(0, 255).to(torch.uint8)
        v = (0.615 * r - 0.51499 * g - 0.10001 * b + 128).clamp(0, 255).to(torch.uint8)

        # torch -> cupy (共享 GPU 内存，零拷贝)
        y_cp = cp.asarray(y)
        u_cp = cp.asarray(u)
        v_cp = cp.asarray(v)

        y_size = height * width

        # 直接填充预分配的 buffer（无新分配）
        self._buf[:y_size] = y_cp.flatten()

        # UV 平面
        uv = self._buf[y_size:].reshape(height // 2, width)
        uv[:, 0::2] = u_cp[::2, ::2]  # U
        uv[:, 1::2] = v_cp[::2, ::2]  # V

        return self

    def cuda(self):
        """返回 CUDA Array Interface"""
        return [self.y_plane, self.uv_plane]


class GpuNV12Frame:
    """
    GPU 上的 NV12 帧封装类（每帧分配新内存，性能较差）
    保留此类用于对比测试
    """
    def __init__(self, rgba_tensor):
        """
        从 RGBA tensor 创建 NV12 帧
        :param rgba_tensor: (1, 4, H, W) RGBA tensor，值范围 [0, 1]，位于 GPU
        """
        import cupy as cp

        # 提取 RGB 通道，转换为 float
        r = (rgba_tensor[0, 0, :, :] * 255).float()
        g = (rgba_tensor[0, 1, :, :] * 255).float()
        b = (rgba_tensor[0, 2, :, :] * 255).float()

        height, width = r.shape

        # RGB -> YUV (BT.601)
        y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)
        u = (-0.14713 * r - 0.28886 * g + 0.436 * b + 128).clamp(0, 255).to(torch.uint8)
        v = (0.615 * r - 0.51499 * g - 0.10001 * b + 128).clamp(0, 255).to(torch.uint8)

        # 转换为 cupy array (零拷贝，共享 GPU 内存)
        y_cp = cp.asarray(y)
        u_cp = cp.asarray(u)
        v_cp = cp.asarray(v)

        # 分配连续 GPU 内存（每帧都分配，性能瓶颈！）
        y_size = height * width
        uv_size = (height // 2) * width
        self._buf = cp.empty(y_size + uv_size, dtype=cp.uint8)

        # 填充 Y 平面
        self._buf[:y_size] = y_cp.flatten()

        # 填充 UV 平面 (U/V 交错，2x2 下采样)
        uv = self._buf[y_size:].reshape(height // 2, width)
        uv[:, 0::2] = u_cp[::2, ::2]  # U
        uv[:, 1::2] = v_cp[::2, ::2]  # V

        # 创建视图
        self.y_plane = self._buf[:y_size].reshape(height, width, 1)
        self.uv_plane = self._buf[y_size:].reshape(height // 2, width // 2, 2)

    def cuda(self):
        """返回 CUDA Array Interface"""
        return [self.y_plane, self.uv_plane]


def test_pynvcodec_gpu(frames_gpu, output_path):
    """
    测试 PyNvVideoCodec GPU Buffer 模式编码速度（使用预分配内存池优化）
    这是目标优化方案，全程在 GPU 上处理，无 GPU->CPU 传输
    """
    import PyNvVideoCodec as nvc

    print("\n" + "=" * 60)
    print("   [方案 3] PyNvVideoCodec GPU Buffer 模式 (预分配内存池优化)")
    print("=" * 60)

    h264_path = output_path.replace('.h264', '_nvenc_gpu.h264')
    mp4_path = output_path.replace('.h264', '_nvenc_gpu.mp4')

    # 创建编码器
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=False,  # GPU Buffer 模式
        gpu_id=0,
        codec="h264",
        fps=FPS,  # 添加帧率设置
    )

    # 预分配 GPU 内存池（关键优化：避免每帧分配）
    frame_pool = GpuNV12FramePool(WIDTH, HEIGHT)

    encoded_bytes = 0

    # 计时开始
    t_start = time.time()

    with open(h264_path, "wb") as f:
        for i, frame_tensor in enumerate(frames_gpu):
            # 复用预分配内存，无 GPU->CPU 传输，无新内存分配
            gpu_nv12_frame = frame_pool.update_from_rgba(frame_tensor)

            # 编码
            bitstream = nvenc.Encode(gpu_nv12_frame)
            if bitstream:
                data = bytearray(bitstream)
                f.write(data)
                encoded_bytes += len(data)

            if (i + 1) % 50 == 0:
                print(f"  编码进度: {i+1}/{NUM_FRAMES}")

        # 刷新编码器
        bitstream = nvenc.EndEncode()
        if bitstream:
            data = bytearray(bitstream)
            f.write(data)
            encoded_bytes += len(data)

    # 计时结束
    elapsed = time.time() - t_start
    fps_speed = NUM_FRAMES / elapsed

    # 使用 ffmpeg 封装为 MP4
    print("  正在封装 MP4...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", h264_path, "-c:v", "copy", "-movflags", "+faststart",
        mp4_path, "-loglevel", "quiet"
    ], check=True)
    os.remove(h264_path)

    # 获取文件大小
    file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB

    print(f"\n  [结果] 编码完成!")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {fps_speed:.1f} fps")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  输出文件: {mp4_path}")

    return {
        'method': 'PyNvVideoCodec GPU',
        'elapsed': elapsed,
        'fps': fps_speed,
        'file_size_mb': file_size,
        'output_path': mp4_path
    }


# ============================================================================
# 主测试函数
# ============================================================================
def main():
    results = []

    # 输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(output_dir, "benchmark_output.h264")

    # ---------------------------
    # 生成测试数据 (统一使用 GPU tensor)
    # ---------------------------
    print("\n" + "-" * 60)
    print("  步骤 1/4: 生成测试数据 (统一 GPU tensor)")
    print("-" * 60)

    # 统一生成 GPU 帧数据，所有方案共用
    frames_gpu = generate_random_frames_gpu(NUM_FRAMES, WIDTH, HEIGHT)
    time.sleep(10)
    # ---------------------------
    # 方案 1: cv2.VideoWriter
    # ---------------------------
    print("\n" + "-" * 60)
    print("  步骤 2/4: 测试 cv2.VideoWriter")
    print("-" * 60)
    try:
        result1 = test_cv2_videowriter(frames_gpu, output_base)
        results.append(result1)
    except Exception as e:
        print(f"  [错误] cv2.VideoWriter 测试失败: {e}")
        results.append({'method': 'cv2.VideoWriter', 'elapsed': None, 'fps': None, 'error': str(e)})
    time.sleep(10)
    # ---------------------------
    # 方案 2: PyNvVideoCodec CPU
    # ---------------------------
    print("\n" + "-" * 60)
    print("  步骤 3/4: 测试 PyNvVideoCodec CPU Buffer 模式")
    print("-" * 60)
    try:
        result2 = test_pynvcodec_cpu(frames_gpu, output_base)
        results.append(result2)
    except Exception as e:
        print(f"  [错误] PyNvVideoCodec CPU 测试失败: {e}")
        results.append({'method': 'PyNvVideoCodec CPU', 'elapsed': None, 'fps': None, 'error': str(e)})
    time.sleep(10)
    # ---------------------------
    # 方案 3: PyNvVideoCodec GPU
    # ---------------------------
    print("\n" + "-" * 60)
    print("  步骤 4/4: 测试 PyNvVideoCodec GPU Buffer 模式")
    print("-" * 60)
    try:
        result3 = test_pynvcodec_gpu(frames_gpu, output_base)
        results.append(result3)
    except Exception as e:
        print(f"  [错误] PyNvVideoCodec GPU 测试失败: {e}")
        results.append({'method': 'PyNvVideoCodec GPU', 'elapsed': None, 'fps': None, 'error': str(e)})

    # ---------------------------
    # 打印汇总结果
    # ---------------------------
    print("\n")
    print("=" * 60)
    print("   性能测试结果汇总")
    print("=" * 60)
    print(f"  测试参数: {WIDTH}x{HEIGHT}, {FPS}fps, {DURATION}秒, 共 {NUM_FRAMES} 帧")
    print("-" * 60)
    print(f"  {'编码方式':<25} {'耗时(秒)':<12} {'速度(fps)':<12} {'文件大小(MB)':<12}")
    print("-" * 60)

    baseline_fps = None
    for r in results:
        if r.get('elapsed') is not None:
            if baseline_fps is None:
                baseline_fps = r['fps']
            speedup = r['fps'] / baseline_fps if baseline_fps else 1.0
            print(f"  {r['method']:<25} {r['elapsed']:<12.2f} {r['fps']:<12.1f} {r['file_size_mb']:<12.2f}  ({speedup:.2f}x)")
        else:
            print(f"  {r['method']:<25} {'失败':<12} {'-':<12} {'-':<12}")

    print("=" * 60)

    # 计算加速比
    if len(results) >= 3 and results[0].get('fps') and results[2].get('fps'):
        speedup = results[2]['fps'] / results[0]['fps']
        print(f"\n  GPU Buffer 模式 vs cv2.VideoWriter: {speedup:.2f}x 加速")

    if len(results) >= 3 and results[1].get('fps') and results[2].get('fps'):
        speedup = results[2]['fps'] / results[1]['fps']
        print(f"  GPU Buffer 模式 vs CPU Buffer 模式: {speedup:.2f}x 加速")

    print("\n  测试完成!")


if __name__ == "__main__":
    main()