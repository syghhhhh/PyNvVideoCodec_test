#!/usr/bin/env python3
"""
PyNvVideoCodec 编码器测试脚本
生成纯色渐变视频: 红(Red) → 黄(Yellow) → 蓝(Blue)
分辨率: 1920×1080 | 帧率: 25fps | 总帧数: 100
支持 CPU 模式 和 GPU 模式，输出 MP4 文件

用法:
    python test_encode.py --mode cpu    # 仅测试 CPU 模式
    python test_encode.py --mode gpu    # 仅测试 GPU 模式
    python test_encode.py --mode both   # 两种都测试（默认）

依赖:
    pip install PyNvVideoCodec numpy
    pip install cupy-cuda12x          # GPU 模式需要
    系统需安装 ffmpeg（用于封装 MP4）
"""

import numpy as np
import PyNvVideoCodec as nvc
import subprocess
import time
import argparse
import os

# ======================== 全局参数 ========================
WIDTH      = 1920
HEIGHT     = 1080
NUM_FRAMES = 1000
FPS        = 25
CODEC      = "h264"
GPU_ID     = 0


# ======================== 颜色与格式工具 ========================

def get_frame_rgb(frame_idx, total_frames):
    """
    根据帧序号计算 RGB 颜色值
      前半段 (0  ~ 49):  红(255,0,0)   → 黄(255,255,0)
      后半段 (50 ~ 99):  黄(255,255,0) → 蓝(0,0,255)
    """
    half = total_frames // 2
    if frame_idx < half:
        t = frame_idx / max(half - 1, 1)
        r, g, b = 255, int(255 * t), 0
    else:
        t = (frame_idx - half) / max(total_frames - half - 1, 1)
        r = int(255 * (1 - t))
        g = int(255 * (1 - t))
        b = int(255 * t)
    return r, g, b


def rgb_to_yuv(r, g, b):
    """RGB → YUV (BT.601 标准)"""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
    return (
        int(np.clip(round(y), 0, 255)),
        int(np.clip(round(u), 0, 255)),
        int(np.clip(round(v), 0, 255)),
    )


def make_nv12_cpu(y_val, u_val, v_val, width, height):
    """
    在 CPU 上生成一帧纯色 NV12 数据 (numpy uint8 数组)
    NV12 内存布局:
      - Y  平面: height × width         (亮度)
      - UV 平面: (height/2) × width      (色度, U/V 交错排列)
    """
    y_plane = np.full(height * width, y_val, dtype=np.uint8)

    uv_size = (height // 2) * width
    uv_plane = np.empty(uv_size, dtype=np.uint8)
    uv_plane[0::2] = u_val   # U 分量
    uv_plane[1::2] = v_val   # V 分量

    return np.concatenate([y_plane, uv_plane])


# ======================== GPU 帧封装类 ========================

class GpuNV12Frame:
    """
    GPU 上的 NV12 帧。
    关键: Y 和 UV 必须位于同一块连续 GPU 内存中,
    因为编码器内部通过 Y 指针 + pitch*height 偏移来定位 UV 数据。
    """

    def __init__(self, y_val, u_val, v_val, width, height):
        import cupy as cp

        y_size  = height * width                  # 1920 * 1080 = 2073600
        uv_size = (height // 2) * width           # 540  * 1920 = 1036800

        # ★ 分配一整块连续的 GPU 显存
        self._buf = cp.empty(y_size + uv_size, dtype=cp.uint8)

        # 填充 Y 平面
        self._buf[:y_size] = y_val

        # 填充 UV 平面 (U/V 交错)
        uv = self._buf[y_size:]
        uv[0::2] = u_val
        uv[1::2] = v_val

        # 创建视图 (view)，不拷贝数据，指针仍指向同一块内存
        self.y_plane  = self._buf[:y_size].reshape(height, width, 1)
        self.uv_plane = self._buf[y_size:].reshape(height // 2, width // 2, 2)

    def cuda(self):
        return [self.y_plane, self.uv_plane]


# ======================== 封装 MP4 ========================

def wrap_to_mp4(h264_path, mp4_path, fps):
    """
    调用 ffmpeg 将裸 H.264 码流封装为 MP4 容器
    (仅做封装, 不重新编码, 速度极快)
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", h264_path,
        "-c:v", "copy",
        "-movflags", "+faststart",
        mp4_path,
    ]
    print(f"  [ffmpeg] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ffmpeg 错误] {result.stderr}")
        raise RuntimeError("ffmpeg 封装失败")
    # 封装成功后删除中间的 .h264 文件
    os.remove(h264_path)
    file_size = os.path.getsize(mp4_path) / 1024
    print(f"  ✅ MP4 已保存: {mp4_path}  ({file_size:.1f} KB)")


# ======================== CPU 模式编码 ========================

def encode_cpu(output_prefix="output_cpu"):
    """使用 CPU 缓冲模式进行编码"""
    print()
    print("=" * 60)
    print("   CPU 缓冲模式编码测试")
    print("=" * 60)

    h264_path = f"{output_prefix}.h264"
    mp4_path  = f"{output_prefix}.mp4"

    # 创建编码器 —— usecpuinputbuffer=True 表示输入帧在 CPU 内存
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=True,
        gpu_id=GPU_ID,
        codec=CODEC,
    )
    print(f"  编码器已创建: {WIDTH}x{HEIGHT}, NV12, {CODEC}, CPU 输入模式")

    encoded_bytes = 0
    t_start = time.time()

    with open(h264_path, "wb") as f:
        for i in range(NUM_FRAMES):
            # 1. 计算当前帧颜色
            r, g, b = get_frame_rgb(i, NUM_FRAMES)
            y, u, v = rgb_to_yuv(r, g, b)

            # 2. 生成 NV12 帧数据 (numpy array)
            frame = make_nv12_cpu(y, u, v, WIDTH, HEIGHT)

            # 3. 编码
            bitstream = nvenc.Encode(frame)
            if bitstream:
                data = bytearray(bitstream)
                f.write(data)
                encoded_bytes += len(data)

            # 每 25 帧打印一次进度
            if (i + 1) % 25 == 0:
                print(f"  帧 {i+1:3d}/{NUM_FRAMES}  "
                      f"RGB=({r:3d},{g:3d},{b:3d})  "
                      f"YUV=({y:3d},{u:3d},{v:3d})")

        # 4. 刷新编码器缓冲区 —— 必须调用, 否则丢失尾部帧
        bitstream = nvenc.EndEncode()
        if bitstream:
            data = bytearray(bitstream)
            f.write(data)
            encoded_bytes += len(data)

    elapsed = time.time() - t_start
    print(f"  编码完成: 耗时 {elapsed:.3f}s, "
          f"速度 {NUM_FRAMES / elapsed:.1f} fps, "
          f"码流大小 {encoded_bytes / 1024:.1f} KB")

    # 5. 封装为 MP4
    wrap_to_mp4(h264_path, mp4_path, FPS)
    return mp4_path


# ======================== GPU 模式编码 ========================

def encode_gpu(output_prefix="output_gpu"):
    """使用 GPU 缓冲模式进行编码"""
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        print("\n  ⚠️  GPU 模式需要 CuPy, 请安装: pip install cupy-cuda12x")
        print("  跳过 GPU 模式测试。")
        return None

    print()
    print("=" * 60)
    print("   GPU 缓冲模式编码测试")
    print("=" * 60)

    h264_path = f"{output_prefix}.h264"
    mp4_path  = f"{output_prefix}.mp4"

    # 创建编码器 —— usecpuinputbuffer=False 表示输入帧在 GPU 显存
    nvenc = nvc.CreateEncoder(
        width=WIDTH,
        height=HEIGHT,
        fmt="NV12",
        usecpuinputbuffer=False,
        gpu_id=GPU_ID,
        codec=CODEC,
    )
    print(f"  编码器已创建: {WIDTH}x{HEIGHT}, NV12, {CODEC}, GPU 输入模式")

    encoded_bytes = 0
    t_start = time.time()

    with open(h264_path, "wb") as f:
        for i in range(NUM_FRAMES):
            # 1. 计算当前帧颜色
            r, g, b = get_frame_rgb(i, NUM_FRAMES)
            y, u, v = rgb_to_yuv(r, g, b)

            # 2. 在 GPU 上生成 NV12 帧 (通过 GpuNV12Frame 封装)
            gpu_frame = GpuNV12Frame(y, u, v, WIDTH, HEIGHT)

            # 3. 编码 —— 传入实现了 cuda() 接口的 GPU 帧对象
            bitstream = nvenc.Encode(gpu_frame)
            if bitstream:
                data = bytearray(bitstream)
                f.write(data)
                encoded_bytes += len(data)

            if (i + 1) % 25 == 0:
                print(f"  帧 {i+1:3d}/{NUM_FRAMES}  "
                      f"RGB=({r:3d},{g:3d},{b:3d})  "
                      f"YUV=({y:3d},{u:3d},{v:3d})")

        # 4. 刷新编码器
        bitstream = nvenc.EndEncode()
        if bitstream:
            data = bytearray(bitstream)
            f.write(data)
            encoded_bytes += len(data)

    elapsed = time.time() - t_start
    print(f"  编码完成: 耗时 {elapsed:.3f}s, "
          f"速度 {NUM_FRAMES / elapsed:.1f} fps, "
          f"码流大小 {encoded_bytes / 1024:.1f} KB")

    # 5. 封装为 MP4
    wrap_to_mp4(h264_path, mp4_path, FPS)
    return mp4_path


# ======================== 主入口 ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyNvVideoCodec 编码测试 — 纯色渐变视频 (红→黄→蓝)"
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="编码模式: cpu / gpu / both (默认 both)",
    )
    args = parser.parse_args()

    print(f"📹 视频参数: {WIDTH}×{HEIGHT}, {FPS}fps, {NUM_FRAMES}帧, {CODEC}")
    print(f"🎨 颜色渐变: 红(255,0,0) → 黄(255,255,0) → 蓝(0,0,255)")

    results = {}

    if args.mode in ("cpu", "both"):
        results["cpu"] = encode_cpu("gradient_cpu")

    if args.mode in ("gpu", "both"):
        results["gpu"] = encode_gpu("gradient_gpu")

    # 汇总
    print()
    print("=" * 60)
    print("   测试结果汇总")
    print("=" * 60)
    for mode, path in results.items():
        if path:
            size_kb = os.path.getsize(path) / 1024
            print(f"  [{mode.upper()}] {path}  ({size_kb:.1f} KB)")
        else:
            print(f"  [{mode.upper()}] 跳过")
    print()
    print("全部完成! 🎉")