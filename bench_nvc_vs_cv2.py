import os
import time
import numpy as np
import torch
import cv2
import PyNvVideoCodec as nvc
import pycuda.driver as cuda
from pycuda.autoinit import context  # 初始化CUDA上下文

# ===================== 全局配置参数 =====================
GPU_ID = 0  # 使用的GPU编号
W, H = 1440, 2560  # 分辨率：宽1440，高2560
FPS = 25  # 帧率
TOTAL_FRAMES = 100  # 测试总帧数
CODEC = "h264"  # 编码格式
BITRATE = 20 * 1024 * 1024  # 码率20Mbps

# 完整输出路径：脚本同目录下的test_output文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绝对路径
OUTPUT_PYNVC = os.path.join(OUTPUT_DIR, "pynvc_hw_encoder.mp4")
OUTPUT_CV2 = os.path.join(OUTPUT_DIR, "cv2_cpu_encoder.mp4")
OUTPUT_COMPARE = os.path.join(OUTPUT_DIR, "encode_compare.mp4")

# 素材路径
BG_IMG_PATH = os.path.join(SCRIPT_DIR, "bg.png")
WOMEN_IMG_PATH = os.path.join(SCRIPT_DIR, "women.png")

# ===================== 工具函数：加载并预处理素材（GPU显存） =====================
def load_and_preprocess_materials() -> tuple[torch.Tensor, torch.Tensor]:
    """
    加载背景图和人物图，预处理为GPU显存中的RGBA张量：
    - 背景图：bg.png (1440×2560) → RGBA (4, H, W)，CUDA uint8
    - 人物图：women.png (1080×1920) → 拉伸至1440×2560 → RGBA (4, H, W)，CUDA uint8
    """
    if not os.path.exists(BG_IMG_PATH):
        raise FileNotFoundError(f"背景图不存在：{BG_IMG_PATH}（请确保bg.png在脚本同目录）")
    if not os.path.exists(WOMEN_IMG_PATH):
        raise FileNotFoundError(f"人物图不存在：{WOMEN_IMG_PATH}（请确保women.png在脚本同目录）")
    
    # 加载背景图（BGRA → RGBA）
    bg_img = cv2.imread(BG_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if bg_img.shape[:2] != (H, W):
        raise ValueError(f"背景图分辨率错误，要求{W}×{H}，实际{bg_img.shape[1]}×{bg_img.shape[0]}")
    if bg_img.ndim == 3 and bg_img.shape[-1] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
    bg_rgba = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2RGBA)
    bg_tensor = torch.from_numpy(bg_rgba).permute(2, 0, 1).to(torch.uint8).to(f"cuda:{GPU_ID}")
    
    # 加载人物图（拉伸+BGRA → RGBA）
    women_img = cv2.imread(WOMEN_IMG_PATH, cv2.IMREAD_UNCHANGED)
    women_img_resized = cv2.resize(women_img, (W, H), interpolation=cv2.INTER_LINEAR)
    if women_img_resized.ndim == 3 and women_img_resized.shape[-1] == 3:
        women_img_resized = cv2.cvtColor(women_img_resized, cv2.COLOR_BGR2BGRA)
    women_rgba = cv2.cvtColor(women_img_resized, cv2.COLOR_BGRA2RGBA)
    women_tensor = torch.from_numpy(women_rgba).permute(2, 0, 1).to(torch.uint8).to(f"cuda:{GPU_ID}")
    
    return bg_tensor, women_tensor

# ===================== 工具函数：生成测试帧（GPU显存，背景+人物叠加） =====================
def generate_test_frame(bg_tensor: torch.Tensor, women_tensor: torch.Tensor) -> torch.Tensor:
    """
    生成GPU显存中的测试帧：背景图 + 拉伸填满的人物图（RGBA叠加）
    :return: (1, 4, H, W) 的RGBA CUDA uint8张量
    """
    # 人物图Alpha通道归一化（0-255 → 0-1）
    alpha = women_tensor[3:4, :, :].to(torch.float32) / 255.0
    # 背景图和人物图叠加（Alpha混合）
    frame = torch.zeros((1, 4, H, W), dtype=torch.uint8, device=f"cuda:{GPU_ID}")
    # RGB通道混合：人物*alpha + 背景*(1-alpha)
    frame[:, 0:3, :, :] = (women_tensor[0:3, :, :].to(torch.float32) * alpha + bg_tensor[0:3, :, :].to(torch.float32) * (1 - alpha)).to(torch.uint8)
    # Alpha通道：取不透明（255）
    frame[:, 3, :, :] = 255
    
    return frame

# ===================== 工具函数：GPU端RGBA转ABGR（修正通道顺序，解决偏红） =====================
def rgba_to_abgr_contiguous_hwc_gpu(rgba_tensor: torch.Tensor) -> torch.Tensor:
    """
    修正通道重排逻辑：
    NV官方ABGR格式定义：[B, G, R, A]（H,W,C）
    转换逻辑：RGBA (C,H,W) → ABGR (H,W,C) 连续内存张量
    """
    # 1. RGBA (C=4,H,W) → BGR A (C=4,H,W) （先调整通道为BGR+A）
    # RGBA: [R(0), G(1), B(2), A(3)] → ABGR(NV定义): [B(2), G(1), R(0), A(3)]
    abgr_chw = torch.empty_like(rgba_tensor)
    abgr_chw[0, :, :] = rgba_tensor[2, :, :]  # B → 第0通道（核心修正：之前错放成A）
    abgr_chw[1, :, :] = rgba_tensor[1, :, :]  # G → 第1通道
    abgr_chw[2, :, :] = rgba_tensor[0, :, :]  # R → 第2通道（核心修正：之前错放成最后）
    abgr_chw[3, :, :] = rgba_tensor[3, :, :]  # A → 第3通道
    
    # 2. 维度转置：(C,H,W) → (H,W,C) + 连续内存（修正步长）
    abgr_hwc_contiguous = abgr_chw.permute(1, 2, 0).contiguous()
    
    # 调试：打印通道值（可选，验证转换是否正确）
    # print(f"转换前R通道均值：{rgba_tensor[0, :, :].mean().item()}")
    # print(f"转换后R通道位置均值：{abgr_chw[2, :, :].mean().item()}")
    
    return abgr_hwc_contiguous

# ===================== 类：PyNvVideoCodec硬件编码器（修正fmt参数+色差） =====================
class PyNvVideoCodecHWEncoder:
    def __init__(self, gpu_id: int, w: int, h: int, fps: int, bitrate: int, codec: str = "h264"):
        self.gpu_id = gpu_id
        self.w = w
        self.h = h
        self.fps = fps
        self.bitrate = bitrate
        self.codec = codec
        self.encoder = None
        self._init_encoder()

    def _init_encoder(self):
        """初始化编码器：修正参数名format→fmt + 正确ABGR格式"""
        config_params = {
            "gpu_id": self.gpu_id,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "fps": self.fps,
            "preset": "P1",  # 最快编码预设
            "tuning_info": "low_latency",
            "rc": "cbr",  # 恒定码率
            "idrperiod": 30,  # 每30帧一个I帧
        }
        # 核心：使用NV官方定义的ABGR格式（fmt参数）
        self.encoder = nvc.CreateEncoder(
            width=self.w,
            height=self.h,
            fmt="ABGR",  # 修正：参数名是fmt，格式是NV官方ABGR
            usecpuinputbuffer=False,  # GPU输入
            **config_params
        )
        self.fp = open(OUTPUT_PYNVC, "wb")

    def encode_frame(self, torch_rgba: torch.Tensor):
        """编码单帧GPU RGBA数据（修正色差）"""
        # 去掉batch维度，得到 (4, H, W) 的RGBA张量
        rgba_chw = torch_rgba.squeeze(0).to(f"cuda:{self.gpu_id}")
        
        # 关键：转换为NV官方ABGR格式（H,W,C）+ 连续内存
        abgr_hwc_contiguous = rgba_to_abgr_contiguous_hwc_gpu(rgba_chw)
        
        # 编码
        bitstream = self.encoder.Encode(abgr_hwc_contiguous)
        if bitstream:
            self.fp.write(bytearray(bitstream))

    def release(self):
        """释放编码器"""
        bitstream = self.encoder.EndEncode()
        if bitstream:
            self.fp.write(bytearray(bitstream))
        self.fp.close()
        print(f"[PyNvVideoCodec] 硬件编码视频已保存：{OUTPUT_PYNVC}")

# ===================== 类：OpenCV CPU编码器（基准对比） =====================
class OpenCVCpuEncoder:
    def __init__(self, w: int, h: int, fps: int, codec: str = "mp4v"):
        self.w = w
        self.h = h
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            OUTPUT_CV2,
            self.fourcc,
            self.fps,
            (w, h),
            isColor=True
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV编码器初始化失败，输出路径：{OUTPUT_CV2}")

    def encode_frame(self, torch_rgba: torch.Tensor):
        """编码单帧GPU RGBA数据"""
        # GPU→CPU，转置为(H,W,C)，RGBA→BGR
        frame_cpu = torch_rgba.squeeze(0).permute(1,2,0).cpu().numpy()  # (H,W,4) RGBA
        frame_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGBA2BGR)        # OpenCV默认BGR
        self.writer.write(frame_bgr)

    def release(self):
        """释放编码器"""
        self.writer.release()
        print(f"[OpenCV] CPU编码视频已保存：{OUTPUT_CV2}")

# ===================== 性能测试主函数 =====================
def run_encoder_benchmark():
    # 1. 加载素材（预热GPU）
    print("===== 加载测试素材 =====")
    bg_tensor, women_tensor = load_and_preprocess_materials()
    torch.cuda.synchronize(GPU_ID)
    print(f"素材加载完成：背景图{BG_IMG_PATH}，人物图{WOMEN_IMG_PATH}")

    # 2. PyNvVideoCodec硬件编码测试
    print(f"\n[1/2] 启动PyNvVideoCodec {CODEC} 硬件编码（GPU {GPU_ID}）...")
    pynvc_encoder = PyNvVideoCodecHWEncoder(GPU_ID, W, H, FPS, BITRATE, CODEC)
    start_time = time.time()
    for idx in range(TOTAL_FRAMES):
        frame = generate_test_frame(bg_tensor, women_tensor)
        pynvc_encoder.encode_frame(frame)
        if (idx + 1) % 20 == 0:
            print(f"  已编码 {idx+1}/{TOTAL_FRAMES} 帧")
    pynvc_encoder.release()
    torch.cuda.synchronize(GPU_ID)
    pynvc_time = time.time() - start_time
    pynvc_fps = TOTAL_FRAMES / pynvc_time

    # 3. OpenCV CPU编码测试
    print(f"\n[2/2] 启动OpenCV H264 CPU编码...")
    cv2_encoder = OpenCVCpuEncoder(W, H, FPS)
    start_time = time.time()
    for idx in range(TOTAL_FRAMES):
        frame = generate_test_frame(bg_tensor, women_tensor)
        cv2_encoder.encode_frame(frame)
        if (idx + 1) % 20 == 0:
            print(f"  已编码 {idx+1}/{TOTAL_FRAMES} 帧")
    cv2_encoder.release()
    cv2_time = time.time() - start_time
    cv2_fps = TOTAL_FRAMES / cv2_time

    # 4. 打印性能结果
    print("\n===== 编码性能测试结果 =====")
    print(f"总帧数：{TOTAL_FRAMES} | 分辨率：{W}×{H} | 帧率：{FPS} FPS")
    print(f"PyNvVideoCodec硬件编码：耗时 {pynvc_time:.2f}s | FPS {pynvc_fps:.2f}")
    print(f"OpenCV CPU编码：耗时 {cv2_time:.2f}s | FPS {cv2_fps:.2f}")
    print(f"硬件编码提速：{(cv2_time/pynvc_time - 1)*100:.1f}%")

    # 5. FFmpeg拼接对比视频
    print(f"\n===== 拼接对比视频 =====")
    ffmpeg_cmd = (
        f"ffmpeg -y -i {OUTPUT_PYNVC} -i {OUTPUT_CV2} "
        f"-filter_complex '[0:v]scale={W}:{H}[left];[1:v]scale={W}:{H}[right];[left][right]hstack=2' "
        f"-c:v h264 -b:v {BITRATE} -r {FPS} {OUTPUT_COMPARE}"
    )
    print(f"执行FFmpeg命令：{ffmpeg_cmd}")
    os.system(ffmpeg_cmd)
    if os.path.exists(OUTPUT_COMPARE):
        print(f"对比视频已保存：{OUTPUT_COMPARE}（左=硬件编码，右=CPU编码）")
    else:
        print("FFmpeg拼接失败，请检查FFmpeg是否安装")

# ===================== 执行测试 =====================
if __name__ == "__main__":
    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU")
    # 检查PyNvVideoCodec
    try:
        nvc.CreateDemuxer
    except Exception as e:
        raise RuntimeError(f"PyNvVideoCodec初始化失败：{e}")
    # 运行测试
    run_encoder_benchmark()