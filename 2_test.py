import os
from pathlib import Path
import re
import cv2
import numpy as np

# -----------------------------
# 配置
# -----------------------------
OUTPUT_DIR = Path("compare_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PNG = Path(r"H:\PyNvVideoCodec_test\color_test_output\test_frame.png")

VIDEO_PATHS = {
    "CV2 (libx264 chain)": Path(r"H:\PyNvVideoCodec_test\color_test_output\cv2.mp4"),
    "PyNvVideoCodec cpu":  Path(r"H:\PyNvVideoCodec_test\color_test_output\pynvc_cpu.mp4"),
    "PyNvVideoCodec gpu":  Path(r"H:\PyNvVideoCodec_test\color_test_output\pynvc_gpu.mp4"),
}

FRAME_INDEX = 0
ALLOW_RESIZE = False   # 建议 False：尺寸不一致就直接报错，避免 resize 掩盖问题
HEATMAP_CLIP = (1, 99) # 按百分位拉伸，避免少量异常点影响整幅图


# -----------------------------
# 工具函数
# -----------------------------
def slug(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip(), flags=re.UNICODE)
    return s[:120] if len(s) > 120 else s

def read_image_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

def read_video_frame_bgr(video_path: Path, idx: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    # 尽量定位到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def ensure_same_shape(ref, frm, allow_resize=False):
    if ref.shape == frm.shape:
        return frm
    if not allow_resize:
        raise ValueError(f"shape mismatch: ref={ref.shape}, frame={frm.shape}")
    return cv2.resize(frm, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)

def psnr_u8(a_u8, b_u8):
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))

def delta_e76_opencv_float_lab(bgr1_u8, bgr2_u8):
    """
    用 float(0..1) 输入 OpenCV 的 BGR2Lab，得到更接近标准 Lab 范围：
    L: 0..100, a/b: 约 -127..127
    然后做 CIE76 (Euclidean)。
    """
    f1 = bgr1_u8.astype(np.float32) / 255.0
    f2 = bgr2_u8.astype(np.float32) / 255.0
    lab1 = cv2.cvtColor(f1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(f2, cv2.COLOR_BGR2LAB)
    de = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    return float(np.mean(de))

def compute_metrics(ref_u8, frm_u8):
    ref = ref_u8.astype(np.float32)
    frm = frm_u8.astype(np.float32)
    diff = frm - ref
    adiff = np.abs(diff)

    # BGR MAE
    mae_bgr = adiff.reshape(-1, 3).mean(axis=0)  # [B,G,R]
    mae = float(adiff.mean())
    max_abs = float(adiff.max())
    p95 = float(np.percentile(adiff, 95))
    p99 = float(np.percentile(adiff, 99))

    # PSNR
    psnr = psnr_u8(ref_u8, frm_u8)

    # DeltaE (Lab, CIE76)
    de76 = delta_e76_opencv_float_lab(ref_u8, frm_u8)

    # 额外：看 Y/Cr/Cb（更好定位“色度偏差 vs 亮度偏差”）
    ycc_ref = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc_frm = cv2.cvtColor(frm_u8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc_mae = np.mean(np.abs(ycc_frm - ycc_ref), axis=(0, 1))  # [Y,Cr,Cb]

    return {
        "mae_b": float(mae_bgr[0]),
        "mae_g": float(mae_bgr[1]),
        "mae_r": float(mae_bgr[2]),
        "mae_all": mae,
        "p95": p95,
        "p99": p99,
        "max": max_abs,
        "psnr": psnr,
        "deltae76": de76,
        "mae_y": float(ycc_mae[0]),
        "mae_cr": float(ycc_mae[1]),
        "mae_cb": float(ycc_mae[2]),
    }

def save_heatmap(ref_u8, frm_u8, out_path: Path, clip=HEATMAP_CLIP):
    # 绝对差（灰度）
    adiff = np.abs(frm_u8.astype(np.int16) - ref_u8.astype(np.int16)).astype(np.uint8)
    diff_gray = np.mean(adiff.astype(np.float32), axis=2)

    lo, hi = np.percentile(diff_gray, clip)
    if hi <= lo + 1e-6:
        norm = np.zeros_like(diff_gray, dtype=np.uint8)
    else:
        norm = np.clip((diff_gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), heatmap)

def save_absdiff_gray(ref_u8, frm_u8, out_path: Path, clip=HEATMAP_CLIP):
    adiff = np.abs(frm_u8.astype(np.int16) - ref_u8.astype(np.int16)).astype(np.uint8)
    diff_gray = np.mean(adiff.astype(np.float32), axis=2)

    lo, hi = np.percentile(diff_gray, clip)
    if hi <= lo + 1e-6:
        norm = np.zeros_like(diff_gray, dtype=np.uint8)
    else:
        norm = np.clip((diff_gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    cv2.imwrite(str(out_path), norm)

def save_side_by_side(images, titles, out_path: Path, title_h=48):
    assert len(images) == len(titles)
    h, w = images[0].shape[:2]
    canvas = np.zeros((h + title_h, w * len(images), 3), dtype=np.uint8)

    for i, (img, t) in enumerate(zip(images, titles)):
        canvas[title_h:title_h + h, i*w:(i+1)*w] = img
        cv2.putText(canvas, t, (i*w + 12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


# -----------------------------
# 主流程
# -----------------------------
def main():
    ref = read_image_bgr(REFERENCE_PNG)
    if ref is None:
        raise FileNotFoundError(f"cannot read reference png: {REFERENCE_PNG}")

    # 保存一份 reference 拷贝
    cv2.imwrite(str(OUTPUT_DIR / "reference_test_frame.png"), ref)

    frames = {}
    for name, vpath in VIDEO_PATHS.items():
        frame = read_video_frame_bgr(vpath, FRAME_INDEX)
        if frame is None:
            print(f"[FAIL] read frame {FRAME_INDEX}: {name} -> {vpath}")
            continue

        frame = ensure_same_shape(ref, frame, allow_resize=ALLOW_RESIZE)
        frames[name] = frame

        # 落盘保存提取帧
        out_png = OUTPUT_DIR / f"frame_{FRAME_INDEX:04d}_{slug(name)}.png"
        cv2.imwrite(str(out_png), frame)
        print(f"[OK] {name} frame saved: {out_png}")

    if not frames:
        print("No frames loaded. Exit.")
        return

    # 对每个视频：和 reference 比较
    results = {}
    for name, frm in frames.items():
        m = compute_metrics(ref, frm)
        results[name] = m

        # 可视化输出
        save_heatmap(ref, frm, OUTPUT_DIR / f"heatmap_{slug(name)}.png")
        save_absdiff_gray(ref, frm, OUTPUT_DIR / f"absdiff_{slug(name)}.png")

        # 三联图：ref / current / heatmap
        heat = cv2.imread(str(OUTPUT_DIR / f"heatmap_{slug(name)}.png"))
        save_side_by_side(
            [ref, frm, heat],
            ["reference", name, "diff heatmap"],
            OUTPUT_DIR / f"triplet_{slug(name)}.png"
        )

    # 汇总打印
    print("\n" + "=" * 90)
    print("Metrics vs reference (test_frame.png)")
    print("=" * 90)
    header = f"{'name':<26} {'MAE(B,G,R)':<20} {'MAE':>6} {'P95':>6} {'P99':>6} {'Max':>6} {'PSNR':>8} {'dE76':>7} {'Y/Cr/Cb MAE':>18}"
    print(header)
    print("-" * 90)

    def fmt3(a, b, c): return f"{a:4.2f},{b:4.2f},{c:4.2f}"

    for name, m in results.items():
        print(
            f"{name:<26} "
            f"{fmt3(m['mae_b'], m['mae_g'], m['mae_r']):<20} "
            f"{m['mae_all']:6.2f} {m['p95']:6.2f} {m['p99']:6.2f} {m['max']:6.2f} "
            f"{m['psnr']:8.2f} {m['deltae76']:7.2f} "
            f"{fmt3(m['mae_y'], m['mae_cr'], m['mae_cb']):>18}"
        )

    # 找最接近 reference 的（按 DeltaE）
    best = min(results.items(), key=lambda kv: kv[1]["deltae76"])
    print("\nBest (min DeltaE76):")
    print(f"  {best[0]}")
    print(f"  DeltaE76={best[1]['deltae76']:.3f}, PSNR={best[1]['psnr']:.2f} dB, MAE={best[1]['mae_all']:.3f}")

    print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()