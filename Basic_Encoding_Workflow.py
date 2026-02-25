# Raw Frames → Buffer Preparation → Encoder → Encoded Bitstream
import numpy as np
import PyNvVideoCodec as nvc

width = 1920
height = 1080
num_frames = 100
# ==========================Step 1: Prepare Buffer for Encoding==========================
# Prepare input buffers based on your buffer mode. For CPU buffers, read raw YUV data into a numpy array. For GPU buffers, use CUDA device memory objects.

# CPU Buffer Mode
# Calculate frame size based on format (NV12 = height * 1.5)
frame_size = int(width * height * 1.5)
# Read raw YUV frame into numpy array
with open("input.yuv", "rb") as dec_file:
    chunk = np.fromfile(dec_file, np.uint8, count=frame_size)

# GPU Buffer Mode
# For GPU buffers, use objects implementing CUDA Array Interface
# The object must expose a cuda() method returning device pointers
class AppFrame:
    def __init__(self, width, height, fmt):
        self.frameSize = int(width * height * 1.5)  # NV12
        # Allocate CUDA device memory
        
    def cuda(self):
        # Return CUDA Array Interface for each plane
        return [self.luma_cuda_interface, self.chroma_cuda_interface]
input_frame = AppFrame(width, height, "NV12")


# ==========================Step 2: Configure and Create Encoder==========================
# Create an encoder with CreateEncoder() specifying resolution, format, buffer mode, and encoding parameters. See CreateEncoder API Reference for all available parameters.

# Encoder configuration parameters
config_params = {
    "gpu_id": 0,
    "codec": "h264",
    # Additional optional parameters (bitrate, preset, etc.)
}
# Create encoder: usecpuinputbuffer=True for CPU, False for GPU
nvenc = nvc.CreateEncoder(
    width=1920,
    height=1080,
    format="NV12",
    usecpuinputbuffer=True,  # True=CPU buffers, False=GPU buffers
    **config_params
)


# ==========================Step 3: Encode Frames and Flush==========================
# Pass frames to Encode() to get encoded bitstream. After processing all frames, call EndEncode() to flush remaining data from the encoder queue. See Encode API Reference EndEncode API Reference.

with open("output.h264", "wb") as enc_file:
    # Encode each frame
    for i in range(num_frames):
        chunk = np.fromfile(dec_file, np.uint8, count=frame_size)
        if chunk.size == 0:
            break
        # Encode frame - returns bitstream data
        bitstream = nvenc.Encode(chunk)
        enc_file.write(bytearray(bitstream))

    # Flush encoder queue - REQUIRED to get remaining frames
    bitstream = nvenc.EndEncode()
    enc_file.write(bytearray(bitstream))


# ========================Step 4: Runtime Reconfiguration (Optional)========================
# Change encoder parameters at runtime without recreating the encoder session using Reconfigure(). This is useful for adaptive bitrate streaming or handling network conditions. See Reconfigure API Reference for supported parameters.


# Get current encoder parameters
reconfig_params = nvenc.GetEncodeReconfigureParams()
# Modify parameters (e.g., change bitrate)
reconfig_params["averageBitrate"] = 5000000  # 5 Mbps
# Apply new configuration
nvenc.Reconfigure(reconfig_params)
