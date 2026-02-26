# PyNvVideoCodec_test

优化合成视频时候最后写入视频文件的速度,修改原工程文件 utils\video_merge.py 中从 cv2.VideoWriter 改成 PyNvVideoCodec.CreateEncoder