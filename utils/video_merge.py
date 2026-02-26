# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 18:00
# @Author  : 施昀谷
# @File    : video_merge.py
from utils.data_prepare import *
from utils.file_transfer import upload_oss
import subprocess


class GpuNV12FramePool:
    """预分配 GPU 内存池，用于 PyNvVideoCodec GPU Buffer 模式"""

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
        uv = self._buf[y_size:].reshape(height // 2, width)
        uv[:, 0::2] = v_cp[::2, ::2]  # V 在偶数位置
        uv[:, 1::2] = u_cp[::2, ::2]  # U 在奇数位置

        return self

    def cuda(self):
        return [self.y_plane, self.uv_plane]


def merge_video(item, id_merge, ret_dict, config_dict, log_file):
    """
    合成视频
    :param item: 任务信息
    :param id_merge: 合成任务的id
    :param ret_dict: {path_background, add_name_list, add_path_dict, subtitle_list, music_path, frame_num}
    :param config_dict: {background_type, subtitle, watermark, child, last_child, music, show_people}
    :param log_file: 日志文件
    :return:
    """
    log_content_write(log_file, 'start merge')
    merge_folder = join(parameters['workspace'], id_merge)
    nodes = item.nodes
    show_people = config_dict['show_people']
    width = int(item.width)
    height = int(item.height)
    frame_num = int(ret_dict['frame_num'])
    sceneId = item.sceneId
    path_background = ret_dict['path_background']
    inf_data_body_folder = join(parameters['inf_data_folder'], sceneId, 'body')
    human_frame_count = len(listdir(inf_data_body_folder))

    # 文字水印处理
    log_content_write(log_file, 'start watermark')
    watermark_wenzi = cv2.imread(parameters['watermark_wenzi_path'], cv2.IMREAD_UNCHANGED)
    watermark_wenzi_tensor = torch.tensor(watermark_wenzi, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255  # 转成tensor
    watermark_wenzi_h, watermark_wenzi_w = watermark_wenzi.shape[:2]  # 读取宽高
    # 字幕背景图片处理
    if config_dict['subtitle_style']:
        log_content_write(log_file, 'start subtitle bg picture')
        png_subtitle = cv2.imread(ret_dict['subtitle_bg_path'], cv2.IMREAD_UNCHANGED)
        # png_subtitle_w = width
        # png_subtitle_h = int(png_subtitle.shape[0] * width / png_subtitle.shape[1])
        png_subtitle_w = int(item.subtitle_style[0]['pos']['width'])
        png_subtitle_h = int(item.subtitle_style[0]['pos']['height'])
        tensor_subtitle_bg = get_frame_tensor(png_subtitle, png_subtitle_w, png_subtitle_h)
        # 计算贴图的坐标
        png_subtitle_x = int(item.subtitle_style[0]['pos']['left'])
        png_subtitle_y = int(item.subtitle_style[0]['pos']['top'])
        png_subtitle_add_coordinates_list, png_subtitle_bg_coordinates_list = get_correct_coordinates(png_subtitle_x, png_subtitle_y, png_subtitle_w, png_subtitle_h, width, height)

    # 置顶文字处理
    if config_dict['floatTexts']:
        log_content_write(log_file, 'start floatTexts')
        png_floatTexts = cv2.imread(ret_dict['floatTexts_path'], cv2.IMREAD_UNCHANGED)
        tensor_floatTexts = get_frame_tensor(png_floatTexts, width, height)
    # 全屏水印
    if config_dict['watermark']:
        log_content_write(log_file, 'start watermark')
        watermark_360 = cv2.imread(parameters['watermark_path'], cv2.IMREAD_UNCHANGED)
        w_360_num = width // watermark_360.shape[1] + 1
        h_360_num = height // watermark_360.shape[0] + 1
        # 转成tensor
        watermark_360_tensor = torch.tensor(watermark_360, dtype=torch.float32).cuda().permute(2, 0, 1).unsqueeze(0) / 255
        # tensor横向复制w_360_num次，纵向复制h_360_num次
        watermark_360_tensor = watermark_360_tensor.repeat(1, 1, h_360_num, 1).repeat(1, 1, 1, w_360_num)
        # 截取需要的部分
        watermark_360_tensor = watermark_360_tensor[:, :, :height, :width]

    # 添加的素材
    if len(nodes) > 0:
        log_content_write(log_file, 'start nodes')
        nodes = sorted(nodes, key=lambda x: int(x['level']))
    node_bottom_gen_list, node_top_gen_list = [], []
    for node in nodes:
        if int(node['level']) < 0:
            node_bottom_gen_list.append(proxy_one_add_gen(item, node, ret_dict, log_file))
        else:
            node_top_gen_list.append(proxy_one_add_gen(item, node, ret_dict, log_file))

    if config_dict['background_type'] == 'picture':
        background = cv2.imread(path_background)
        background = get_frame_tensor(background, width, height)
    else:
        background_cap = cv2.VideoCapture(path_background)
        # 获取视频的总帧数
        frame_count = int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_frame = 0
        background_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame)

    # 在cfg_json中添加当前子任务的end_bg_frame背景结束帧和end_human_frame人物结束帧
    if config_dict['child']:
        # 找到当前子任务的cfg_json文件
        cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
        assert exists(cfg_json)
        with open(cfg_json, 'r') as f:
            cfg_dict_list = json.load(f)
        dict_list = [x for x in cfg_dict_list if (x['childId'] == item.childId)]
        assert len(dict_list) == 1
        dict_this = dict_list[0]
        # 背景结束帧
        if config_dict['background_type'] == 'picture':
            dict_this['end_bg_frame'] = 0
        else:
            # 重新调整开始的背景帧数
            bg_frame = dict_this['start_bg_frame']
            bg_frame = bg_frame % frame_count
            background_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame)
            # 计算结束的背景帧数
            dict_this['end_bg_frame'] = (frame_num + bg_frame) % frame_count
        # 人物结束帧
        start_human_frame = dict_this['start_human_frame']
        dict_this['end_human_frame'] = (frame_num + start_human_frame) % (2 * human_frame_count)
        if not show_people:
            dict_this['end_human_frame'] = 0
        with open(cfg_json, 'w') as f:
            json.dump(cfg_dict_list, f)
    else:
        # 单个任务
        start_human_frame = 0

    # 开始合成 - 使用 PyNvVideoCodec GPU 编码
    import PyNvVideoCodec as nvc

    # 解析码率
    bitrate_str = str(item.bitRate) if hasattr(item, 'bitRate') else '16M'
    bitrate_value = bitrate_str.upper()
    if bitrate_value.endswith('M'):
        bitrate_num = int(float(bitrate_value[:-1]) * 1_000_000)
    elif bitrate_value.endswith('K'):
        bitrate_num = int(float(bitrate_value[:-1]) * 1_000)
    else:
        bitrate_num = int(bitrate_value)

    # 创建编码器
    nvenc = nvc.CreateEncoder(
        width=width,
        height=height,
        fmt="NV12",
        usecpuinputbuffer=False,
        gpu_id=0,
        codec="h264",
        fps=25,
        bitrate=bitrate_num,
        maxbitrate=int(bitrate_num * 1.5),
        preset="P1",
        tuning_info="high_quality",
        profile="high",
        rc="vbr",
        gop=50,
    )

    # 预分配 GPU 内存池
    frame_pool = GpuNV12FramePool(width, height)

    # 临时 H264 文件路径
    h264_path = join(merge_folder, "speaker_25fps_16k_merged.h264")
    mp4_path = join(merge_folder, "speaker_25fps_16k_merged.mp4")

    # 打开 H264 文件用于写入
    h264_file = open(h264_path, "wb")

    # 如果背景为图片,则只需处理一次滤镜
    bool_videoFilter = (len(item.videoFilter) > 0)
    if bool_videoFilter:
        if config_dict['background_type'] == 'picture':
            videoFilter = item.videoFilter[0]
            background = image_filter(background, videoFilter)
    # 如果show_people为False,则不显示人物

    if show_people:
        human_frame_gen = proxy_human_gen(item, merge_folder, start_human_frame, human_frame_count, log_file)
    # 逐帧合成
    log_title_write(log_file, 'start merge video frames')
    for i in range(frame_num):
        # ----------------------------------------------------开始处理背景----------------------------------------------------
        if config_dict['background_type'] == 'picture':
            frame = background.clone()
        else:
            # 视频背景,循环读取
            ret, frame = background_cap.read()
            if (not ret) or (frame is None):
                bg_frame = 0
                background_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame)
                ret, frame = background_cap.read()
            bg_frame += 1
            if bg_frame >= frame_count:
                bg_frame = 0
                background_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame)
            frame = get_frame_tensor(frame, width, height)
            if bool_videoFilter:
                videoFilter = item.videoFilter[0]
                frame = image_filter(frame, videoFilter)
        # log_content_write(log_file, f'end bg, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加底层素材----------------------------------------------------
        if len(node_bottom_gen_list) > 0:
            for node_bottom_gen in node_bottom_gen_list:
                add_frame_tensor, add_coordinates_list, bg_coordinates_list = next(node_bottom_gen)
                frame = merge_bg_add(frame, add_frame_tensor, add_coordinates_list, bg_coordinates_list)
        # log_content_write(log_file, f'end bottom, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加人物----------------------------------------------------
        if show_people:
            body_frame_tensor, human_add_coordinates_list, human_bg_coordinates_list = next(human_frame_gen)
            if bool_videoFilter:
                videoFilter = item.videoFilter[0]
                body_frame_tensor = image_filter(body_frame_tensor, videoFilter)
            # log_content_write(log_file, f'i:{i}, body_frame_tensor.shape = {body_frame_tensor.shape}, human_add_coordinates_list = {human_add_coordinates_list}, human_bg_coordinates_list = {human_bg_coordinates_list}')
            frame = merge_bg_add(frame, body_frame_tensor, human_add_coordinates_list, human_bg_coordinates_list)
            # log_content_write(log_file, f'end human, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加顶层素材----------------------------------------------------
        if len(node_top_gen_list) > 0:
            for node_top_gen in node_top_gen_list:
                add_frame_tensor, add_coordinates_list, bg_coordinates_list = next(node_top_gen)
                # log_content_write(log_file, f'add_frame_tensor.shape = {add_frame_tensor.shape}, add_coordinates_list = {add_coordinates_list}, bg_coordinates_list = {bg_coordinates_list}')
                frame = merge_bg_add(frame, add_frame_tensor, add_coordinates_list, bg_coordinates_list)
        # log_content_write(log_file, f'end top, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加字幕背景图片----------------------------------------------------
        if config_dict['subtitle_style']:
            frame = merge_bg_add(frame, tensor_subtitle_bg, png_subtitle_add_coordinates_list, png_subtitle_bg_coordinates_list)
        # log_content_write(log_file, f'end subtitle, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加置顶文字----------------------------------------------------
        if config_dict['floatTexts']:
            frame = merge_bg_add(frame, tensor_floatTexts, [0, width, 0, height], [0, width, 0, height])
        # log_content_write(log_file, f'end floatTexts, frame.shape = {frame.shape}')
        # ----------------------------------------------------开始添加水印----------------------------------------------------
        if config_dict['watermark']:
            frame = merge_bg_add(frame, watermark_360_tensor, [0, width, 0, height], [0, width, 0, height])
        # if int(item.childId) == 1:
        #     if 0 <= i < 5 * 25:
        #         frame = merge_bg_add(frame, watermark_wenzi_tensor, [0, watermark_wenzi_w, 0, watermark_wenzi_h], [0, watermark_wenzi_w, 0, watermark_wenzi_h])
        # log_content_write(log_file, f'end watermark, frame.shape = {frame.shape}')
        # ----------------------------------------------------保存第一张预览图----------------------------------------------------
        if (not config_dict['child']) or (config_dict['child'] and config_dict['last_child']):
            if i == 0:
                # frame 转回 cv2 格式用于保存预览图
                frame_preview = frame.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                frame_preview = frame_preview.astype(np.uint8)[..., :3]
                cv2.imwrite(join(merge_folder, 'first_frame.jpg'), frame_preview, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # 使用 GPU 编码 (无需 GPU->CPU 传输)
        gpu_nv12_frame = frame_pool.update_from_rgba(frame)
        bitstream = nvenc.Encode(gpu_nv12_frame)
        if bitstream:
            h264_file.write(bytearray(bitstream))

    # 刷新编码器
    bitstream = nvenc.EndEncode()
    if bitstream:
        h264_file.write(bytearray(bitstream))
    h264_file.close()

    # 封装 MP4
    log_content_write(log_file, 'start ffmpeg muxing to mp4')
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "25",
        "-i", h264_path, "-c:v", "copy", "-movflags", "+faststart",
        mp4_path, "-loglevel", "quiet"
    ], check=True)

    # 删除临时 H264 文件
    if exists(h264_path):
        os.remove(h264_path)

    duration = frame_num / 25
    return duration


def merge(id_merge, item, status_file, config_dict, log_file):
    # 下载并处理所有素材文件
    folder_merge = join(parameters['workspace'], id_merge)
    ret_dict, bool_dowload, error_reason = download_materials(item, config_dict, folder_merge, log_file)
    if not bool_dowload:
        return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 合成视频
    try:
        duration = merge_video(item, id_merge, ret_dict, config_dict, log_file)
    except BaseException as e:
        error_traceback = traceback.format_exc()
        error_reason = 'merge_video error' + '\n' + error_traceback
        return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 添加字幕
    if config_dict['subtitle']:
        try:
            mp4_path = join(folder_merge, 'speaker_25fps_16k_merged.mp4')
            mp4_old_path = join(folder_merge, 'speaker_25fps_16k_merged_old.mp4')
            rename(mp4_path, mp4_old_path)
            file_srt = ret_dict['subtitle_list'][0]
            # 把file_srt从D:\Generating_offline_2D_lip-sync_videos\workspace\240315150441845564DINet26072\subtitle.ass变成D\:\\Generating_offline_2D_lip-sync_videos\\workspace\\240315150441845564DINet26072\\subtitle.ass
            file_srt = file_srt.replace('\\', '\\\\').replace(':', '\\:')
            cmd = f"ffmpeg -i {mp4_old_path} -vf \"subtitles='{file_srt}'\" -pix_fmt yuv420p -q:v 0 -b:v {item.bitRate} -c:v libx264 -profile:v high -level 5.1 -preset medium -c:a copy -c:s mov_text {mp4_path} -loglevel quiet"
            os_system(cmd, log_file)
        except BaseException as e:
            error_traceback = traceback.format_exc()
            error_reason = 'merge subtitle' + '\n' + error_traceback
            return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 合并所有merge_wav里的音频文件,再和视频合并
    try:
        log_title_write(log_file, 'start merge audio')
        merge_audio(id_merge, item, folder_merge, config_dict, log_file)
    except BaseException as e:
        error_traceback = traceback.format_exc()
        error_reason = 'merge_audio error' + '\n' + error_traceback
        return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 若为单一任务,把result.mp4添加背景音乐
    if not config_dict['child']:
        if config_dict['music']:
            # 如果存在背景音乐，就把背景音乐和result.mp4合并成result_oss.mp4
            try:
                log_title_write(log_file, 'start merge background music')
                path_bg_music = ret_dict['music_path']
                path_result = join(parameters['workspace'], id_merge, 'result.mp4')
                path_result_oss = join(parameters['workspace'], id_merge, 'result_oss.mp4')
                merge_bg_music(log_file, path_bg_music, path_result, path_result_oss)
            except BaseException as e:
                error_traceback = traceback.format_exc()
                error_reason = 'merge_bg_music error' + '\n' + error_traceback
                return error_handling(item, id_merge, status_file, log_file, error_reason)
        else:
            # 如果不存在背景音乐，就把result.mp4复制成result_oss.mp4
            try:
                log_title_write(log_file, 'start copy result.mp4 to result_oss.mp4')
                shutil.copy(join(folder_merge, 'result.mp4'), join(folder_merge, 'result_oss.mp4'))
            except BaseException as e:
                error_traceback = traceback.format_exc()
                error_reason = 'copy_result_mp4 error' + '\n' + error_traceback
                return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 如果为多个子任务中的最后一个子任务，就把所有子任务的视频合并成一个视频, 再添加会循环播放的背景音乐
    if config_dict['child'] and config_dict['last_child']:
        # 先读取所有子任务生成的result.mp4的路径,按照childId排序组成list
        cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
        with open(cfg_json, 'r') as f:
            cfg_dict_list = json.load(f)
        cfg_dict_list = sorted(cfg_dict_list, key=lambda x: x['childId'])
        result_mp4_list = [join(parameters['workspace'], x['id_merge'], 'result.mp4') for x in cfg_dict_list]
        # 检查所有result.mp4是否存在
        for result_mp4 in result_mp4_list:
            if not exists(result_mp4):
                # 检查status_folder里的status文件
                status = 'error'
                status_file_path = join(parameters['status_folder'], basename(dirname(result_mp4)) + '.json')
                if exists(status_file_path):
                    with open(status_file_path, 'r') as f:
                        status_dict = json.load(f)
                    assert 'status' in status_dict.keys()
                    if status_dict['status'] == 'merging':
                        cycle_time = 0
                        while True:
                            cycle_time += 1
                            time.sleep(10)  # 1分钟检查6次,等待1小时,就是360次检查
                            with open(status_file_path, 'r') as f:
                                status_dict = json.load(f)
                            if status_dict['status'] == 'success':
                                assert exists(result_mp4)
                                status = 'success'
                                break
                            if cycle_time >= 360:
                                break
                if status == 'error':
                    error_reason = 'result.mp4 not exists'
                    return error_handling(item, id_merge, status_file, log_file, error_reason)
            # 把视频转成16k音频采样率,保证音频采样率相同
            result_mp4_16k = result_mp4[:-4] + '_16k.mp4'
            os_system(f'ffmpeg -i {result_mp4} -vcodec copy -acodec aac -ar 16000 {result_mp4_16k} -loglevel quiet', log_file)
        # 把所有result.mp4合并成一个视频,再添加会循环播放的背景音乐,保存为result_oss.mp4
        try:
            log_title_write(log_file, 'start merge all result.mp4')
            # 新建一个txt文件，里面写入所有result.mp4的路径,每行一个
            txt_path = join(folder_merge, 'result_mp4_list.txt')
            with open(txt_path, 'w') as f:
                for i, result_mp4 in enumerate(result_mp4_list):
                    result_mp4_16k = result_mp4[:-4] + '_16k.mp4'
                    f.write(f"file '{result_mp4_16k}'\n")
            os_system(f'ffmpeg -f concat -safe 0 -i {txt_path} -c copy {join(folder_merge, "result_oss.mp4")} -loglevel quiet', log_file)
            # 如果存在背景音乐，就把背景音乐和result_oss.mp4合并成result_oss.mp4
            if config_dict['music']:
                log_title_write(log_file, 'start merge background music')
                path_bg_music = ret_dict['music_path']
                path_result = join(parameters['workspace'], id_merge, 'result_oss.mp4')
                path_result_oss = join(parameters['workspace'], id_merge, 'result_oss_new.mp4')
                merge_bg_music(log_file, path_bg_music, path_result, path_result_oss)
                remove(path_result)
                rename(path_result_oss, path_result)
        except BaseException as e:
            error_traceback = traceback.format_exc()
            error_reason = 'merge_all_result_mp4 error' + '\n' + error_traceback
            return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 上传结果视频到oss
    if (not config_dict['child']) or (config_dict['child'] and config_dict['last_child']):
        try:
            log_title_write(log_file, 'start upload video to oss')
            upload_oss(join(parameters['workspace'], id_merge, 'result_oss.mp4'), parameters, f'{id_merge.replace("DINet", "DN")}.mp4')
            upload_oss(join(parameters['workspace'], id_merge, 'first_frame.jpg'), parameters, f'{id_merge.replace("DINet", "DN")}.jpg')
        except BaseException as e:
            error_traceback = traceback.format_exc()
            error_reason = 'upload_oss error' + '\n' + error_traceback
            return error_handling(item, id_merge, status_file, log_file, error_reason)
        videoUrl = f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{id_merge.replace("DINet", "DN")}.mp4'
    else:
        videoUrl = ''
    # videoUrl = ''

    # 回调
    try:
        log_title_write(log_file, f'start callback')
        log_content_write(log_file, f'callbackUrl = {item.callbackUrl}, merge_id = {id_merge}, duration = {duration}, videoUrl = {videoUrl}, videoName = {id_merge.replace("DINet", "DN")}.mp4, result = success, localPath = {join(parameters["workspace"], id_merge, "result_oss.mp4")}, failReason = "", horizontal = {item.width}, vertical = {item.height}, coverUrl = https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{id_merge.replace("DINet", "DN")}.jpg')
        callback_local_path = join(parameters['workspace'], id_merge, 'result_oss.mp4')
        if config_dict['child']:
            if not config_dict['last_child']:
                callback_local_path = ''
        callback_status = callback_merge(
            callbackUrl=item.callbackUrl,
            merge_id=id_merge,
            duration=duration,
            videoUrl=videoUrl,
            videoName=f'{id_merge.replace("DINet", "DN")}.mp4',
            result='success',
            localPath=callback_local_path,
            failReason='',
            horizontal=item.width,
            vertical=item.height,
            coverUrl=f'https://bsddata.oss-cn-hangzhou.aliyuncs.com/virtual_live/AIface/{id_merge.replace("DINet", "DN")}.jpg',
        )
        if callback_status:
            status_file = join(parameters['status_folder'], id_merge + '.json')
            with open(status_file, 'w') as f:
                f.write(json.dumps({'status': 'success'}))
    except BaseException as e:
        error_traceback = traceback.format_exc()
        error_reason = 'callback error' + '\n' + error_traceback
        return error_handling(item, id_merge, status_file, log_file, error_reason)

    # 如果为单一任务或者为多个子任务中的最后一个子任务，就删除所有工作素材
    if not parameters['keep_files']:
        log_title_write(log_file, 'start clean workspace')
        if callback_status:
            if (not config_dict['child']) or (config_dict['child'] and config_dict['last_child']):
                shutil.rmtree(join(parameters['workspace'], id_merge))
                if config_dict['child']:
                    # 删除cfg文件
                    cfg_json = join(parameters['cfg_folder'], item.uniqid + '.json')
                    os.remove(cfg_json)


if __name__ == '__main__':
    pass
