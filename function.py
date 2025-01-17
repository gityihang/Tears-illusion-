'''
Description: Sports video background modeling, foreground removal
Author: byh呀呀呀
version: 
Date: 2025-01-03 20:42:15
LastEditors: byh呀呀呀
LastEditTime: 2025-01-17 23:57:52
'''


import os 
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import utils.statistic as statistic
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed

class remove_foreground():
    def __init__(self, **kwargs):
        """
        :param save_path: video path to save
        :param video_path: video path to remove
        :param kwargs: other parameters
        """
        self.save_path = kwargs.get('save_path')
        os.makedirs(self.save_path, exist_ok=True)
        self.video_path = kwargs.get('video_path')
        self.duration_sec = kwargs.get('duration_sec', 30)  # Default to 30 seconds
        self.fps_override = kwargs.get('fps_override', None)  # Default to None, meaning use video's FPS
        self.save_frames = kwargs.get('save_frames', True)  # Default to False, not saving frames
        self.mean_compensation = kwargs.get('mean_compensation', False)

        # 缓存均值和方差
        self.images_list = self.extract_frames_from_video()
        self.mean, self.var = statistic.compute_image_statistics(self.images_list)
        self.grey_mean = cv2.cvtColor(self.mean.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.grey_var = cv2.cvtColor(self.var.astype(np.uint8), cv2.COLOR_BGR2GRAY)


  


    def extract_frames_from_video(self,):
        """
        从视频中提取帧。

        参数：
            video_path: 视频路径。
            duration_sec: 提取的时间长度（秒）。
            fps_override: 重写视频的 FPS。
            save_frames: 是否保存帧。

        返回：
            frames_list: 帧列表。
        """

        # 加载视频
        cap = cv2.VideoCapture(self.video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error opening video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if self.fps_override:
            fps = self.fps_override
            print(f"Overriding video FPS with specified value: {self.fps_override}")
        print(f"Video FPS: {fps}")

        # 获取视频的总帧数和分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Resolution: {width}x{height}")
        print(f"Total Frames: {total_frames}")
        if self.duration_sec is not None:
        # 计算需要提取的帧数（最多提取视频的总帧数或设定的时间长度）
            frames_to_extract = min(fps * self.duration_sec, total_frames)

        else:
            frames_to_extract = total_frames

        self.fps = fps
        self.total_frames = frames_to_extract
        self.width = width
        self.height = height

        frames_list = []

        # 遍历视频的每一帧，提取前 N 秒的帧
        print('Extracting frames...')
        for i in tqdm(range(frames_to_extract), desc="Extracting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        return frames_list
    
    # 获取mask  
    def get_mask(self, images_array):
        """
        对视频帧进行处理，获取 mask。
        
        参数：
            images_array: 原始图像数组。
        返回：
            remove: mask 列表。
        """
        print('get mask ...')
        remove = []
        for i in tqdm(range(min(images_array.shape[0], self.total_frames)), desc="get mask"):
            # 使用 remove_function 对每一帧进行去除操作
            if i < 6:
                gray_first_frame = cv2.cvtColor(images_array[30], cv2.COLOR_BGR2GRAY)   
            else :    
                gray_first_frame = cv2.cvtColor(images_array[0], cv2.COLOR_BGR2GRAY)   
            gray_frame = cv2.cvtColor(images_array[i], cv2.COLOR_BGR2GRAY)
            outliers = (statistic.threshold(gray_frame, gray_first_frame, self.grey_mean, self.grey_var, self.total_frames - 1)).astype(bool)
            one = np.zeros_like(gray_frame, dtype=np.uint8)    # 直接设置类型为 uint8
            one[outliers] = 255
            remove.append(one)
        return remove
    

    # add parallel make mask
    def get_mask_parallel(self, images_array):
        print('get mask ...')
        remove = []
        with ThreadPoolExecutor() as executor:
            # 创建一个字典，用于保存future对象和其对应的索引
            future_to_index = {executor.submit(self.remove_function, i, images_array): i for i in range(min(images_array.shape[0], self.total_frames))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    mask = future.result()
                    remove.append(mask)
                except Exception as exc:
                    print(f'Frame {index} generated an exception: {exc}')
        return remove

    

    def process_video_frames(self, images_array, mask_list):
        """
        对视频帧进行处理，使用 remove_function 对每一帧进行去除操作，并根据 save_results 参数决定是否保存处理结果。

        参数：
            images_array: 原始图像数组。
            mask_list: 对应的 mask 列表。

        返回：
            image_files: 处理后的图像列表。
        """
        image_files = [None] * len(images_array) 

        def process_frame(args):
            i, images_array, mask_list, n_frame = args
            image = self.remove_function(images_array, mask_list, i, n_frame)
            return i, image

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_frame, (i, images_array, mask_list, self.total_frames))
                for i in range(min(images_array.shape[0], self.total_frames))
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
                i, image = future.result()  
                image_files[i] = image  
        return image_files



    def remove_function(self, image_files, remove, frame_idx, n_frame=30):
        frame = image_files[frame_idx]
        current_mask = remove[frame_idx]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算边界索引
        lower_idx = max(0, frame_idx - n_frame)
        upper_idx = min(len(remove), frame_idx + n_frame + 1)

        # 创建一个与 frame 大小相同的布尔数组
        current_white_area = current_mask == 255
        current_white_area_3d = np.repeat(current_white_area[:, :, np.newaxis], 3, axis=2)

        # 只在当前白色区域不为空时进行操作
        if np.any(current_white_area):
            for offset in range(1, n_frame + 1):
                # 向前遍历
                prev_idx = frame_idx - offset
                if lower_idx <= prev_idx < frame_idx and prev_idx != 20 and prev_idx != 0:
                    if np.any(current_white_area & (remove[prev_idx] == 0)):
                        frame[current_white_area_3d & (remove[prev_idx][:, :, np.newaxis] == 0)] = image_files[prev_idx][current_white_area_3d & (remove[prev_idx][:, :, np.newaxis] == 0)]
                        current_white_area &= ~(remove[prev_idx] == 0)

                # 向后遍历
                next_idx = frame_idx + offset
                if frame_idx < next_idx < upper_idx and next_idx != 20 and prev_idx != 0:
                    if np.any(current_white_area & (remove[next_idx] == 0)):
                        frame[current_white_area_3d & (remove[next_idx][:, :, np.newaxis] == 0)] = image_files[next_idx][current_white_area_3d & (remove[next_idx][:, :, np.newaxis] == 0)]
                        current_white_area &= ~(remove[next_idx] == 0)

            # 如果还有剩余的白色区域，用均值替换
            if np.any(current_white_area):
                if self.mean_compensation:
                    mean = self.mean + np.sqrt(self.var / 255) * (- self.statistic.adjust_mean(gray_frame, remove[frame_idx], self.mean, self.var))
                else:
                    mean = self.mean
                frame[current_white_area_3d] = mean[current_white_area_3d]

        return frame



    def save_images_to_video(self,image_files):
        """
        将图像列表保存为视频。

        参数：
            image_files (list): 图像列表，每个图像是一个 NumPy 数组，形状为 (height, width, 3) 或 (height, width, 1)。
            save_path (str): 保存视频的路径。
            video_name (str): 输出视频的文件名，默认为 "output_video.mp4"。
            fps (int): 视频帧率，默认为 30。
            width (int, optional): 视频的宽度。如果为 None，则使用图像的宽度。
            height (int, optional): 视频的高度。如果为 None，则使用图像的高度。

        返回：
            None
        """
        
        # 创建视频写入器
        
        video_name = os.path.basename(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
        output_video_path = os.path.join(self.save_path, video_name)  # 输出视频路径
        video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        # 遍历每一帧图像并写入视频
        for image in tqdm(image_files, desc="Saving video frames"):
            # image = adjust_white_balance_gw(image)
            if image.shape[-1] == 1:  # 如果是灰度图（单通道），转换为 RGB 图像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.dtype != np.uint8:  # 如果图像的类型不是 uint8，进行转换
                image = image.astype(np.uint8)
            video_writer.write(image)  # 写入视频
        # 释放资源
        video_writer.release()
        print(f"Video saved to {output_video_path}")


    # 保存文件        
    def save_process_parm(self, Reconstruct_image, mask_list):
        os.makedirs(os.path.join(self.save_path, 'statistics'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'frame'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Reconstruct_image'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'mask'), exist_ok=True)
        # 保存均值，方差
        cv2.imwrite(f'{self.save_path}/statistics/mean_image.png', self.mean.astype(np.uint8))
        cv2.imwrite(f'{self.save_path}/statistics/variance_image.png', self.var.astype(np.uint8))
        # 保存视频帧
        for i, (frame, reconstruct_frame, mask) in enumerate(zip_longest(self.images_list, Reconstruct_image,mask_list, fillvalue=None)):
            # 保存原始帧
            cv2.imwrite(f'{self.save_path}/frame/{i:05d}.jpg', frame)
            # 保存修复后的帧
            cv2.imwrite(f'{self.save_path}/Reconstruct_image/{i:05d}.jpg', reconstruct_frame)
            cv2.imwrite(f'{self.save_path}/mask/{i:05d}.jpg', mask)
        print(f"Saved all frames.")


    def __call__(self,):
        images_list = self.images_list
        mask_list = self.get_mask(np.array(images_list))
        Reconstruct_image = self.process_video_frames(np.array(images_list), mask_list)
        output = self.save_images_to_video(Reconstruct_image)
        if self.save_frames == True:
            self.save_process_parm(Reconstruct_image, mask_list)
        return output

if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # 添加参数


    parser.add_argument('--video_path', type=str, default='./data/example/example1.mp4',
                        help='path containing mp4')   
    
    parser.add_argument('--save_path', type=str, default='./data/output',
                        help='save_path')
    
    parser.add_argument('--duration_sec', type=int, default=30,
                        help='Duration in seconds (default: 30)')
    
    parser.add_argument('--fps_override', type=str, nargs='?', const=None, default=None,
                        help='FPS Override (default: None)')
    
    parser.add_argument('--save_frames', default=False, choices=['True', 'False'],
                        help='Save frames (default: Fasle)')
    
    # 解析命令行参数
    args = parser.parse_args()

    kwargs = {
        'video_path':args.video_path,
        'save_path':args.save_path,
        'duration_sec': args.duration_sec,
        'fps_override': args.fps_override,
        'save_frames': args.save_frames
    }
    remove = remove_foreground(**kwargs)
    remove() 