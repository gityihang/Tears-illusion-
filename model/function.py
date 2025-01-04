'''
Description: Sports video background modeling, foreground removal
Author: byh呀呀呀
version: 
Date: 2025-01-03 20:42:15
LastEditors: byh呀呀呀
LastEditTime: 2025-01-03 23:12:00
'''


import os 
import cv2
import numpy as np
from tqdm import tqdm
import argparse

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
        self.duration_sec = kwargs.get('duration_sec', 30)  # Default to 10 seconds
        self.fps_override = kwargs.get('fps_override', None)  # Default to None, meaning use video's FPS
        self.save_frames = kwargs.get('save_frames', True)  # Default to False, not saving frames
        self.mean_compensation = kwargs.get('mean_compensation', True)

        # 缓存均值和方差
        self.mean, self.var = self.compute_image_statistics()
        self.grey_mean = cv2.cvtColor(self.mean.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        self.grey_var = cv2.cvtColor(self.var.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        

    def adjust_mean(self, gray_images,  mask):
        # 调整均值
        # 计算掩码区域内和掩码区域外的灰度值平均值
        # 掩码区域内的像素
        masked_pixels = gray_images[mask == 1]
        avg_gray_masked = np.mean(masked_pixels) if masked_pixels.size > 0 else 0
        
        # 掩码区域外的像素
        unmasked_pixels = self.grey_mean[mask == 0]
        avg_gray_unmasked = np.mean(unmasked_pixels) if unmasked_pixels.size > 0 else 0
        
        # # 输出计算出的两个平均值（掩码区域内和外的平均灰度值）
        # print("掩码区域内的平均灰度值:", avg_gray_masked)
        # print("掩码区域外的平均灰度值:", avg_gray_unmasked)
        return ((avg_gray_masked - avg_gray_unmasked) / np.mean(np.sqrt(self.var))) * 2.5


    def extract_frames_from_video(self,):
        '''
        @des  :  
            Extract the first {duration_sec} seconds of video frames from the video and save them to the specified path, while obtaining the frame rate, total frame rate, and resolution of the video.
               
        @params  : 
            video_path (str): Enter the path of the video file.
            save_path (str): The folder path used to save extracted frames.
            duration_sec (int): The duration to be extracted (in seconds)
            fps_override (int, optional): If provided, extract using the specified frame rate; Otherwise, use the frame rate from the video.
            save_frames (bool, optional): If True, save each frame as an image; Default False does not save. 
                 
        @return  :
            frames_list (list): List of extracted video frames.
                  
        '''


        # 加载视频
        cap = cv2.VideoCapture(self.video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error opening video file")
            return [], None, None, None, None

        # 获取视频帧率（fps）
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
        self.total_frames = total_frames
        self.width = width
        self.height = height

        # 创建一个空列表来存储提取的视频帧
        frames_list = []

        # 遍历视频的每一帧，提取前 N 秒的帧
        print('Extracting frames...')
        for i in tqdm(range(frames_to_extract), desc="Extracting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
            if self.save_frames:         
                save_path = os.path.join(self.save_path, 'frame')
                os.makedirs(save_path, exist_ok=True)
                frame_filename = os.path.join(save_path, f"{i:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
        # 释放视频捕获对象
        cap.release()
        print(f"Total frames extracted: {len(frames_list)}")
        return frames_list
    
    def compute_image_statistics(self,):
        """
        处理视频帧，计算均值、方差。

        参数:

        返回:
            mean_image (ndarray): 均值。
            variance_image (ndarray): 方差。
        """
        # 将帧列表转换为 numpy 数组
        frames_list = self.extract_frames_from_video()
        self.frames_list = frames_list
        images_array = np.array(frames_list)
        
        # 计算每个像素点的均值和方差
        pixel_mean = np.mean(images_array, axis=0)
        pixel_variance = np.var(images_array, axis=0)
        # 保存均值图像
        if self.save_frames:
            save_path = os.path.join(self.save_path, 'statistics')
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(f'{save_path}/mean_image.png', pixel_mean.astype(np.uint8))
            cv2.imwrite(f'{save_path}/variance_image.png', pixel_variance.astype(np.uint8))

        return pixel_mean, pixel_variance
    
    def process_video_frames(self,):
        """
        对视频帧进行处理，使用 remove_function 对每一帧进行去除操作，并根据 save_results 参数决定是否保存处理结果。

        参数：

        返回：
            image_files (list): 处理后的图像文件路径列表（仅在 save_results 为 True 时有效）。
            remove (list): 处理后的 "去除" 图像列表。
        """
        # 确保保存目录存在，如果需要保存结果
        if self.save_frames:
            os.makedirs(os.path.join(self.save_path, 'move_path'), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, 'Reconstruct_image'), exist_ok=True)

        # 用于存储结果
        image_files = []
        remove = []
        images_array = np.array(self.frames_list)
        # 遍历每一帧
        print('Reconstruct image ...')
        for i in tqdm(range(min(images_array.shape[0], self.total_frames)), desc="Processing frames"):
            # 使用 remove_function 对每一帧进行去除操作
            if i  == 0:
                self.first_frame = images_array[i-20]
                self.gray_first_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_RGB2GRAY)
                print(f'{i} chang to {i-20}')
            one, image = self.remove_function(images_array[i])

            # 存储处理后的结果
            remove.append(one)
            image_files.append(image)

            # 如果需要保存结果，保存每一帧的处理结果
            if self.save_frames:
                cv2.imwrite(f'{self.save_path}/move_path/{i:05d}.jpg', one)
                cv2.imwrite(f'{self.save_path}/Reconstruct_image/{i:05d}.jpg', image)

        # 返回处理结果
        return image_files, remove


    def detect_connected_components(self,image, min_area=80, top_n=2):
        """
        对单张二值图像进行联通区域检测，保留最大的top_n个区域，并标注区域边界和质心。

        参数：
            image: 输入的二值图像（黑白掩码，dtype=np.uint8）。
            min_area: 过滤掉小于此面积的区域。
            top_n: 保留最大的top_n个区域。

        返回：
            output_image: 标注区域边界和质心的彩色图像。
            stats: 每个区域的统计信息 [x, y, width, height, area]。
            centroids: 每个区域的质心坐标。
            largest_regions: 最大的top_n个区域的二值化图像。
        """
        # 检测联通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        
        # 将输入图像转为彩色，便于绘制标注
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 创建一个空的二值图像，用于保存最大的top_n个区域
        largest_regions = np.zeros_like(image)

        # 存储每个区域的面积及其标签
        areas = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                areas.append((label, area))

        # 根据面积排序，选择最大的top_n个区域
        areas = sorted(areas, key=lambda x: x[1], reverse=True)[:top_n]

        # 遍历保留的区域
        for label, area in areas:
            # 提取区域统计信息
            x, y, w, h, _ = stats[label]
            cx, cy = centroids[label]

            # 将该区域添加到largest_regions图像中
            largest_regions[labels == label] = 255

            # 绘制边界框
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制质心
            cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            # 在区域旁标注索引
            cv2.putText(output_image, f"ID: {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return largest_regions


    def remove_function(self,frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 计算每一像素点是否为离群点，并使用 remove_function 去除
        outliers = (self.standard_function(gray_frame, self.gray_first_frame, self.grey_mean, self.grey_var) / 255).astype(bool)

        # mask 
        one = np.zeros_like(gray_frame, dtype=np.uint8)  # 直接设置类型为 uint8
        one[outliers] = 255
        # image
        outliers_3d = np.repeat(outliers[:, :, np.newaxis], 3, axis=2)
        if self.mean_compensation:
            mean = self.mean + np.sqrt(self.var / 255) * (- self.adjust_mean(gray_frame,outliers))
        else:
            mean = self.mean
        frame[outliers_3d] = mean[outliers_3d]
        return one, frame

    def standard_function(self,current_frame,first_frame,mean_frame,var):
        a = abs(current_frame - first_frame)
        b = abs(current_frame - mean_frame)
        outliersa = (a > 10)  
        kernel = np.ones((3, 3), np.uint8) 
        outliersa = np.array(outliersa, dtype=np.uint8) * 255
        outliersa = cv2.erode(outliersa, kernel=kernel, iterations=3)  # 腐蚀
        outliersa = self.detect_connected_components(outliersa, 100, top_n=10)
        outliersa = outliersa / 255
        # outliersb = (b > np.sqrt(var) * 1.5)
        outliers = outliersa #* outliersb
        outliers = cv2.dilate(np.array(outliers, dtype=np.uint8) * 255, np.ones((25, 25), np.uint8) , iterations=3)
        return outliers

    def save_images_to_video(self,):
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
        image_files, remove_image  = self.process_video_frames()
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

    def __call__(self,):
        self.save_images_to_video()

if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # 添加参数
    parser.add_argument('--save_path', type=str, default='/home/byh/Tears-illusion-/data/output',
                        help='save_path')

    parser.add_argument('--video_path', type=str, default='/home/byh/Tears-illusion-/data/example/example1.mp4',help='path containing mp4')   
    
    parser.add_argument('--duration_sec', type=int, default=30,
                        help='Duration in seconds (default: 30)')
    
    parser.add_argument('--fps_override', type=int, nargs='?', const=None, default=None,
                        help='FPS Override (default: None)')
    
    parser.add_argument('--save_frames', action='store_false',
                        help='Save frames (default: True)')
    
    # 解析命令行参数
    args = parser.parse_args()

    kwargs = {
        'save_path':args.save_path,
        'video_path':args.video_path,
        'duration_sec': args.duration_sec,
        'fps_override': args.fps_override,
        'save_frames': args.save_frames
    }
    remove = remove_foreground(**kwargs)

    a = remove()
