'''
Description: stactistic functions
Author: byh呀呀呀
version: 
Date: 2025-01-03 22:29:20
LastEditors: byh呀呀呀
LastEditTime: 2025-01-07 22:36:20
'''

import cv2
import numpy as np

# 初始化全局参数
global_param1 = 1  # 腐蚀次数
global_param2 = 2  # 膨胀次数
global_param3 = 5  # 卷积核大小
global_pixel_threshold = 10  # 阈值1
global_z = 1.96  # z_alpha/0.025 = 1.96

def compute_image_statistics(frames_list):
    """
    处理视频帧，计算均值、方差。

    参数:

    返回:
        mean_image (ndarray): 均值。
        variance_image (ndarray): 方差。
    """
    # 将帧列表转换为 numpy 数组
    images_array = np.array(frames_list)
    
    # 计算每个像素点的均值和方差
    pixel_mean = np.mean(images_array, axis=0)
    pixel_variance = np.var(images_array, axis=0)
    # 保存均值图像
    return pixel_mean, pixel_variance


def t_test(X_bar, mean, variance, n):
    '''
    t检验
    :param X_bar: 样本均值
    :param mean: 总体均值
    :param variance: 样本方差
    :param n: 样本大小
    :return: t统计量
    '''
    # 双边t检验的公式:
    # t = (x̄ - μ) / (s / √n)
    # 其中 x̄ 是样本均值, μ 是总体均值, s 是样本标准差, n 是样本大小

    t_statistic = (X_bar - mean) / (((variance + 1e-10) / n ** 0.5))

    return t_statistic

# 创新点1：多阈值法检测异常帧
# 可视化前景分割见 实验实验实验./test/test.ipynb
def threshold(X_bar, standard_X, mean, var, n):
    '''
    阈值法检测异常帧
    :param X_bar: 样本均值
    :param standard_X: 标准对照值（按照顺序依次取 n-20 )
    :param mean: 总体均值
    :param var: 总体方差
    :param n: 样本大小
    '''
    # 公式 first_X - mean
    # 公式 (first_X - X_bar) + (X_bar - mean)
    # 拆分为threshold1 和 threshold2
    # 其中 threshold1 = first_X - X_bar 表示当前帧与第一帧的差值
    # 其中 threshold2 = X_bar - mean 表示当前帧与均值的差值
    # threshold1 表示为与标准差的波动表示为像素级别的波动，这里的 threshold1 可以取固定值 

    threshold1 = abs(X_bar - standard_X)
    threshold2 = abs(X_bar - mean)
    # 当 threshold1 > 10 , 则认为是异常像素点
    # 阈值1
    outliers1 = (threshold1 > global_pixel_threshold)  
    # 但是，由于视频的每一帧都伴随有光线的变化，所以 threshold2 也会有波动，所以 threshold2 也需要设置一个阈值
    # threshold2 是由于两个均值之间的误差，这里使用t检验
    # z = (x̄ - μ) / (s / √n)
    # 这里的 alpha 取 0.05， z_alpha/0.025 = 1.96
    # 当 z > 1.96 时，认为是异常像素点
    # 阈值2
    outliers2 = (t_test(X_bar, mean, var, n) > global_z)
    # 阈值3 # e_d(outliers1,1,3,3)
    mask1_d, mask2_d = e_d(outliers1,global_param1,global_param2,global_param3), e_d(outliers2,global_param1,global_param2,global_param3)
    combined_mask = mask1_d + mask2_d 
    # Step 2: 查找交集区域 (即值为 2 的区域)
    intersection_mask = (combined_mask == 2).astype(np.uint8)  # 交集区域

    # Step 3: 使用膨胀操作找到被包围的区域 (值为 1 且被交集包围的区域)
    kernel = np.ones((23, 23), np.uint8)  
    dilated_mask = cv2.dilate(intersection_mask, kernel, iterations=3)  # 膨胀交集区域

    # Step 4: 仅保留那些在膨胀后的区域内的 1
    bordering_mask = (combined_mask == 1) & (dilated_mask == 1)  # 找到交集区域外的区域，但被包围

    # Step 5: 合并交集区域和包围区域
    outliers = intersection_mask | bordering_mask.astype(np.uint8)
    return outliers

# 创新点2：调整均值
def adjust_mean(gray_images, mask, grey_mean, var):
    # 调整均值
    # 计算掩码区域内和掩码区域外的灰度值平均值
    # 掩码区域内的像素
    masked_pixels = gray_images[mask == 1]
    avg_gray_masked = np.mean(masked_pixels) if masked_pixels.size > 0 else 0
    
    # 掩码区域外的像素
    unmasked_pixels = grey_mean[mask == 0]
    avg_gray_unmasked = np.mean(unmasked_pixels) if unmasked_pixels.size > 0 else 0
    
    # # 输出计算出的两个平均值（掩码区域内和外的平均灰度值）
    return ((avg_gray_masked - avg_gray_unmasked) / np.mean(np.sqrt(var))) * 2.5


# 进行膨胀腐蚀的操作,可以改动卷积核(3,3)——>(5,5)
def e_d(mask,a,b,c):
    # 原始mask 
    mask1_O = mask.astype(np.uint8) * 255
    # 腐蚀
    mask_e = cv2.erode(mask1_O, 
                       kernel=np.ones((c, c), np.uint8), 
                       iterations=a)  
    # 联通区域检测
    mask_dcc,_,_ = detect_connected_components(mask_e, 20, 100)
    # 膨胀
    mask_d = cv2.dilate(np.array(mask_dcc, dtype=np.uint8) * 255, 
                        kernel=np.ones((15, 15), np.uint8), 
                        iterations=b)
    return mask_d


# 联通区域检测
def detect_connected_components(image, min_area=80, top_n=2):
    '''
    检测联通区域
    :param image: 输入图像
    :param min_area: 最小区域面积
    :param top_n: 保留的最大区域数量
    :return: 保留最大区域的灰度图像, 带有边界框的图像, 边界框坐标列表
    '''
    # 连通区域检测
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    largest_regions = np.zeros_like(image)

    areas = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            areas.append((label, area))

    areas = sorted(areas, key=lambda x: x[1], reverse=True)[:top_n]

    bounding_boxes = []
    for label, area in areas:
        # 保留最大区域
        largest_regions[labels == label] = 255
        x, y, w, h, _ = stats[label]
        # 绘制边界框
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 记录边界框坐标
        bounding_boxes.append((x, y, x + w, y + h))

    return largest_regions, output_image, bounding_boxes