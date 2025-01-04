'''
Description: stactistic functions
Author: byh呀呀呀
version: 
Date: 2025-01-03 22:29:20
LastEditors: byh呀呀呀
LastEditTime: 2025-01-04 10:52:01
'''

import numpy as np


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

    t_statistic = (X_bar - mean) / (variance / (n ** 0.5))

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
    outliers1 = (threshold1 > 10)  
    # 但是，由于视频的每一帧都伴随有光线的变化，所以 threshold2 也会有波动，所以 threshold2 也需要设置一个阈值
    # threshold2 是由于两个均值之间的误差，这里使用t检验
    # z = (x̄ - μ) / (s / √n)
    # 这里的 alpha 取 0.05， z_alpha/0.025 = 1.96
    # 当 z > 1.96 时，认为是异常像素点
    outliers2 = (t_test(X_bar, mean, var, n) > 1.96)
