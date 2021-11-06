'''
Author: Adam XL Wang
Date: 2021-10-02 17:11:08
LastEditors: Please set LastEditors
LastEditTime: 2021-11-06 15:19:27
FilePath: /yijyun-jupyter/jingluo.py
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize
import os
import matplotlib


def cut(x: np.ndarray):
    '''
    description: 滤波 -> 截取中间的1000个数据点
    param {*}
    return {*}
    '''
    x = filter(padding(x))  # 滤波
    mid = int((len(x)) / 2)
    extent = 500  # 范围
    return x[mid - extent:mid + extent]  # 只取中间的1000个数据点=5秒


def padding(x: np.ndarray):
    if len(x) < 10983:
        x = np.concatenate([[0] * (10983 - len(x) + 1), x])
    return x


def analyze(x: np.ndarray, param: np.ndarray = [1, 1], m: int = 8):
    """预测第1到第m-1经络的结果，也就是C1-Cm-1，暂时先无视C0
        流程：filter滤波原始数据 -> findTroughs找谷底索引 -> compute根据谷底索引对每单个周期计算平均幅值等统计量
            -> selectRatio根据测试者参数选择标准幅值比例 -> match根据标准幅值比例进行拟合
            -> 对比拟合后的数据幅值比例和标准幅值比例
        参数：
            x：数据
            param：测试人的参数
            m：对前m条经络进行计算
            visualize：画出对比图if True

        返回：
            pred：预测的结果，单位和金姆的一样，例如：1代表10%，-3代表-30%

        """

    # 预处理
    x = cut(x)
    locs = find_troughs(x)  # 寻找谷底索引

    # 画图测试
    matplotlib.use('Agg')
    plt.plot(x)
    plt.plot(locs, x[locs], '.')
    plt.savefig('test.png')

    # 根据谷底定位计算幅值和相位的平均值和std%
    amp_means, phase_means = compute(x, locs, m)

    # 根据标准幅值比例拟合并预测结果
    standard_amp_ratios, standars_phases = get_standards()
    return compare(
        amp_means,
        standard_amp_ratios,
    ), compare(
        phase_means,
        standars_phases,
    )  # 预测结果


def compare(x, standards, matching=True):
    # 暂时忽略第一经络心包经
    return list((x - standards) * 100 / standards)  # 默认心包经差异为0


def get_standards():
    standard_amp_ratios = np.array([
        1., 1., 0.42563353, 0.30427939, 0.17541439, 0.12563837, 0.08758069,
        0.05727361
    ])
    standard_phases = np.array(
        [0, 2.0413, 2.8055, 3.0445, -2.2744, -1.6200, -1.4833, -0.2213])
    return standard_amp_ratios, standard_phases


def filter(x: np.ndarray):
    """对数据进行高通滤波

    参数：
        x：数据

    返回：
        x：滤波后数据

    """

    # b,a 滤波器分子分母是从Matlab获取的
    b = np.array(pd.read_csv('b.txt')).flatten()
    a = 1
    x = x - np.mean(x)  # 去均值，否则容易出现数值运算的问题
    x = signal.filtfilt(b, a, x)  # 传入滤波器
    return x


def fft(x: np.ndarray):
    """对数据进行快速傅立叶变换

    参数：
        x：数据

    返回：
        amp：幅值
        phase：相位

    """
    fft_result = np.fft.fft(x)
    amp = np.abs(fft_result) / len(x)
    amp[1:] = amp[1:] * 2
    phase = np.angle(fft_result)
    return amp, phase


def compute(x: np.ndarray, locs: np.ndarray, m: int = 8):
    """根据谷底定位计算幅值和相位的平均值

    参数：
        x：数据

    返回：
        ampMean：平均幅值
        phaseMean：平均相位

    """

    # 储存幅值和相位
    amps = np.empty(8)
    phases = np.empty(8)
    for i in range(len(locs) - 1):  # 每个谷底
        amp, phase = fft(x[locs[i]:locs[i + 1]])
        amps = np.row_stack([amps, amp[:m]])
        phases = np.row_stack([phases, phase[:m]])
    amps = amps[1:, :]
    phases = phases[1:, :]

    # writer = pd.ExcelWriter('jingluo-fft.xlsx')
    # pd.DataFrame(amps).to_excel(writer, sheet_name='amps', index=False)
    # pd.DataFrame(phases).to_excel(writer, sheet_name='phases', index=False)
    # writer.save()

    # amp_means = np.mean(amps, axis=0)  # 幅值平均
    # phase_means = np.mean(phases, axis=0)  # 相位平均
    return amps, phases


def select_ratio(param: np.ndarray):
    """根据参数（手指、性别）判断标准幅值比例%

    参数：
        param：测试人的参数，param[0]=手指，param[1]=性别

    返回：
         amp_ratios：根据测试人的参数选择的标准幅值比例

    """
    finger = param[0]
    sex = param[1]
    if finger <= 5 and sex == 1:
        amp_ratios = [1.0000, 0.4367, 0.3793, 0.1896, 0.1451, 0.0940,
                      0.0496]  # 男左
    elif finger > 5 and sex == 1:
        amp_ratios = [
            1.0000, 0.4367, 0.3793 / 1.07, 0.1896, 0.1451, 0.0940, 0.0496
        ]  # 男右
    elif finger <= 5 and sex == 2:
        amp_ratios = [1.0000, 0.4055, 0.2971, 0.1901, 0.1810, 0.1273,
                      0.0714]  # 女左
    else:
        amp_ratios = [1.0000, 0.3894, 0.2732, 0.1757, 0.1664, 0.1223,
                      0.0756]  # 女右
    amp_ratios = [
        1., 0.37829378, 0.2676348, 0.16825397, 0.1081648, 0.08782907,
        0.06953167
    ]
    return np.array(amp_ratios)


def find_troughs(x: np.ndarray, visualize=False):
    """谷底定位

    参数：
        x：数据

    返回：
        locs：谷底索引

    """

    # medianTrough = np.median(x[signal.find_peaks(-x, distance=100)[0]])
    # medianPeak = np.median(x[signal.find_peaks(x, distance=100)[0]])
    # minHeight = medianTrough + 0.5 * (medianPeak - medianTrough)
    # locs = signal.find_peaks(-x, distance=100, height=minHeight)[0]
    locs = signal.find_peaks(-x, distance=100)[0]

    # 可视化
    if visualize:
        plt.plot(x)
        plt.plot(locs, x[locs], marker=10, linestyle='None')
        plt.show()

    return locs


def match(x: np.ndarray, y: np.ndarray):
    """拟合，计算出一个常量b，让x*b很接近y，也就是用b等比例放大或缩小x，最后返回x*b

    参数：
        x：将要被拟合的一组数
        y：拟合的目标

    返回：
        x*b.x：经过拟合后的x

    """
    x = x / x[0]
    y = np.array(y)

    def fun(b):
        percentDiff = x * b / y - 1  # 百分比差异
        return np.sum(np.square(percentDiff))  # 平方的和

    b = minimize(fun, np.array(1))  # 最小化：百分比差异平方的和
    return x * b.x
