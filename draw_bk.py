"""-*- 老婆最美 -*-"""
# @Project : Paper3  
# @File : DrawPointLine.py 
# @Time : 2023/10/22 21:39 
# @Author : WaiL 
# @Software: PyCharm
"""
功能：绘制并连线成对的点
"""
import numpy as np
import matplotlib.pyplot as plt

def PointLine(KpsA, KpsB, SizeKps, LineWidth, path):
    # 输入:
    # KpsA, KpsB: Numpy数据格式，[n*2]尺寸
    # SizeKps:点的大小
    # LineWidth: 线的粗细
    #
    # 创建一个新的图形
    plt.figure()
    # 绘制线连接对应的坐标
    for i in range(KpsA.shape[0]):
        plt.plot([KpsA[i, 0], KpsB[i, 0]], [KpsA[i, 1], KpsB[i, 1]], color='r', linewidth=LineWidth, zorder=1)
    # 绘制点集合
    plt.scatter(KpsA[:, 0], KpsA[:, 1], color='g', label='KpsA', s=SizeKps, zorder=2)
    plt.scatter(KpsB[:, 0], KpsB[:, 1], color='b', label='KpsB', s=SizeKps, zorder=3)
    # 设置Y轴翻转
    plt.gca().invert_yaxis()
    # 设置纵横比为1:1
    plt.gca().set_aspect('equal')
    # 添加图例
    plt.legend()
    # 显示图形
    # plt.show()
    plt.savefig(path)


if __name__=="__main__":
    # 创建示例的 A 和 B
    print('__name__==__main__')
    KpsA = np.random.rand(10, 2)  # 10行两列的随机点集合
    KpsB = np.random.rand(10, 2)  # 10行两列的随机点集合
    PointLine(KpsA, KpsB, SizeKps=50, LineWidth=1)








"""
# 语法：
1、

参数：
输出：
"""
# .............................................................................#
"""
# 小知识：
1、

"""
# .............................................................................#
"""
# 实施例：

"""
