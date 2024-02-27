"""-*- 老婆最美 -*-"""
# @Project : accuracy_evaluation  
# @File : AI_MatchLocation.py 
# @Time : 2024/1/5 11:08 
# @Author : WaiL 
# @Software: PyCharm
"""
功能：根据匹配结果计算鼠标坐标的变换位置位置,支持仿射和TPS变换方式
"""
import numpy as np
import cv2
import faiss
import matplotlib.pyplot as plt

def estimate_affine_transformation(A_keypoint, B_keypoint):
    """
        功能:
        - 估计仿射变换矩阵
        输入:
        - A_keypoint, B_keypoint: 图A上的匹配坐标,图B上的匹配坐标,行向对应,Numpy、List数据格式
        输出：
        - A_keypoint, B_keypoint: Numpy数据格式
        - aff_m: 从A_keypoint到B_keypoint的仿射变换矩阵M。
                 此外从B_keypoint到A_keypoint的仿射变换矩阵H=cv2.invertAffineTransform(M)
        - mask: 内点索引,Numpy数据格式
    """
    #
    # 使用RANSAC方法估计仿射变换矩阵
    src_pts = np.float32(A_keypoint).reshape(-1, 1, 2)
    dst_pts = np.float32(B_keypoint).reshape(-1, 1, 2)
    aff_m, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)  # 估计刚性变换  # cv2.RANSAC cv2.LMEDS
    # aff_m, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC)  # 估计仿射变换
    # 将内点提取出来
    mask = (mask.astype(bool)).flatten()  # array.flatten()和.ravel()区别在于,前者是复制后者是引用
    A_keypoint = A_keypoint[mask, :]
    B_keypoint = B_keypoint[mask, :]
    return A_keypoint, B_keypoint, aff_m, mask

def transform_point(mouse_coordinates, trans_mat):
    """
        功能:
        - 使用变换矩阵,变换坐标点的位置,支持同时输入多个鼠标坐标
        输入:
        - mouse_coordinates: 鼠标点击图像A的坐标位置,格式[x,y],Numpy数据格式
        - trans_mat:变换矩阵,注意变换矩阵的变换方向,从A到B还是从B到A
        输出:
        - transformed_coordinates:  经过trans_mat变换后的坐标,Numpy数据格式,[n*2]尺寸
    """
    #
    # 旋转和平移校准浮动图像上的关键点
    transformed_coordinates = cv2.transform(mouse_coordinates.reshape((-1, 1, 2)), trans_mat)  # 坐标点变换
    transformed_coordinates = np.squeeze(transformed_coordinates, axis=1)  # 删除指定维度为1的矩阵维度
    return transformed_coordinates

def tps_transform_location(mouse_coordinates, A_keypoint, B_keypoint):
    """
        功能:
        - Thin Plate Spline (TPS) 变换函数,支持同时输入多个鼠标坐标
        输入:
        - mouse_coordinates: 鼠标点击图像A的坐标位置,格式二维[n,2]的[x,y]坐标
        - A_keypoint,B_keypoint: 图A上的匹配坐标,图B上的匹配坐标,行向对应,格式二维[n,2]的[x,y]坐标
        输出:
        - transformed_coordinates: 经过 TPS 变换后的坐标,格式二维[n,2]的[x,y]坐标
    """
    print('TPS runing')
    # 初始化数据格式
    A_keypoint = np.array(A_keypoint, dtype=np.float32).reshape(1, -1, 2)
    B_keypoint = np.array(B_keypoint, dtype=np.float32).reshape(1, -1, 2)
    mouse_coordinates = np.array(mouse_coordinates, dtype=np.float32).reshape(1, -1, 2)
    # 初始化匹配关系
    matches = [cv2.DMatch(i, i, 0) for i in range(A_keypoint.shape[1])]
    # 创建 TPS 变换器
    Image_TPS = cv2.createThinPlateSplineShapeTransformer()
    Image_TPS.estimateTransformation(A_keypoint, B_keypoint, matches)
    # 应用 TPS 变换
    print(mouse_coordinates)
    M = Image_TPS.applyTransformation(mouse_coordinates)
    # 提取坐标
    transformed_coordinates = M[1][0]
    print('TPS end')
    return transformed_coordinates

def near_keypoint(keypoint, num_near, search_keypoint):
    """
        功能:
        - 在keypoint中,搜索search_keypoint周围num_near临近点,输出其在keypoint中的索引,以及该点到临近点之间的各个距离
        参数:
        - 见上介绍
        返回:
        - distance: numpy数据格式,[n*m],n对应于search_keypoint中点数目,m对应于num_near设置的值
        - index_near: numpy数据格式,[n*m],n对应于search_keypoint中点数目,m对应于num_near设置的值
    """
    #
    # 利用L2距离(欧氏距离)来查询KP坐标的近邻域
    index_L2 = faiss.IndexFlatL2(2)  # 2表示关键点维度
    # KP坐标填充
    keypoint = np.asarray(keypoint).astype('float32')
    index_L2.add(keypoint)  # keypoint默认序号0,1,2,3...
    # print('打印index_l2包含的关键点数量:', index_L2.ntotal)
    # 搜索各个KP周围num_near个近邻KPs, 并返回距离【输出是升序的】
    distance, index_near = index_L2.search(search_keypoint.astype('float32'), num_near)
    return distance, index_near

def adjacent_point_reinforcement(A_match_near, B_match_near):
    """
        功能:
        - 使用四象限区域分割紧邻匹配的点对, 构建稳健的匹配坐标,为准确变换坐标做准备
        参数:
        - A_match_near, B_match_near: 紧邻匹配的关键点对
        返回:
        - A_reinforcement_point, B_reinforcement_point: 使用四象限加固的匹配点对,大概率是四个匹配点对,至少是三个匹配点对,行向对应
    """
    #
    # 中心点
    point_center = np.mean(B_match_near, axis=0, dtype=np.float32)
    # 四分域【依据四象限划分】
    quadrant = B_match_near - point_center[np.newaxis,:]
    index_1 = quadrant[:, 0] > 0  # H_Y坐标
    index_2 = ~index_1
    index_3 = quadrant[:,1] > 0   # W_X坐标
    index_4 = ~index_3
    # 根据四分域索引构建匹配坐标[大概率是四个匹配点对,至少是三个匹配点对]
    all_indexes = [index_1, index_2, index_3, index_4]
    use_indexes = [index_x for index_x in all_indexes if index_x.any()]
    A_point_list = list()
    B_point_list = list()
    for use_index in use_indexes:
        A_point = np.mean(A_match_near[use_index], axis=0, dtype=np.float32)
        B_point = np.mean(B_match_near[use_index], axis=0, dtype=np.float32)
        A_point_list.append(A_point)
        B_point_list.append(B_point)
    # numpy格式
    A_reinforcement_point = np.array(A_point_list, dtype=np.float32)
    B_reinforcement_point = np.array(B_point_list, dtype=np.float32)
    return A_reinforcement_point, B_reinforcement_point

def transform_match_location(mouse_location, A_match, B_match, num_near, mode='affine'):
    """
        功能:
        - 根据匹配结果变换鼠标点击位置获得其匹配位置
        输入:
        - mouse_location: 鼠标点击图像A的坐标位置,格式[x,y]
        - A_match,B_match: 图A上的匹配坐标,图B上的匹配坐标,行向对应
        - mode: 变换匹配位置所使用的变换矩阵类型,支持仿射变换、TPS变换
        输出:
        - transformed_location: 是mouse_location的映射坐标, 图A上mouse_location在图B中的匹配位置
    """
    # 初始化
    transformed_location = np.zeros_like(mouse_location)
    B_match_near_list = np.zeros((len(mouse_location), num_near,2))
    # 变换坐标
    if mode == 'affine':
        # 搜索每个鼠标坐标周围num_near个紧邻匹配点[]
        _, index_nears = near_keypoint(A_match, num_near, mouse_location)
        # 根据紧邻点计算变换坐标
        for item, index_near in enumerate(index_nears):
            # 四象限加固紧邻匹配点
            A_reinforcement_point, B_reinforcement_point = adjacent_point_reinforcement(A_match[index_near], B_match[index_near])
            _, _, trans_mat, _ = estimate_affine_transformation(A_reinforcement_point, B_reinforcement_point)
            transformed_location[item] = transform_point(mouse_location[item], trans_mat)
            B_match_near_list[item] = B_match[index_near]
    elif mode == 'TPS':
        # 搜索每个鼠标坐标周围num_near个紧邻匹配点[]
        _, index_nears = near_keypoint(A_match, num_near, mouse_location)
        # 根据紧邻点计算变换坐标[当临近点不能正确获得TPS变换矩阵时,鼠标坐标变换结果将是错误的(x=0,y=0)]
        for item, index_near in enumerate(index_nears):
            transformed_location[item] = tps_transform_location(mouse_location[item], A_match[index_near], B_match[index_near])
    else:
        raise ValueError("错误: 请正确设置点变换矩阵类型,可选择'affine'或'TPS'中的任意一个.")
    return transformed_location, B_match_near_list


if __name__=="__main__":
    mouse_location = np.array([[1,2],
                               [3,4],
                               [5,6]])
    A_match = np.array([[11,22],
                        [33,44],
                        [55,66],
                        [77,88],
                        [99,111]])
    B_match = np.array([[11, 22],
                        [33, 44],
                        [55, 66],
                        [77, 88],
                        [99, 111]])+10

    transformed_location, _ = transform_match_location(mouse_location, A_match, B_match, num_near=100, mode='affine')
    print(transformed_location)
    