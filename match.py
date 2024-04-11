
# @Project : Paper3  
# @File : Match.py 
# @Time : 2023/10/22 8:01 
# @Author : WaiL 
# @Software: PyCharm
"""
功能：特征匹配算法。【粗匹配->几何一致性匹配->局部匹配->精匹配】
"""
import sys
import numpy as np
import faiss
import heapq
import cv2
import time
import taichi as ti


def Euclidean(eigens_1, eigens_2):
    @ti.kernel
    def _Euclidean3(eigens_1: ti.types.ndarray(), eigens_2: ti.types.ndarray(), result: ti.types.ndarray()):
        for i, j in ti.ndrange((0, result.shape[0]), (0, result.shape[1])):
            sum_value = 0.0
            for index in range(0, eigens_1.shape[1]):
                temp = ti.pow((eigens_1[i, index] - eigens_2[j, index]), 2)
                sum_value += temp
            result[i, j] = sum_value
        
        return

    result = np.zeros((eigens_1.shape[0], eigens_2.shape[0]))
    _Euclidean3(eigens_1, eigens_2, result)

    return result

def RemoveRepetition(Index_Similar, Similarity, Order_Keypoint):
    """
        为全局匹配结果实施降重,注意Similarity是【全局】相似性矩阵
        对匹配结果降重，避免一个关键点与多个关键点匹配，实现关键点一一配对
        如果一个关键点匹配多个关键点视为重复匹配，此时比较重复匹配的相似度，使用高度相似的匹配对降重，其他被删除。
    """
    # 输入：
    # Index_Similar: 关键点的索引，其中可能包含相同的数字【我用于：Index_Similar是含重复值的KPB的索引】
    # Similarity: 一个矩阵存储了所有关键点对应的特征向量之间的欧氏距离【所有特征向量遍历欧式距离的相似性矩阵】
    # Order_Keypoint: 从0开始，对Similarity的行进行编号
    #
    # 输出：
    # Order, Value：为数值索引,两者数值是一一对应的
    # Order: 索引KPA的关键点，即为降重后的KPA结果
    # Value: 索引KPB的关键点，即为降重后的KPB结果
    #
    # 注意理解以下提示：
    # Index_Similar和Order_Keypoint行方向是一一对应的
    #
    # 对[1*n]数组降重并统计各个元素的重复次数
    Value, Index_NoRep, Count = np.unique(Index_Similar, axis=0, return_index=True, return_counts=True)
    Index_Rep = Count > 1            # 重复值的索引
    if Index_Rep.any() == True:      # 判断: 若是存在重复
        Order = Order_Keypoint[Index_NoRep]
        RepValue = Value[Index_Rep]  # 重复的值
        RevisedOrder = []
        for Val in RepValue:
            Index = (Index_Similar == Val)
            Index_MinSimilInRep = np.argmin(Similarity[Index,Val])
            RevisedOrder.append(Order_Keypoint[Index][Index_MinSimilInRep])
        # 含重复匹配，替换为相似度高的索引数值
        Order[Index_Rep] = np.array(RevisedOrder)
    else:
        Order = Order_Keypoint
    return Order, Value

def RemoveRepetition_Local(Index_Similar, Similarity, Index_Similar_Around, Order_Keypoint):
    """
        为局部区域匹配结果实施降重,注意Similarity是【局部】相似性矩阵
        对匹配结果降重，避免一个关键点与多个关键点匹配，实现关键点一一配对
        如果一个关键点匹配多个关键点视为重复匹配，此时比较重复匹配的相似度，使用高度相似的匹配对降重，其他被删除。
    """
    # 输入：
    # Index_Similar: 关键点的索引，其中可能包含相同的数字【我用于：Index_Similar是含重复值的KPB的索引】
    # Similarity: 一个矩阵存储了【所有关键点】与【每个关键点周围多个关键点】对应的特征向量之间的欧氏距离
    # Index_Similar_Around: 局部最相似索引,为Similarity每行中的最小值索引
    # Order_Keypoint: 从0开始，对Similarity的行进行编号
    #
    # 输出：
    # Order, Value：为数值索引,两者数值是一一对应的
    # Order: 索引KPA的关键点，即为降重后的KPA结果
    # Value: 索引KPB的关键点，即为降重后的KPB结果
    #
    # 注意理解以下提示：
    # Index_Similar和Order_Keypoint行方向是一一对应的
    #
    # 对[1*n]数组降重并统计各个元素的重复次数
    Value, Index_NoRep, Count = np.unique(Index_Similar, axis=0, return_index=True, return_counts=True)
    Index_Rep = Count > 1            # 重复值的索引
    if Index_Rep.any() == True:      # 判断: 若是存在重复
        Order = Order_Keypoint[Index_NoRep]
        RepValue = Value[Index_Rep]  # 重复的值
        RevisedOrder = []
        for Val in RepValue:
            Index = (Index_Similar == Val)
            RepSimilarity = Similarity[Index]
            RepSimilarityValue = RepSimilarity[range(RepSimilarity.shape[0]), Index_Similar_Around[Index]]
            Index_MinSimilInRep = np.argmin(RepSimilarityValue)
            RevisedOrder.append(Order_Keypoint[Index][Index_MinSimilInRep])
        # 含重复匹配，替换为相似度高的索引数值
        Order[Index_Rep] = np.array(RevisedOrder)
    else:
        Order = Order_Keypoint
    return Order, Value

def NearKeypoint(Keypoint, NumNear, SearchKeypoint):
    """
        在Keypoint中，搜索SearchKeypoint周围NumNear个临近点，输出其在Keypoint中的索引，以及该点到临近点之间的各个距离
    """
    # 输入: 见上介绍
    # 输出:
    # Distance: Numpy数据格式，[n*m],n对应于SearchKeypoint中点数目，m对应于NumNear设置的值
    # IndexNear: Numpy数据格式，[n*m],n对应于SearchKeypoint中点数目，m对应于NumNear设置的值
    #
    # 利用L2距离(欧氏距离)来查询KP坐标的近邻域
    Index_L2 = faiss.IndexFlatL2(2)  # 2表示关键点维度
    # KP坐标填充
    Keypoint = np.asarray(Keypoint).astype('float32')
    Index_L2.add(Keypoint)  # Keypoint默认序号0,1,2,3...与Similarity的行对应
    # print('打印index_l2包含的关键点数量:', Index_L2.ntotal)
    # 搜索各个KP周围包括本身在内的Num_NearKP+1个近邻KPs, 并返回距离【输出是升序的】
    Distance, IndexNear = Index_L2.search(SearchKeypoint.astype('float32'), NumNear)
    # 去掉无用的KP自身的近邻

    return Distance, IndexNear

def EstimateAffineTransformation(KeypointA, KeypointB):
    """
        估计仿射变换矩阵
    """
    # 输入:
    # KeypointA, KeypointB: Numpy、List数据格式
    #
    # 输出：
    # KeypointA, KeypointB: Numpy数据格式
    # Aff_M: 从KeypointA到KeypointB的仿射变换矩阵M。
    #        此外从KeypointB到KeypointA的仿射变换矩阵H=cv2.invertAffineTransform(M)
    # Mask: 内点索引，Numpy数据格式
    #
    # 使用RANSAC方法估计仿射变换矩阵
    src_pts = np.float32(KeypointA).reshape(-1, 1, 2)
    dst_pts = np.float32(KeypointB).reshape(-1, 1, 2)
    # Aff_M, Mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)  # 估计刚性变换  # cv2.RANSAC cv2.LMEDS
    Aff_M, Mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC)  # 估计仿射变换
    # 将内点提取出来
    Mask = (Mask.astype(bool)).flatten()  # array.flatten()和.ravel()区别在于，前者是复制后者是引用
    KeypointA = KeypointA[Mask, :]
    KeypointB = KeypointB[Mask, :]
    return KeypointA, KeypointB, Aff_M, Mask

def TransformPoint(Point, TransMat):
    """
        使用变换矩阵，变换坐标点的位置
    """
    # 输入:
    # Point: Numpy数据格式
    # TransMat:变换矩阵，注意变换矩阵的变换方向，从A到B还是从B到A
    # 输出:
    # Point_Trans: Numpy数据格式，[n*2]尺寸
    #
    # 旋转和平移校准浮动图像上的关键点
    Point_Trans = cv2.transform(Point.reshape((-1, 1, 2)), TransMat)  # 坐标点变换
    Point_Trans = np.squeeze(Point_Trans, axis=1)  # 删除指定维度为1的矩阵维度
    return Point_Trans

def GeometricConsistency_Guide(KeypointA, KeypointB, Magnification):
    """
        基于几何差异，迭代出几何一致的成对的关键
    """
    # 输入:
    # KeypointA, KeypointB: Numpy数据格式,各行一一对应
    # Magnification: 引导匹配与当前关键点，他们所对应的图像尺寸放大倍数
    # 输出：
    # KeypointA, KeypointB: 是满足结构一致性的匹配结果【放大前的输入关键点对应的结果】
    # Aff_M: 是引导匹配所需的坐标变换矩阵,与图像magnification有关,与输出的KeypointA, KeypointB不是直接相关
    #
    # 初始化参数
    T_NumKP = 100                 # 控制最后一次成对关键点的数目
    T_IterativeResolution = 0.7  # 该值属于[0~1],取0与1非法,值越小迭代次数越少，结果越不稳定
    # Num_GoodKP = T_NumKP + 1   # 确保Num>T_NumKP首次执行满足条件
    Loop = 0                     # 统计迭代次数
    # 迭代
    E_Start = time.time()
    while True:
        # 遍历搜索，获得关键点之间的距离
        DistanceA, _ = NearKeypoint(KeypointA, KeypointA.shape[0], KeypointA)
        DistanceB, _ = NearKeypoint(KeypointB, KeypointB.shape[0], KeypointB)
        # 几何差异【图"间"距离筛选】【DD: Distance Difference】
        DDiff = np.abs(np.mean(DistanceA, axis=1) - np.mean(DistanceB, axis=1))
        # 找出T_IterativeResolution倍的成对关键，输出其索引
        Indexed_DDiffValue = [(Value, Index) for Index, Value in enumerate(DDiff)]  # 创建带有索引的元组列表
        Index_GoodDD = [Index for (Value, Index) in heapq.nsmallest(int(T_IterativeResolution*DDiff.shape[0]), Indexed_DDiffValue, key=lambda x: x[0])]
        # 更新迭代所需的数据
        KeypointA = KeypointA[Index_GoodDD]
        KeypointB = KeypointB[Index_GoodDD]
        # 关键点数目【KeypointA与KeypointB的关键点数目始终相同】
        Num_GoodKP = KeypointA.shape[0]
        Loop += 1  # 统计迭代次数
        # 判定是否结束迭代,并准备输出值
        if Num_GoodKP < T_NumKP:
            # 两点设计:
            # 防止误匹配,使用EstimateAffineTransformation()函数
            # 计算引导匹配的关键点和变换矩阵【注意：这里的magnification是需要引导的图与当前图的尺寸倍数】
            KPS_A, KPS_B, Aff_M, Mask = EstimateAffineTransformation(Magnification*KeypointA, Magnification*KeypointB) # 迭代结束后使用变换矩阵是为了防止迭代结束后可能存在误匹配，尽管这种情况极小
            KeypointA, KeypointB = KPS_A / Magnification, KPS_B / Magnification
            Num_GoodKP = KeypointA.shape[0]
            break
    # 报告迭代信息
    E_End = time.time()
    print(f'全局迭代信息: 共计迭代{Loop}次,耗时{E_End-E_Start:.3f}秒，从暴力匹配中，迭代产生{Num_GoodKP}对关键点')
    return KeypointA, KeypointB, Aff_M

def GeometricConsistencyForRegion(KeypointA, KeypointB):
    """
        基于几何差异，迭代出几何一致的成对的关键
    """
    # 输入:
    # KeypointA, KeypointB: Numpy数据格式,各行一一对应
    #
    # 初始化参数
    T_NumKP = 100                   # 控制最后一次成对关键点的数目
    T_IterativeResolution = 0.7     # 该值属于[0~1],取0与1非法,值越小迭代次数越少，结果越不稳定
    T_NearSearch_InRegion = 150     # 在Region中设置各点搜索最近点的至少数目【不怕设置大就怕设置小,要比T_NumKP值大】
    Loop = 2                        # 设置迭代次数，迭代次数越多结果越好，但输出的成对点数目越少
    # 迭代
    Num_NearSearch_InRegion = KeypointA.shape[0]
    for I in range(Loop):
        # 遍历搜索，获得关键点之间的距离
        DistanceA, _ = NearKeypoint(KeypointA, Num_NearSearch_InRegion, KeypointA)
        DistanceB, _ = NearKeypoint(KeypointB, Num_NearSearch_InRegion, KeypointB)
        # 几何差异【DD: Distance Difference】
        DDiff = np.abs(np.sum(DistanceA, axis=1) - np.sum(DistanceB, axis=1))
        # 找出T_IterativeResolution倍的成对关键，输出其索引
        Indexed_DDiffValue = [(Value, Index) for Index, Value in enumerate(DDiff)]  # 创建带有索引的元组列表
        Index_GoodDD = [Index for (Value, Index) in heapq.nsmallest(int(T_IterativeResolution*DDiff.shape[0]), Indexed_DDiffValue)]
        # 更新迭代所需的数据
        KeypointA = KeypointA[Index_GoodDD]
        KeypointB = KeypointB[Index_GoodDD]
        # 关键点数目【KeypointA与KeypointB的关键点数目始终相同】
        Num_GoodKP = KeypointA.shape[0]
        # 判定是否结束迭代
        if Num_GoodKP < T_NumKP:
            break
        # 参数修订
        if T_NearSearch_InRegion < Num_GoodKP:
            Num_NearSearch_InRegion = int(0.5 * Num_GoodKP)  # 总数目的一半
        else:
            Num_NearSearch_InRegion = Num_GoodKP
    return KeypointA, KeypointB

def LastGeometricConsistency(KeypointA, KeypointB):
    """
        基于几何差异，迭代出几何一致的成对的关键
    """
    # 输入:
    # KeypointA, KeypointB: Numpy数据格式,各行一一对应
    #
    # 初始化参数
    T_IterativeResolution = 0.50  # 0.5-0.8该值属于[0~1],取0与1非法, 从成对的关键点中保留几何一致的前90%个
    #
    # 搜索范围
    Num_NearSearch_InRegion = KeypointA.shape[0] // 9  # 9宫格范围
    # 遍历搜索，获得关键点之间的距离
    DistanceA, _ = NearKeypoint(KeypointA, Num_NearSearch_InRegion, KeypointA)
    DistanceB, _ = NearKeypoint(KeypointB, Num_NearSearch_InRegion, KeypointB)
    # 几何差异【DD: Distance Difference】
    DDiff = np.abs(np.sum(DistanceA, axis=1) - np.sum(DistanceB, axis=1))
    # 找出T_IterativeResolution倍的成对关键，输出其索引
    Indexed_DDiffValue = [(Value, Index) for Index, Value in enumerate(DDiff)]  # 创建带有索引的元组列表
    Index_GoodDD = [Index for (Value, Index) in heapq.nsmallest(int(T_IterativeResolution*DDiff.shape[0]), Indexed_DDiffValue)]
    # 更新迭代所需的数据
    KeypointA = KeypointA[Index_GoodDD]
    KeypointB = KeypointB[Index_GoodDD]
    print(f'在全局几何一致性约束中，匹配结果包含{KeypointA.shape[0]}个关键点，这是最终的匹配结果')
    return KeypointA, KeypointB

def DivideRegion(Point):
    """
        根据点坐标位置划分区域
    """
    # 输入:
    # Point: Numpy数据格式
    #
    # 初始化参数
    T = 0.3              # T取值范围[0-1]，取0或1非法，中心点周围1/3的点在圆域中
    Num = Point.shape[0]
    # 中心点
    Center = np.float32([np.mean(Point, axis=0, dtype=np.float32)])
    # 圆形域【在近似圆形域中查找中心点周围的点的索引】
    _, IndexCircle = NearKeypoint(Point, int(T*Num), Center)
    # 四分域【依据四象限划分】
    Quadrant = Point - Center
    Index1 = Quadrant[:, 0] > 0  # H_Y坐标
    Index2 = ~Index1
    Index3 = Quadrant[:,1] > 0   # W_X坐标
    Index4 = ~Index3
    # 划分的区域[索引对应输入的Point]
    Index_Plus_Plus = Index2 & Index3
    Index_Minus_Plus = Index2 & Index4
    Index_Minus_Minus = Index1 & Index4
    Index_Plus_Minus = Index1 & Index3
    Index_Region = {'Center':IndexCircle[0],'Plus_Plus':Index_Plus_Plus, 'Minus_Plus':Index_Minus_Plus, 'Minus_Minus':Index_Minus_Minus, 'Plus_Minus':Index_Plus_Minus}
    return Index_Region

def KeypintInRegion(PointA_NoRep, PointB_NoRep):
    """
        构建重叠区域，并索引出各个重叠区域内的成对关键点
    """
    # 划分关键点区域【注意使用降重后的成对关键点才正确】
    Index_D = DivideRegion(PointA_NoRep)
    Region_Middle = Index_D['Center']
    Region_Up = Index_D['Plus_Plus'] | Index_D['Minus_Plus']
    Region_Down = Index_D['Minus_Minus'] | Index_D['Plus_Minus']
    Region_Lift = Index_D['Minus_Plus'] | Index_D['Minus_Minus']
    Region_Right = Index_D['Plus_Plus'] | Index_D['Plus_Minus']
    # 准备各区域中的成对的关键点【区域重叠使得成对的关键点大量重复】
    PointA_MR = PointA_NoRep[Region_Middle]  # PointA_InMiddleRegion
    PointA_UR = PointA_NoRep[Region_Up]      # PointA_InUpRegion
    PointA_DR = PointA_NoRep[Region_Down]    # PointA_InDownRegion
    PointA_LR = PointA_NoRep[Region_Lift]    # PointA_InLiftRegion
    PointA_RR = PointA_NoRep[Region_Right]   # PointA_InRightRegion
    PointB_MR = PointB_NoRep[Region_Middle]  # PointB_InMiddleRegion
    PointB_UR = PointB_NoRep[Region_Up]      # PointB_InUpRegion
    PointB_DR = PointB_NoRep[Region_Down]    # PointB_InDownRegion
    PointB_LR = PointB_NoRep[Region_Lift]    # PointB_InLiftRegion
    PointB_RR = PointB_NoRep[Region_Right]   # PointB_InRightRegion
    PointA = {'MR':PointA_MR, 'UR':PointA_UR, 'DR':PointA_DR, 'LR':PointA_LR, 'RR':PointA_RR}
    PointB = {'MR':PointB_MR, 'UR':PointB_UR, 'DR':PointB_DR, 'LR':PointB_LR, 'RR':PointB_RR}
    
    return PointA, PointB

def foo2(eigens_1, eigens_2, index_2):
    """Memory friendly function"""
    @ti.kernel
    def _foo2(eigens_1: ti.types.ndarray(), eigens_2: ti.types.ndarray(), index_2: ti.types.ndarray(), result: ti.types.ndarray()):
        for i, j in ti.ndrange((0, result.shape[0]), (0, result.shape[1])):
            coor = index_2[i, j]
            sum_value = ti.cast(0, ti.f64)
            for k in range(0, eigens_2.shape[1]):
                sum_value += ti.pow((eigens_1[i, k] - eigens_2[coor, k]), 2)
            
            result[i, j] = ti.sqrt(sum_value)

        return

    result = np.zeros_like(index_2, dtype=np.float64)
    _foo2(eigens_1, eigens_2, index_2, result)
    
    return result

def LocalMatching(PointA, EncodeA, TransMat, PointB, EncodeB, W_LocalRegionSize=0.2):
    """
        在局部区域内【局部区域是指映射坐标周围的区域，该区域由指定数目的点划定，与四象限和圆域无关】执行点匹配
    """
    # 输入:
    # PointA, PointB: Numpy数据格式
    # TransMat: 变换矩阵,PointA向PointB方向映射
    # EncodeA, EncodeB: Numpy数据格式,分别是PointA,PointB的描述子向量
    #
    # 输出:
    # MatchPointA, MatchPointB: Numpy数据格式,局部匹配结果,行向一一对应
    #
    # 初始化参数
    # W_LocalRegionSize = 0.02       # 使用一定比例的点数目近似局部区域的尺寸，局部区域近似为目标区域的10%大小
    Num_PointB = PointB.shape[0]   # 点数目
    Num_NearPoint = int(W_LocalRegionSize*Num_PointB)  # 在每个映射坐标周围找出Num个最近的关键点
    print(f'在局部匹配中，设置局部区域尺寸权重为{W_LocalRegionSize}时，使得每个映射坐标所在的局部区域包含{Num_NearPoint}个关键点')
    #
    # 转变关键点的坐标【从关键点少的向关键点多的方向映射坐标】
    Point_Map = TransformPoint(PointA, TransMat)
    # 搜索映射坐标周围的关键点索引
    Distance, Index_PointB_Near = NearKeypoint(PointB, Num_NearPoint, Point_Map)  # Num_NearPoint
    # ++映射坐标与其周围关键点的相似性
    Similarity_Around = foo2(EncodeA, EncodeB, Index_PointB_Near)
    # 获得局部最相似索引
    Index_Similar_Around = np.argmin(Similarity_Around, axis=1)
    # 依据局部最相似，索引出与KPA粗匹配的KPB
    Index_PointB_Around = Index_PointB_Near[range(Index_Similar_Around.shape[0]), Index_Similar_Around]
    # 降重
    Index_PointA, Index_PointB = RemoveRepetition_Local(Index_PointB_Around, Similarity_Around, Index_Similar_Around, np.arange(PointA.shape[0]))
    PointA_NoRep = PointA[Index_PointA]
    PointB_NoRep = PointB[Index_PointB]
    print(f'局部粗匹配获得{Index_PointA.shape[0]}个成对的关键点,此时还未执行基于区域的几何一致性评估')
    # 区域内成对的关键点
    PA, PB = KeypintInRegion(PointA_NoRep, PointB_NoRep)
    # 在划分区域的内分别进行几何一致性评估
    PointA_MR, PointB_MR = GeometricConsistencyForRegion(PA['MR'], PB['MR'])
    PointA_UR, PointB_UR = GeometricConsistencyForRegion(PA['UR'], PB['UR'])
    PointA_DR, PointB_DR = GeometricConsistencyForRegion(PA['DR'], PB['DR'])
    PointA_LR, PointB_LR = GeometricConsistencyForRegion(PA['LR'], PB['LR'])
    PointA_RR, PointB_RR = GeometricConsistencyForRegion(PA['RR'], PB['RR'])
    # 降重
    MatchPointA = np.vstack((PointA_MR, PointA_UR, PointA_DR, PointA_LR, PointA_RR))
    MatchPointB = np.vstack((PointB_MR, PointB_UR, PointB_DR, PointB_LR, PointB_RR))
    MatchPointA, Index_NoRepKP = np.unique(MatchPointA, axis=0, return_index=True)
    MatchPointB = MatchPointB[Index_NoRepKP]
    print(f'局部精匹配获得{Index_NoRepKP.shape[0]}个成对的关键点,这是执行基于区域的几何一致性评估后的结果')
    return MatchPointA, MatchPointB

def DoLocalMatching(KeypointA, EncodeA, TransMat, KeypointB, EncodeB, W_LocalRegionSize):
    """
        执行局部匹配
    """
    # 初始化数据【判断KeypointA和KeypointB的数目哪个更少】
    if KeypointA.shape[0] <= KeypointB.shape[0]:
        MatchPointA, MatchPointB = LocalMatching(KeypointA, EncodeA, TransMat, KeypointB, EncodeB, W_LocalRegionSize)
    else:
        MatchPointB, MatchPointA = LocalMatching(KeypointB, EncodeB, cv2.invertAffineTransform(TransMat), KeypointA, EncodeA, W_LocalRegionSize)  # 逆变换与转置
    return MatchPointA, MatchPointB

def Matching_Origin(KeypointA, EncodeA, KeypointB, EncodeB, Magnification):
    """
        依据关键点及其描述子向量，计算匹配结果
        【提示】如果结果不理想，考虑优先修改GeometricConsistencyForRegion()函数的参数
    """
    # 输入：
    # KeypointA, EncodeA, KeypointB, EncodeB: Numpy数据格式
    # KeypointA, KeypointB: [n*2],n与关键点的数目相等
    # EncodeA, EncodeB: [n*m],m与特征向量[1*m]的长度相等
    # Magnification: 引导匹配与当前关键点，他们所对应的图像尺寸放大倍数
    # 输出:
    # Transform: 变换矩阵,根据原始点描述子数据获得,但是属于引导匹配的变换矩阵,不能用于变换原始点描述子
    #
    # 给KPA坐标编号
    Order_KeypointA = np.arange(KeypointA.shape[0])
    # 计算对应于KPA和KPB的EncodeA, EncodeB之间的全相似性强度矩阵【便于索引】
    Similarity = Euclidean(EncodeA.astype(np.float32), EncodeB.astype(np.float32))
    # 基于相似性，索引出与KPA粗匹配的KPB
    Index_Similar = np.argmin(Similarity, axis=1)
    # 降重
    Index_KeypointA, Index_KeypointB = RemoveRepetition(Index_Similar, Similarity, Order_KeypointA)
    # 几何一致性匹配
    KeypointA_GC, KeypointB_GC, Transform = GeometricConsistency_Guide(KeypointA[Index_KeypointA], KeypointB[Index_KeypointB], Magnification)
    return KeypointA_GC, KeypointB_GC, Transform

def Matching_Guide(Transform, KeypointA, EncodeA, KeypointB, EncodeB, W_LocalRegionSize):
    """
        依据关键点及其描述子向量，计算匹配结果
        【提示】如果结果不理想，考虑优先修改GeometricConsistencyForRegion()函数的参数
    """
    # 输入：
    # Transform: 已知KeypointA和KeypointB的变换矩阵
    # KeypointA, EncodeA, KeypointB, EncodeB: Numpy数据格式
    # KeypointA, KeypointB: [n*2],n与关键点的数目相等
    # EncodeA, EncodeB: [n*m],m与特征向量[1*m]的长度相等
    # 输出:
    # # MatchResultA, MatchResultB: Numpy数据格式,匹配结果,行向一一对应
    # Transform: 刚性变换矩阵,根据原始点描述子数据获得
    #
    # 局部匹配
    MatchPointA, MatchPointB = DoLocalMatching(KeypointA, EncodeA, Transform, KeypointB, EncodeB, W_LocalRegionSize)
    # 估计变换矩阵
    MatchResultA, MatchResultB, Transform, Mask = EstimateAffineTransformation(MatchPointA, MatchPointB)
    return MatchResultA, MatchResultB, Transform


def Matching_TwoMapping(KeypointA_Origin, EncodeA_Origin, KeypointB_Origin, EncodeB_Origin, Magnification,
                          KeypointA_Guide1, EncodeA_Guide1, KeypointB_Guide1, EncodeB_Guide1):
    """
        实施三级映射匹配, 使用函数Matching_Origin()和Matching_Guide()实现
        第一级获得变换矩阵, 第二级根据第一级变换矩阵提升匹配结果, 第三级根据第二级变换矩阵匹配大尺寸图像, 最终获得精确匹配结果
    """
    """
    KeypointA_Origin：HE小图的关键点检测结果
    EncodeA_Origin：HE小图的特征描述子
    KeypointB_Origin：IHC大小图的关键点检测结果
    EncodeB_Origin：IHC小图的特征描述子
    Magnification：大图与小图之间的缩放比例
    KeypointA_Guide1：HE大图的关键点检测结果
    EncodeA_Guide1：HE大图的特征描述子
    KeypointB_Guide1：IHC大图的关键点检测结果
    EncodeB_Guide1：IHC大图的特征描述子
    """
    # 输入：
    # 参考Matching_Origin()和Matching_Guide()的解释
    # 输出：
    # :匹配结果
    #
    ti.init(arch=ti.gpu)
    E_Start = time.time()
    # 实施第一级起点匹配
    ResultA_Origin, ResultB_Origin, Transform_Origin = Matching_Origin(KeypointA_Origin, EncodeA_Origin, KeypointB_Origin, EncodeB_Origin, Magnification)
    # 实施第二级引导匹配
    ResultA_Guide2, ResultB_Guide2 = DoLocalMatching(KeypointA_Guide1, EncodeA_Guide1, Transform_Origin, KeypointB_Guide1, EncodeB_Guide1, W_LocalRegionSize=0.02)
    # 在获得匹配结果之前最后一次要求全局几何一致性
    MatchResultA, MatchResultB = LastGeometricConsistency(ResultA_Guide2, ResultB_Guide2)
    #
    E_End = time.time()
    print(f'三级映射匹配: 总耗时{E_End - E_Start:.3f}秒')
    return ResultA_Origin, ResultB_Origin, ResultA_Guide2, ResultB_Guide2, MatchResultA, MatchResultB

if __name__ == '__main__':
    import pickle
    with open("data.pkl", "rb") as file:
        kps_1_small, eigens_1_small, kps_2_small, eigens_2_small, ratio, kps_1_large, eigens_1_large, kps_2_large, eigens_2_large = pickle.load(file)
    
    _, _, _, _, match_kps_1, match_kps_2 = Matching_TwoMapping(kps_1_small, 
                                                               eigens_1_small, 
                                                               kps_2_small, 
                                                               eigens_2_small, 
                                                               ratio, 
                                                               kps_1_large, 
                                                               eigens_1_large, 
                                                               kps_2_large, 
                                                               eigens_2_large)
    print(match_kps_1.shape)
    print(match_kps_1.shape)
