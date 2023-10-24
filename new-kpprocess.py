
import numba
import pickle
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _get_kp(img_up, 
            img_self, 
            img_down):
    img_height, img_width = img_self.shape[:2]
    result = np.zeros_like(img_self)
    for i in prange(1, img_height):
        for j in prange(1, img_width):
            img_up_part = img_up[i-1:i+2, j-1:j+2]
            img_down_part = img_down[i-1:i+2, j-1:j+2]
            img_self_part = img_self[i-1:i+2, j-1:j+2]
            img_up_max = np.max(img_up_part)
            img_up_min = np.min(img_up_part)
            img_down_max = np.max(img_down_part)
            img_down_min = np.min(img_down_part)
            
            pixel = img_self[i, j]
            temp_max = pixel == np.max(img_self_part)
            temp_min = pixel == np.min(img_self_part)
            
            flag_max = temp_max and (pixel > img_up_max) and (pixel > img_down_max)
            flag_min = temp_min and (pixel < img_up_min) and (pixel < img_down_min)

            if (flag_max):
                result[i, j] = 1
            elif (flag_min):
                result[i, j] = -1
            
    return result


def get_keypoint_list(diff_list):
    keypoint_list = list()
    diff_depth = len(diff_list)
    # 对除了底部和顶部的两张图片外的，开始找极值点
    for i in range(1, diff_depth - 1):
        # 获取上下和自己一共三张图像
        img_self = diff_list[i]
        img_up = diff_list[i + 1]
        img_low = diff_list[i - 1]
        
        kps = _get_kp(img_self, img_up, img_low)
        keypoint_list.append(kps)
        
        print(f"Detecting level {i}")
        
    return keypoint_list

# Test case
if __name__ == "__main__":
    with open("diff_list.pkl", "rb") as file:
        diff_list = pickle.load(file)
        get_keypoint_list(diff_list)
        