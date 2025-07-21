import cv2
import numpy as np
import os
import json
def make_color_wheel(bins=None):
    """Build a color wheel.

    Args:
        bins(list or tuple, optional): Specify the number of bins for each
            color range, corresponding to six ranges: red -> yellow,
            yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
            magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
            (see Middlebury).

    Returns:
        ndarray: Color wheel of shape (total_bins, 3).
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins) 

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T
def flow2rgb_3d(flow, color_wheel=None, unknown_thr=1e6):
    assert flow.ndim == 3 and flow.shape[-1] == 3
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()
    dz = flow[:, :, 2].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | np.isnan(dz) |
        (np.abs(dx) > unknown_thr) | (np.abs(dy) > unknown_thr) | (np.abs(dz) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0
    dz[ignore_inds] = 0

    rad_xy = np.sqrt(dx**2 + dy**2)
    rad_z = np.sqrt(dz**2)
    if np.any(rad_xy > np.finfo(float).eps):
        max_rad_xy = np.max(rad_xy)
        dx /= max_rad_xy
        dy /= max_rad_xy

    if np.any(rad_z > np.finfo(float).eps):
        max_rad_z = np.max(rad_z)
        dz /= max_rad_z

    rad = np.sqrt(dx**2 + dy**2 + dz**2)
    angle_xy = np.arctan2(-dy, -dx) / np.pi
    angle_z = np.arctan2(-dz, np.sqrt(dx**2 + dy**2)) / np.pi

    bin_real_xy = (angle_xy + 1) / 2 * (num_bins - 1)
    bin_real_z = (angle_z + 1) / 2 * (num_bins - 1)
    bin_left_xy = np.floor(bin_real_xy).astype(int)
    bin_left_z = np.floor(bin_real_z).astype(int)
    bin_right_xy = (bin_left_xy + 1) % num_bins
    bin_right_z = (bin_left_z + 1) % num_bins
    w_xy = (bin_real_xy - bin_left_xy.astype(np.float32))[..., None]
    w_z = (bin_real_z - bin_left_z.astype(np.float32))[..., None]
    flow_img = (1 - w_xy) * color_wheel[bin_left_xy, :] + w_xy * color_wheel[bin_right_xy, :]
    flow_img += (1 - w_z) * color_wheel[bin_left_z, :] + w_z * color_wheel[bin_right_z, :]
    flow_img /= 2

    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img
def flow3d(path):
    apex = cv2.imread(os.path.join(path,'crop_color','apex.jpg'))
    x_img = cv2.imread(os.path.join(path,'direction','x_flow_stand_nomove.png'))
    y_img = cv2.imread(os.path.join(path,'direction','y_flow_stand_nomove.png'))
    z_img = cv2.imread(os.path.join(path,'direction','z_flow_stand_nomove.png'))
    flow_img = np.zeros((apex.shape[0],apex.shape[1],3))
    flow_img[:,:,0] += x_img[:,:,0]
    flow_img[:,:,0] -= x_img[:,:,1]
    flow_img[:,:,1] += y_img[:,:,0]
    flow_img[:,:,1] -= y_img[:,:,1]
    flow_img[:,:,2] += z_img[:,:,0]
    flow_img[:,:,2] -= z_img[:,:,1]
    img_3d = flow2rgb_3d(flow_img)
    cv2.imwrite(os.path.join(path,'flow','flow3d.jpg'),img_3d*255.)
    new_img = img_3d*100. + apex
    cv2.imwrite(os.path.join(path,'flow','new_img.jpg'), cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX))
def process_data(origin_path,root_path):
    with open(os.path.join(origin_path,root_path,"exp7_data_"+root_path+"_cropped"+".json"), 'r') as f:
        data = json.load(f)
    for name in data:
        print(name)
        root_ = name.split("_")
        path = os.path.join(root_[0],root_[1],root_[2]+"-"+root_[3])
        try:
            flow3d(os.path.join(origin_path,root_path,path))
        except Exception as e:
            print(e)
            continue

if __name__ == '__main__':
    for ty in ['macro','micro']:
        process_data('/home/disk2_12t/DATASET/MEGC/CASME3_PART/Crop_new_face',ty)
    # flow3d(r'F:\code\n176c2327\n1d250')