# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import cv2
def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 -
                w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img


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
class point_prosses():
    def __init__(self,origin_path, focal_length=1324.65, scalingfactor=1000):
        self.mae_path = r'cas(me)3_part_A_MaE_label_JpgIndex_v2_emotion.xlsx'
        self.me_path = r'cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'
        self.origin_path = origin_path
        self.focal_length = focal_length
        self.scalingfactor = scalingfactor
    def del_zero(self, depon_image):
        h,w = depon_image.shape 
        tep_de  = depon_image.reshape(-1)
        for i in range(len(tep_de)):
            if tep_de[i] == 0:
                tep_de[i] = tep_de[i-1]
        return tep_de.reshape((h,w))
    def del_fewpoint(self, img,n=150):
        if n > 0:   
            nt,bins,patchs = plt.hist(img, bins=11, color=sns.desaturate("indianred", .8), alpha=.4)
            # nt1,bins1,patchs1 = plt.hist(img, bins=11, color=sns.desaturate("indianred", .8), alpha=.4)
            
            for i in range(len(nt)):
                if nt[i] <n:
                    img[(img >= bins[i]) & (img< bins[i+1]+0.0001)] = bins[i-1] - 48 
            if nt[-1] <n:
                    img[(img >= bins[-2]) & (img< bins[-1]+1)] = bins[-2]-48
                    # import pdb;pdb.set_trace()
            img  = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).reshape(-1)
            # n -= 1
            # img = self.del_fewpoint(img,n)
            return img
        else:
            return img
    def del_out_point(self,pos,color, on_img = np.zeros(1)):
        df = np.asarray(pos)
        color = np.asarray(color)
        non_zero_rows = df[np.any(df != 0, axis=1)]
        x_mean,y_mean,z_mean = np.mean(non_zero_rows[:,0:1]),np.mean(non_zero_rows[:,1:2]),np.mean(non_zero_rows[:,2:3])
        total_dis = 0
        for i in df:
            total_dis = total_dis + ((i[0]-x_mean)**2+(i[1]-y_mean)**2+(i[2]-z_mean)**2)**(1/2)
        mean_dis = total_dis/len(df[:,0:1])
        del_list = []
        for i in range(len(df[:,0:1])): 
            if np.any(df[i] != 0):
                if ((df[i,0:1]-x_mean)**2+(df[i,1:2]-y_mean)**2+(df[i,2:3]-z_mean)**2)**(1/2) > 1.5*mean_dis:
                    del_list.append(i)
        df[del_list] = np.zeros(3)
        pose_all = df
        color[del_list] = np.zeros(3)
        color_all = color
        df = np.delete(df, del_list, axis=0)
        color = np.delete(color, del_list, axis=0)
        if on_img.all() != np.zeros(1):
            on_img = np.delete(on_img, del_list, axis=0)
        return df,color,on_img,pose_all,color_all
    
    def move_face(self,point,of,diff, max_list):        
        stand_onset_point = self.calculate_just_point(point)[:,:3]
        new_of = cv2.resize(of, (point.shape[1],point.shape[0]), interpolation=cv2.INTER_NEAREST)
        new_diff = cv2.resize(diff, (point.shape[1],point.shape[0]), interpolation=cv2.INTER_NEAREST)
        new_point = np.zeros(stand_onset_point.shape)
        new_point[:,0] = stand_onset_point[:, 0] + new_of.reshape(-1,2)[:, 0] * max_list[0] /1000
        new_point[:,1] = stand_onset_point[:, 1] + new_of.reshape(-1,2)[:, 1] * max_list[1] /1000
        new_point[:,2] = stand_onset_point[:, 2] + new_diff.reshape(-1) * max_list[2] /1000
        return new_point, new_of, new_diff
    def calculate_just_point(self, depth_file):
        self.width = depth_file.shape[1]
        self.height = depth_file.shape[0]
        depth = np.asarray(depth_file).T
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - self.width / 2) * self.Z) / self.focal_length
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - self.height / 2) * self.Z) / self.focal_length

        df=np.zeros((6,self.width*self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = -self.Y.T.reshape(-1)
        df[2] = -self.Z.T.reshape(-1)  
        
        return df.T
    def calculate_point(self,rgb, depth_file):
        self.width = depth_file.shape[1]
        self.height = depth_file.shape[0]
        self.rgb = rgb
        depth = np.asarray(depth_file).T
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - self.width / 2) * self.Z) / self.focal_length
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - self.height / 2) * self.Z) / self.focal_length

        df=np.zeros((6,self.width*self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = -self.Y.T.reshape(-1)
        df[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        df[3] = img[:,  0:1].reshape(-1)
        df[4] = img[:,  1:2].reshape(-1)
        df[5] = img[:,  2:3].reshape(-1)      
        
        return df.T
    def adjust_gamma(self, image, gamma=1.0):
        invgamma = 1/gamma
        brighter_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
        return brighter_image
    def gen_7dict(self):
        micro_dict = {}
        macro_dict = {}
        mic = pd.read_excel(io=self.me_path) # 读取数据集的标签
        exp_mic_dict = {'anger':0,'happy':0, 'disgust':0,'fear':0,'sad':0,'surprise':0,'others':0}
        mic_emo = {'anger':0,'happy':1, 'disgust':2,'fear':3,'sad':4,'surprise':5,'others':6}
        for _, row in mic.iterrows():
            color_onset = os.path.join(self.origin_path,row[0],row[1],'color',str(row[2])+'.jpg')
            color_apex = os.path.join(self.origin_path,row[0],row[1],'color', str(row[3])+'.jpg')
            depth_onset = os.path.join(self.origin_path,row[0],row[1],'depth', str(row[2])+'.png')
            depth_apex = os.path.join(self.origin_path,row[0],row[1],'depth', str(row[3])+'.png')
            if row[2] != row[3] and os.path.exists(os.path.join(color_onset)) and os.path.exists(os.path.join(color_apex)) and os.path.exists(os.path.join(depth_onset)) and os.path.exists(os.path.join(depth_apex)):
                exp_mic_dict[row[7].lower()] += 1
                micro_dict[str(row[0])+'_'+str(row[1])+'_'+str(row[2])+'_'+str(row[3])] = mic_emo[row[7].lower()]
        import pdb;pdb.set_trace()
        mac = pd.read_excel(io=self.mae_path) # 读取数据集的标签
        exp_mac_dict = {'anger':0,'happiness':0, 'disgust':0,'fear':0,'sadness':0,'surprise':0,'others':0}
        mac_emo = {'anger':0,'happiness':1, 'disgust':2,'fear':3,'sadness':4,'surprise':5,'others':6}
        for _, row in mac.iterrows():
            color_onset = os.path.join(self.origin_path,row[0],row[1],'color',str(row[2])+'.jpg')
            color_apex = os.path.join(self.origin_path,row[0],row[1],'color', str(row[3])+'.jpg')
            depth_onset = os.path.join(self.origin_path,row[0],row[1],'depth', str(row[2])+'.png')
            depth_apex = os.path.join(self.origin_path,row[0],row[1],'depth', str(row[3])+'.png')
            if row[2] != row[3] and os.path.exists(os.path.join(color_onset)) and os.path.exists(os.path.join(color_apex)) and os.path.exists(os.path.join(depth_onset)) and os.path.exists(os.path.join(depth_apex)):
                exp_mac_dict[row[6].lower()] += 1
                macro_dict[str(row[0])+'_'+str(row[1])+'_'+str(row[2])+'_'+str(row[3])] = mac_emo[row[6].lower()]
        print(exp_mac_dict)
        print(exp_mic_dict)
        return micro_dict, macro_dict
    def nor_and_save(self, pos_crop,img_crop , files_path, save_name):
        pos_crop1  = cv2.normalize(pos_crop, None,-1,1,cv2.NORM_MINMAX)
        img_crop1  = cv2.normalize(img_crop, None,-1,1,cv2.NORM_MINMAX)
        # img_crop1  = img_crop
        diff_positive_xyz_color= o3d.geometry.PointCloud()

        diff_positive_xyz_color.points = o3d.utility.Vector3dVector(np.asarray(pos_crop1))
        diff_positive_xyz_color.colors = o3d.utility.Vector3dVector(np.asarray(img_crop1)*np.where(img_crop1>0,1,0))
        #保存点云	
        os.makedirs(os.path.join(files_path,'ply'),exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(files_path,'ply','diff_positive'+save_name+'.ply'), diff_positive_xyz_color)
        #negative_xyz_color
        diff_negative_xyz_color= o3d.geometry.PointCloud()

        diff_negative_xyz_color.points = o3d.utility.Vector3dVector(np.asarray(pos_crop1))
        diff_negative_xyz_color.colors = o3d.utility.Vector3dVector(-np.asarray(img_crop1)*np.where(img_crop1<0,1,0))
        #保存点云
        o3d.io.write_point_cloud(os.path.join(files_path,'ply','diff_negative'+save_name+'.ply'), diff_negative_xyz_color)
        pos_crop2 = cv2.normalize(pos_crop1,None,-1,1,cv2.NORM_MINMAX)
        os.makedirs(os.path.join(files_path,'txt'),exist_ok=True)
        np.savetxt(os.path.join(files_path,'txt','diff_'+save_name+'.txt'), np.hstack((pos_crop2,img_crop1)), fmt='%f', delimiter=',')
    def get_image_hull_mask(self, image_shape, image_landmarks, ie_polys=None):
        # get the mask of the image
        if image_landmarks.shape[0] != 68:
            raise Exception(
                'get_image_hull_mask works only with 68 landmarks')
        int_lmrks = np.array(image_landmarks, dtype=np.int)

        #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
        hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[0:9],
                            int_lmrks[17:18]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[8:17],
                            int_lmrks[26:27]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:20],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[24:27],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[19:25],
                            int_lmrks[8:9],
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:22],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[22:27],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        # nose
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

        if ie_polys is not None:
            ie_polys.overlay_mask(hull_mask)
        y0 = int((int_lmrks[24][1]+int_lmrks[46][1])*0.6)
        y1 = int_lmrks[30][1]
        hull_mask[y0:y1,:] = 0
        return hull_mask
    def reduce_landmarks(self, imgshape, landmarks):
        mid = [imgshape[0]/2,imgshape[1]/2]
        for i,xy  in enumerate(landmarks[:27]):
            if xy[0] > mid[0]:
                landmarks[i,0] = xy[0] - 7
            else:
                landmarks[i,0] = xy[0] + 7
            if xy[1] > mid[1]:
                landmarks[i,1] = xy[1] - 4
        # for i,xy  in enumerate(landmarks[:2]):            
        #     landmarks[i,0] = xy[0] + 3
        # for i,xy  in enumerate(landmarks[15:17]):            
        #     landmarks[i,0] = xy[0] - 3 
        # for i,xy  in enumerate(landmarks[17:27]):
        #     if xy[0] > mid[0]:
        #         landmarks[i,0] = xy[0] - 3
        #     else:
        #         landmarks[i,0] = xy[0] + 3  
        return landmarks
def flow3d(path):
    pp = point_prosses('')
    flow = np.load(os.path.join(path,'flow','afflow1.npy'))
    flow[:,:,0] = flow[:,:,0]/abs(flow[:,:,0]).max()
    flow[:,:,1] = flow[:,:,1]/abs(flow[:,:,1]).max()
    depth_onset = cv2.imread(os.path.join(path,'crop_depth','onset.png'), cv2.CV_16UC1)
    depth_apex = cv2.imread(os.path.join(path,'crop_depth','apex.png'), cv2.CV_16UC1)
    # import pdb;pdb.set_trace()
    new_dep_on = depth_onset.astype(np.float64)
    new_dep_ap = depth_apex.astype(np.float64)
    diff_img = (new_dep_ap - new_dep_on)/abs((new_dep_ap - new_dep_on)).max()
    print(diff_img.shape)
    z_flow = np.zeros((flow.shape))
    z_flow[:,:,0] = pp.del_fewpoint(pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(diff_img*np.where(diff_img>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)).reshape(diff_img.shape)
    z_flow[:,:,1] = pp.del_fewpoint(pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-diff_img*np.where(diff_img<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)).reshape(diff_img.shape)
    
    flow_img = np.zeros((diff_img.shape[0],diff_img.shape[1],3))
    flow_img[:,:,0] = flow[:,:,0] 
    flow_img[:,:,1] = flow[:,:,1] 
    flow_img[:,:,2] += cv2.normalize(z_flow[:,:,0],None,0,diff_img.max(),cv2.NORM_MINMAX)
    flow_img[:,:,2] += cv2.normalize(-z_flow[:,:,1],None,diff_img.min(),0,cv2.NORM_MINMAX)
    print(flow_img[:,:,0].max(),flow_img[:,:,0].min())
    print(flow_img[:,:,1].max(),flow_img[:,:,1].min())
    print(flow_img[:,:,2].max(),flow_img[:,:,2].min())
    img = flow2rgb(flow)
    img_3d = flow2rgb_3d(flow_img)
    print(img.shape)
    cv2.imwrite(os.path.join(path,'flow','flow2d.jpg'),img*255.)
    cv2.imwrite(os.path.join(path,'flow','flow3d.jpg'),img_3d*255.)
    pc = pp.calculate_point(img_3d,new_dep_ap)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:,3:6])
    o3d.io.write_point_cloud(os.path.join(path,'flow','flow3d.ply'), pcd)
    np.savetxt(os.path.join(path,'flow','flow3d.txt'), pc, fmt='%f', delimiter=',')
    # o3d.visualization.draw_geometries([pcd])

def save_flow_3ch(path):
    print(path)
    pp = point_prosses('')
    flow = np.load(os.path.join(path,'flow','afflow1.npy'))
    depth_onset = cv2.imread(os.path.join(path,'crop_depth','onset.png'), cv2.CV_16UC1)
    depth_apex = cv2.imread(os.path.join(path,'crop_depth','apex.png'), cv2.CV_16UC1)
    new_dep_on = depth_onset.astype(np.float64)
    new_dep_ap = depth_apex.astype(np.float64)
    diff_img = (new_dep_ap - new_dep_on)*np.where(np.abs(new_dep_ap - new_dep_on)>8,0,1)
    reduced_landmark = pp.reduce_landmarks(diff_img.shape, np.load(os.path.join(path,'p68','align_p68_landmarks.npy')))
    hull_mask = pp.get_image_hull_mask(diff_img.shape, reduced_landmark)
    diff_img = diff_img*hull_mask.squeeze()
    max_list = [abs(flow[:,:,0]).max(),abs(flow[:,:,1]).max()]
    flow[:,:,0] = flow[:,:,0]/abs(flow[:,:,0]).max()*hull_mask.squeeze()
    flow[:,:,1] = flow[:,:,1]/abs(flow[:,:,1]).max()*hull_mask.squeeze()
    z_max_xy = reduced_landmark[30]
    h,w = diff_img.shape
    z_max_point = z_max_xy[1]*w + z_max_xy[0]
    try:
        print(depth_onset[z_max_xy[0],z_max_xy[1]]-8)
    except:
        return 0

    # cv2.imwrite('diff_img.png',cv2.normalize(diff_img*hull_mask.squeeze(),None,0,255,cv2.NORM_MINMAX))

    # import pdb;pdb.set_trace()
    if np.all(diff_img == 0):
        return 0
    max_list.append(abs(diff_img).max())
    diff_img = diff_img /abs((diff_img)).max()
    
    depth_onset[depth_onset[:,:] < depth_onset[z_max_xy[0],z_max_xy[1]]-8] = 0
    stand_onset_point, flow, diff_img = pp.move_face(depth_onset*hull_mask.squeeze(),flow,diff_img,max_list)
    
    color_fea = flow.reshape(-1,2)
    img3 = np.zeros((color_fea.shape[0],3))
    img3[:,0] = color_fea[:,0]
    img3[:,1] = color_fea[:,1]
    img3[:,2] = diff_img.reshape(-1)
    apex_point = pp.calculate_just_point(depth_apex*hull_mask.squeeze()) #stand_onset_point
    _,_,_,pose_all,color_all = pp.del_out_point(apex_point[:,:3],img3)
    x_flow, y_flow, z_flow = np.zeros((color_all.shape)), np.zeros((color_all.shape)), np.zeros((color_all.shape))
    z_pos = cv2.normalize(color_all[:,2]*np.where(color_all[:,2]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_neg = cv2.normalize(-color_all[:,2]*np.where(color_all[:,2]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_pos = cv2.normalize(z_pos*np.where(z_pos>204,0.8,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_neg = cv2.normalize(z_neg*np.where(z_neg>204,0.8,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(z_pos),gamma=0.3)
    z_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(z_neg),gamma=0.3)
    z_flow[:,0] = cv2.normalize(z_flow[:,0]*np.where(z_flow[:,0]>230,0.9,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_flow[:,1] = cv2.normalize(z_flow[:,1]*np.where(z_flow[:,1]>230,0.9,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    # z_flow[:,0] = cv2.normalize(z_flow[:,0]*np.where(z_flow[:,0]>200,0.6,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    # z_flow[:,1] = cv2.normalize(z_flow[:,1]*np.where(z_flow[:,1]>200,0.6,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)

    x_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(color_all[:,0]*np.where(color_all[:,0]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    x_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-color_all[:,0]*np.where(color_all[:,0]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    y_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(color_all[:,1]*np.where(color_all[:,1]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    y_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-color_all[:,1]*np.where(color_all[:,1]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    os.makedirs(os.path.join(path,'direction'),exist_ok=True)
    cv2.imwrite(os.path.join(path,'direction','x_flow_stand_nomove.png'), cv2.normalize(x_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','y_flow_stand_nomove.png'), cv2.normalize(y_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','z_flow_stand_nomove.png'), cv2.normalize(z_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    img4 = np.zeros((color_all.shape))
    img4[:,0] += cv2.normalize(x_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].max()
    img4[:,0] += cv2.normalize(-x_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].min()
    img4[:,1] += cv2.normalize(y_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].max()
    img4[:,1] += cv2.normalize(-y_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].min()
    img4[:,2] += cv2.normalize(z_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #diff_img.max()
    img4[:,2] += cv2.normalize(-z_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #diff_img.min()
    
    pp.nor_and_save(pose_all,img4,os.path.join(path),'flow3ch_stand_nomove')
    # import pdb;pdb.set_trace()
def save_flow_angel(path):
    print(path)
    pp = point_prosses('')
    flow = np.load(os.path.join(path,'flow','afflow1.npy'))
    flow[:,:,0] = flow[:,:,0]/abs(flow[:,:,0]).max()
    flow[:,:,1] = flow[:,:,1]/abs(flow[:,:,1]).max()
    depth_onset = cv2.imread(os.path.join(path,'crop_depth','onset.png'), cv2.CV_16UC1)
    depth_apex = cv2.imread(os.path.join(path,'crop_depth','apex.png'), cv2.CV_16UC1)
    new_dep_on = depth_onset.astype(np.float64)
    new_dep_ap = depth_apex.astype(np.float64)
    diff_img = (new_dep_ap - new_dep_on)*np.where(np.abs(new_dep_ap - new_dep_on)>8,0,1)
    diff_img = diff_img /abs((diff_img)).max()
    # import pdb;pdb.set_trace()
    
    color_fea = flow.reshape(-1,2)
    img3 = np.zeros((color_fea.shape[0],3))
    img3[:,0] = color_fea[:,0]
    img3[:,1] = color_fea[:,1]
    img3[:,2] = diff_img.reshape(-1)
    apex_point = pp.calculate_point(img3,depth_apex)
    _,_,_,pose_all,color_all = pp.del_out_point(apex_point[:,:3],img3)
    x_flow, y_flow, z_flow = np.zeros((color_all.shape)), np.zeros((color_all.shape)), np.zeros((color_all.shape))
    z_pos = cv2.normalize(color_all[:,2]*np.where(color_all[:,2]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_neg = cv2.normalize(-color_all[:,2]*np.where(color_all[:,2]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_pos = cv2.normalize(z_pos*np.where(z_pos>204,0.8,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_neg = cv2.normalize(z_neg*np.where(z_neg>204,0.8,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(z_pos),gamma=0.3)
    z_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(z_neg),gamma=0.3)
    z_flow[:,0] = cv2.normalize(z_flow[:,0]*np.where(z_flow[:,0]>230,0.9,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    z_flow[:,1] = cv2.normalize(z_flow[:,1]*np.where(z_flow[:,1]>230,0.9,1),None,0,255,cv2.NORM_MINMAX).reshape(-1)
    x_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(color_all[:,0]*np.where(color_all[:,0]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    x_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-color_all[:,0]*np.where(color_all[:,0]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    y_flow[:,0] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(color_all[:,1]*np.where(color_all[:,1]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    y_flow[:,1] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-color_all[:,1]*np.where(color_all[:,1]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5)
    os.makedirs(os.path.join(path,'direction'),exist_ok=True)
    cv2.imwrite(os.path.join(path,'direction','x_flow230.png'), cv2.normalize(x_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','y_flow230.png'), cv2.normalize(y_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','z_flow230.png'), cv2.normalize(z_flow.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    img4 = np.zeros((color_all.shape))
    img4[:,0] += cv2.normalize(x_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].max()
    img4[:,0] += cv2.normalize(-x_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].min()
    img4[:,1] += cv2.normalize(y_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].max()
    img4[:,1] += cv2.normalize(-y_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].min()
    img4[:,2] += cv2.normalize(z_flow[:,0],None,0,1,cv2.NORM_MINMAX).reshape(-1) #diff_img.max()
    img4[:,2] += cv2.normalize(-z_flow[:,1],None,-1,0,cv2.NORM_MINMAX).reshape(-1) #diff_img.min()

    x_flow1, y_flow1, z_flow1 = np.zeros((color_all.shape)), np.zeros((color_all.shape)), np.zeros((color_all.shape))
    img5 = np.zeros((color_all.shape))
    norm = np.linalg.norm([x_flow[:,0], y_flow[:,0], z_flow[:,0]], axis=0)
    norm1 = np.linalg.norm([x_flow[:,1], y_flow[:,1], z_flow[:,1]], axis=0)
    img5[:,0] += cv2.normalize(norm,None,0,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].max()
    img5[:,0] += cv2.normalize(-norm1,None,-1,0,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].min()
    img5[:,1] = np.arctan2(img4[:,1],-img4[:,0])/np.pi
    img5[:,2] = np.arcsin(-img4[:,2])/np.pi*2
    
    
    x_flow1[:,0] = cv2.normalize(norm,None,0,255,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].max()
    x_flow1[:,1] = cv2.normalize(norm1,None,0,255,cv2.NORM_MINMAX).reshape(-1) #color_all[:,0].min()

    y_flow1[:,0] = cv2.normalize(-img5[:,1]*np.where(img5[:,1]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].max()
    y_flow1[:,1] = cv2.normalize(-img5[:,1]*np.where(img5[:,1]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].min()

    z_flow1[:,0] = cv2.normalize(img5[:,2]*np.where(img5[:,2]>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1) #diff_img.max()
    z_flow1[:,1] = cv2.normalize(-img5[:,2]*np.where(img5[:,2]<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1) #diff_img.min()

    cv2.imwrite(os.path.join(path,'direction','x_flow_angel_230.png'), cv2.normalize(x_flow1.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','y_flow_angel_230.png'), cv2.normalize(y_flow1.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(path,'direction','z_flow_angel_230.png'), cv2.normalize(z_flow1.reshape(flow.shape[0],flow.shape[1],3),None,0,255,cv2.NORM_MINMAX))

    import pdb;pdb.set_trace()
    img5[:,1] = cv2.normalize(img5[:,1],None,-1,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].max()
    img5[:,2] = cv2.normalize(img5[:,2],None,-1,1,cv2.NORM_MINMAX).reshape(-1) #color_all[:,1].min()
    import pdb;pdb.set_trace()

    pp.nor_and_save(pose_all,img4,os.path.join(path),'flow3ch_230')
if __name__ == "__main__":
    # pp = point_prosses('')
    # flow = np.load('afflow1.npy')
    # flow[:,:,0] = flow[:,:,0]/abs(flow[:,:,0]).max()
    # flow[:,:,1] = flow[:,:,1]/abs(flow[:,:,1]).max()
    # depth_onset = cv2.imread(os.path.join('crop_depth', 'onset.png'), cv2.CV_16UC1)
    # depth_apex = cv2.imread(os.path.join('crop_depth', 'apex.png'), cv2.CV_16UC1)
    # new_dep_on = depth_onset.astype(np.float64)
    # new_dep_ap = depth_apex.astype(np.float64)
    # diff_img = (new_dep_ap - new_dep_on)/abs((new_dep_ap - new_dep_on)).max()
    # print(diff_img.shape)
    # z_flow = np.zeros((flow.shape))
    # z_flow[:,:,0] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(diff_img*np.where(diff_img>0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.5).reshape(diff_img.shape)
    # z_flow[:,:,1] = pp.adjust_gamma(pp.del_fewpoint(cv2.normalize(-diff_img*np.where(diff_img<0,1,0),None,0,255,cv2.NORM_MINMAX).reshape(-1)),gamma=0.3).reshape(diff_img.shape)
    
    # flow_img = np.zeros((diff_img.shape[0],diff_img.shape[1],3))
    # flow_img[:,:,0] = flow[:,:,0] 
    # flow_img[:,:,1] = flow[:,:,1] 
    # flow_img[:,:,2] += cv2.normalize(z_flow[:,:,0],None,0,diff_img.max(),cv2.NORM_MINMAX)
    # flow_img[:,:,2] += cv2.normalize(-z_flow[:,:,1],None,diff_img.min(),0,cv2.NORM_MINMAX)
    # print(flow_img[:,:,0].max(),flow_img[:,:,0].min())
    # print(flow_img[:,:,1].max(),flow_img[:,:,1].min())
    # print(flow_img[:,:,2].max(),flow_img[:,:,2].min())
    # img = flow2rgb(flow)
    # img_3d = flow2rgb_3d(flow_img)
    # print(img.shape)
    # cv2.imshow('img',img)
    # cv2.imshow('img3d',img_3d)
    # cv2.waitKey()
    save_flow_3ch('/home/disk2_12t/DATASET/MEGC/CASME3_PART/macro_cropped/spNO.11/d/19-54')
    # save_flow_angel('/home/disk1/zr/code/MEGC2019-NEWCASMEII/OpticalFlow/macro_cropped/spNO.1/a/268-289')