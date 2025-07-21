import os
import dlib
import cv2
import numpy as np
import pandas as pd
import yaml

# 加载配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 全局配置
CONFIG = load_config()

def align_p68(center, landmarks, angle=0):
    """
    将面部关键点根据指定角度进行对齐旋转
    
    Args:
        center: 旋转中心点坐标
        landmarks: 面部关键点坐标
        angle: 旋转角度
    
    Returns:
        rotated_landmarks: 旋转后的关键点坐标
    """
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    landmarks = np.array(landmarks, dtype=np.float32)

    # 添加一列全为1的数组，以便进行仿射变换
    ones_column = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    landmarks = np.hstack((landmarks, ones_column))

    # 应用旋转矩阵到关键点
    rotated_landmarks = np.dot(rotation_matrix, landmarks.T).T

    # 将关键点转换回整数类型
    rotated_landmarks = rotated_landmarks.astype(np.int)
    return rotated_landmarks

def single_face_alignment(face, landmarks, depth_path):
    """
    对面部图像和深度图进行对齐
    
    Args:
        face: 人脸图像
        landmarks: 面部关键点
        depth_path: 深度图路径
    
    Returns:
        align_face: 对齐后的人脸图像
        align_p68_landmarks: 对齐后的关键点坐标
        align_depth: 对齐后的深度图
    """
    # 计算两眼的中心坐标
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,
                 (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    
    # 计算眼睛之间的差值
    dx = (landmarks[45, 0] - landmarks[36, 0])
    dy = (landmarks[45, 1] - landmarks[36, 1])
    
    # 读取深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # 计算旋转角度并进行图像旋转
    angle = np.arctan2(dy, dx) * 180. / np.pi
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    align_face = cv2.warpAffine(face, rotate_matrix, (face.shape[0], face.shape[1]))
    align_depth = cv2.warpAffine(depth, rotate_matrix, (depth.shape[0], depth.shape[1]))

    # 旋转关键点坐标
    align_p68_landmarks = align_p68(eye_center, landmarks, angle)
    return align_face, align_p68_landmarks, align_depth

def det_face(test_img_path, depth):
    """
    检测人脸并提取关键点
    
    Args:
        test_img_path: 测试图像路径
        depth: 深度图路径
    
    Returns:
        crop_image: 裁剪后的人脸图像
        xy: 裁剪范围坐标
        align_p68_landmarks: 对齐后的关键点坐标
        align_depth: 对齐后的深度图
        flag: 是否检测到人脸的标志
    """
    # 加载模型
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(CONFIG['files']['face_landmark_model'])

    # 读取并转换图像
    img = cv2.imread(test_img_path)
    if img is None:
        print(f"无法读取图像: {test_img_path}")
        return img, (0,0,0,0), 0, depth, 0
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 检测人脸
    rects = detector(img, 0)
    if len(rects) == 0:
        print(f"未检测到人脸: {test_img_path}")
        return img, (0,0,0,0), 0, depth, 0
    
    # 提取关键点
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    
    # 对人脸进行对齐
    align_face, align_p68_landmarks, align_depth = single_face_alignment(img, landmarks, depth)
    
    # 计算裁剪区域
    x = align_p68_landmarks[:,0]
    y = align_p68_landmarks[:,1]
    x0, x1 = min(x), max(x)
    y0, y1 = min(y), max(y)
    
    # 将对齐后的图像转回BGR颜色空间
    crop_image = cv2.cvtColor(align_face, cv2.COLOR_RGB2BGR)
    
    return crop_image, (x0, x1, y0, y1), align_p68_landmarks, align_depth, 1

def crop_landmark_face(path, onset_path, apex_path, onset_depth_path, apex_depth_path):
    """
    裁剪人脸并保存对齐后的图像和关键点
    
    Args:
        path: 保存路径
        onset_path: 起始帧图像路径
        apex_path: 顶点帧图像路径
        onset_depth_path: 起始帧深度图路径
        apex_depth_path: 顶点帧深度图路径
    
    Returns:
        0: 如果处理失败
        1: 如果处理成功
    """
    # 检测并对齐人脸
    crop_image1, xy, align_p68_landmarks, onset_depth, flag = det_face(onset_path, onset_depth_path)
    if flag == 0:
        print(f"起始帧处理失败: {onset_path}")
        return 0
        
    crop_image2, _, _, apex_depth, flag = det_face(apex_path, apex_depth_path)
    if flag == 0:
        print(f"顶点帧处理失败: {apex_path}")
        return 0
    
    # 创建保存目录
    os.makedirs(os.path.join(path, CONFIG['folders']['crop_color']), exist_ok=True)
    os.makedirs(os.path.join(path, CONFIG['folders']['crop_depth']), exist_ok=True)
    os.makedirs(os.path.join(path, CONFIG['folders']['p68']), exist_ok=True)
    
    try:
        # 保存裁剪后的图像和深度图
        cv2.imwrite(os.path.join(path, CONFIG['folders']['crop_color'], "onset.jpg"), crop_image1[xy[2]:xy[3], xy[0]:xy[1]])
        cv2.imwrite(os.path.join(path, CONFIG['folders']['crop_color'], "apex.jpg"), crop_image2[xy[2]:xy[3], xy[0]:xy[1]])
        cv2.imwrite(os.path.join(path, CONFIG['folders']['crop_depth'], "onset.png"), onset_depth[xy[2]:xy[3], xy[0]:xy[1]])
        cv2.imwrite(os.path.join(path, CONFIG['folders']['crop_depth'], "apex.png"), apex_depth[xy[2]:xy[3], xy[0]:xy[1]])
        
        # 调整关键点坐标并保存
        align_p68_landmarks[:,0] = align_p68_landmarks[:,0] - xy[0]
        align_p68_landmarks[:,1] = align_p68_landmarks[:,1] - xy[2]
        align_p68_landmarks = align_p68_landmarks.astype(np.int)
        np.save(os.path.join(path, CONFIG['folders']['p68'], "align_p68_landmarks.npy"), align_p68_landmarks)
        return 1
    except Exception as e:
        print(f"保存处理后的图像时发生错误: {str(e)}")
        return 0

def main():
    """
    主函数：处理微表情和宏表情数据
    """
    # 从配置文件获取路径
    origin_path = CONFIG['paths']['origin_data']
    output_base_path = CONFIG['paths']['processed_data']
    
    # 获取Excel文件路径
    me_path = CONFIG['files']['micro_label']
    mae_path = CONFIG['files']['macro_label']
    
    # 处理微表情数据
    print("处理微表情数据...")
    process_expression_data(CONFIG['expression_types']['micro'], me_path, origin_path, output_base_path)
    
    # 处理宏表情数据
    print("处理宏表情数据...")
    process_expression_data(CONFIG['expression_types']['macro'], mae_path, origin_path, output_base_path)

def process_expression_data(exp_type, excel_path, origin_path, output_base_path):
    """
    处理表情数据
    
    Args:
        exp_type: 表情类型 ('micro' 或 'macro')
        excel_path: Excel文件路径
        origin_path: 原始数据路径
        output_base_path: 输出基础路径
    """
    # 读取Excel文件
    data = pd.read_excel(excel_path)
    
    # 遍历每一行数据
    for _, row in data.iterrows():
        if row[2] != row[3]:  # 确保起始帧和顶点帧不同
            subject = str(row[0])
            clip = str(row[1])
            onset = str(row[2])
            apex = str(row[3])
            
            # 构建文件路径
            color_onset_path = os.path.join(origin_path, subject, clip, 'color', f"{onset}{CONFIG['formats']['color_image']}")
            color_apex_path = os.path.join(origin_path, subject, clip, 'color', f"{apex}{CONFIG['formats']['color_image']}")
            depth_onset_path = os.path.join(origin_path, subject, clip, 'depth', f"{onset}{CONFIG['formats']['depth_image']}")
            depth_apex_path = os.path.join(origin_path, subject, clip, 'depth', f"{apex}{CONFIG['formats']['depth_image']}")
            
            # 输出路径
            output_path = os.path.join(output_base_path, exp_type, subject, clip, f"{onset}-{apex}")
            
            # 检查是否已处理
            if os.path.exists(os.path.join(output_path, CONFIG['folders']['p68'], f"align_p68_landmarks{CONFIG['formats']['landmarks']}")):
                print(f"已处理过: {output_path}")
                continue
                
            print(f"处理{exp_type}表情: {subject}/{clip}/{onset}-{apex}")
            # 检查文件是否存在
            if not os.path.exists(color_onset_path) or not os.path.exists(color_apex_path) or \
               not os.path.exists(depth_onset_path) or not os.path.exists(depth_apex_path):
                print(f"文件不存在，跳过: {subject}/{clip}/{onset}-{apex}")
                continue
                
            crop_landmark_face(output_path, color_onset_path, color_apex_path, depth_onset_path, depth_apex_path)

if __name__ == '__main__':
    main()

    
    
    
    
    
    
