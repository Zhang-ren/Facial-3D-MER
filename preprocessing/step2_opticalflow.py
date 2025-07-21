import os
import cv2
import numpy as np
import pandas as pd
import transplant
import scipy.io
import yaml
from of_img import save_flow_3ch

# 加载配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 全局配置
CONFIG = load_config()

# 初始化MATLAB引擎
matlab = transplant.Matlab(jvm=False, desktop=False)

def process_landmarks(landmarks, img_shape):
    """
    处理面部关键点坐标，确保在图像范围内
    
    Args:
        landmarks: 面部关键点坐标数组
        img_shape: 图像尺寸 (高度, 宽度, 通道数)
    
    Returns:
        x, y: 处理后的x和y坐标数组
    """
    height, width = img_shape[:2]
    x, y = [], []
    
    # 处理x坐标
    for i in landmarks:
        if i[0] <= 0:
            x.append(1)
        elif i[0] >= width:
            x.append(width - 1)
        else:
            x.append(i[0])
    
    # 处理y坐标
    for i in landmarks:
        if i[1] <= 0:
            y.append(1)
        elif i[1] >= height:
            y.append(height - 1)
        else:
            y.append(i[1])
            
    return np.array(x), np.array(y)

def compute_optical_flow(onset_path, apex_path, landmarks_path, output_dir):
    """
    计算光流并保存结果
    
    Args:
        onset_path: 起始帧图像路径
        apex_path: 顶点帧图像路径
        landmarks_path: 面部关键点文件路径
        output_dir: 输出目录
    
    Returns:
        success: 是否成功处理
    """
    # 创建输出目录
    flow_dir = os.path.join(output_dir, CONFIG['folders']['flow'])
    os.makedirs(flow_dir, exist_ok=True)
    
    # 设置输出文件路径
    flow_save_path = os.path.join(flow_dir, f"flow1{CONFIG['formats']['flow_image']}")
    afflow_pic_path = os.path.join(flow_dir, f"afflow1{CONFIG['formats']['flow_image']}")
    afflow_xy_path = os.path.join(flow_dir, f"afflow1{CONFIG['formats']['flow_data']}")
    
    # 检查输入文件是否存在
    if not os.path.exists(onset_path) or not os.path.exists(apex_path) or not os.path.exists(landmarks_path):
        print(f"输入文件不存在，跳过: {output_dir}")
        return False
        
    try:
        # 读取图像和关键点
        onset_img = cv2.imread(onset_path)
        if onset_img is None:
            print(f"无法读取起始帧图像: {onset_path}")
            return False
            
        landmarks = np.load(landmarks_path)
        
        # 处理关键点坐标
        x, y = process_landmarks(landmarks, onset_img.shape)
        
        # 计算光流
        matlab.test(x, y, onset_path, apex_path, flow_save_path, afflow_pic_path, afflow_xy_path)
        
        # 读取光流数据并保存为npy格式
        data = scipy.io.loadmat(afflow_xy_path)
        afflow = data['afflow']
        np.save(os.path.join(flow_dir, f"afflow1{CONFIG['formats']['flow_npy']}"), afflow)
        
        # 生成3通道光流
        save_flow_3ch(output_dir)
        
        return True
    except Exception as e:
        print(f"处理光流时出错: {str(e)}")
        with open(CONFIG['paths']['error_log'], 'a') as f:
            f.write(f"光流处理错误 {output_dir}: {str(e)}\n")
        return False

def main():
    """
    主函数：从Excel文件读取数据并处理光流
    """
    # 设置路径
    base_path = CONFIG['paths']['processed_data']
    
    # Excel文件路径
    me_path = CONFIG['files']['micro_label']
    mae_path = CONFIG['files']['macro_label']
    
    # 处理微表情数据
    print("处理微表情数据的光流...")
    process_expression_data(CONFIG['expression_types']['micro'], me_path, base_path)
    
    # 处理宏表情数据
    print("处理宏表情数据的光流...")
    process_expression_data(CONFIG['expression_types']['macro'], mae_path, base_path)

def process_expression_data(exp_type, excel_path, base_path):
    """
    处理表情数据
    
    Args:
        exp_type: 表情类型 ('micro' 或 'macro')
        excel_path: Excel文件路径
        base_path: 基础路径
    """
    try:
        # 读取Excel文件
        data = pd.read_excel(excel_path)
        
        # 遍历每一行数据
        for _, row in data.iterrows():
            if row[2] == row[3]:  # 跳过起始帧和顶点帧相同的数据
                continue
                
            # 提取信息
            subject = str(row[0])
            clip = str(row[1])
            onset = str(row[2])
            apex = str(row[3])
            
            # 构建路径
            sample_dir = os.path.join(base_path, exp_type, subject, clip, f"{onset}-{apex}")
            onset_path = os.path.join(sample_dir, CONFIG['folders']['crop_color'], f"onset{CONFIG['formats']['color_image']}")
            apex_path = os.path.join(sample_dir, CONFIG['folders']['crop_color'], f"apex{CONFIG['formats']['color_image']}")
            landmarks_path = os.path.join(sample_dir, CONFIG['folders']['p68'], f"align_p68_landmarks{CONFIG['formats']['landmarks']}")
            
            # 检查必要文件是否存在
            if not os.path.exists(onset_path) or not os.path.exists(apex_path) or not os.path.exists(landmarks_path):
                continue
                
            # 检查是否已经处理过
            flow_mat_path = os.path.join(sample_dir, CONFIG['folders']['flow'], f"afflow1{CONFIG['formats']['flow_data']}")
            if os.path.exists(flow_mat_path):
                print(f"已处理过: {sample_dir}")
                continue
                
            print(f"处理: {subject}/{clip}/{onset}-{apex}")
            compute_optical_flow(onset_path, apex_path, landmarks_path, sample_dir)
    
    except Exception as e:
        print(f"处理{exp_type}数据时出错: {str(e)}")
        with open(CONFIG['paths']['error_log'], 'a') as f:
            f.write(f"处理{exp_type}数据时出错: {str(e)}\n")

if __name__ == '__main__':
    main()

            # import pdb;pdb.set_trace()
    
    
    
    
    
    
