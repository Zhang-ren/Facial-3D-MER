import numpy as np
import cv2
import open3d as o3d
import os
import pandas as pd
import yaml
import json

# 加载配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 全局配置
CONFIG = load_config()

class FacePart:
    """
    用于生成面部不同区域的掩码
    """
    def __init__(self, image_shape, landmarks):
        """
        初始化面部区域分割类
        
        Args:
            image_shape: 图像尺寸 (高度, 宽度)
            landmarks: 68个面部关键点坐标
        """
        self.image_shape = image_shape
        
        # 检查关键点数量
        if landmarks.shape[0] != 68:
            raise ValueError('面部关键点必须是68个')
            
        self.landmarks = np.array(landmarks, dtype=np.int)
        self.hull_mask = np.full(self.image_shape[0:2] + (1,), 0, dtype=np.float32)
        
        # 定义面部区域分界线
        self.y0 = int((self.landmarks[24][1] + self.landmarks[46][1]) * 0.6)
        self.y1 = self.landmarks[29][1] + 5

    def _reset_mask(self):
        """重置掩码为全零"""
        self.hull_mask = np.full(self.image_shape[0:2] + (1,), 0, dtype=np.float32)
        
    def left_face(self):
        """生成左脸区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[1:5],
                           [[self.landmarks[31][0]-10, self.landmarks[1][1]]],
                           self.landmarks[31:32]))), (1,))
        self.hull_mask[self.y0:self.y1, :] = 0
        return self.hull_mask
        
    def right_face(self):
        """生成右脸区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[12:16],
                           [[self.landmarks[35][0]+10, self.landmarks[15][1]]],
                           self.landmarks[35:36]))), (1,))
        self.hull_mask[self.y0:self.y1, :] = 0
        return self.hull_mask
        
    def left_mouse(self):
        """生成左嘴角区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[4:7],
                           self.landmarks[31:32],
                           self.landmarks[48:49]))), (1,))
        self.hull_mask[self.y0:self.y1, :] = 0
        return self.hull_mask
        
    def right_mouse(self):
        """生成右嘴角区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[10:13],
                           self.landmarks[35:36],
                           self.landmarks[54:55]))), (1,))
        self.hull_mask[self.y0:self.y1, :] = 0
        return self.hull_mask
        
    def chin(self):
        """生成下巴区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[6:11],
                           self.landmarks[55:60]))), (1,))
        self.hull_mask[self.y0:self.y1, :] = 0
        return self.hull_mask
        
    def mouse(self):
        """生成嘴部区域掩码"""
        self._reset_mask()
        cv2.fillConvexPoly(self.hull_mask, cv2.convexHull(
            np.concatenate((self.landmarks[48:55],                            
                           [[self.landmarks[54][0], self.landmarks[54][1]+15]],
                           self.landmarks[54:59],
                           [[self.landmarks[48][0], self.landmarks[48][1]+15]]))), (1,))
        return self.hull_mask
        
    def left_eye(self):
        """生成左眼区域掩码"""
        self._reset_mask()
        self.hull_mask[:self.y0, :] = 1
        self.hull_mask[:, int(self.image_shape[1]/2):] = 0
        self.hull_mask[self.y0:, :] = 0
        return self.hull_mask
        
    def right_eye(self):
        """生成右眼区域掩码"""
        self._reset_mask()
        self.hull_mask[:self.y0, :] = 1
        self.hull_mask[:, :int(self.image_shape[1]/2)] = 0
        self.hull_mask[self.y0:, :] = 0
        return self.hull_mask


class PointProcessor:
    """
    处理3D点云数据
    """
    def __init__(self):
        """
        初始化点云处理器
        """
        # 从配置文件获取参数
        self.focal_length = CONFIG['point_cloud']['focal_length']
        self.scaling_factor = CONFIG['point_cloud']['scaling_factor']
        self.num_points = CONFIG['point_cloud']['num_points']
        
        # 从配置文件获取情绪标签
        self.emotion_labels = CONFIG['emotion_labels']
        
        # 初始化计数器和映射字典
        self.count = {label: 0 for label in self.emotion_labels.values()}
        self.save_dict = {}

    def read_txt(self, txt_file):
        """
        读取点云文本文件
        
        Args:
            txt_file: 点云文件路径
            
        Returns:
            point_cloud: 点云数组
            pcd: Open3D点云对象
        """
        point_cloud = np.loadtxt(txt_file, delimiter=',').astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
        return point_cloud, pcd

    def calculate_point(self, depth_file):
        """
        从深度图计算点云
        
        Args:
            depth_file: 深度图
            
        Returns:
            points: 计算得到的点云坐标
        """
        width = depth_file.shape[1]
        height = depth_file.shape[0]
        depth = np.asarray(depth_file).T
        Z = depth / self.scaling_factor
        
        # 计算X坐标
        X = np.zeros((width, height))
        for i in range(width):
            X[i, :] = np.full(X.shape[1], i)
        X = ((X - width / 2) * Z) / self.focal_length
        
        # 计算Y坐标
        Y = np.zeros((width, height))
        for i in range(height):
            Y[:, i] = np.full(Y.shape[0], i)
        Y = ((Y - height / 2) * Z) / self.focal_length

        # 组合XYZ坐标
        points = np.zeros((3, width * height))
        points[0] = X.T.reshape(-1)
        points[1] = -Y.T.reshape(-1)
        points[2] = -Z.T.reshape(-1)

        return points.T

    def count_nonzero_rows(self, matrix):
        """
        筛选非零行并按照光流强度排序
        
        Args:
            matrix: 包含点云和光流的矩阵
            
        Returns:
            sorted_matrix: 排序后的矩阵，取前num_points个点
        """
        # 筛选非零行
        non_zero_rows = matrix[np.any(matrix[:, :3] != 0, axis=1)]
        
        # 计算平均Z值并筛选
        mean_z = np.mean(non_zero_rows[:, 2])
        non_zero_rows = non_zero_rows[non_zero_rows[:, 2] - mean_z < 0.1]
        
        # 按光流强度排序
        norms = np.linalg.norm(non_zero_rows[:, 3:6], axis=1)
        sorted_indices = np.argsort(norms)[::-1]
        
        # 取前num_points个点
        sorted_matrix = non_zero_rows[sorted_indices][:self.num_points]
        return sorted_matrix

    def reduce_landmarks(self, img_shape, landmarks):
        """
        调整面部关键点位置
        
        Args:
            img_shape: 图像尺寸
            landmarks: 面部关键点
            
        Returns:
            landmarks: 调整后的关键点
        """
        mid = [img_shape[0]/2, img_shape[1]/2]
        for i, xy in enumerate(landmarks[:7]):
            if xy[0] > mid[0]:
                landmarks[i, 0] = xy[0] - 7
            else:
                landmarks[i, 0] = xy[0] + 7
                
            if xy[1] > mid[1]:
                landmarks[i, 1] = xy[1] - 7
                
        return landmarks

    def process_sample(self, subject, clip, onset, apex, emotion_id, base_path, output_base_path):
        """
        处理单个样本
        
        Args:
            subject: 主体ID
            clip: 片段ID
            onset: 起始帧
            apex: 顶点帧
            emotion_id: 情绪ID
            base_path: 基础路径
            output_base_path: 输出基础路径
            
        Returns:
            success: 是否成功处理
        """
        try:
            # 构建路径
            sample_path = os.path.join(base_path, subject, clip, f"{onset}-{apex}")
            
            # 读取点云和关键点
            point_txt_path = os.path.join(sample_path, CONFIG['folders']['txt'], f"diff_flow3ch_stand_nomove{CONFIG['formats']['point_cloud']}")
            p68_path = os.path.join(sample_path, CONFIG['folders']['p68'], f"align_p68_landmarks{CONFIG['formats']['landmarks']}")
            depth_path = os.path.join(sample_path, CONFIG['folders']['crop_depth'], f"apex{CONFIG['formats']['depth_image']}")
            
            # 检查文件是否存在
            if not os.path.exists(point_txt_path) or not os.path.exists(p68_path) or not os.path.exists(depth_path):
                print(f"文件不存在: {sample_path}")
                return False
                
            # 读取数据
            point_txt, _ = self.read_txt(point_txt_path)
            p68 = np.load(p68_path)
            depth = cv2.imread(depth_path, cv2.CV_16UC1)
            
            # 处理数据
            width = depth.shape[1]
            height = depth.shape[0]
            
            # 检查点云大小
            if width * height < CONFIG['point_cloud']['min_point_cloud_size']:
                print(f"{subject}/{clip}/{onset}-{apex}: 点云过小 ({width*height} < {CONFIG['point_cloud']['min_point_cloud_size']})")
                return False
                
            # 调整关键点并创建面部区域
            p68 = self.reduce_landmarks((height, width), p68)
            face = FacePart((height, width), p68)
            
            # 归一化点云坐标
            point_txt[:, :3] = cv2.normalize(point_txt[:, :3], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # 将点云重塑为图像形状
            point_reshaped = point_txt.reshape(height, width, 6)
            
            # 处理各个面部区域
            face_parts = []
            
            # 左眼
            left_eye = cv2.multiply(point_reshaped, np.repeat(face.left_eye(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(left_eye))
            
            # 右眼
            right_eye = cv2.multiply(point_reshaped, np.repeat(face.right_eye(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(right_eye))
            
            # 左脸
            left_face = cv2.multiply(point_reshaped, np.repeat(face.left_face(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(left_face))
            
            # 右脸
            right_face = cv2.multiply(point_reshaped, np.repeat(face.right_face(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(right_face))
            
            # 左嘴角
            left_mouse = cv2.multiply(point_reshaped, np.repeat(face.left_mouse(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(left_mouse))
            
            # 右嘴角
            right_mouse = cv2.multiply(point_reshaped, np.repeat(face.right_mouse(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(right_mouse))
            
            # 下巴
            chin = cv2.multiply(point_reshaped, np.repeat(face.chin(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(chin))
            
            # 嘴部
            mouse = cv2.multiply(point_reshaped, np.repeat(face.mouse(), 6, axis=2)).reshape(-1, 6)
            face_parts.append(self.count_nonzero_rows(mouse))
            
            # 合并所有面部区域
            merged_matrix = np.concatenate(face_parts, axis=0)
            
            # 检查点数是否符合要求
            expected_points = self.num_points * 8
            if merged_matrix.shape[0] != expected_points:
                print(f"{subject}/{clip}/{onset}-{apex}: 点云数量不符 ({merged_matrix.shape[0]} != {expected_points})")
                return False
                
            # 保存结果
            emotion_label = self.emotion_labels[emotion_id]
            save_dir = os.path.join(output_base_path, CONFIG['folders']['dataset_seg_part'], emotion_label)
            os.makedirs(save_dir, exist_ok=True)
            
            # 更新计数器
            self.count[emotion_label] += 1
            file_name = f"{emotion_label}_{str(self.count[emotion_label]).zfill(4)}{CONFIG['formats']['point_cloud']}"
            
            # 记录映射关系
            path_key = os.path.join(subject, clip, f"{onset}-{apex}")
            self.save_dict[path_key] = os.path.join(emotion_label, file_name)
            
            # 保存点云文件
            np.savetxt(os.path.join(save_dir, file_name), merged_matrix, fmt='%.8f', delimiter=',')
            
            print(f"成功处理: {subject}/{clip}/{onset}-{apex} -> {emotion_label}")
            return True
            
        except Exception as e:
            print(f"处理样本时出错 {subject}/{clip}/{onset}-{apex}: {str(e)}")
            with open(CONFIG['paths']['error_log'], 'a') as f:
                f.write(f"{subject}/{clip}/{onset}-{apex}: {str(e)}\n")
            return False

    def process_expression_data(self, exp_type, excel_path, base_path, output_base_path):
        """
        处理表情数据
        
        Args:
            exp_type: 表情类型 ('micro' 或 'macro')
            excel_path: Excel文件路径
            base_path: 基础路径
            output_base_path: 输出基础路径
        """
        try:
            # 读取Excel文件
            print(f"读取{exp_type}表情数据: {excel_path}")
            data = pd.read_excel(excel_path)
            
            # 处理计数
            processed_count = 0
            success_count = 0
            
            # 遍历每一行数据
            for _, row in data.iterrows():
                if row[2] == row[3]:  # 跳过起始帧和顶点帧相同的数据
                    continue
                    
                # 提取信息
                subject = str(row[0])
                clip = str(row[1])
                onset = str(row[2])
                apex = str(row[3])
                
                # 提取情绪标签
                # 微表情和宏表情的情绪标签在不同列
                if exp_type == CONFIG['expression_types']['micro']:
                    emotion_name = row[7].lower() if isinstance(row[7], str) else 'others'
                    emotion_id = CONFIG['micro_emotion_map'].get(emotion_name, 6)
                else:
                    emotion_name = row[6].lower() if isinstance(row[6], str) else 'others'
                    emotion_id = CONFIG['macro_emotion_map'].get(emotion_name, 6)
                
                processed_count += 1
                if self.process_sample(subject, clip, onset, apex, emotion_id, os.path.join(base_path, exp_type), output_base_path):
                    success_count += 1
                    
            print(f"处理完成: {exp_type}表情, 共处理{processed_count}个样本, 成功{success_count}个")
            
            # 保存映射关系
            mapping_file = os.path.join(output_base_path, CONFIG['folders']['dataset_seg_part'], f"{exp_type}_cropped_exp.json")
            with open(mapping_file, 'w') as f:
                json.dump(self.save_dict, f)
                
        except Exception as e:
            print(f"处理{exp_type}数据时出错: {str(e)}")
            with open(CONFIG['paths']['error_log'], 'a') as f:
                f.write(f"处理{exp_type}数据时出错: {str(e)}\n")


def main():
    """
    主函数：处理微表情和宏表情数据
    """
    # 设置路径
    base_path = CONFIG['paths']['processed_data']
    output_base_path = CONFIG['paths']['output_base']
    
    # Excel文件路径
    me_path = CONFIG['files']['micro_label']
    mae_path = CONFIG['files']['macro_label']
    
    # 初始化点云处理器
    processor = PointProcessor()
    
    # 处理微表情数据
    processor.process_expression_data(CONFIG['expression_types']['micro'], me_path, base_path, output_base_path)
    
    # 重置计数器和映射字典
    processor.save_dict = {}
    
    # 处理宏表情数据
    processor.process_expression_data(CONFIG['expression_types']['macro'], mae_path, base_path, output_base_path)


if __name__ == '__main__':
    main()
   