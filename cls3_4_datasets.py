import os
from collections import defaultdict
import json
import shutil
def rename_txt_files(root_dir):
    # 创建一个字典来存储每个目录中的.txt文件数量
    dir_counts = defaultdict(int)
    emo3 = {'anger':'Negative','disgust':'Negative','fear':'Negative','happy':'Positive','others':'Others','sad':'Negative','surprise':'Surprise'}
    # 遍历root_dir及其所有子目录
    with open("/home/disk1/zr/code/Pointnet_Pointnet2_pytorch/data/dataset_seg_part_nomove/micro_cropped_exp.json", 'r') as f:
        data = json.load(f)
    new_path = os.path.join(os.path.dirname(root_dir), 'dataset_seg_part_nomove_3cls')
    micro_dict3 = {}
    micro_dict4 = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 遍历所有文件
        for filename in filenames:
            # 如果文件是.txt文件
            if filename.endswith('.txt'):
                # 在字典中增加这个目录的计数
                base_path = os.path.basename(dirpath)
                # print(base_path,)
                
                # print(f"{base_path}/{filename}")
                
                try:
                    if f"{base_path}/{filename}" in data.values():
                        emo = emo3[base_path]
                        dir_counts[emo] += 1
                    
                        # print(emo3[base_path])
                        # 构造新的文件名
                        new_filename = f"{emo}_{str(dir_counts[emo]).zfill(4)}.txt"
                        new_save_path = os.path.join(new_path,emo)
                        if not os.path.exists(new_save_path):
                            os.makedirs(new_save_path)
                        
                        # 获取文件的完整原始路径
                        old_file_path = os.path.join(dirpath, filename)
                        # 获取文件的新路径
                        new_file_path = os.path.join(new_save_path, new_filename)
                        # 复制文件
                        shutil.copy( old_file_path, new_file_path)
                        micro_dict4[list(data.keys())[list(data.values()).index(f"{base_path}/{filename}")]] = f"{emo}/{new_filename}"
                        if emo != 'Others':
                            micro_dict3[list(data.keys())[list(data.values()).index(f"{base_path}/{filename}")]] = f"{emo}/{new_filename}"
                        
                        

                except KeyError:
                    print('error')
    print(len(micro_dict4))
    print(len(micro_dict3))
    with open(os.path.join(new_path, 'micro_cropped_exp_3cls.json'), 'w') as f:
        json.dump(micro_dict3, f)
    with open(os.path.join(new_path, 'micro_cropped_exp_4cls.json'), 'w') as f:
        json.dump(micro_dict4, f)
        

# 使用示例
rename_txt_files("/home/disk1/zr/code/Pointnet_Pointnet2_pytorch/data/dataset_seg_part_nomove")