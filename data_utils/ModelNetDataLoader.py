'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
import cv2
warnings.filterwarnings('ignore')
np.random.seed(0)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, ind, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        elif self.num_category == 40:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'me_shape_name.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids_raw = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        elif self.num_category == 40:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        elif self.num_category == 7:
            with open(os.path.join(self.root,'macro_cropped_exp.json'),'r',encoding='utf8')as fp:
                macro_data = json.load(fp)
            with open(os.path.join(self.root,'micro_cropped_exp.json'),'r',encoding='utf8')as fp:
                micro_data = json.load(fp)
            shape_ids_raw['train'] = [macro_data[key].split('/')[1][:-4] for key in macro_data]
            shape_ids_raw['test'] = [micro_data[key].split('/')[1][:-4] for key in micro_data]
            # shape_ids['train'] = shape_ids_raw['train'][:int(len(shape_ids_raw['train'])*0.8)] + shape_ids_raw['test'][:int(len(shape_ids_raw['test'])*0.4)]+ shape_ids_raw['test'][int(len(shape_ids_raw['test'])*0.6):]
            # shape_ids['test'] =shape_ids_raw['test'][int(len(shape_ids_raw['test'])*0.4):int(len(shape_ids_raw['test'])*0.6)]
            # shape_ids['train'] = np.concatenate([np.random.choice(shape_ids_raw['train'], int(len(shape_ids_raw['train'])* 0.8 ), replace=False),
            #                                      np.random.choice(shape_ids_raw['test'][:int(len(shape_ids_raw['test']) * 0.8)], int(len(shape_ids_raw['test']) * 0.8), replace=False)])
            # shape_ids['test'] = np.random.choice(shape_ids_raw['test'][int(len(shape_ids_raw['test'])*0.8):], int(len(shape_ids_raw['test']) * 0.2 ), replace=False)
            # shape_ids['train'] = shape_ids_raw['train'] 
            # shape_ids['test'] = shape_ids_raw['test']
            shape_ids['train'] = np.concatenate([np.random.choice(shape_ids_raw['train'], int(len(shape_ids_raw['train'])* 0.8 ), replace=False),
    np.random.choice(shape_ids_raw['test'], int(len(shape_ids_raw['test']) * 0.3), replace=False)])
            shape_ids['test'] =[x for x in shape_ids_raw['test'] if x not in shape_ids['train']] # [x for x in shape_ids_raw['train'] if x not in shape_ids['train']] + 
            

        else :
            # with open(os.path.join(self.root,'corr_macro.json'),'r',encoding='utf8')as fp:
            #     macro_data = json.load(fp)
            with open(os.path.join(self.root,'new_exp_micro.json'),'r',encoding='utf8')as fp:
                json_data = json.load(fp)
            shape_ids['train'] = [json_data[key].split('/')[1][:-4] for key in json_data if key.split('_')[0] not in ind] #+ [macro_data[keys].split('/')[1][:-4] for keys in macro_data]
            if ind[0] == -10:
                shape_ids['test'] = [json_data[key].split('/')[1][:-4] for key in json_data]
            else:
                shape_ids['test'] = [json_data[key].split('/')[1][:-4] for key in json_data if key.split('_')[0] in ind]
            


        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        # point_set[:,0] = cv2.normalize(point_set[:,0], None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX).reshape(-1)
        # point_set[:,1] = cv2.normalize(point_set[:,1], None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX).reshape(-1)
        # point_set[:,2] = cv2.normalize(point_set[:,2], None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX).reshape(-1)        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set[:, 3:] = pc_normalize(point_set[:, 3:])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        # point_set_list = [point_set[2048*(i):2048*(i+1)] for i in range(8)]
        # point_set = np.stack(point_set_list, axis=0)

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
