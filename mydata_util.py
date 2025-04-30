import pandas as pd
import os
import torch
from glob import glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import numpy as np
from torchvision import transforms
import config



class SkinData(Dataset):
    def __init__(self, df, transform=None, classes=None, target=None):
        self.df = df
        self.transform = transform
        self.classes = classes
        self.target = target
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y


def dataset_iid(dataset, num_users):
    np.random.seed(config.seed)
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dataset_train_non_iid(dataset, num_users, per):
    num_items = int(len(dataset)*per)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])  #每个非重复
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def read_data():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.Pad(3),
                                           transforms.RandomRotation(10),
                                           transforms.CenterCrop(64),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)
                                           ])

    test_transforms = transforms.Compose([
        transforms.Pad(3),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    df = pd.read_csv('data/HAM10000_metadata.csv')

    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join("data", '*', '*.jpg'))}
    # print("path---------------------------------------", imageid_path.get)
    df['path'] = df['image_id'].map(imageid_path.get)  # 添加path 用map函数
    df['cell_type'] = df['dx'].map(lesion_type.get)  # 添加cell_type 用map函数
    df['target'] = pd.Categorical(df['cell_type']).codes  # 添加target 用pd.Categorical函数

    print(df['cell_type'].value_counts())
    print(df['target'].value_counts())
    print(df.head())
    train_df, test_df = train_test_split(df, test_size=config.test_size, random_state=config.seed, stratify=df['target'])
    train_df = train_df.reset_index()  # 重置索引 并将原来的索引作为新的一列
    test_df = test_df.reset_index()
    _,Q_test_data_df = train_test_split(test_df, test_size=config.test_size, random_state=config.seed, stratify=test_df['target'])
    Q_test_data_df = Q_test_data_df.reset_index()

    # With augmentation
    dataset_train = SkinData(train_df, transform=train_transforms, classes=np.array([0,1,2,3,4,5,6]), target=train_df['target'])
    dataset_test = SkinData(test_df, transform=test_transforms, classes=np.array([0,1,2,3,4,5,6]), target=test_df['target'])
    # dataset_Q_test_data = SkinData(Q_test_data_df, transform=test_transforms, classes=np.array([0,1,2,3,4,5,6]), target=Q_test_data_df['target'])
    # dict_users = dataset_train_non_iid(dataset_train, config.num_users, config.per)
    dict_users = dataset_iid(dataset_train, config.num_clients)
    dict_users_test = dataset_iid(dataset_test, 1)
    return dataset_train, dataset_test, dict_users, dict_users_test

def random_get_dict(dict_users, p):
    random.seed(config.seed)
    sampled_dict = {}

    # 遍历每个set
    for key, dict_set in dict_users.items():
        min_num = int(p * len(dict_users[key]))
        # 随机选择采样数量
        num_samples = min_num
        # num_samples = random.randint(min_num, len(dict_set))
        # 随机采样
        sampled_ints = set(random.sample(dict_set, num_samples))
        # 将采样结果添加到新字典中
        sampled_dict[key] = sampled_ints
    # print("采样结果：", sampled_dict)
    return sampled_dict

# dataset_train, dataset_test, dict_users, dict_users_test = read_data()
# sampled_dict = random_get_dict(dict_users)
# sampled_dict = {}
# for key, dict_set in dict_users_test.items():
#     # 随机采样
#     sampled_ints = set(random.sample(dict_set, 128))
#     # 将采样结果添加到新字典中
#     sampled_dict[key] = sampled_ints
# quality = DataLoader(DatasetSplit(dataset_test,sampled_dict),batch_size = 128, shuffle=True)
# a=0