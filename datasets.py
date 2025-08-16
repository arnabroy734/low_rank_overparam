import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import warnings
from PIL import ImageFilter
import random
import yaml
from pathlib import Path
warnings.filterwarnings('ignore')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
            csv_path,
            image_root_path='',
            class_index=-1,
            use_frontal=True,
            use_upsampling=False,
            flip_label=False,
            shuffle=True,
            seed=123,
            transform=None,
            verbose=True,
            upsampling_cols=['Cardiomegaly', 'Consolidation'],
            train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
            mode='train',
            pretraining=False
        ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)


        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        # transform
        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        
        image = Image.open(self._images_list[idx]).convert('RGB')

        image = self.transform(image)

        if self.class_index != -1:  # multi-class mode
            label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
        else:
            label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

        if self.pretraining:
            label = -1

        return image, label


def create_dataset(name: str, mode: str, config: dict):
    """
    'name' should be 'chestexpert' or 
    'mode' should be 'train', 'cross_valid' or 'valid'
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.0, 0.0)  
        ], p=0.8),
        transforms.RandomRotation(degrees=(0, 15)),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.ToTensor(),
        transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    ])

    if name == 'chestexpert':
        if mode == 'train':
            upsample = True
            csvpath = config['data']['chestexpert']['traincsvpath']
            transform = train_transform
        elif mode == 'valid':
            upsample = False
            csvpath = config['data']['chestexpert']['validcsvpath']
            transform = valid_transform
        elif mode == 'cross_valid':
            upsample = False
            csvpath = config['data']['chestexpert']['crossvalidcsvpath']
            transform = valid_transform
        ds = CheXpert(
            csv_path=str(csvpath),
            image_root_path=str(config['data']['chestexpert']['imageroot'])+"/",
            mode=mode,
            use_upsampling=upsample,
            transform=transform
        )
        return ds
