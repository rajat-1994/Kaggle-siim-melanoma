import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from utils import load_config
from logger import get_logger

cfg = load_config('config.yaml')
logger = get_logger()


class SiimDataset:
    def __init__(self, is_valid=False, is_test=False):
        """Dataset class for training and validation data

        Agrs:
            is_valid (bool) : Flag for distinguise between train and valid data

        Returns:
            dict : Batch of images and their labels
        """

        self.cfg = cfg
        self.img_height = self.cfg['img_height']
        self.img_width = self.cfg['img_width']
        self.mean = self.cfg['input_mean']
        self.std = self.cfg['input_std']
        self.is_test = is_test

        if self.is_test:
            df = pd.read_csv(self.cfg['test']['csv_file'])
            self.image_ids = df.image_name.values
            self.img_folder = self.cfg['test_folder']
        else:
            df = pd.read_csv(self.cfg['train']['csv_file'])

            # Checking if class is called for train or valid dataset
            if is_valid:
                folds = self.cfg['valid']['folds']
            else:
                folds = self.cfg['train']['folds']

            df = df[df.kfold.isin(folds)].reset_index(drop=True)
            self.image_ids = df.image_name.values
            self.target = df.target.values
            self.img_folder = self.cfg['train_folder']

        # Augmentation
        if is_valid or self.is_test:
            self.aug = A.Compose([
                # A.CenterCrop(600, 600, always_apply=True, p=1.0),
                A.Resize(self.img_height, self.img_width, always_apply=True),
                A.Normalize(self.mean, self.std, always_apply=True)
            ])
            logger.opt(colors=True).info(
                f"Augmentations used for <green>Testing/Validation</green>: {self.aug}")

        else:
            self.aug = A.Compose([
                A.RandomCrop(height=self.img_height, width=self.img_width),
                A.Resize(self.img_height, self.img_width, always_apply=True),
                A.RandomRotate90(),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.10,
                                   scale_limit=0.0,
                                   rotate_limit=30,
                                   border_mode=cv2.BORDER_REFLECT,
                                   p=0.9),
                A.Normalize(self.mean, self.std, always_apply=True),
            ])
            logger.opt(colors=True).info(
                f"Augmentations used for <red>Training</red>: {self.aug}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):

        image = cv2.imread(
            f"{self.img_folder}/{self.image_ids[item]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.is_test:
            batch_test = {
                "image": torch.tensor(image, dtype=torch.float),
            }
            return batch_test

        batch_train = {
            "image": torch.tensor(image, dtype=torch.float),
            "target": torch.tensor(self.target[item], dtype=torch.float)
        }
        return batch_train
