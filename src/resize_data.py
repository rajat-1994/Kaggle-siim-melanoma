import cv2
import os
import glob
import numpy as np
import albumentations as A
from joblib import Parallel, delayed


def read_save(load_path, folder, shape=(768, 768)):
    """save cropped image"""

    img = cv2.imread(load_path)
    hw = min(img.shape[0], img.shape[1])
    aug = A.Compose([A.CenterCrop(hw, hw, always_apply=True, p=1.0)])
    img = aug(image=np.array(img))["image"]
    img = cv2.resize(img, shape)
    save_path = f"../data/{folder}_images/{load_path.split('/')[-1]}"
    cv2.imwrite(save_path, img)


def process(path, folder, shape, njobs=6):

    print(f"Saving {folder}Images...")
    # Saving Images
    _ = Parallel(n_jobs=njobs)(delayed(read_save)(i, folder, shape)
                               for i in files)


if __name__ == "__main__":
    shape = (1024, 1024)
    folders = ['train', 'test']
    for folder in folders:
        # making folders to save cropped images
        if not os.path.exists(f"../data/{folder}_images/"):
            os.mkdir(f"../data/{folder}_images/")
        files = []
        for file in glob.glob(f"../data/jpeg/{folder}/*"):
            files.append(file)
        process(files, folder, shape, njobs=6)
