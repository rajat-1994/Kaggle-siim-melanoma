# SIIM-ISIC Melanoma Classification
This repo contains my code for training classification model for kaggle recently held competition [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/).This code reproduce one of my best single model.

# Table of contents

- [SIIM-ISIC Melanoma Classification](#introduction)
- [Table of contents](#table-of-contents)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)


# Project Structure
[(top)](#table-of-contents)

    ```
    ./
    ├── config.yaml
    ├── data
    ├── logs
    ├── README.md
    ├── requirements.txt
    ├── src
    │   ├── classifier.py
    │   ├── create_folds.py
    │   ├── dataset.py
    │   ├── early_stopping.py
    │   ├── logger.py
    │   ├── models.py
    │   ├── resize_data.py
    │   ├── train.py
    │   ├── TTA.py
    │   └── utils.py
    ├── subs
    └── weights
    ```

# Installation
[(top)](#table-of-contents)
1. Setup virtual environment(using conda or venv).

    ```
    conda create -n kaggle python=3.7
    conda activate kaggle
    pip install -r requirements.txt
    ```

2. Download the dataset(jpeg folder) and zip it in *data* folder from [kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/data).After downloading *data* folder should look like this

    ```
    data/
    ├── jpeg
    │   ├── test
    │   └── train
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
    ```

# Training
[(top)](#table-of-contents)

1. Create folds.

    ```python create_folds.py```

2. Resize images. Original images are quite large so loading large images while training will slow training. So resize dataset to smaller size (384,512,1024 etc.) by change *resize_data.py* and then run

    ```python resize_data.py```

3. Start training

    ```python train.py```

    weights, logs and submission file will be saved in *weights*, *logs*, *subs* folder respectively
