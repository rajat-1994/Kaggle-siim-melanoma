import pandas as pd
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":
    df = pd.read_csv(
        "./data/train.csv")
    df.loc[:, 'kfold'] = -1
    print(df.head())

    numfolds = 5
    print(f"Creating {numfolds} folds...")

    df = df.sample(frac=1).reset_index(drop=True)
    gfk = GroupKFold(n_splits=numfolds)

    for fold, (trn_, val_) in enumerate(gfk.split(X=df.image_id, groups=df.patient_id)):
        print("TRAIN: ", trn_, "VAL: ", val_)
        df.loc[val_, "kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv(f"./data/train_folds{numfolds}.csv",
              index=False)
