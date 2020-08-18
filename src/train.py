import os
import numpy as np
import torch
import wandb
import models
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping
from torchsummary import summary
from dataset import SiimDataset
from classifier import Engine
from utils import save_prediction, load_config
from logger import get_logger
from torch.cuda import amp

print(f"Cuda device name : {torch.cuda.get_device_name(0)}")
print(f"Is cuda available : {torch.cuda.is_available()} ")
print(f"Pytorch version : {torch.__version__} ")

cfg = load_config('config.yaml')
if cfg['dryrun']:
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init()

PATH_SUFFIX = str(wandb.run.id)

logger = get_logger(PATH_SUFFIX)


MODEL_PATH = f"{cfg['weights_folder']}/{cfg['base_model']}_fold{cfg['valid']['folds'][0]}x{cfg['img_height']}_{PATH_SUFFIX}.pth"
SUB_CSV = f"{cfg['subs_folder']}/{cfg['base_model']}x{cfg['img_height']}_{PATH_SUFFIX}.csv"

EPOCHS = cfg['train']['epochs']
DEVICE = cfg['device']


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def data_loaders(cfg):

    train_dataset = SiimDataset()
    valid_dataset = SiimDataset(is_valid=True)
    test_dataset = SiimDataset(is_test=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['n_jobs'])

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=cfg['test']['batch_size'],
                              shuffle=False,
                              num_workers=cfg['n_jobs'])

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=False,
                             num_workers=cfg['n_jobs'])

    return train_loader, valid_loader, test_loader


def trainer():
    num_meta = None

    # Dataloaders
    train_loader, valid_loader, test_loader = data_loaders(cfg)
    # Model
    logger.info(f"Loading {cfg['base_model']}")
    if cfg['use_meta']:
        num_meta = train_loader.dataset.meta_features.shape[1]

    if 'efficientnet' in cfg['base_model']:
        model = models.EfficientNets(
            model_name=cfg['base_model'], num_meta=num_meta)

    else:
        model = models.PretrainedNets(
            model_name=cfg['base_model'], num_meta=num_meta)

    model.to(DEVICE)
    # summary(model, input_size=(3, cfg['img_height'], cfg['img_width']))

    # Learning Rate
    lr = cfg['train']['base_lr']

    # Optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=2,
                                                           min_lr=1e-6,
                                                           factor=0.1,
                                                           verbose=True)

    # Early Stopping Callback
    early_stopping = EarlyStopping(patience=5, mode='max', verbose=True)

    logger.info(f"FOLD :{cfg['valid']['folds'][0]}")
    scaler = amp.GradScaler()

    # Training
    for epoch in range(1, EPOCHS + 1):

        train_loss, train_score = Engine.train(model=model,
                                               dataloader=train_loader,
                                               optimizer=optimizer,
                                               scaler=scaler,
                                               use_meta=cfg['use_meta'],)
        val_loss, val_score = Engine.evaluate(model=model,
                                              dataloader=valid_loader,
                                              use_meta=cfg['use_meta'],)

        scheduler.step(val_loss)

        # wandb logging
        wandb.log({'epoch': epoch, 'val_loss': val_loss,
                   'val_acc': val_score, 'loss': train_loss})

        epoch_len = len(str(EPOCHS))
        epoch_msg = (f"EPOCH [{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] "
                     f"train_loss: {train_loss:.5f} "
                     f"train_score: {train_score:.5f} "
                     f"valid_loss: {val_loss:.5f} "
                     f"valid_score: {val_score:.5f}")
        logger.info(epoch_msg)
        early_stopping(val_score, model, MODEL_PATH)
        if early_stopping.early_stop:
            logger.info('Early stopping')
            break

    # Prediction on test data
    prediction = Engine.predict(model=model,
                                model_path=MODEL_PATH,
                                dataloader=test_loader,
                                use_meta=cfg['use_meta'],)

    logger.info(f"Saving test prediction at {SUB_CSV}")
    save_prediction(prediction, SUB_CSV)


if __name__ == '__main__':
    trainer()
