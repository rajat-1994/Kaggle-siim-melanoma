import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import metric
from torch.cuda import amp


class Engine():
    """
    This class contains train,evalute and predict functions for the model
    """
    @staticmethod
    def train(model,
              dataloader,
              optimizer,
              scaler,
              device="cuda"):
        model.train()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []

        print(f"{'='*15}Training{'='*15}")
        for _, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
            counter = counter + 1

            image = data["image"]
            targets = data["target"]
            image = image.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(image)

                loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            final_loss += loss

            final_outputs.append(outputs)
            final_targets.append(targets)

        final_outputs = torch.cat(final_outputs).detach().cpu().numpy()
        final_targets = torch.cat(final_targets).detach().cpu().numpy()

        auc_metric = metric(
            final_targets, final_outputs)

        return final_loss / counter, auc_metric

    @staticmethod
    def evaluate(model, dataloader, device="cuda"):
        with torch.no_grad():
            model.eval()
            final_loss = 0
            counter = 0
            final_outputs = []
            final_targets = []
            print(f"{'='*15}Evaluating{'='*15}")
            for _, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
                counter = counter + 1
                image = data["image"]
                target = data["target"]

                image = image.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                outputs = model(image)

                loss = nn.BCEWithLogitsLoss()(outputs, target.view(-1, 1))

                final_loss += loss

                final_outputs.append(outputs)
                final_targets.append(target)

            final_outputs = torch.cat(final_outputs).cpu().numpy()
            final_targets = torch.cat(final_targets).cpu().numpy()

            auc_metric = metric(final_targets, final_outputs)

        return final_loss / counter, auc_metric

    @staticmethod
    def predict(model, model_path, dataloader, device="cuda"):

        # load model
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        print(f"Predicting from {model_path}")
        # no_grad so that we dont keep track of gradient during inference
        with torch.no_grad():
            model.eval()
            counter = 0
            final_outputs = []
            for _, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
                counter = counter + 1
                image = data["image"]

                image = image.to(device, dtype=torch.float)

                outputs = model(image)

                final_outputs.append(outputs)

            final_outputs = torch.cat(final_outputs).cpu().numpy()

        return np.array(final_outputs)
