from trainer.util import get_dataloader, CustomLoss, CONFIG, get_lr
from model import create_basemodel
from overparam.utils import overparameterize
from datasets import create_dataset
from torch.utils.data import random_split
import torch
from pathlib import Path
import numpy as np
import json

class Trainer:
    def __init__(
        self,
        dataset_name: str, # 'chestexpert' or ..
        loadpath: bool = False , # load model from loadpath
        overparam: dict = None, # {'depth': int, 'overparam': str ('conv', 'fc' or 'all')}

    ):
        # get the dataloader - 10% from traindata will be used for validation
        ds = create_dataset(name=dataset_name, mode='train', config=CONFIG)
        val_size = int(0.1 * len(ds))
        train_size = len(ds) - val_size


        train_dataset, val_dataset = random_split(ds, [train_size, val_size])
        self.trainloader = get_dataloader(train_dataset)
        self.valloader = get_dataloader(val_dataset)

        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # save from checkpoint or initiate new
        if loadpath:
            savepath = 'weights/' + dataset_name + f'_overparam_depth_{overparam['depth']}_{overparam['overparam']}'
            self.model = torch.load(Path.cwd()/savepath/'best.pth', weights_only=False)
            print(f"Saved model loaded successfully")
            with open(Path.cwd()/savepath/'losses.json', 'r') as f:
                self.losses = json.load(f) # keys: 'train', 'valid'
                f.close()
        else:
            self.model = create_basemodel()
            print(f" ---- New Model Initialised ---- ")
            savepath = 'weights/'+ dataset_name 
            if overparam is not None:
                self.model = overparameterize(
                    model=self.model,
                    depth=overparam['depth'],
                    overparam=overparam['overparam']
                )
                savepath += f'_overparam_depth_{overparam['depth']}_{overparam['overparam']}'
            self.losses = {'train': [], 'valid': []}
            
        self.savepath = Path.cwd()/savepath
        Path.mkdir(self.savepath, parents=True, exist_ok=True)
        
        # define loss function
        self.loss_fn = CustomLoss()

        # optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters())

        # epochs and steps
        self.epochs = CONFIG['train_config']['epochs']
        self.total_steps = self.epochs*len(self.trainloader)
        
    
    def train_step(self, X, y, step):
        # split in train and validation randomly
        X = X.to(self.device)
        y = y.to(self.device)

        pred_scores = self.model(X)
        loss = self.loss_fn(pred_scores, y)

        lr = get_lr(step, self.total_steps)
        for pg in self.optimiser.param_groups:
            pg['lr'] = lr

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()
    
    def validation(self):
        losses = []
        for X, y in self.valloader:

            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                pred_scores = self.model(X)
                loss = self.loss_fn(pred_scores, y)
                losses.append(loss.item())
        return np.mean(losses)
    
    def save_model(self):
        if np.min(self.losses['valid']) == self.losses['valid'][-1]:
            torch.save(self.model, self.savepath/'best.pth')
            print(f"---- Best model saved ------")

        if len(self.losses['train']) % 5 == 0:
            epochs = len(self.losses['train'])
            torch.save(self.model, self.savepath/f'epoch_{epochs}.pth')
            print("---- Latest model saved ------")
            with open(self.savepath/'losses.json', 'w') as f:
                json.dump(self.losses, f)
                f.close()
            print(f" --- losses saved ---")
        

    def train(self):
        self.model.to(self.device)
        steps_done = 0
        for epoch in range(self.epochs):
            train_losses = []
            for j, (X,y) in enumerate(self.trainloader):
                steps_done += 1
                train_loss = self.train_step(X, y, steps_done)
                train_losses.append(train_loss)
                if (j+1)%100 == 0:
                    print(f"Step {j+1}| epoch {epoch+1} | train loss: {train_loss: 0.3f}")
            
            # epoch train loss
            epoch_train_loss = np.mean(train_losses)
            self.losses['train'].append(epoch_train_loss)
            # Validation loss
            epoch_val_loss = self.validation()
            self.losses['valid'].append(epoch_val_loss)

            print("-"*30, "\n")
            print(f"EPOCH - {epoch+1} | train loss: {epoch_train_loss} | validation loss: {epoch_val_loss}\n")
            self.save_model()
            print("-"*30, "\n")



