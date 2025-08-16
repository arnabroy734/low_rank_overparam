import torch
from datasets import create_dataset
from trainer.util import get_dataloader
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn

CHESTEXPERT_LABELS = {0:'Cardiomegaly', 1:'Edema', 2:'Consolidation', 3:'Atelectasis', 4:'Pleural Effusion'}

def evaluate_model(modelpath: str, dataset_name: str):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        f.close()
    ds = create_dataset(name=dataset_name, mode='valid', config=config)
    loader = get_dataloader(ds)
    model = torch.load(modelpath, weights_only=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    actual_labels = None
    pred_scores = None
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            scores = model(X)
            scores = nn.Sigmoid()(scores)
            y = y.cpu().numpy()
            scores = scores.cpu().numpy()
            if actual_labels is None:
                actual_labels = y
                pred_scores = scores
            else:
                actual_labels = np.vstack((actual_labels, y))
                pred_scores = np.vstack((pred_scores, scores))
    evaluation = {}
    for c in range(actual_labels.shape[1]):
        if dataset_name == 'chestexpert':
            name = CHESTEXPERT_LABELS[c]
        evaluation[name] = roc_auc_score(actual_labels[:, c], pred_scores[:,c])
    evaluation['microAUC'] = roc_auc_score(actual_labels, pred_scores, average='micro')
    return evaluation
