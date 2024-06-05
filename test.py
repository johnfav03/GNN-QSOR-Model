import torch
from torch_geometric.loader import DataLoader
from model import GCN
import numpy as np
import random
import pandas as pd
from torchmetrics.functional.classification import auroc
import warnings
import pprint

warnings.simplefilter('ignore')

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    bin_auroc_score = 0
    mlt_auroc_score = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss = 0
            loss = criterion(out.float(), data.y.float())
            total_loss += loss.item()
            bin_auroc_score = auroc(out, data.y, task='binary', num_classes=138)
            mlt_auroc_score = auroc(out, data.y, task='multilabel', num_labels=138)
    return total_loss / len(loader), bin_auroc_score, mlt_auroc_score

if __name__ == "__main__":
    trainset = 'goodscents'
    dataset = 'leffingwell'

    model = GCN()
    model.load_state_dict(torch.load('{}_gcn_model.pth'.format(trainset)))
    criterion = torch.nn.BCELoss()

    test_data = torch.load('processed-data/{}.pt'.format(dataset))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    test_loss, bin_auroc_score, mlt_auroc_score = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Binary AUROC: {bin_auroc_score:.4f}, AUROC: {mlt_auroc_score:.4f}')

