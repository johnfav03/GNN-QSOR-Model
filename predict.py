import torch
from model import GCN
import numpy as np
import random
import pandas as pd

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss = 0
            loss = criterion(out.float(), data.y.float())
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    model = GCN()
    model.load_state_dict(torch.load('gcn_model.pth'))

    data_list = torch.load('processed-data/data.pt')
    victim = random.choice(data_list)

    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)
    
    model.eval()
    out = model(victim.x, victim.edge_index, victim.batch)

    output_feats = out[0].detach().numpy()
    truth_feats = victim.y[0].numpy()
    
    output_feats = [1 if x >= 0.5 else 0 for x in output_feats]

    df = pd.read_csv('processed-data/dravnieks.csv')
    targets = df.columns[6:].values

    out_smells = []
    corr_smells = []
    for i in range(len(targets)):
        if output_feats[i] == 1:
            out_smells.append(targets[i])
        if truth_feats[i] == 1:
            corr_smells.append(targets[i])

    print(f'ODOR PREDICTION TRIAL: {victim.name}')
    print(f'Predicted Odors: {out_smells}')
    print(f'Actual Odors: {corr_smells}')

