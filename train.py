import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from model import GCN
import numpy as np
import random
from torchmetrics.functional.classification import auroc

NUM_EPOCHS = 300

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.float(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.float(), data.y.float())
            total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    auroc_score = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss = 0
            loss = criterion(out.float(), data.y.float())
            total_loss += loss.item()
            auroc_score = auroc(out, data.y, task='binary', num_classes=146)
    return total_loss / len(loader), auroc_score

if __name__ == "__main__":
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.BCELoss()

    seed = 298
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    data_list = torch.load('processed-data/data.pt')
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=seed)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    test_loss, auroc_score = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Binary AUROC: {auroc_score:.4f}')

    torch.save(model.state_dict(), 'gcn_model.pth')