from model import GCN


if __name__ == "__main__":
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    out = model(graph)
    loss = F.mse_loss(out, target)  # Define your target
    loss.backward()
    optimizer.step()