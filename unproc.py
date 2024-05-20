import torch
import pandas as pd

# Load the .pt file
data = torch.load('processed-data/main.pt')

# Check the type of the loaded data
df = pd.DataFrame()

# Check the type of the loaded data
if isinstance(data[1], dict):
    # If it's a dictionary, we need to handle each item
    for key, value in data[1].items():
        if isinstance(value, torch.Tensor):
            # Convert each tensor to a DataFrame
            tensor_df = pd.DataFrame(value.numpy())
            # Add key as prefix to the column names to avoid duplicate column names
            tensor_df.columns = [f"{key}_{col}" for col in tensor_df.columns]
            # Concatenate to the main DataFrame
            df = pd.concat([df, tensor_df], axis=1)
        else:
            print(f"Skipping {key} as it's not a tensor.")
elif isinstance(data, torch.Tensor):
    # If it's a single tensor, convert it directly
    df = pd.DataFrame(data.numpy())
    df.to_csv('data.csv', index=False)
    print('Saved data.csv')

df.to_csv('data.csv', index=False)