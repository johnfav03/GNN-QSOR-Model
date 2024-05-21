import pandas as pd
import pandasql as psql
from rdkit import Chem
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "F": 5, "Cl": 6, "Br": 7}


def process_goodscents():
    behavior = pd.read_csv('goodscents-data/behavior.csv')
    molecules = pd.read_csv('goodscents-data/molecules.csv')
    stimuli = pd.read_csv('goodscents-data/stimuli.csv')

    query = """
    SELECT molecules.CID, molecules.MolecularWeight, molecules.IsomericSMILES, molecules.IUPACName, molecules.name, behavior.Descriptors
    FROM molecules
    INNER JOIN stimuli ON molecules.CID = stimuli.CID
    INNER JOIN behavior ON stimuli.Stimulus = behavior.Stimulus
    """

    final_df = psql.sqldf(query, locals())
    return final_df

def process_dravnieks():
    behavior = pd.read_csv('dravnieks-data/behavior.csv')
    molecules = pd.read_csv('dravnieks-data/molecules.csv')
    stimuli = pd.read_csv('dravnieks-data/stimuli.csv')

    query = """
    SELECT molecules.CID, molecules.MolecularWeight, molecules.IsomericSMILES, molecules.IUPACName, molecules.name, stimuli.Stimulus
    FROM molecules
    INNER JOIN stimuli ON molecules.CID = stimuli.CID
    """

    result_df = psql.sqldf(query, locals())
    final_df = result_df.merge(behavior, on='Stimulus')
    return final_df

def convert_to_graph(df):
    result = []
    for i in range(len(df)):
        row = df.iloc[i]
        smile = row["IsomericSMILES"]
        name = row["IUPACName"]
        atomic_idxs = []
        atomic_nums = []
        atomic_mass = []
        atomic_aroma = []
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ATOM_TYPES.keys():
                continue
            atomic_idxs.append(ATOM_TYPES[atom.GetSymbol()])
            atomic_nums.append(atom.GetAtomicNum())
            atomic_mass.append(atom.GetMass())
            atomic_aroma.append(atom.GetIsAromatic())
        
        row = []
        col = []
        for bond in mol.GetBonds():
            beg = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row += [beg, end]
            col += [end, beg]

        edge_idx = torch.tensor([row, col], dtype=torch.long)
        perm = (edge_idx[0] * mol.GetNumAtoms() + edge_idx[1]).argsort()
        edge_idx = edge_idx[:, perm]

        x = F.one_hot(torch.tensor(atomic_idxs), num_classes=len(ATOM_TYPES)).to(torch.float)
        adtl = torch.tensor([atomic_nums, atomic_mass, atomic_aroma], dtype=torch.float).t().contiguous()
        x = torch.cat([x, adtl], dim=-1)
        
        targets = df.columns[6:].values
        for t in targets:
            df[t] = df[t].to_numpy() / df[t].to_numpy().max()
            df[t] = df[t].apply(lambda x: 1 if x >= 0.5 else 0)
        y = torch.unsqueeze(torch.tensor(df[targets].iloc[i].values), 0)

        data = Data(
            x=x,
            y=y,
            edge_index=edge_idx,
            name=name,
        )

        result.append(data)

    torch.save(result, "processed-data/data.pt")

if __name__ == "__main__":
    goodscents = process_goodscents()
    dravnieks = process_dravnieks()
    goodscents.to_csv('processed-data/goodscents.csv', index=False)
    dravnieks.to_csv('processed-data/dravnieks.csv', index=False)
    convert_to_graph(dravnieks)