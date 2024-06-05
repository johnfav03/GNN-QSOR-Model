import pandas as pd
import pandasql as psql
from rdkit import Chem
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import json
import warnings
import pprint

warnings.simplefilter('ignore')


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

    df = psql.sqldf(query, locals())
    df['Descriptors'] = df['Descriptors'].str.split(';')

    victims = []
    data_file = 'goodscents-data/stddict.json'
    with open(data_file, 'r') as file:
        transforms = json.load(file)
        for target in transforms:
            df[target] = 0
        for target in transforms:
            for k in range(df.shape[0]):
                if df.loc[k, 'Descriptors']:
                    for item in transforms[target]: 
                        if item in df.loc[k, 'Descriptors']:
                            df.loc[k, target] = 1
                            break
                else:
                    victims.append(k)

    df = df.drop(index=victims)
    df = df.drop(columns='Descriptors')
    return df

def process_dravnieks():
    behavior = pd.read_csv('dravnieks-data/behavior.csv')
    molecules = pd.read_csv('dravnieks-data/molecules.csv')
    stimuli = pd.read_csv('dravnieks-data/stimuli.csv')

    query = """
    SELECT molecules.CID, molecules.MolecularWeight, molecules.IsomericSMILES, molecules.IUPACName, molecules.name, stimuli.Stimulus
    FROM molecules
    INNER JOIN stimuli ON molecules.CID = stimuli.CID
    """

    df = psql.sqldf(query, locals())
    df = df.merge(behavior, on='Stimulus')
    df = df.drop(columns='Stimulus')
    targets = df.columns[5:].values

    for row in range(df.shape[0]):  # dravnieks 146
        df.loc[row, targets] = df.loc[row, targets].apply(lambda x: 1 if x >= df.loc[row, targets].mean() else 0)

    data_file = 'dravnieks-data/stddict.json'
    with open(data_file, 'r') as file:
        transforms = json.load(file)
        for target in transforms:
            df[target] = 0
        for target in transforms:
            for k in range(df.shape[0]):
                for item in transforms[target]:
                    if isinstance(item, list):
                        # and case
                        flag = len(item)
                        for i in item:
                            if df.loc[k, i] == 1:
                                flag -= 1
                        if flag == 0:
                            df.loc[k, target] = 1    
                    else:
                        # or case
                        if df.loc[k, item] == 1:
                            df.loc[k, target] = 1
                            break

    df = df.drop(columns=targets)
    return df

def process_leffingwells():
    behavior = pd.read_csv('leffingwells-data/behavior.csv')
    molecules = pd.read_csv('leffingwells-data/molecules.csv')

    query = """
    SELECT molecules.CID, molecules.MolecularWeight, molecules.IsomericSMILES, molecules.IUPACName, molecules.name, behavior.Descriptors
    FROM molecules
    INNER JOIN behavior ON molecules.IsomericSMILES = behavior.IsomericSMILES
    """

    df = psql.sqldf(query, locals())
    df['Descriptors'] = df['Descriptors'].str.split(';')

    victims = []
    data_file = 'leffingwells-data/stddict.json'
    out_blob = {}
    with open(data_file, 'r') as file:
        transforms = json.load(file)
        for target in transforms:
            df[target] = 0
            out_blob[target] = [f"{target}"]
        for target in transforms:
            flag = False
            for k in range(df.shape[0]):
                if df.loc[k, 'Descriptors']:
                    if target in df.loc[k, 'Descriptors']:
                        df.loc[k, target] = 1

                        flag = True
                else:
                    victims.append(k)
    with open('leffingwells-data/stddict.json', 'w') as file:
        json_str = json.dumps(out_blob, indent=4)
        file.write(json_str)

    df = df.drop(index=victims)
    df = df.drop(columns='Descriptors')
    return df


def convert_to_graph(file, df):
    result = []
    targets = df.columns[5:].values
    for i in range(len(df)):
        print(f"\r[{file}]=> {i} / {len(df) - 1}", end='')
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

        x = F.one_hot(torch.tensor(atomic_idxs).long(), num_classes=len(ATOM_TYPES)).to(torch.float)
        adtl = torch.tensor([atomic_nums, atomic_mass, atomic_aroma], dtype=torch.float).t().contiguous()
        x = torch.cat([x, adtl], dim=-1)

        if edge_idx.numel() == 0 or edge_idx.max() >= len(x):
            continue
        
        for t in targets:
            df[t] = df[t].to_numpy() / df[t].to_numpy().max()
            df[t] = df[t].apply(lambda x: 1 if x >= 0.5 else 0)
        y = torch.unsqueeze(torch.tensor(df[targets].iloc[i].values), 0)

        if not name:
            name = "placeholder"
        data = Data(
            x=x,
            y=y,
            edge_index=edge_idx,
            name=name,
        )

        result.append(data)

    torch.save(result, file)
    print("")

if __name__ == "__main__":
    # gs = "goodscents-data/stddict.json"
    # dn = "dravnieks-data/stddict.json"
    # gs_data = {}
    # dn_data = {}
    # with open(gs, 'r') as file:
    #     gs_data = json.load(file)
    # with open(dn, 'r') as file:
    #     dn_data = json.load(file)
    # with open(gs, 'w') as file:
    #     pretty_json_str = pprint.pformat(gs_data, compact=True, width=500).replace("'",'"')
    #     file.write(pretty_json_str)
    # with open(dn, 'w') as file:
    #     pretty_json_str = pprint.pformat(dn_data, compact=True, width=500).replace("'",'"')
    #     file.write(pretty_json_str)

    # goodscents = process_goodscents()
    # dravnieks = process_dravnieks()
    leffingwells = process_leffingwells()
    # goodscents.to_csv('processed-data/goodscents.csv', index=False)
    # dravnieks.to_csv('processed-data/dravnieks.csv', index=False)
    leffingwells.to_csv('processed-data/leffingwells.csv', index=False)
    # goodscents = pd.read_csv('processed-data/goodscents.csv')
    # dravnieks = pd.read_csv('processed-data/dravnieks.csv')
    # leffingwells = pd.read_csv('processed-data/leffingwells.csv')
    # convert_to_graph("processed-data/goodscents.pt", goodscents)
    # convert_to_graph("processed-data/dravnieks.pt", dravnieks)
    # convert_to_graph("processed-data/leffingwells.pt", leffingwells)