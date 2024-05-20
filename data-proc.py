import pandas as pd
import pandasql as psql
import pyrfume
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data

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
    for row in df:
        smile = row["IsometricSMILES"]
        atom_idxs = []
        mol = Chem.MolFromSmiles(mol)
        mol = Chem.AddHs(mol)
        for atom in mol.getAtoms():
            if atom.GetSymbol() not in ATOM_TYPES.keys():
                flag = 1
                continue
            atom_idxs.append(
                ATOM_TYPES[atom.GetSymbol()]
            )  # encoded number for atom symbol
            atomic_number.append(atom.GetAtomicNum())


def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()

    return x, edge_index
 

if __name__ == "__main__":
    goodscents = process_goodscents()
    dravnieks = process_dravnieks()
    print(len(dravnieks.columns))
    goodscents.to_csv('processed-data/goodscents.csv', index=False)
    dravnieks.to_csv('processed-data/dravnieks.csv', index=False)
