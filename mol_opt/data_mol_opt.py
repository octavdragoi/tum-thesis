import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from rdkit.Chem import MolFromSmiles

import numpy as np

np.random.seed(69)
# some inspiration from otgnn prop_dataset.py


class MolOptDataset(Dataset):
    def __init__(self, data_dir, data_type = "train_pairs", same_number_atoms = False, 
                ratio = None):
        self.data = []
        self.pairs = "pairs" in data_type

        with open('{}/{}.txt'.format(data_dir, data_type), 'r+') as datafile:
            for line in datafile.readlines():
                if ratio is not None:
                    if np.random.rand() > ratio:
                        continue
                if 'smiles' not in line:
                    smiles = line.strip().split(' ')

                    if self.pairs:
                        assert len(smiles) == 2
                        if same_number_atoms:
                            if MolFromSmiles(smiles[0]).GetNumAtoms() == \
                            MolFromSmiles(smiles[1]).GetNumAtoms():
                                self.data.append((smiles[0], smiles[1]))
                        else:
                            self.data.append((smiles[0], smiles[1]))
                    else:
                        assert len(smiles) == 1
                        self.data.append(smiles[0])

        if self.pairs:
            self.initial, self.optim = zip(*self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def get_loader(data_dir, data_type, batch_size, same_number_atoms = False,
               shuffle=False, num_workers=1, ratio = None):
    molopt_dataset = MolOptDataset(data_dir, data_type, same_number_atoms, ratio)

    if molopt_dataset.pairs:
        def combine_data(data):
            batch_initial, batch_optim = zip(*data)
            return batch_initial, batch_optim
    else:
        combine_data = None

    data_loader = DataLoader(
        molopt_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=combine_data,
        num_workers=num_workers)
    return data_loader