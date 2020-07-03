import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from rdkit.Chem import MolFromSmiles

# some inspiration from otgnn prop_dataset.py


class MolOptDataset(Dataset):
    def __init__(self, data_dir, same_number_atoms = False):
        self.data = []

        with open('%s/train_pairs.txt' % data_dir, 'r+') as datafile:
            for line in datafile.readlines():
                if 'smiles' not in line:
                    smiles = line.strip().split(' ')
                    assert len(smiles) == 2
                    if same_number_atoms:
                        if MolFromSmiles(smiles[0]).GetNumAtoms() == \
                           MolFromSmiles(smiles[1]).GetNumAtoms():
                            self.data.append((smiles[0], smiles[1]))
                    else:
                        self.data.append((smiles[0], smiles[1]))

        self.initial, self.optim = zip(*self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def get_loader(data_dir, batch_size, same_number_atoms = False,
               shuffle=False, num_workers=1):
    molopt_dataset = MolOptDataset(data_dir, same_number_atoms)

    def combine_data(data):
        batch_initial, batch_optim = zip(*data)
        return batch_initial, batch_optim

    data_loader = DataLoader(
        molopt_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=combine_data,
        num_workers=num_workers)
    return data_loader