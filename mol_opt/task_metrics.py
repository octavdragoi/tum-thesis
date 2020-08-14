import os
if "ntbk" in os.getcwd():
    os.chdir("..")

import sys
sys.path.append(os.path.join(os.getcwd(), "iclr19-graph2graph", "props"))

import importlib  
props = importlib.import_module("iclr19-graph2graph.props")

from molgen.dataloading.feat2smiles import feat2smiles
from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES
from rdkit import Chem

def measure_task(X, pred_pack):
    # unpack
    yhat_labels, _, scope = pred_pack
    symbols_labels, charges_labels, bonds_labels = yhat_labels

    # get predicted smiles
    bond_idx = 0
    res = {}
    res["similarity"]=0
    res["QED"]=0
    res["penlog"]=0
    for mol_idx, (st, le) in enumerate(scope):
        symbols_labels_crt = symbols_labels[st:st+le]
        charges_labels_crt = charges_labels[st:st+le]
        bonds_labels_crt = bonds_labels[bond_idx:bond_idx+le*le].view(le, le, -1)
        
        pmol = feat2smiles(SYMBOLS, FORMAL_CHARGES, BOND_TYPES, 
                symbols_labels_crt, charges_labels_crt, bonds_labels_crt)
        
        bond_idx += le * le

        res["similarity"] += props.similarity(pmol, Chem.MolToSmiles(X.rd_mols[mol_idx]))
        res["QED"] += props.qed(pmol)
        res["penlog"] += props.penalized_logp(pmol)

    # return unaveraged dict
    return res
