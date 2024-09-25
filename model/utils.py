import torch
import random as rd
import sys

amino_acid_to_number = {'A': 0,
     'R': 1,
     'N': 2,
     'D': 3,
     'C': 4,
     'E': 5,
     'Q': 6,
     'G': 7,
     'H': 8,
     'I': 9,
     'L': 10,
     'K': 11,
     'M': 12,
     'F': 13,
     'P': 14,
     'S': 15,
     'T': 16,
     'W': 17,
     'Y': 18,
     'V': 19, 
     '_': 20                  
       }

def tokenize(seq):
    """
    Generate tokenized aa sequence from raw AA sequence
    """
    numeric_sequence = [amino_acid_to_number[aa] for aa in seq]
    return numeric_sequence

def get_design(X, order, n_terms):

    L, AA_size = X.shape[1:]
    X = X.float()
    X1hf = X.view(-1, L*AA_size)
    
    Feat = []
    for _ in range(n_terms):
        feat_vec = torch.zeros(L, AA_size)
        seq_idx = rd.randint(0, X.shape[0])
        pos_idx = rd.sample(range(L), order)
        feat_vec[pos_idx] = X[seq_idx, pos_idx]
        Feat.append(feat_vec.flatten())

    Feat = torch.stack(Feat).T

    design = (torch.matmul(X1hf, Feat) == order).float()
    
    return design

def simulate_pheno_epi(X, n_terms, stds):
    designs = {}
    coeffs = {}
    phenos = {}
    for order in n_terms.keys():
        designs[order] = get_design(X, order, n_terms[order])
        coeffs[order] = torch.normal(0., stds[order], (n_terms[order], ))
        phenos[order] = torch.matmul(designs[order], coeffs[order])
        
    return designs, torch.sum(torch.stack(list(phenos.values())), 0)

def simulate_pheno_add(X):
    L, AA_size = X.shape[1:]
    X1hf = X.view(-1, L*AA_size)
    coeffs = torch.normal(0., 1, (L*AA_size, ))
    return torch.matmul(X1hf, coeffs)

class Tee:
    """
    A file-like object to redirect stdout to both a file and the console. 
    It will write and flush immediately to both outputs.
    """
    def __init__(self, file):
        self.file = file
        self.console = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.console.write(message)
        self.console.flush()

    def flush(self):
        self.file.flush()
        self.console.flush()
