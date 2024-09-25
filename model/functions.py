import pandas as pd
import torch
import numpy as np
from torch.nn.functional import one_hot



def get_A2N_list(seqs):
    """
    Generate dictionary for tokenizing protein sequences
    """
    seqs_ = list([list(seq) for seq in seqs])
    seqs__ = pd.DataFrame(seqs_)
    alphabet_by_site = [list(seqs__.iloc[:, i].unique()) for i in range(seqs__.shape[1])]
    A2N_list = [dict(zip(alphabet, range(len(alphabet)))) for alphabet in alphabet_by_site]        
    return A2N_list

def tokenize(seq, A2N_list):
    """
    Generate tokenized sequences given seqs and AA to integer dictionary
    """
    numeric_sequence = [A2N_list[i][seq[i]] for i in range(len(A2N_list))]
    return numeric_sequence

def make_train_val_test_lists_rand(datafile, num_train, num_test, seed=None):
    """
    Generate nonoverlapping lists for train, test, and val
    """
    if seed is None:
        pass
    else: 
        np.random.seed(seed)
        
    sub_list = np.random.choice(range(len(datafile)), num_train, replace=False) #random list containing train and val
    comp_list = list(set(range(len(datafile))).difference(sub_list)) # random list containing test
    val_list = np.random.choice(sub_list, min(5000, int(.1*len(sub_list))), replace=False)
    train_list = list(set(sub_list).difference(val_list))
    test_list = np.random.choice(comp_list, num_test, replace=False)
    
    return train_list, val_list, test_list, sub_list

def prepare_data(datafile):
    """
    Prepares phenotype and sequence data from the input datafile for modeling.

    This function extracts phenotypes and tokenizes sequences from the provided datafile. 
    It performs the following steps:
        1. Extracts and processes phenotype data.
        2. Tokenizes mutated sequences.
        3. Filters and processes the sequences to retain variable sites.
        4. Converts sequences to one-hot encoding.

    Args:
        datafile (pandas.DataFrame): A DataFrame containing the input data with the columns
                                     'DMS_score' for phenotypes and 'mutated_sequence' for sequences.
    
    """
    
    
    phenotypes = torch.tensor(list(datafile.DMS_score)).float() 
    phenotypes = phenotypes.unsqueeze(1)
    
    A2N_list = get_A2N_list(datafile.mutated_sequence) # Get position-wise alphabet
    
    # Tokenize sequence, only variable sites and site specific set of AAs are used
    seqs = [tokenize(mutant, A2N_list) for mutant in datafile.mutated_sequence]
    seqs = np.array(seqs)
    seqs_df = pd.DataFrame(seqs)
    counts = {i: seqs_df[i].value_counts() for i in range(seqs_df.shape[1])}
    n_aas = [len(counts[i]) for i in range(len(counts))]
    sites_var = np.where(np.array(n_aas) != 1)
    seqs = seqs[:, sites_var]
    seqs = torch.tensor(seqs)
    seqs = seqs.squeeze(1)
    seqs1h = one_hot(seqs)

    return phenotypes, seqs, seqs1h