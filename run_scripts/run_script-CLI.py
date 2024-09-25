source_path = "/blue/juannanzhou/ProteinLLE/"
data_path = "/blue/juannanzhou/ProteinLLE/Data/Data_prepared/"

import argparse

def main(device, data_name, train_percent):
    print(f"Device: {device}")
    print(f"Data Name: {data_name}")
    print(f"Training Percentage: {train_percent}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument('--device', type=str, required=True, help='Device to use')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the data set')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix of the study')
    parser.add_argument('--train_percent', type=float, required=True, help='Percentage of data to use for training')
    parser.add_argument('--test_distance', action="store_true")    
    parser.add_argument('--seed', type=int, help='Random seed for sampling')
    parser.add_argument('--specify_params', action='store_true', default=False, help='If use prespecified hyparameters')
    parser.add_argument('--best_params', type=str, help='Path to text file containing paths to best hyper-parameters')
    parser.add_argument('--fit_linear', action='store_true', default=False, help='If use prespecified hyparameters')
    parser.add_argument('--specify_train', action='store_true', default=False, help='If use prespecified train test splits')
    parser.add_argument('--train_list', type=str, help='Path to pickle file containing paths train_list')
    parser.add_argument('--specify_val', action='store_true', default=False, help='If use prespecified train test splits')    
    parser.add_argument('--val_list', type=str, help='Path to pickle file containing paths val_list')
    parser.add_argument('--specify_test', action='store_true', default=False, help='If use prespecified train test splits')    
    parser.add_argument('--test_list', type=str, help='Path to pickle file containing paths test_list')    
    parser.add_argument('--iter2', default=100, type=int, help='Optuna iterations for pairwise model')
    parser.add_argument('--iter4', default=100, type=int, help='Optuna iterations for pairwise model')
    parser.add_argument('--iter8', default=100, type=int, help='Optuna iterations for pairwise model')

    args = parser.parse_args()
    main(args.device, args.data_name, args.train_percent)

print("running script")
import os
import sys
import shutil
import csv
import numpy as np
import math
import json
import pickle
from functools import partial
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import gc
import random
sys.path.append(source_path + 'model')
from utils import amino_acid_to_number, tokenize, Tee
from functions import get_A2N_list, tokenize, make_train_val_test_lists_rand, prepare_data
from models import make_predictions, ProtDataset, Transformer_2k, LinearModel

def split_list(list_to_split, ratio):
    length_total = len(list_to_split)
    length_part1 = int(length_total * ratio)
    length_part2 = length_total - length_part1
    random.shuffle(list_to_split)
    part1 = list_to_split[:length_part1]
    part2 = list_to_split[length_part1:]
    return part1, part2

current_pid = os.getpid()
print("Process PID:", current_pid)

outpath = "../output/"
device, data_name, prefix, train_percent = args.device, args.data_name, args.prefix, args.train_percent

# Load hyperparameters
if args.specify_params:
    print("Using pre-specified hyperparameters")
    with open(args.best_params, 'r') as file:
        best_params_paths = file.read().split()

    best_params_TF = {}

    for num_layers in [1, 2, 3]:
        with open(best_params_paths[num_layers - 1], 'rb') as file:
            best_params = pickle.load(file)
        best_params_TF[num_layers] = best_params

# Set random seed for sampling train, test
if args.seed is not None:
    seed = args.seed
else:
    seed = None

print(f"using dataset {data_name}")

layers_to_test = [1, 2, 3]
linear_model_n_trials = 20
n_trials = {1:args.iter2, 2:args.iter4, 3:args.iter8}


# make folder for storing analysis outputs
study_id = "_".join([data_name, prefix, str(train_percent) + "%"])
print(f"performing study {study_id}")

matching_folders = [folder for folder in os.listdir(outpath) 
                    if study_id in folder and os.path.isdir(os.path.join(outpath, folder)) ]
if len(matching_folders) == 0:
    rep = 0
else: rep = np.max([int(folder.split("_")[-1]) for folder in matching_folders]) + 1

try:
    results_path = outpath + "_".join([study_id, "rep", str(rep)])
    print(f"saving results to {results_path}")
    os.makedirs(results_path)

except:
    rep += 1
    results_path = outpath + "_".join([study_id, "rep", str(rep)])
    print(f"saving results to {results_path}")
    os.makedirs(results_path)
    
with open(os.path.join(results_path, study_id), 'w') as file:
    pass


# Save a summary file
summary_path = os.path.join(results_path, "summary.txt")
with open(summary_path, 'w') as file:
    file.write(args.data_name + '\n')
with open(summary_path, 'a') as file:
    file.write(str(args.seed) + '\n')
if args.train_list is not None:
    with open(summary_path, 'a') as file:
        file.write(args.train_list + '\n')
if args.val_list is not None:
    with open(summary_path, 'a') as file:
        file.write(args.val_list + '\n')


# Save models.py and run scripts for reference
source_script_path = source_path + "run_scripts/run_script-CLI.py"
filename = os.path.basename(source_script_path)
shutil.copy(source_script_path, results_path)
source_script_path = source_path + "model/models.py"
filename = os.path.basename(source_script_path)
shutil.copy(source_script_path, results_path)

# Direct output to a text file in real time
log_file = open(os.path.join(results_path, "output.txt"), "w")
sys.stdout = Tee(log_file)

# Make csv file storing R2 values
R2s = pd.DataFrame(columns=['Model', 'R2'])
R2s.to_csv(os.path.join(results_path, 'R2s.csv'), index=False)

# prepare data
in_path = data_path + data_name + ".csv"
datafile = pd.read_csv(in_path, index_col=None)
phenotypes, seqs, seqs1h = prepare_data(datafile)
n, L, AA_size = seqs1h.shape
print(f"sequence length = {L}; ", f"AA_size = {AA_size}")

        
        
if args.specify_train and not args.specify_val:
    with open(args.train_list, 'rb') as file:
        train_val_list = pickle.load(file)
    print("Using prespecified train list")
    train_list, val_list = split_list(train_val_list, .9)
    
    comp_list = list(set(range(n)).difference(train_val_list))
    if args.specify_test:
        with open(args.test_list, 'rb') as file:
            test_list = pickle.load(file)
        num_test = len(test_list)
    else:
        num_test = min(10000, len(comp_list))
        test_list = random.sample(comp_list, num_test)
    
elif args.specify_train and args.specify_val:
    with open(args.train_list, 'rb') as file:
        train_list = pickle.load(file)
    print("Using prespecified train list")
    with open(args.val_list, 'rb') as file:
        val_list = pickle.load(file)
    print("Using prespecified validation list")
    comp_list = list(set(range(n)).difference(train_list + val_list)) 
    if args.specify_test:
        with open(args.test_list, 'rb') as file:
            test_list = pickle.load(file)
        num_test = len(test_list)
    else:
        num_test = min(10000, len(comp_list))
        test_list = random.sample(comp_list, num_test)
            
else: 
    num_train = int(.01*train_percent*len(datafile))
    num_test = min(10000, len(datafile) - num_train)
    if args.test_distance:
        print("preparing test samples by sampling distance classes")
        test_list_by_d = {}
        for d in datafile.hd.unique():
            inds = np.where(np.array(datafile['hd'] == d))[0]
            min_num = int(.8*(len(inds)))
            test_list_by_d[d] = np.random.choice(inds, min(500, min_num), replace=False)
        test_list = np.concatenate(list(test_list_by_d.values()))

        comp_list = list(set(range(n)).difference(test_list))
        train_val_list = random.sample(comp_list, min(num_train, len(comp_list)))

        train_list, val_list = split_list(train_val_list, .9)
    else: 
        train_list, val_list, test_list, train_val_list = make_train_val_test_lists_rand(datafile, num_train, num_test, seed)    

        
print(f"number of training samples = {len(train_list)}", 
      f"number of validation samples = {len(val_list)}", 
      f"number of test samples = {len(test_list)}")

with open(os.path.join(results_path, 'train_list.pkl'), 'wb') as file:
    pickle.dump(train_list, file)
with open(os.path.join(results_path, 'val_list.pkl'), 'wb') as file:
    pickle.dump(val_list, file)
with open(os.path.join(results_path, 'test_list.pkl'), 'wb') as file:
    pickle.dump(test_list, file)


#############################
#############################
#############################
#############################
#############################
####### LINEAR MODEL ########
#############################
#############################
#############################
#############################
#############################
#############################

if args.fit_linear:
    print("Fitting linear regression")
    model_name = "Linear"
    X = seqs1h.float().to(device)
    y = phenotypes.to(device)

    X_train, y_train = X[train_list], y[train_list]
    X_val, y_val = X[val_list], y[val_list]
    X_test, y_test = X[test_list], y[test_list]

    train_dataset = ProtDataset(X_train, y_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=1000,
                                   shuffle=True,
                                   drop_last=False)

    val_dataset = ProtDataset(X_val, y_val)
    val_loader = data.DataLoader(val_dataset,
                                batch_size=1000,
                                shuffle=False,
                                drop_last=False)

    test_dataset = ProtDataset(X_test, y_test)
    test_loader = data.DataLoader(test_dataset,
                                batch_size=1000,
                                shuffle=False,
                                drop_last=False)

    def lin_objective(trial):
        global criterion_best, model_best

        dropout_p = trial.suggest_float('dropout_p', 0.00, 0.1)
        fc_out_norm = trial.suggest_categorical('fc_out_norm', [True, False])
        n_epochs = trial.suggest_int('n_epochs', 30, 300)

        learning_rate = trial.suggest_float('learning_rate', .0001, 0.01, log=True)
        model = LinearModel(L, AA_size, dropout_p, fc_out_norm).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        r2_test_log = []
        for epoch in range(n_epochs):

                model.train()
                total_loss = 0
                for batch_inputs, batch_targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader)}")
                    model.eval()
                    pred, true = make_predictions(model, val_loader)
                    print(pearsonr(pred, true)[0]**2)
                    if pearsonr(pred, true)[0]**2 == "nan":
                        break
                    r2_test_log.append(pearsonr(pred, true)[0]**2)
                    if (len(r2_test_log) > 2) & ((np.array(r2_test_log)[-3:] < .1).sum() == 3):
                        print("no improvement in last three steps, aborting...")
                        break

        model.eval()
        pred, true = make_predictions(model, val_loader)
        criterion = pearsonr(pred, true)[0]**2     
        print(f"criterion = {criterion}")
        if criterion > criterion_best:
            print("Found better hyperparameter, update model")
            criterion_best = criterion
            model_best = model

        return criterion


    ##########################
    ##########################
    ##########################
    ##########################
    # Use Optuna to find best hyperparameter

    criterion_best = 0.
    study = optuna.create_study(direction='maximize')
    study.optimize(lin_objective, n_trials=linear_model_n_trials)

    # Print the best hyperparameters
    best_trial = study.best_trial
    print("Best Trial:")
    print(f"  Criterion: {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")  

    best_hyper_parameters = {}
    for key, value in best_trial.params.items():
        best_hyper_parameters[key] = value
    torch.save(model_best, os.path.join(results_path, "Linear" + "_BestModel"))        

    model_best.eval()
    pred, true = model_best(X_test.flatten(1)).flatten().detach().cpu().numpy(), y_test.flatten().detach().cpu().numpy()
    # save predictions
    pd.DataFrame({"prediction": pred, "true": true}).to_csv(os.path.join(results_path, model_name + "_predictions.csv"), index=False)

    r2_test = pearsonr(pred, true)[0]**2
    print(f"{model_name} achieved R2 = {r2_test}")

    with open(os.path.join(results_path, "R2s.csv"), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[model_name, r2_test]])

    del r2_test, model_best
    
#############################
#############################
#############################
#############################
#### TRANSFORMER MODELS #####
#############################
#############################
#############################
#############################
#############################

seqs_ex = seqs + AA_size*torch.tensor(range(L))
X = seqs_ex.to(device)
y = phenotypes.to(device)
X_train, y_train = X[train_list], y[train_list]
X_val, y_val = X[val_list], y[val_list]
X_test, y_test = X[test_list], y[test_list]

train_dataset = ProtDataset(X_train, y_train)
val_dataset = ProtDataset(X_val, y_val)
val_loader = data.DataLoader(val_dataset,
                            batch_size=500,
                            shuffle=False,
                            drop_last=False)

test_dataset = ProtDataset(X_test, y_test)
test_loader = data.DataLoader(test_dataset,
                            batch_size=100,
                            shuffle=False,
                            drop_last=False)

num_heads = 4
input_dim = AA_size*L
output_dim = 1
learning_rate = 0.001

def objective(trial):
    global criterion_best, model_best
    hidden_dim_h = trial.suggest_int('hidden_dim_h', 10, 200)
    dropout = trial.suggest_float('dropout', 0.001, 0.35)
    batch_size = trial.suggest_int('batch_size', 200, 1000)
    # num_heads = trial.suggest_int('num_heads', 1, 4)
    n_epochs = trial.suggest_int('n_epochs', 30, 300)
    # learning_rate = trial.suggest_float('learning_rate', .0001, 0.001, log=True)
    print(f"Build model with {num_layers} layers of attention")
    model = model_class(L, input_dim, hidden_dim_h*num_heads, num_layers, num_heads, dropout).to(device)
    
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    r2_test_log = []
    try: 
        for epoch in range(n_epochs):
                model.train()
                total_loss = 0
                for batch_inputs, batch_targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader)}")
                    model.eval()
                    pred, true = make_predictions(model, val_loader)
                    val_r2 = pearsonr(pred, true)[0]**2
                    print(val_r2)
                    if math.isnan(val_r2):
                        print("nan encountered")
                        break
                    r2_test_log.append(val_r2)
                    stop = (len(r2_test_log) > 2) & ((np.array(r2_test_log)[-3:] < .1).sum() == 3)
                    if stop:
                        print("no improvement in last three steps, aborting...")
                        break
                    gc.collect()
                    torch.cuda.empty_cache()
    except: print("training failed")
    try:
        model.eval()
        pred, true = make_predictions(model, val_loader)
        criterion = pearsonr(pred, true)[0]**2    
        # write to Optuna log file 
        with open(os.path.join(results_path, optuna_log_file), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[criterion]])
        # Update best model
        print(f"criterion = {criterion}")
        if criterion > criterion_best:
            print("Found better hyperparameter, update model")
            criterion_best = criterion
            model_best = model
        return criterion
    except:
        return 0

    del model
    gc.collect()
    torch.cuda.empty_cache()

    
for num_layers in layers_to_test:
    model_class = Transformer_2k
    model_name = "TF_" + str(num_layers)
    
    #Use Optuna to find best hyperparameters
    ##########################
    ##########################
    ##########################
    ##########################
    ##########################    
    if not args.specify_params:
        print("Using Optuna to find best hyerparameters")
        # create empy Optuna log file
        optuna_log_file = "optuna_log_layers=" + str(num_layers) + ".csv"
        with open(os.path.join(results_path, optuna_log_file), mode='w', newline='') as file:
            csv_writer = csv.writer(file)

        criterion_best = 0.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials[num_layers])

        # Print the best hyperparameters
        best_trial = study.best_trial
        print("Best Trial:")
        print(f"  Criterion: {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")  

        best_hyper_parameters = {}
        for key, value in best_trial.params.items():
            best_hyper_parameters[key] = value

        with open(os.path.join(results_path, model_name + "_BestHyperParameters.pkl"), 'wb') as file:
            pickle.dump(best_hyper_parameters, file)

    # Directly use hyperparameter
    ##########################
    ##########################
    ##########################
    ##########################
    ##########################            
    if args.specify_params:
        print("Using pre-specified hyerparameters")
        best_params = best_params_TF[num_layers]
        hidden_dim_h = best_params['hidden_dim_h']
        dropout = best_params['dropout']
        batch_size = best_params['batch_size']
        n_epochs = best_params['n_epochs']

        print(f"Build model with {num_layers} layers of attention")
        model = model_class(L, input_dim, hidden_dim_h*num_heads, num_layers, num_heads, dropout).to(device)    
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0            
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader)}")
                model.eval()
                pred, true = make_predictions(model, val_loader)
                val_r2 = pearsonr(pred, true)[0]**2
                print(val_r2)
                if math.isnan(val_r2):
                    print("nan encountered")
                    break
        model_best = model

    ##########################
    ##########################
    ##########################
    ##########################
    ##########################
    # Evaluate test performance
    torch.save(model_best, os.path.join(results_path, model_name + "_BestModel"))            
    gc.collect()
    torch.cuda.empty_cache()
    model_best.eval()
    try: 
        pred, true = make_predictions(model_best, test_loader)
        r2_test = pearsonr(pred, true)[0]**2
        print(f"{model_name} achieved R2 = {r2_test}")
        # save test R2 score
        with open(os.path.join(results_path, "R2s.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[model_name, r2_test]])

        # save predictions
        pd.DataFrame({"prediction": pred, "true": true}).to_csv(os.path.join(results_path, model_name + "_predictions.csv"), index=False)
        torch.save(model_best.state_dict(), os.path.join(results_path, model_name + "_BestModel_params"))
    except:
        print("Prediction failed!")

with open(os.path.join(results_path, 'done'), 'w') as file:
    pass

