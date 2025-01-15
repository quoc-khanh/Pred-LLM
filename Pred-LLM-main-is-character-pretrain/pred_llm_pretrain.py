import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tabular_llm.predllm_utils import _encode_row_partial
from tabular_llm.predllm import PredLLM
import read_data
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="iris", type=str, nargs='?', help='dataset name')
parser.add_argument('--method', default="pred_llm", type=str, nargs='?', help='generative method')
parser.add_argument('--trainsize', default="1.0", type=str, nargs='?', help='size of training set')
parser.add_argument('--testsize', default="0.2", type=str, nargs='?', help='size of test set')
parser.add_argument('--gensize', default="1.0", type=str, nargs='?', help='size of generation set')
parser.add_argument('--runs', default="3", type=str, nargs='?', help='no of times to run algorithm')
args = parser.parse_args()
print("dataset: {}, method: {}, train_size: {}, test_size: {}, gen_size: {}".
      format(args.dataset, args.method, args.trainsize, args.testsize, args.gensize))

dataset_input = args.dataset
method_input = args.method
train_size = float(args.trainsize)
test_size = float(args.testsize)
gen_size = float(args.gensize)
n_run = int(args.runs)

llm_batch_size = 32
llm_epochs = 50

if dataset_input == "classification":
    # datasets = ["iris", "breast_cancer", "australian",
    #             "blood_transfusion", "steel_plates_fault",
    #             "qsar_biodeg", "phoneme", "waveform",
    #             "churn", "cardiotocography",
    #             "kc1", "kc2", "balance_scale",
    #             "diabetes", "compas", "bank", "adult"]
    
    datasets = ['australian', 'travel', 'german_credit', 'compas', 'bank', 'heloc', 'adult', 'credit']#, 'credit_card_fraud', 'home_credit', 'lending_club', 'paysim', 'ieee_cis', 'yahoo_finance', 'fred', 'churn']

else:
    datasets = [dataset_input]
print("datasets: {}".format(datasets))

if method_input == "all":
    methods = ["original", "pred_llm"]
else:
    methods = [method_input]
print("methods: {}".format(methods))
list_data = []
for dataset in datasets:
    print("dataset: {}".format(dataset))
    # compute no of generated samples
    _, _, _, _, n_generative, _, n_feature, n_class, feature_names = read_data.gen_train_test_data(dataset,
                                                                                                   train_size=gen_size,
                                                                                                   normalize_x=None)
    for method in methods:
        for run in range(n_run):
            print("run: {}".format(run))
            np.random.seed(run)
            X_train, y_train, X_test, y_test, n_train, n_test, _, _, _ = \
                read_data.gen_train_test_data(dataset, train_size, test_size, normalize_x=True, seed=run)
            if method == "original":
                X_train_new, y_train_new = X_train, y_train
            if method == "pred_llm":
                X_y_train = np.append(X_train, y_train.reshape(-1, 1), axis=1) #
                X_y_train_df = pd.DataFrame(X_y_train)
                X_y_train_df.columns = np.append(feature_names, "target")
                list_data.append(X_y_train_df)

predllm = PredLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)#TODO  distilgpt2
predllm.pretrain(list_data)

import torch
torch.save(predllm.model.state_dict(), "saved.pt")
