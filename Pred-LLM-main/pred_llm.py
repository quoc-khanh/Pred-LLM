import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tabular_llm.predllm_utils import _encode_row_partial
from tabular_llm.predllm import PredLLM
import read_data
import torch

import logging

# Configure logging
logging.basicConfig(
    filename='experiment_log.txt',  # Log file name
    level=logging.INFO,             # Log level
    format='%(asctime)s - %(message)s'  # Log format
)

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

xgb_dataset_method_run = []
recall_dataset_method_run = []
f1_dataset_method_run = []
auc_dataset_method_run = []
for dataset in datasets:
    print("dataset: {}".format(dataset))
    # compute no of generated samples
    _, _, _, _, n_generative, _, n_feature, n_class, feature_names = read_data.gen_train_test_data(dataset,
                                                                                                   train_size=gen_size,
                                                                                                   normalize_x=None)
    xgb_method_run = []
    recall_method_run = []
    f1_method_run = []
    auc_method_run = []
    for method in methods:
        print("method: {}".format(method))
        xgb_run = np.zeros(n_run)
        recall_run = np.zeros(n_run)
        f1_run = np.zeros(n_run)
        auc_run = np.zeros(n_run)
        for run in range(n_run):
            print("run: {}".format(run))
            np.random.seed(run)
            X_train, y_train, X_test, y_test, n_train, n_test, _, _, _ = \
                read_data.gen_train_test_data(dataset, train_size, test_size, normalize_x=True, seed=run)
            # train a classifier to predict labels of synthetic samples
            xgb_org = XGBClassifier(random_state=run)
            xgb_org.fit(X_train, y_train)
            y_pred = xgb_org.predict(X_test)
            acc_org = round(accuracy_score(y_test, y_pred), 4)
            print("original accuracy-{}: {}".format(run, acc_org))
            # train a generative method
            if method == "original":
                X_train_new, y_train_new = X_train, y_train
            if method == "pred_llm":
                X_y_train = np.append(X_train, y_train.reshape(-1, 1), axis=1)
                X_y_train_df = pd.DataFrame(X_y_train)
                X_y_train_df.columns = np.append(feature_names, "target")
                predllm = PredLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)#TODO  distilgpt2
                
                # predllm.model.load_state_dict(torch.load("adult_king_insurance_intrusion_covtype_pretrained.pt"), strict=False)
                
                predllm.fit(X_y_train_df)
                # compute length of input sequence
                encoded_text = _encode_row_partial(X_y_train_df.iloc[0], shuffle=False) #TODO
                prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
                X_y_train_new = predllm.sample_new(n_samples=n_generative, max_length=prompt_len, task="classification")
                X_train_new = X_y_train_new.iloc[:, :-1].to_numpy(dtype=float).reshape(-1, n_feature)
                y_train_new = X_y_train_new.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )
                unique_values = np.unique(y_train_new)
                print(f"Unique values in y_train_new: {unique_values}")
                print(f"Number of unique values: {len(unique_values)}")
                # TODO
                print(f"Feature old: {X_y_train_df.columns[:-1].tolist()}")
                print(f"Feature new: {X_y_train_new.columns[:-1].tolist()}")
                
                # Print first 3 rows of original and generated data
                print("\nFirst 3 rows of original training data:")
                print(X_y_train_df.iloc[0:3])
                print("\nFirst 3 rows of generated data:")
                print(X_y_train_new.iloc[0:3])
                
                # Concatenate the generated data with the original training data
# Rename features of X_y_train_new to match X_y_train_df
                X_y_train_new.columns = X_y_train_df.columns #rename feature of X_y_train_new the same as X_y_train_df TODO
                X_y_train_new_ORI = pd.concat([X_y_train_new, X_y_train_df[X_y_train_new.columns]], axis=0, ignore_index=True)
                X_train_new_ORI = X_y_train_new.iloc[:, :-1].to_numpy(dtype=float).reshape(-1, n_feature)
                y_train_new_ORI = X_y_train_new.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )

                # file_name = "ds{}_tr{}_te{}_ge{}_run{}".format(dataset, train_size, test_size, gen_size, run)
                # with open("./results/_classification/{}/X_gen_ORI_{}.npz".format(method, file_name), "wb") as f:
                #     np.save(f, X_train_new_ORI)
                # with open("./results/_classification/{}/y_gen_ORI_{}.npz".format(method, file_name), "wb") as f:
                #     np.save(f, y_train_new_ORI)
                

            print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
            print("n_generative: {}".format(n_generative))
            print("X_train_new: {}, y_train_new: {}".format(X_train_new.shape, y_train_new.shape))
            # convert labels from float to int
            y_train_new = np.array([int(y) for y in y_train_new])
            # save result to text file
            file_name = "ds{}_tr{}_te{}_ge{}_run{}".format(dataset, train_size, test_size, gen_size, run)
            with open("./results/_classification/{}/X_gen_{}.npz".format(method, file_name), "wb") as f:
                np.save(f, X_train_new)
            with open("./results/_classification/{}/y_gen_{}.npz".format(method, file_name), "wb") as f:
                np.save(f, y_train_new)
                
            ###TODO 10/1 save the syn+ori dataset
      
            ###

            # get number of generative classes
            n_class_generative = len(np.unique(y_train_new))
            # train a classifier
            if n_class_generative != n_class:
                print("generate less/more than the number of real classes")
                acc_new = 0
            else:
                xgb_new = XGBClassifier(random_state=run)
                xgb_new.fit(X_train_new, y_train_new)
                y_pred = xgb_new.predict(X_test)
                acc_new = round(accuracy_score(y_test, y_pred), 4)
                recall_new = round(recall_score(y_test, y_pred, average='macro'), 4)
                f1_new = round(f1_score(y_test, y_pred, average='macro'), 4)
                y_pred_proba = xgb_new.predict_proba(X_test)
                n_classes = len(np.unique(y_test))

                if n_classes == 2:  # Bài toán nhị phân
                    y_pred_prob = y_pred_proba[:, 1]  # Xác suất của lớp dương
                    auc_new = round(roc_auc_score(y_test, y_pred_prob), 4)
                else:  # Bài toán đa lớp
                    auc_new = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4)
                # Print and log
                log_message = (
                      "dataset: {}, method: {}, run: {}, accuracy: {}, recall: {}, f1: {}, auc: {}"
                      .format(dataset, method, run, acc_new, recall_new, f1_new, auc_new)
                  )
                  
                  # Print to console
                print(log_message)
                  
                  # Write to log file
                logging.info(log_message)
            # save metrics of each run of each method for each train_size in each dataset
            xgb_run[run] = acc_new
            recall_run[run] = recall_new
            f1_run[run] = f1_new 
            auc_run[run] = auc_new
            # save result to text file 
            if run == (n_run - 1):
                with open('./results/_classification/{}/metrics_{}.txt'.format(method, file_name), 'w') as f:
                    acc_avg = round(np.mean(xgb_run), 4)
                    acc_std = round(np.std(xgb_run), 4)
                    recall_avg = round(np.mean(recall_run), 4)
                    recall_std = round(np.std(recall_run), 4)
                    f1_avg = round(np.mean(f1_run), 4)
                    f1_std = round(np.std(f1_run), 4)
                    auc_avg = round(np.mean(auc_run), 4)
                    auc_std = round(np.std(auc_run), 4)
                    
                    f.write("Metrics for Original Model:\n")
                    f.write("Accuracy: {}\n\n".format(acc_org))
                    
                    f.write("Metrics for Generated Model:\n")
                    f.write("Accuracy: {}\n".format(acc_new))
                    f.write("Recall (macro): {}\n".format(recall_new))
                    f.write("F1 Score (macro): {}\n".format(f1_new))
                    f.write("ROC AUC (ovr): {}\n\n".format(auc_new))
                    
                    f.write("Average Metrics Across Runs:\n")
                    f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
                    f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
                    f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
                    f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))
        # save metrics of n_run of each method in each dataset
        xgb_method_run.append(xgb_run)
        recall_method_run.append(recall_run)
        f1_method_run.append(f1_run)
        auc_method_run.append(auc_run)
    # save metrics of n_run of all methods in each dataset  
    xgb_dataset_method_run.append(xgb_method_run)
    recall_dataset_method_run.append(recall_method_run)
    f1_dataset_method_run.append(f1_method_run)
    auc_dataset_method_run.append(auc_method_run)

# save all results to csv file
n_dataset = len(datasets)
n_method = len(methods)
file_result = './results/_classification/metrics_ds{}_me{}_tr{}_te{}_ge{}'.\
    format(dataset_input, method_input, train_size, test_size, gen_size)
# with open(file_result + ".csv", 'w') as f: #TODO
#     f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy,recall,f1,auc\n")
#     for data_id in range(n_dataset):
#         for method_id in range(n_method):
#             for run_id in range(n_run):
#                 dataset_name = datasets[data_id]
#                 method_name = methods[method_id]
#                 classifier = "xgb"
#                 xgb_new = XGBClassifier(random_state=run_id)
#                 xgb_new.fit(X_train_new, y_train_new)
#                 y_pred = xgb_new.predict(X_test)
#                 acc = round(accuracy_score(y_test, y_pred), 4)
#                 recall = round(recall_score(y_test, y_pred, average='macro'), 4)
#                 f1 = round(f1_score(y_test, y_pred, average='macro'), 4)
#                 y_pred_proba = xgb_new.predict_proba(X_test)
#                 n_classes = len(np.unique(y_test))

#                 if n_classes == 2:  # Bài toán nhị phân
#                     y_pred_prob = y_pred_proba[:, 1]  # Xác suất của lớp dương
#                     auc = round(roc_auc_score(y_test, y_pred_prob), 4)
#                 else:  # Bài toán đa lớp
#                     auc = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4)
#                 line = dataset_name + "," + method_name + "," + classifier + "," + \
#                        str(train_size) + "," + str(test_size) + "," + str(gen_size) + "," + \
#                        str(run_id) + "," + str(acc) + "," + str(recall) + "," + \
#                        str(f1) + "," + str(auc) + "\n"
#                 f.write(line)
with open(file_result + ".csv", 'w') as f:
    f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy,recall,f1,auc\n")
    for data_id in range(n_dataset):
        for method_id in range(n_method):
            for run_id in range(n_run):
                dataset_name = datasets[data_id]
                method_name = methods[method_id]
                classifier = "xgb"
                acc = xgb_dataset_method_run[data_id][method_id][run_id]
                recall = recall_dataset_method_run[data_id][method_id][run_id]
                f1 = f1_dataset_method_run[data_id][method_id][run_id]
                auc = auc_dataset_method_run[data_id][method_id][run_id]
                line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc},{recall},{f1},{auc}\n"
                f.write(line)
            acc_avg = round(np.mean(xgb_dataset_method_run), 4)
            acc_std = round(np.std(xgb_dataset_method_run), 4)
            recall_avg = round(np.mean(recall_dataset_method_run), 4)
            recall_std = round(np.std(recall_dataset_method_run), 4)
            f1_avg = round(np.mean(f1_dataset_method_run), 4)
            f1_std = round(np.std(f1_dataset_method_run), 4)
            auc_avg = round(np.mean(auc_dataset_method_run), 4)
            auc_std = round(np.std(auc_dataset_method_run), 4)
            run_id = 'avg'
            line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc_avg},{recall_avg},{f1_avg},{auc_avg}\n"
            f.write(line)

# save metrics of all datasets to text file 
with open(file_result + ".txt", 'w') as f:
    acc_avg = round(np.mean(xgb_dataset_method_run), 4)
    acc_std = round(np.std(xgb_dataset_method_run), 4)
    recall_avg = round(np.mean(recall_dataset_method_run), 4)
    recall_std = round(np.std(recall_dataset_method_run), 4)
    f1_avg = round(np.mean(f1_dataset_method_run), 4)
    f1_std = round(np.std(f1_dataset_method_run), 4)
    auc_avg = round(np.mean(auc_dataset_method_run), 4)
    auc_std = round(np.std(auc_dataset_method_run), 4)
    f.write("Metrics Summary:\n")
    f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
    f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
    f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
    f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))
