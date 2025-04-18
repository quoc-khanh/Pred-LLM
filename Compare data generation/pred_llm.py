import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tabular_llm.predllm_utils import _encode_row_partial
from tabular_llm.predllm import PredLLM
import read_data
import torch
import tempfile
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score

###TAPTAP
from TapTap_master.taptap.taptap import Taptap
from TapTap_master.taptap.exp_utils import lightgbm_hpo
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

###GREAT
from be_great import GReaT
from P_TA_main.main import train_pipeline
from DPO.main import train_dpo

###CTGAN COPULA_GAN TVAE 
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

import logging

# Configure logging
logging.basicConfig(
    filename='experiment_log.txt',  # Log file name
    level=logging.INFO,             # Log level
    format='%(asctime)s - %(message)s'  # Log format
)

def get_score(train_data, test_data, target_col, best_params):
    train_x = train_data.drop(columns=target_col).copy()
    test_x = test_data.drop(columns=target_col).copy()
    train_y = train_data[target_col]
    test_y = test_data[target_col]
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    gbm = lgb.LGBMClassifier(**best_params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = gbm.predict(test_x)
    score = f1_score(test_y, pred)
    return score, gbm


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

llm_batch_size = 8
llm_epochs = 50
nn_batch_size = 512
nn_epochs = 1000

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
precision_dataset_method_run = []
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
    precision_method_run = []
    f1_method_run = []
    auc_method_run = []
    
    for method in methods:
        print("method: {}".format(method))
        xgb_run = np.zeros(n_run)
        recall_run = np.zeros(n_run)
        precision_run = np.zeros(n_run)
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
            else:
                # Combine training features and target using column_stack for better efficiency
                X_y_train = np.column_stack((X_train, y_train))
                X_y_train_df = pd.DataFrame(X_y_train, columns=np.append(feature_names, "target"))
                
                # Make metadata for sdv models
                metadata = Metadata.detect_from_dataframe(data=X_y_train_df, table_name=dataset)
                
                # Combine testing features and target similarly
                X_y_test = np.column_stack((X_test, y_test))
                X_y_test_df = pd.DataFrame(X_y_test, columns=np.append(feature_names, "target"))

                
                if method == "pred_llm":
                    
                    predllm = PredLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)#TODO  distilgpt2
                    
                    # predllm.model.load_state_dict(torch.load("adult_king_insurance_intrusion_covtype_pretrained.pt"), strict=False)
                    
                    predllm.fit(X_y_train_df)
                    # compute length of input sequence
                    encoded_text = _encode_row_partial(X_y_train_df.iloc[0], shuffle=False) #TODO
                    prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
                    X_y_train_new = predllm.sample_new(n_samples=n_generative, max_length=prompt_len+200, task="classification")
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
                if method == "great_dpo":          
                    # Tạo một file CSV tạm thời. delete=False để file không bị xóa khi đóng.
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                    csv_path = tmp_file.name  # Lấy đường dẫn trước khi đóng file
                    tmp_file.close()  # Đóng file để tránh lỗi khi ghi dữ liệu
                
                    X_y_train_df.to_csv(csv_path, index=False)  # Ghi DataFrame vào file CSV
                    X_y_train_new = train_dpo(csv_path = csv_path)
                if method == "great_ppo":          
                    # Tạo một file CSV tạm thời. delete=False để file không bị xóa khi đóng.
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                    csv_path = tmp_file.name  # Lấy đường dẫn trước khi đóng file
                    tmp_file.close()  # Đóng file để tránh lỗi khi ghi dữ liệu
                
                    X_y_train_df.to_csv(csv_path, index=False)  # Ghi DataFrame vào file CSV
                    X_y_train_new = train_pipeline(csv_path = csv_path)
                if method == "taptap":
                    train_data, test_data = X_y_train_df, X_y_test_df
                    target_col = 'target'
                    task = 'classification'
                    best_params = lightgbm_hpo(
                        data=train_data, target_col=target_col, task=task, n_trials=10, n_jobs=16
                    )
                    original_score, gbm = get_score(
                        train_data, test_data, target_col=target_col, best_params=best_params
                    )
                    print("The score training by the original data is", original_score)
    
                    model = Taptap(llm='ztphs980/taptap-distill',
                                experiment_dir='./experiment_taptap/',
                                steps=llm_epochs,
                                batch_size=llm_batch_size,
                                numerical_modeling='split',
                                gradient_accumulation_steps=2)
    
                    # Fine-tuning
                    model.fit(train_data, target_col=target_col, task=task)
    
                    # Sampling
                    X_y_train_new = model.sample(n_samples=n_generative,
                                                data=train_data,
                                                task=task,
                                                max_length=1024)
    
                    # Label generation
                    X_y_train_new[target_col] = gbm.predict(X_y_train_new.drop(columns=[target_col]))
                    
                    
                    # X_train_new = X_y_train_new.iloc[:, :-1].to_numpy(dtype=float).reshape(-1, n_feature)
                    # y_train_new = X_y_train_new.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )
                    # X_y_train_new.columns = X_y_train_df.columns
                    # unique_values = np.unique(y_train_new)
                    # print(f"Unique values in y_train_new: {unique_values}")
                    # print(f"Number of unique values: {len(unique_values)}")
                    # # TODO
                    # print(f"Feature old: {X_y_train_df.columns[:-1].tolist()}")
                    # print(f"Feature new: {X_y_train_new.columns[:-1].tolist()}")
                if method == "great":
                    model = GReaT(llm='distilgpt2', batch_size=llm_batch_size,  epochs=llm_epochs, fp16=True)
                    model.fit(X_y_train_df)
                    X_y_train_new = model.sample(n_samples=n_generative)
                if method == "ctgan":
                    synthesizer = CTGANSynthesizer(metadata, # required
                                                    enforce_rounding=False,
                                                    epochs=nn_epochs,
                                                    verbose=False
                                                )
                    synthesizer.fit(X_y_train_df)
                    X_y_train_new = synthesizer.sample(num_rows=n_generative) 
                if method == "copula_gan": 
                    synthesizer = CopulaGANSynthesizer(metadata, # required
                                                        enforce_min_max_values=True,
                                                        enforce_rounding=False,
                                                        epochs=nn_epochs,
                                                        verbose=False
                                                    )
                    synthesizer.fit(X_y_train_df)
                    X_y_train_new = synthesizer.sample(num_rows=n_generative)
                if method == "tvae": 
                    synthesizer = TVAESynthesizer(metadata, # required
                                                    enforce_min_max_values=True,
                                                    enforce_rounding=False,
                                                    epochs=nn_epochs
                                                )
                    synthesizer.fit(X_y_train_df)
                    X_y_train_new = synthesizer.sample(num_rows=n_generative)

                X_train_new = X_y_train_new.iloc[:, :-1].to_numpy(dtype=float).reshape(-1, n_feature)
                y_train_new = X_y_train_new.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )
                X_y_train_new.columns = X_y_train_df.columns
                
                file_ori = "./results/_classification/{}/ds{}_ORI_tr{}_te{}_ge{}".format(method, dataset, train_size, test_size, gen_size)#, "wb"#"ds{}_ORI_tr{}_te{}_ge{}".format(dataset, train_size, test_size, gen_size)
                if not os.path.exists(file_ori + ".csv"):
                    X_y_train_df.to_csv(file_ori + ".csv", index=False)
                file_result = "./results/_classification/{}/ds{}_NEW_tr{}_te{}_ge{}_run{}".format(method, dataset, train_size, test_size, gen_size, run)#"ds{}_NEW_tr{}_te{}_ge{}_run{}".format(dataset, train_size, test_size, gen_size, run)
                X_y_train_new.to_csv(file_result + ".csv", index=False)
                
                unique_values = np.unique(y_train_new)
                print(f"Unique values in y_train_new: {unique_values}")
                print(f"Number of unique values: {len(unique_values)}")
                # TODO
                print(f"Feature old: {X_y_train_df.columns[:-1].tolist()}")
                print(f"Feature new: {X_y_train_new.columns[:-1].tolist()}")
                

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
            
            # get number of generative classes
            n_class_generative = len(np.unique(y_train_new))
            # train a classifier using 5-fold CV with additional precision metric and custom AUC scorer
            if n_class_generative != n_class:
                print("generate less/more than the number of real classes")
                acc_new = 0
                precision_new = 0
                recall_new = 0
                f1_new = 0
                auc_new = 0
            else:                
                # Custom AUC function to handle binary and multi-class cases
                def custom_auc(y_true, y_proba, **kwargs):
                    classes = np.unique(y_true)
                    if len(classes) == 2:
                        return roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        return roc_auc_score(y_true, y_proba, multi_class='ovr')
                
                auc_scorer = make_scorer(custom_auc, needs_proba=True)
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': 'precision_macro',
                    'recall': 'recall_macro',
                    'f1': 'f1_macro',
                    'auc': auc_scorer
                }
                
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
                xgb_new = XGBClassifier(random_state=run)
                cv_results = cross_validate(xgb_new, X_train_new, y_train_new, cv=skf, scoring=scoring, return_train_score=False)
                
                acc_new = round(np.mean(cv_results['test_accuracy']), 4)
                precision_new = round(np.mean(cv_results['test_precision']), 4)
                recall_new = round(np.mean(cv_results['test_recall']), 4)
                f1_new = round(np.mean(cv_results['test_f1']), 4)
                auc_new = round(np.mean(cv_results['test_auc']), 4)
                
                log_message = (
                    "dataset: {}, method: {}, run: {}, accuracy: {}, precision: {}, recall: {}, f1: {}, auc: {}"
                    .format(dataset, method, run, acc_new, precision_new, recall_new, f1_new, auc_new)
                )
                print(log_message)
                logging.info(log_message)
            
            # save metrics for this run of each method for each train_size in each dataset
            xgb_run[run] = acc_new
            recall_run[run] = recall_new
            precision_run[run] = precision_new
            f1_run[run] = f1_new 
            auc_run[run] = auc_new
            # save result to text file 
            if run == (n_run - 1):
                with open('./results/_classification/{}/metrics_{}.txt'.format(method, file_name), 'w') as f:
                    acc_avg = round(np.mean(xgb_run), 4)
                    acc_std = round(np.std(xgb_run), 4)
                    recall_avg = round(np.mean(recall_run), 4)
                    recall_std = round(np.std(recall_run), 4)
                    precision_avg = round(np.mean(precision_run), 4)
                    precision_std = round(np.std(precision_run), 4)
                    f1_avg = round(np.mean(f1_run), 4)
                    f1_std = round(np.std(f1_run), 4)
                    auc_avg = round(np.mean(auc_run), 4)
                    auc_std = round(np.std(auc_run), 4)
                    
                    f.write("Metrics for Original Model:\n")
                    f.write("Accuracy: {}\n\n".format(acc_org))
                    
                    f.write("Metrics for Generated Model:\n")
                    f.write("Accuracy: {}\n".format(acc_new))
                    f.write("Recall (macro): {}\n".format(recall_new))
                    f.write("Precision (macro): {}\n".format(precision_new))
                    f.write("F1 Score (macro): {}\n".format(f1_new))
                    f.write("ROC AUC (ovr): {}\n\n".format(auc_new))
                    
                    f.write("Average Metrics Across Runs:\n")
                    f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
                    f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
                    f.write("Precision: {} (±{})\n".format(precision_avg, precision_std))
                    f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
                    f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))
            
            # save metrics of n_run of each method in each dataset
            xgb_method_run.append(xgb_run)
            recall_method_run.append(recall_run)
            precision_method_run.append(precision_run)
            f1_method_run.append(f1_run)
            auc_method_run.append(auc_run)
    # save metrics of n_run of all methods in each dataset  
    xgb_dataset_method_run.append(xgb_method_run)
    recall_dataset_method_run.append(recall_method_run)
    precision_dataset_method_run.append(precision_method_run)
    f1_dataset_method_run.append(f1_method_run)
    auc_dataset_method_run.append(auc_method_run)

# save all results to csv file
n_dataset = len(datasets)
n_method = len(methods)
file_result = './results/_classification/metrics_ds{}_me{}_tr{}_te{}_ge{}'.\
    format(dataset_input, method_input, train_size, test_size, gen_size)
with open(file_result + ".csv", 'w') as f:
    f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy,recall,precision,f1,auc\n")
    for data_id in range(n_dataset):
        for method_id in range(n_method):
            for run_id in range(n_run):
                dataset_name = datasets[data_id]
                method_name = methods[method_id]
                classifier = "xgb"
                acc = xgb_dataset_method_run[data_id][method_id][run_id]
                recall = recall_dataset_method_run[data_id][method_id][run_id]
                precision = precision_dataset_method_run[data_id][method_id][run_id]
                f1 = f1_dataset_method_run[data_id][method_id][run_id]
                auc = auc_dataset_method_run[data_id][method_id][run_id]
                line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc},{recall},{precision},{f1},{auc}\n"
                f.write(line)
            acc_avg = round(np.mean(xgb_dataset_method_run), 4)
            acc_std = round(np.std(xgb_dataset_method_run), 4)
            recall_avg = round(np.mean(recall_dataset_method_run), 4)
            recall_std = round(np.std(recall_dataset_method_run), 4)
            precision_avg = round(np.mean(precision_dataset_method_run), 4)
            precision_std = round(np.std(precision_dataset_method_run), 4)
            f1_avg = round(np.mean(f1_dataset_method_run), 4)
            f1_std = round(np.std(f1_dataset_method_run), 4)
            auc_avg = round(np.mean(auc_dataset_method_run), 4)
            auc_std = round(np.std(auc_dataset_method_run), 4)
            run_id = 'avg'
            line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc_avg},{recall_avg},{precision_avg},{f1_avg},{auc_avg}\n"
            f.write(line)

# save metrics of all datasets to text file 
with open(file_result + ".txt", 'w') as f:
    acc_avg = round(np.mean(xgb_dataset_method_run), 4)
    acc_std = round(np.std(xgb_dataset_method_run), 4)
    recall_avg = round(np.mean(recall_dataset_method_run), 4)
    recall_std = round(np.std(recall_dataset_method_run), 4)
    precision_avg = round(np.mean(precision_dataset_method_run), 4)
    precision_std = round(np.std(precision_dataset_method_run), 4)
    f1_avg = round(np.mean(f1_dataset_method_run), 4)
    f1_std = round(np.std(f1_dataset_method_run), 4)
    auc_avg = round(np.mean(auc_dataset_method_run), 4)
    auc_std = round(np.std(auc_dataset_method_run), 4)
    f.write("Metrics Summary:\n")
    f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
    f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
    f.write("Precision: {} (±{})\n".format(precision_avg, precision_std))
    f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
    f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))


            ###TODO 10/1 save the syn+ori dataset
      
            ###

#             # get number of generative classes
#             n_class_generative = len(np.unique(y_train_new))
#             # train a classifier
#             if n_class_generative != n_class:
#                 print("generate less/more than the number of real classes")
#                 acc_new = 0
#                 recall_new = 0
#                 f1_new = 0
#                 auc_new = 0
#             else:
#                 xgb_new = XGBClassifier(random_state=run)
#                 xgb_new.fit(X_train_new, y_train_new)
#                 y_pred = xgb_new.predict(X_test)
#                 acc_new = round(accuracy_score(y_test, y_pred), 4)
#                 recall_new = round(recall_score(y_test, y_pred, average='macro'), 4)
#                 f1_new = round(f1_score(y_test, y_pred, average='macro'), 4)
#                 y_pred_proba = xgb_new.predict_proba(X_test)
#                 n_classes = len(np.unique(y_test))

#                 if n_classes == 2:  # Bài toán nhị phân
#                     y_pred_prob = y_pred_proba[:, 1]  # Xác suất của lớp dương
#                     auc_new = round(roc_auc_score(y_test, y_pred_prob), 4)
#                 else:  # Bài toán đa lớp
#                     auc_new = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4)
#                 # Print and log
#                 log_message = (
#                       "dataset: {}, method: {}, run: {}, accuracy: {}, recall: {}, f1: {}, auc: {}"
#                       .format(dataset, method, run, acc_new, recall_new, f1_new, auc_new)
#                   )
                  
#                   # Print to console
#                 print(log_message)
                  
#                   # Write to log file
#                 logging.info(log_message)
#             # save metrics of each run of each method for each train_size in each dataset
#             xgb_run[run] = acc_new
#             recall_run[run] = recall_new
#             f1_run[run] = f1_new 
#             auc_run[run] = auc_new
#             # save result to text file 
#             if run == (n_run - 1):
#                 with open('./results/_classification/{}/metrics_{}.txt'.format(method, file_name), 'w') as f:
#                     acc_avg = round(np.mean(xgb_run), 4)
#                     acc_std = round(np.std(xgb_run), 4)
#                     recall_avg = round(np.mean(recall_run), 4)
#                     recall_std = round(np.std(recall_run), 4)
#                     f1_avg = round(np.mean(f1_run), 4)
#                     f1_std = round(np.std(f1_run), 4)
#                     auc_avg = round(np.mean(auc_run), 4)
#                     auc_std = round(np.std(auc_run), 4)
                    
#                     f.write("Metrics for Original Model:\n")
#                     f.write("Accuracy: {}\n\n".format(acc_org))
                    
#                     f.write("Metrics for Generated Model:\n")
#                     f.write("Accuracy: {}\n".format(acc_new))
#                     f.write("Recall (macro): {}\n".format(recall_new))
#                     f.write("F1 Score (macro): {}\n".format(f1_new))
#                     f.write("ROC AUC (ovr): {}\n\n".format(auc_new))
                    
#                     f.write("Average Metrics Across Runs:\n")
#                     f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
#                     f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
#                     f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
#                     f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))
#         # save metrics of n_run of each method in each dataset
#         xgb_method_run.append(xgb_run)
#         recall_method_run.append(recall_run)
#         f1_method_run.append(f1_run)
#         auc_method_run.append(auc_run)
#     # save metrics of n_run of all methods in each dataset  
#     xgb_dataset_method_run.append(xgb_method_run)
#     recall_dataset_method_run.append(recall_method_run)
#     f1_dataset_method_run.append(f1_method_run)
#     auc_dataset_method_run.append(auc_method_run)

# # save all results to csv file
# n_dataset = len(datasets)
# n_method = len(methods)
# file_result = './results/_classification/metrics_ds{}_me{}_tr{}_te{}_ge{}'.\
#     format(dataset_input, method_input, train_size, test_size, gen_size)
# # with open(file_result + ".csv", 'w') as f: #TODO
# #     f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy,recall,f1,auc\n")
# #     for data_id in range(n_dataset):
# #         for method_id in range(n_method):
# #             for run_id in range(n_run):
# #                 dataset_name = datasets[data_id]
# #                 method_name = methods[method_id]
# #                 classifier = "xgb"
# #                 xgb_new = XGBClassifier(random_state=run_id)
# #                 xgb_new.fit(X_train_new, y_train_new)
# #                 y_pred = xgb_new.predict(X_test)
# #                 acc = round(accuracy_score(y_test, y_pred), 4)
# #                 recall = round(recall_score(y_test, y_pred, average='macro'), 4)
# #                 f1 = round(f1_score(y_test, y_pred, average='macro'), 4)
# #                 y_pred_proba = xgb_new.predict_proba(X_test)
# #                 n_classes = len(np.unique(y_test))

# #                 if n_classes == 2:  # Bài toán nhị phân
# #                     y_pred_prob = y_pred_proba[:, 1]  # Xác suất của lớp dương
# #                     auc = round(roc_auc_score(y_test, y_pred_prob), 4)
# #                 else:  # Bài toán đa lớp
# #                     auc = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4)
# #                 line = dataset_name + "," + method_name + "," + classifier + "," + \
# #                        str(train_size) + "," + str(test_size) + "," + str(gen_size) + "," + \
# #                        str(run_id) + "," + str(acc) + "," + str(recall) + "," + \
# #                        str(f1) + "," + str(auc) + "\n"
# #                 f.write(line)
# with open(file_result + ".csv", 'w') as f:
#     f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy,recall,f1,auc\n")
#     for data_id in range(n_dataset):
#         for method_id in range(n_method):
#             for run_id in range(n_run):
#                 dataset_name = datasets[data_id]
#                 method_name = methods[method_id]
#                 classifier = "xgb"
#                 acc = xgb_dataset_method_run[data_id][method_id][run_id]
#                 recall = recall_dataset_method_run[data_id][method_id][run_id]
#                 f1 = f1_dataset_method_run[data_id][method_id][run_id]
#                 auc = auc_dataset_method_run[data_id][method_id][run_id]
#                 line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc},{recall},{f1},{auc}\n"
#                 f.write(line)
#             acc_avg = round(np.mean(xgb_dataset_method_run), 4)
#             acc_std = round(np.std(xgb_dataset_method_run), 4)
#             recall_avg = round(np.mean(recall_dataset_method_run), 4)
#             recall_std = round(np.std(recall_dataset_method_run), 4)
#             f1_avg = round(np.mean(f1_dataset_method_run), 4)
#             f1_std = round(np.std(f1_dataset_method_run), 4)
#             auc_avg = round(np.mean(auc_dataset_method_run), 4)
#             auc_std = round(np.std(auc_dataset_method_run), 4)
#             run_id = 'avg'
#             line = f"{dataset_name},{method_name},{classifier},{train_size},{test_size},{gen_size},{run_id},{acc_avg},{recall_avg},{f1_avg},{auc_avg}\n"
#             f.write(line)

# # save metrics of all datasets to text file 
# with open(file_result + ".txt", 'w') as f:
#     acc_avg = round(np.mean(xgb_dataset_method_run), 4)
#     acc_std = round(np.std(xgb_dataset_method_run), 4)
#     recall_avg = round(np.mean(recall_dataset_method_run), 4)
#     recall_std = round(np.std(recall_dataset_method_run), 4)
#     f1_avg = round(np.mean(f1_dataset_method_run), 4)
#     f1_std = round(np.std(f1_dataset_method_run), 4)
#     auc_avg = round(np.mean(auc_dataset_method_run), 4)
#     auc_std = round(np.std(auc_dataset_method_run), 4)
#     f.write("Metrics Summary:\n")
#     f.write("Accuracy: {} (±{})\n".format(acc_avg, acc_std))
#     f.write("Recall: {} (±{})\n".format(recall_avg, recall_std))
#     f.write("F1 Score: {} (±{})\n".format(f1_avg, f1_std))
#     f.write("ROC AUC: {} (±{})\n".format(auc_avg, auc_std))


