import numpy as np
import pandas as pd
from sklearn import datasets
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import openml
import yfinance as yf
from fredapi import Fred
import os

def gen_train_test_data(dataset="", train_size=1.0, test_size=0.2, normalize_x=True, seed=42, fred_api_key=None, data_dir="./data"):
    print("dataset: {}, seed: {}".format(dataset, seed))
    print("train_size: {}, test_size: {}".format(train_size, test_size))

    X, y, names = None, None, None
    df = None

    try:
        #dealing with pre splited datasets
        if dataset in ['CML_Data', 'eiCU_tab_Processed', 'MIMICIII_Grouped']: 
            # Construct paths for the pre-split files
            train_path = os.path.join(data_dir, dataset, "train.csv")
            test_path = os.path.join(data_dir, dataset, "test.csv")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Train or test file not found in {os.path.join(data_dir, dataset)}")
            print("Loading pre-split dataset from:")
            print("  Train:", train_path)
            print("  Test :", test_path)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Assume that a column named 'target' exists and the rest are features.
            if dataset == "CML_Data":
                numerical_cols = ['HR', 'DBP', 'RR', 'BT', 'Glucose']
                categorical_cols = ['Dataset']
                target_col = "target"
            elif dataset == "MIMICIII_Grouped":
                train_df = train_df.drop(columns=['icustay_id'])
                test_df = test_df.drop(columns=['icustay_id'])
                # Convert target column type for both train and test DataFrames
                train_df["label_death_icu"] = train_df["label_death_icu"].astype(int)
                test_df["label_death_icu"] = test_df["label_death_icu"].astype(int)
                numerical_cols = ['heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate',
                                  'tempc', 'spo2', 'albumin', 'bun', 'bilirubin', 'lactate',
                                  'bicarbonate', 'bands', 'chloride', 'creatinine', 'glucose',
                                  'hemoglobin', 'hematocrit', 'platelet', 'potassium', 'ptt', 'sodium',
                                  'wbc']
                categorical_cols = []
                target_col = "label_death_icu"
            elif dataset == "eiCU_tab_Processed":
                train_df = train_df.drop(columns=['patientunitstayid'])
                test_df = test_df.drop(columns=['patientunitstayid'])
                train_df["hospitaldischargestatus"] = train_df["hospitaldischargestatus"].astype(int)
                test_df["hospitaldischargestatus"] = test_df["hospitaldischargestatus"].astype(int)
                numerical_cols = ['itemoffset', 'ethnicity', 'gender', 'GCS Total',
                                  'Eyes', 'Motor', 'Verbal', 'admissionheight', 'admissionweight', 'age',
                                  'Heart Rate', 'MAP (mmHg)', 'Invasive BP Diastolic',
                                  'Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate',
                                  'Temperature (C)', 'glucose', 'FiO2', 'pH', 'unitdischargeoffset']
                categorical_cols = []
                target_col = "hospitaldischargestatus"
            elif dataset == "Tab_Data":
                train_df = train_df.drop(columns=['caseid'])
                test_df = test_df.drop(columns=['caseid'])
                # train_df["hospitaldischargestatus"] = train_df["hospitaldischargestatus"].astype(int)
                # test_df["hospitaldischargestatus"] = test_df["hospitaldischargestatus"].astype(int)
                numerical_cols = ['age', 'preop_hb', 'preop_alb', 'preop_ast', 'preop_cr', 'asa']
                categorical_cols = ['sex', 'department', 'optype', 'approach']
                target_col = "death_inhosp"
            train_df, _ = train_test_split(train_df,
                                train_size=0.1,
                                stratify=train_df[target_col],
                                random_state=seed) #temporary, take too long to train
            # Extract features and target from both train and test
            feature_cols = [col for col in train_df.columns if col != target_col]
            X_train = train_df[feature_cols].copy()
            y_train = train_df[target_col].copy()
            X_test = test_df[feature_cols].copy()
            y_test = test_df[target_col].copy()
            
            # Encode categorical features for both sets
            for col in categorical_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
            
            # Normalize numerical features if requested (fit only on training data)
            if normalize_x and numerical_cols:
                print("Normalizing numerical features based on training data.")
                scaler = MinMaxScaler()
                X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
                X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
                X_train[numerical_cols] = np.around(X_train[numerical_cols], 2)
                X_test[numerical_cols] = np.around(X_test[numerical_cols], 2)
            else:
                print("Not normalizing numerical features.")
            
            # Encode the target variable
            y_train = LabelEncoder().fit_transform(y_train)
            y_test = LabelEncoder().fit_transform(y_test)
            
            names = feature_cols
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            n_feature = X_train.shape[1]
            n_class = len(np.unique(y_train))
            
            print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
            print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
            print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
            print("feature_names: {}".format(names))
            
            return X_train.values, y_train, X_test.values, y_test, n_train, n_test, n_feature, n_class, names
        ###
        if dataset == 'california':
            df = fetch_california_housing(as_frame=True).frame
            numerical_cols = []
            categorical_cols = []
            target_col = "MedHouseVal"
        elif dataset == 'australian':
            statlog_australian_credit_approval = fetch_ucirepo(id=143)
            X = statlog_australian_credit_approval.data.features
            y = statlog_australian_credit_approval.data.targets
            df = pd.concat([X, y], axis=1)
            numerical_cols = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"]
            categorical_cols = []
            target_col = "A15"
        elif dataset == 'iris': 
            df = datasets.load_iris(as_frame=True).frame.head(10)#for fast test
            numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
            categorical_cols = []
            target_col = 'target'
        # File-based datasets
        elif dataset in ["german_credit","credit_card_fraud" , "adult", "compas", "bank", "home_credit", "lending_club", "paysim", "ieee_cis", "churn", 
                         "credit", "travel", "king", "heloc", "modified_admissions", "data01", "heart"]:
            file_path = os.path.join(data_dir, f"{dataset}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}. Please download the dataset.")

            df = pd.read_csv(file_path, header=0, sep=",")
            # df = df.applymap(lambda x: x.replace(' ', '_'))

            print(df.info())

            if dataset == "adult":
                numerical_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
                categorical_cols = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
                target_col = "income-per-year"
            elif dataset == "german_credit":
                numerical_cols = ["Credit amount", "Duration"]
                categorical_cols = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
                target_col = "Risk"
            elif dataset == "compas":
                numerical_cols = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
                categorical_cols = ["race", "sex", "c_charge_degree"]
                target_col = "two_year_recid"
            elif dataset == "bank":
                numerical_cols = ["balance", "duration", "campaign", "pdays", "previous"]
                categorical_cols = ["age", "marital", "job", "education", "default", "housing", "loan", "contact", "poutcome"]
                target_col = "subscribe"
            elif dataset == "home_credit": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "TARGET"
            elif dataset == "credit_card_fraud": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "Class"
            elif dataset == "lending_club": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "loan_status"
            elif dataset == "paysim": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "isFraud"
            elif dataset == "ieee_cis": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "isFraud"
            elif dataset == "heloc": 
                numerical_cols = [
                                    "ExternalRiskEstimate", 
                                    "MSinceOldestTradeOpen", 
                                    "MSinceMostRecentTradeOpen", 
                                    "AverageMInFile", 
                                    "NumSatisfactoryTrades", 
                                    "NumTrades60Ever2DerogPubRec", 
                                    "NumTrades90Ever2DerogPubRec", 
                                    "PercentTradesNeverDelq", 
                                    "MSinceMostRecentDelq", 
                                    "MaxDelq2PublicRecLast12M", 
                                    "MaxDelqEver", 
                                    "NumTotalTrades", 
                                    "NumTradesOpeninLast12M", 
                                    "PercentInstallTrades", 
                                    "MSinceMostRecentInqexcl7days", 
                                    "NumInqLast6M", 
                                    "NumInqLast6Mexcl7days", 
                                    "NetFractionRevolvingBurden", 
                                    "NetFractionInstallBurden", 
                                    "NumRevolvingTradesWBalance", 
                                    "NumInstallTradesWBalance", 
                                    "NumBank2NatlTradesWHighUtilization", 
                                    "PercentTradesWBalance"
                                ]
                categorical_cols = []
                target_col = "RiskPerformance"
            elif dataset == "king": #define later
                numerical_cols = []
                categorical_cols = []
                target_col = "price"
            elif dataset == "travel":
                numerical_cols = [
                                    "Age",
                                    "ServicesOpted"
                                 ]
                categorical_cols = ["FrequentFlyer",
                                    "AnnualIncomeClass",
                                    "AccountSyncedToSocialMedia",
                                    "BookedHotelOrNot"]
                target_col = "Target"
            elif dataset == "credit":
                numerical_cols = [
                                    "RevolvingUtilizationOfUnsecuredLines",
                                    "age",
                                    "NumberOfTime30-59DaysPastDueNotWorse",
                                    "DebtRatio",
                                    "MonthlyIncome",
                                    "NumberOfOpenCreditLinesAndLoans",
                                    "NumberOfTimes90DaysLate",
                                    "NumberRealEstateLoansOrLines",
                                    "NumberOfTime60-89DaysPastDueNotWorse",
                                    "NumberOfDependents"
                                ]
                categorical_cols = []
                target_col = "SeriousDlqin2yrs"
            elif dataset == "churn":
                df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)
                df['TotalCharges'] = df['TotalCharges'].fillna(0)
                numerical_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
                categorical_cols = [
                                    "gender",
                                    "Partner",
                                    "Dependents",
                                    "PhoneService",
                                    "MultipleLines",
                                    "InternetService",
                                    "OnlineSecurity",
                                    "OnlineBackup",
                                    "DeviceProtection",
                                    "TechSupport",
                                    "StreamingTV",
                                    "StreamingMovies",
                                    "Contract",
                                    "PaperlessBilling",
                                    "PaymentMethod"
                                ]
                target_col = "Churn"
            elif dataset == "modified_admissions":
                # df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)
                # df['TotalCharges'] = df['TotalCharges'].fillna(0)
                # df = df.drop(columns=['row_id', 'subject_id', 'hadm_id'])
                categorical_cols = ['sex', 'ethnicity', 'metastatic_cancer', 'diabetes', 'vent', 'sepsis']
                numerical_cols = ['age', 'hospital_elixhauser', 'couch', 'sirs', 'qsofa',
       'anion_gap_medium', 'bocarbonate_medium', 'creatinine_medium',
       'glucose_medium', 'hemoglobin_medium', 'lactate_medium',
       'platelet_means', 'potassium_means', 'inr_means', 'sodium_means',
       'wbc_means', 'heart_rate_means', 'sys_bp_means', 'dias_bp_means',
       'resp_rate_means', 'temp_c_means', 'spo2_medians', 'urine_output']
                target_col = "hospital_expire_flag"
            elif dataset == "data01":
                categorical_cols = ['outcome', 'gendera', 'hypertensive',
       'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
       'depression', 'Hyperlipemia', 'Renal failure', 'COPD']
                numerical_cols =  ['age', 'BMI', 'heart rate',
       'Systolic blood pressure', 'Diastolic blood pressure',
       'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
       'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte',
       'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
       'NT-proBNP', 'Creatine kinase', 'Creatinine', 'Urea nitrogen',
       'glucose', 'Blood potassium', 'Blood sodium', 'Blood calcium',
       'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbonate',
       'Lactic acid', 'PCO2', 'EF']
                target_col = 'group'
                df["group"] = df["group"].replace({1: 0, 2: 1})
            elif dataset == "heart":
                categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
                numerical_cols =  ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                target_col = 'target'
                # df["group"] = df["group"].replace({1: 0, 2: 1})
            else:
                raise ValueError(f"No configuration for dataset: {dataset}")

        # Other datasets
        # elif dataset == "credit_card_fraud":
        #     df = pd.read_csv("https://www.kaggleusercontent.com/datasets/mlg-ulb/creditcardfraud/creditcard.csv")
        #     X = df.drop("Class", axis=1).values
        #     y = df["Class"].values
        #     names = df.columns.to_list()[:-1]
        elif dataset == "yahoo_finance":
            stock = "AAPL"
            df = yf.download(stock, period="5y", interval="1d")
            X = df.drop(columns=["Adj Close", "Close"]).values
            y = df["Close"].values
            names = df.columns.to_list()
        elif dataset == "fred":
            fred_api_key = 'fdb6ef3369b122ace95ed0f0c6783462'
            if not fred_api_key:
                raise ValueError("FRED API key is required for this dataset.")
            fred = Fred(api_key=fred_api_key)
            data = fred.get_series("GDP")
            X = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            names = ["Time"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # numerical_cols = [x.replace(' ', '_') for x in numerical_cols]
        # categorical_cols = [x.replace(' ', '_') for x in categorical_cols]

        # Process the dataset if loaded from a file
        if df is not None:
            X = df[numerical_cols].values
            for col in categorical_cols:
                le = LabelEncoder()
                X = np.hstack((X, le.fit_transform(df[col]).reshape(-1, 1)))
            y = LabelEncoder().fit_transform(df[target_col])
            # Convert y to a pandas Series TODO
            y_series = pd.Series(y)
            
            # 1. Describe (basic statistics)
            print("Basic Statistics:")
            print(y_series.describe())
            
            # 2. Unique values
            print("\nUnique Values:")
            print(y_series.unique())
            
            # 3. Check for NaN values
            print("\nMissing Values (NaN):")
            print(y_series.isnull().sum())
            names = numerical_cols + categorical_cols

        # Normalize X if required
        if normalize_x:
            print("normalize X")
            X = MinMaxScaler().fit_transform(X)
            X = np.around(X, 2)
        else:
            print("don't normalize X")

        # Split the dataset into train and test sets
        if X is not None and y is not None:
            if train_size == 2.0: #2 mean no split
                y = np.array(y)
                X_train = X
                y_train = y
                X_test = None
                y_test = None
            else:
                X, y = train_test_split(X,
                    train_size=0.1,
                    stratify=y,
                    random_state=seed) #temporary, take too long to train
                if dataset in ["fred", "yahoo_finance", "king", "california"]:  # regression tasks
                    y = y.reshape(-1, 1)
                    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
                else:  # classification tasks
                    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=seed)

            # Further split the training set if train_size < 1.0
            if train_size < 1.0:
                X_train, _, y_train, _ = train_test_split(X_train_, y_train_, test_size=(1.0 - train_size), shuffle=True, stratify=y_train_, random_state=seed)
            elif train_size == 1.0:
                X_train, y_train = X_train_, y_train_

            # Summary
            n_train, n_feature, n_class = X_train.shape[0], X_train.shape[1], len(np.unique(y_train))
            print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
            if X_test is not None:
                n_test = X_test.shape[0]
                print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
            else: n_test = None
            print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
            print("feature_names: {}".format(names))

            return X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, names

    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return None, None, None, None, None, None, None, None, None
