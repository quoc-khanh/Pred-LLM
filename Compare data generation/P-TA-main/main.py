# from great import *
# from classifier import *
# from gan import *
# import torch
# import random
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AdamW
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from utils import format_row


# def main():
#     csv_path = "heloc.csv"
#     model_name = 'gpt2'
#     save_path="./gpt2_finetuned"
#     classifier_save_path="./classifier.pth"
#     N=2
#     total_epoch_num=1
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     df = pd.read_csv(csv_path)
#     for epoch in range(total_epoch_num):
#         print("----- EPOCH: ", epoch)
#         print("LLM Training..")
#         if epoch!=0:
#             train_great(csv_path, save_path, save_path)
#         else:
#             train_great(csv_path, model_name, save_path)
#         print("Classifier Training..")
#         df=classifier_train(csv_pth=csv_path, N = N, model_path=save_path, model_name=model_name, classifier_save_pth=classifier_save_path)

#         classifier = TextClassifier(model_name=model_name).cuda()
#         classifier.load_state_dict(torch.load("./classifier.pth"))
#         classifier.eval()

#         model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(save_path)
#         if not tokenizer.pad_token:
#             tokenizer.pad_token = tokenizer.eos_token

#         print("GAN Training..")
#         train_gpt2_with_gan(df, model, tokenizer, classifier)
#         model.save_pretrained(save_path)
#         tokenizer.save_pretrained(save_path)

#         print("LLM Finetuning..")
#         train_great(csv_path, save_path, save_path)
#         print("Data Generating..")
#         classifier_train(csv_pth=csv_path, N = N, model_path=save_path, model_name=model_name, classifier_save_pth=classifier_save_path, write_csv=True)


# if __name__=="__main__":
#     main()

from great import *
from classifier import *
from gan import *
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def train_pipeline(csv_path: str) -> pd.DataFrame:
    model_name = 'gpt2'
    save_path = "./gpt2_finetuned"
    classifier_save_path = "./classifier.pth"
    N = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    print("LLM Training..")
    train_great(csv_path, model_name, save_path)
    
    print("Classifier Training..")
    df = classifier_train(csv_pth=csv_path, N=N, model_path=save_path, model_name=model_name, classifier_save_pth=classifier_save_path)

    classifier = TextClassifier(model_name=model_name).to(device)
    classifier.load_state_dict(torch.load(classifier_save_path))
    classifier.eval()

    model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("GAN Training..")
    train_gpt2_with_gan(df, model, tokenizer, classifier)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("LLM Finetuning..")
    train_great(csv_path, save_path, save_path)
    print("Data Generating..")
    df = classifier_train(csv_pth=csv_path, N=N, model_path=save_path, model_name=model_name, classifier_save_pth=classifier_save_path, write_csv=True)
    
    return df
