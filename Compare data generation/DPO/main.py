# %cd /content/Pred-LLM/Compare data generation
#/P_TA_main

import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'P_TA_main'))
from P_TA_main.great import *
from P_TA_main.classifier import *
from P_TA_main.gan import *

#########################################
# Helper: compute log probability of generated text given a prompt.
#########################################
def compute_generated_log_prob(prompt, generated, model, tokenizer):
    """
    Computes the total log probability of the generated tokens conditioned on the prompt.
    It does this by computing the loss (negative log likelihood) on the generated tokens only.
    """
    full_text = prompt + " " + generated
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = prompt_ids.size(1)

    labels = inputs.input_ids.clone()
    labels[:, :prompt_length] = -100  # mask the prompt tokens

    outputs = model(input_ids=inputs.input_ids, labels=labels)
    num_generated_tokens = inputs.input_ids.size(1) - prompt_length
    total_loss = outputs.loss * num_generated_tokens

    log_prob = -total_loss
    return log_prob

#########################################
# DPO loss function.
#########################################
def dpo_loss(prompt, win_text, lose_text, model, ref_model, tokenizer, beta=1.0):
    """
    Compute the DPO loss for a given prompt and a pair of outputs (win vs. lose).
    It calculates the log probability difference for the candidate outputs under both the current model and the fixed reference,
    and then applies a logistic loss.
    """
    logp_model_win = compute_generated_log_prob(prompt, win_text, model, tokenizer)
    logp_model_lose = compute_generated_log_prob(prompt, lose_text, model, tokenizer)

    with torch.no_grad():
        logp_ref_win = compute_generated_log_prob(prompt, win_text, ref_model, tokenizer)
        logp_ref_lose = compute_generated_log_prob(prompt, lose_text, ref_model, tokenizer)

    delta_model = logp_model_win - logp_model_lose
    delta_ref = logp_ref_win - logp_ref_lose

    loss = - torch.log(torch.sigmoid(beta * (delta_model - delta_ref)))
    return loss

#########################################
# DPO training loop.
#########################################
def train_gpt2_with_dpo(df, model, ref_model, tokenizer, classifier, num_epochs=3, N=2, beta=1.0):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for _, row in df.iterrows():
            input_text = row["formatted_text"]
            corrupted_text, missing_slots = remove_random_values(input_text, num_remove=N)
            prompt = corrupted_text

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            output_candidate1 = model.generate(
                input_ids,
                max_length=tokenizer.model_max_length,
                do_sample=True,
                top_p=0.9
            )
            candidate1 = tokenizer.decode(output_candidate1[0], skip_special_tokens=True)

            output_candidate2 = model.generate(
                input_ids,
                max_length=tokenizer.model_max_length,
                do_sample=True,
                top_p=0.9
            )
            candidate2 = tokenizer.decode(output_candidate2[0], skip_special_tokens=True)

            enc1 = tokenizer(candidate1, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            enc2 = tokenizer(candidate2, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids_1 = enc1["input_ids"].to(model.device)
            attn_mask_1 = enc1["attention_mask"].to(model.device)
            input_ids_2 = enc2["input_ids"].to(model.device)
            attn_mask_2 = enc2["attention_mask"].to(model.device)

            with torch.no_grad():
                logits1 = classifier(input_ids_1, attn_mask_1)
                probs1 = torch.softmax(logits1, dim=-1)
                pred1 = torch.argmax(probs1, dim=-1).item()

                logits2 = classifier(input_ids_2, attn_mask_2)
                probs2 = torch.softmax(logits2, dim=-1)
                pred2 = torch.argmax(probs2, dim=-1).item()

            if pred1 == pred2:
                continue

            if pred1 == 0 and pred2 == 1:
                win_text = candidate1
                lose_text = candidate2
            elif pred1 == 1 and pred2 == 0:
                win_text = candidate2
                lose_text = candidate1
            else:
                continue

            loss = dpo_loss(prompt, win_text, lose_text, model, ref_model, tokenizer, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

#########################################
# Main training pipeline.
#########################################
def train_dpo(csv_path: str) -> pd.DataFrame:
    # csv_path = "/content/Pred-LLM/Compare data generation/data/credit.csv"#"heloc.csv"
    model_name = 'gpt2'
    save_path = "./gpt2_finetuned"
    classifier_save_path = "./classifier.pth"
    N = 2
    total_epoch_num = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)

    # Fine-tune GPT-2 using your original method.
    for epoch in range(total_epoch_num):
        print("----- EPOCH: ", epoch)
        print("LLM Training..")
        if epoch != 0:
            train_great(csv_path, save_path, save_path)  # assumes train_great is defined elsewhere
        else:
            train_great(csv_path, model_name, save_path)

        print("Classifier Training..")
        df = classifier_train(
            csv_pth=csv_path,
            N=N,
            model_path=save_path,
            model_name=model_name,
            classifier_save_pth=classifier_save_path
        )

        classifier = TextClassifier(model_name=model_name).to(device)
        classifier.load_state_dict(torch.load(classifier_save_path))
        classifier.eval()

        model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # Use the finetuned GPT-2 as the reference model
        ref_model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
        ref_model.eval()

        print("DPO Training..")
        train_gpt2_with_dpo(df, model, ref_model, tokenizer, classifier)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("LLM Finetuning..")
        train_great(csv_path, save_path, save_path)
        print("Data Generating..")
        df = classifier_train(
            csv_pth=csv_path,
            N=N,
            model_path=save_path,
            model_name=model_name,
            classifier_save_pth=classifier_save_path,
            write_csv=True
        )
        return df
# if __name__ == "__main__":
#     main()
