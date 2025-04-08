import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.optim import AdamW#, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from P_TA_main.classifier import remove_random_values


def gpt2_loss_function(input_text, missing_slots, classifier, tokenizer, model):
    tokens = input_text.split(", ")
    new_tokens = tokens[:]
    
    for idx, col_name in missing_slots.items():
        prompt = f"{col_name} is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=input_ids.shape[1] + 5, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.9)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_value = generated_text.replace(f"{col_name} is", "").strip().split(",")[0]
        new_tokens[idx] = f"{col_name} is {generated_value}"

    generated_text = ", ".join(new_tokens)
    
    encoding = tokenizer(generated_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    
    with torch.no_grad():
        classifier_outputs = classifier(input_ids, attention_mask)
        logits = classifier_outputs
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    return pred

def train_gpt2_with_gan(df, model, tokenizer, classifier, num_epochs=3, N=2):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for _, row in df.iterrows():
            input_text = row["formatted_text"]
            
            corrupted_text, missing_slots = remove_random_values(input_text, num_remove=N)
            
            pred = gpt2_loss_function(corrupted_text, missing_slots, classifier, tokenizer, model)
            
            if pred == 0:  
                optimizer = AdamW(model.parameters(), lr=5e-5)
                optimizer.zero_grad()

                # Compute GPT-2 loss
                input_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(model.device)
                labels = input_ids.clone().to(model.device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(df)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
