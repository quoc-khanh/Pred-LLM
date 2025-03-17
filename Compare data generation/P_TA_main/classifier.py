import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import format_row


def remove_random_values(input_text, num_remove=2):
    tokens = input_text.split(", ")
    candidates = [i for i in range(len(tokens)) if " is " in tokens[i]]

    if len(candidates) < num_remove:
        num_remove = len(candidates)

    remove_indices = random.sample(candidates, num_remove)
    
    new_tokens = []
    missing_slots = {}

    for i, token in enumerate(tokens):
        if i in remove_indices:
            col_name = token.split(" is ")[0]
            new_tokens.append(f"{col_name} is")  
            missing_slots[i] = col_name
        else:
            new_tokens.append(token)

    return ", ".join(new_tokens), missing_slots


class TextClassifier(nn.Module):
    def __init__(self, model_name):
        super(TextClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(label)}


def prepare_classifier_data(real_texts, generated_texts):
    real_labels = [1] * len(real_texts)  
    generated_labels = [0] * len(generated_texts)  
    
    texts = real_texts + generated_texts
    labels = real_labels + generated_labels
    
    return texts, labels



def train_classifier(dataloader, classifier, optimizer, loss_fn, epochs=3):
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad()

            outputs = classifier(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")



def classifier_train(csv_pth="heloc.csv", N = 2, model_path="./gpt2_finetuned", model_name="gpt2", classifier_save_pth="./classifier.pth", write_csv=False):
    df = pd.read_csv(csv_pth)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def fill_missing_values(input_text, missing_slots):
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
    
        return ", ".join(new_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    model.eval()  
    df["formatted_text"] = df.apply(format_row, axis=1)
    
    df["corrupted_text"] = df["formatted_text"].apply(lambda x: remove_random_values(x, num_remove=N))
    df["repaired_text"] = df["corrupted_text"].apply(lambda x: fill_missing_values(x[0], x[1]))
    if write_csv:
        dict={}
        for sample in df["repaired_text"]:
            key_list=sample.split(", ") #["xx is xx", ...]
            for key in key_list:
                key, key_value=key.split(" is ")
                if key not in dict:
                    dict[key]=[key_value]
                else:
                    dict[key].append(key_value)
        df = pd.DataFrame(dict)
        df.to_csv('output.csv', index=False)
        return
    
    real_texts = df["formatted_text"].tolist()
    generated_texts = df["repaired_text"].tolist()  
    
    texts, labels = prepare_classifier_data(real_texts, generated_texts)
    
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    classifier = TextClassifier(model_name=model_name).cuda()
    optimizer = AdamW(classifier.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    
    train_classifier(dataloader, classifier, optimizer, loss_fn)
    torch.save(classifier.state_dict(), classifier_save_pth)
    return df