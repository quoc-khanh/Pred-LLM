import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from utils import format_row


class GreatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100  

        outputs = model(**inputs, labels=labels)  
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss



def train_great(csv_path = "heloc.csv", model_name = 'gpt2', save_pth="./gpt2_finetuned"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)
    

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_texts = train_df.apply(format_row, axis=1).tolist()
    val_texts = val_df.apply(format_row, axis=1).tolist()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    

    tokenized_datasets = DatasetDict({
        "train": train_dataset.map(tokenize_function, batched=True),
        "validation": val_dataset.map(tokenize_function, batched=True),
    })
    

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    

    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",  
        evaluation_strategy="epoch",  
        save_strategy="epoch", 
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,  
        num_train_epochs=3, 
        weight_decay=0.01,  
        save_total_limit=2, 
        logging_dir="./logs",  
        logging_steps=10, 
    )
    
    
    trainer = GreatTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    trainer.can_return_loss=True
    trainer.train()

    model.save_pretrained(save_pth)
    tokenizer.save_pretrained(save_pth)