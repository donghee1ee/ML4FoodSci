# import sqlite3
# import pandas as pd
# import ast
# from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, TrainingArguments, Trainer
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from transformers import EvalPrediction
from fuzzywuzzy import fuzz
import torch
from tqdm import tqdm
import random
import numpy as np
import logging

from model import ConstrainedT5
from utils import get_constraint_ids, prepare_data, TokenizerWrapper

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

prompt = 'Predict the flavor given the following chemical compounds: '
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


def compute_metrics(p: EvalPrediction): ## TODO - 아마 eval dataset 전부 로드가 되니까 cuda out of memory 뜨는 듯
    # decode
    token_ids = np.argmax(p.predictions[1], axis=-1) ## TODO
    preds = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in token_ids]
    targets = [tokenizer.decode(label, skip_special_tokens=True) for label in p.label_ids]
    
    # Perplexity
    # loss = p.predictions.mean().item()
    # perplexity = torch.exp(torch.tensor(loss)).item()
    
    # BLEU (higher is better) - usually used in long sentence
    bleu_score = sum([sentence_bleu([target], pred) for target, pred in zip(targets, preds)]) / len(preds)

    # Fuzzy Match Ratio (average over all instances) # normalized Edit distance? (higher is better) - 0-100
    avg_fuzzy_ratio = sum([fuzz.ratio(pred, target) for pred, target in zip(preds, targets)]) / len(preds)
    
    return {"bleu": bleu_score, 'fuzzy_ratio': avg_fuzzy_ratio}

def custom_train():
    model = model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for index, row in train_df.iterrows():
            optimizer.zero_grad()
            
            # Encoding input and target
            input_encoding = tokenizer(
                prompt + row['input_text'],
                padding="longest",
                max_length=512,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to('cuda') 
            
            target_encoding = tokenizer(
                row['target_text'],
                padding="longest",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to('cuda')
            
            labels = target_encoding["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward pass
            outputs = model(input_ids=input_encoding["input_ids"], attention_mask=input_encoding["attention_mask"], labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")

        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        for index, row in val_df.iterrows():
            with torch.no_grad():  
                input_encoding = tokenizer(
                    prompt + row['input_text'],
                    padding="longest",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to('cuda') 

                target_encoding = tokenizer(
                    row['target_text'],
                    padding="longest",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                ).to('cuda')
                
                labels = target_encoding["input_ids"]
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(input_ids=input_encoding["input_ids"], attention_mask=input_encoding["attention_mask"], labels=labels)
                
                val_loss = outputs.loss
                total_val_loss += val_loss.item()

                pred_tokens = torch.argmax(outputs.logits, dim=2)
                all_preds.extend([tokenizer.decode(tokens) for tokens in pred_tokens])
                all_targets.extend([row['target_text']])

def train():
    #######
    lr = 1e-04
    epochs = 20
    batch_size = 8
    output_dir="./best_model_with_constraints_large"
    logging_dir = './logs'
    model_name = "google/flan-t5-base"
    #######
    logger.info("####### main start #######")

    train_df, val_df, test_df = prepare_data(prompt)

    # Initialize the tokenizer
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenize the data
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # train_dataset = train_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    # val_dataset = val_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    # test_dataset = test_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer_wrapper = TokenizerWrapper(tokenizer)

    train_dataset = train_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)

    constraint_ids = get_constraint_ids(tokenizer)

    # Load the flan-T5 model
    model = ConstrainedT5.from_pretrained(model_name, constraint_ids=constraint_ids)

    # Define training arguments and initialize trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir=logging_dir,
        logging_strategy = 'epoch',##
        # logging_steps=10,##
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print("* Test start *")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    print("")


def test(model, tokenizer): ## TODO ## test_dataset batchsize 안 정해줘서 그냥 하나씩?
    #######
    lr = 1e-04
    epochs = 2
    batch_size = 8
    output_dir="./best_model_with_constraints2"
    logging_dir = './logs'
    #######

    train_df, val_df, test_df = prepare_data(prompt)

    # Initialize the tokenizer
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenize the data
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer_wrapper = TokenizerWrapper(tokenizer)

    train_dataset = train_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)

    # constraint_ids = get_constraint_ids(tokenizer)

    # Load the flan-T5 model
    # model = ConstrainedT5.from_pretrained("google/flan-t5-base", constraint_ids=constraint_ids)

    # Define training arguments and initialize trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir=logging_dir,
        logging_strategy = 'steps',##
        logging_steps=10,##
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    # Train the model
    # trainer.train()

    # Save tokenizer
    # tokenizer.save_pretrained(output_dir)
    # model.save_pretrained(output_dir)

    print("* Test start *")
    test_results = trainer.evaluate(test_dataset) ## trainer.predict랑 같음.
    print(test_results)



if __name__ == '__main__':
    train()
    # Load the fine-tuned model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = '/nfs_share2/code/donghee/Food/best_model_with_constraints_datamodify'
    # tokenizer = T5Tokenizer.from_pretrained(model_path)
    # constraint_ids = get_constraint_ids(tokenizer)

    # model = ConstrainedT5.from_pretrained(model_path, constraint_ids = constraint_ids)
    # # model = model.to(device)

    # test(model, tokenizer)
