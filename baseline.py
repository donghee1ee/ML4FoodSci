import sqlite3
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from transformers import EvalPrediction
from fuzzywuzzy import fuzz
import torch
from tqdm import tqdm
import random
import numpy as np

prompt = 'Predict the flavor given the following chemical compounds: '
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def prepare_data():
    csv_df = pd.read_csv('entity_mole_group.csv')
    csv_df['molecule_id'] = csv_df['molecule_id'].apply(lambda x: list(ast.literal_eval(x.replace("{", "[").replace("}", "]"))))

    conn = sqlite3.connect('flavordb.db')
    query = "SELECT entity_id, entity_alias_readable FROM food_entities"
    sql_df = pd.read_sql_query(query, conn)

    query = "SELECT id, common_name FROM molecules"
    molecule_map_df = pd.read_sql_query(query, conn)
    conn.close()

    molecule_map_df['common_name'] = molecule_map_df['common_name'].str.replace(' ', '_')
    molecule_map_dict = dict(zip(molecule_map_df.id, molecule_map_df.common_name))
    csv_df['molecule_name'] = csv_df['molecule_id'].apply(lambda ids: [molecule_map_dict[id] for id in ids])


    df = pd.merge(sql_df, csv_df, on='entity_id', how='inner') 
    df = df.rename(columns={'entity_alias_readable': 'entity_name'}) ## X: molecule_name, Y: entity_name

    df['input_text'] = df['molecule_name'].apply(lambda x: ' '.join(sorted(x, key=lambda k: random.random())))
    df['input_text'] = prompt + df['input_text']
    df['labels'] = df['entity_name']
    df['output_text'] = df['entity_name']

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

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

def main():
    #######
    lr = 1e-04
    epochs = 20
    batch_size = 8
    output_dir="./best_model3"
    #######

    train_df, val_df, test_df = prepare_data()

    # Initialize the tokenizer
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenize the data
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # train_dataset = train_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    # val_dataset = val_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    # test_dataset = test_dataset.map(lambda x: tokenizer(x['input_text'], max_length=512, truncation=True), batched=True)
    def tokenize_function(examples):
        tokenized_input = tokenizer(
            examples['input_text'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized_output = tokenizer(
            examples['labels'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized_input.input_ids,
            'attention_mask': tokenized_input.attention_mask,
            'decoder_input_ids': tokenized_output.input_ids,
            'labels': tokenized_output.input_ids  # Including labels for loss computation
        }

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)


    # Load the T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Define training arguments and initialize trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='fuzzy_ratio',
        greater_is_better=True,
        learning_rate=lr,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print("* Test start *")
    test(model, tokenizer, test_dataset)
    print("END")

def test(model, tokenizer, test_dataset): ## TODO ## test_dataset batchsize 안 정해줘서 그냥 하나씩?
    model.eval()

    total_fuzzy_score = 0

    # Iterate over the test dataset
    for row in tqdm(test_dataset, desc="Testing"):
        input_sequence = row["input_text"]
        target_sequence = row["output_text"]

        # Tokenize the input sequence
        input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids ## TODO
        input_ids = input_ids.to('cuda')

        # Generate a prediction
        with torch.no_grad():
            output_ids = model.generate(input_ids)

        # Decode the output IDs to a string
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Calculate Fuzzy Score
        fuzzy_score = fuzz.ratio(output, target_sequence)
        total_fuzzy_score += fuzzy_score

        if random.randint(1,100) < 10:
            print(f"input: {input_sequence}")
            print(f"* Predicted Flavor: {output}")
            print(f"* Target Flavor: {target_sequence}")
            print(f"Fuzzy Score: {fuzzy_score}")
            print("\n"+"="*40+"\n")

    avg_fuzzy_score = total_fuzzy_score / len(test_dataset)
    print(f"Average Fuzzy Score over test dataset: {avg_fuzzy_score}")


if __name__ == '__main__':
    main()
