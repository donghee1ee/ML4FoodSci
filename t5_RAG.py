# import sqlite3
# import pandas as pd
# import ast
# from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, TrainingArguments, Trainer, RagRetriever, AutoTokenizer, RagTokenForGeneration, RagTokenizer
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from transformers import EvalPrediction
from fuzzywuzzy import fuzz
import torch
from tqdm import tqdm
import random
import numpy as np
import logging
import os

from model.modeling_t5 import T5ForConditionalGeneration
from model.model import CustomRagSequenceForGeneration
from model.utils import get_constraint_ids, prepare_data, TokenizerWrapper

from commons.utils import setup_logger

prompt = 'Predict the flavor given the following combination of molecules: '

examples = [
    ('Predict the flavor given the following combination of molecules: Gamma-Terpinene, Thymol methyl ether, Methyl Acetate, Ethyl 3-hydroxyhexanoate, Terpinen-4-ol, Alpha-Pinene, Nootkatone, Methyl Anthranilate, Perillyl acetate, beta-Sinensal, Octanal, 3-Carene, Thymol, 2-Methyl-1-propanol, alpha-TERPINEOL, 2-(4-methylphenyl)propan-2-ol, Citral, Methyl butyrate, (2E,4E)-deca-2,4-dienal, 1-Penten-3-Ol, Ethyl Heptanoate', 'Tangerine'), # GT: Tangerine
    ('Predict the flavor given the following combination of molecules: thiamine, Heptanoic Acid', 'Flour'), # GT: Flour
]


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

def train():
    #######
    # lr = 1e-04
    epochs = 10
    batch_size = 4
    project_name = 'temp'
    output_dir= os.path.join('outputs', project_name)
    logging_dir = os.path.join('logs', project_name)
    # model_name = "google/flan-t5-base"
    # start_token = '<start_chemical>'
    rag_model = 'facebook/rag-token-nq' # facebook/rag-sequence-nq
    logging_steps = 20
    prompt = 'Predict the flavor given the following combination of molecules: '
    n_docs = 5 #
    constraint_decoding = False
    #######
    logging.info("####### main start #######")

    train_df, val_df, test_df = prepare_data(prompt)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    

    tokenizer = RagTokenizer.from_pretrained(rag_model)

    def tokenize_for_rag(examples):
        # This function assumes 'examples' is a batch from your dataset
        input_texts = examples['input_text']
        labels = examples['labels']
        
        # Tokenize inputs and labels. For RAG, you might only need to tokenize the input texts,
        # as the labels (answers) would be generated by the model.
        # However, if you're training RAG, you might need to adjust this to fit your training scheme.
        ## TODO tokenizer.question_encoder... tokenizer.generator...
        
        # tokenized_inputs = tokenizer.question_encoder.batch_encode_plus(input_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        
        # tokenized_labels = tokenizer.batch_encode_plus(text_target=labels, max_length=32, truncation=True, padding="max_length", return_tensors="pt")  # This might need adjustments based on your use case
        tokenized_labels = tokenizer(text_target=labels, max_length=32, truncation=True, padding="max_length", return_tensors="pt")  # This might need adjustments based on your use case # TODO tokenizer.generator.batch_encode_plus 해야하는 거 아닌가..ㅠㅠ

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_labels['input_ids'],
        }

    # Apply the tokenization function to the datasets
    # TODO store locally
    train_dataset = train_dataset.map(tokenize_for_rag, batched=True)
    val_dataset = val_dataset.map(tokenize_for_rag, batched=True)
    test_dataset = test_dataset.map(tokenize_for_rag, batched=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # RAG
    retriever = RagRetriever.from_pretrained(
        rag_model, index_name="exact", use_dummy_dataset=True
    )

    if constraint_decoding:
        constraint_ids = get_constraint_ids(tokenizer.generator)
    else:
        constraint_ids = None
    model = CustomRagSequenceForGeneration.from_pretrained(rag_model, retriever = retriever, constraint_ids = constraint_ids, n_docs = n_docs)

    model.config.reduce_loss = True ### TODO

    # Define training arguments and initialize trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir=logging_dir,
        logging_strategy = 'steps',##
        logging_steps=logging_steps,##
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        # dataloader_num_workers=4,
        # learning_rate=lr,
        warmup_steps=100,
        # weight_decay=0.01,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
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

    logging.info("* Test start *")
    test_results = trainer.evaluate(test_dataset)
    logging.info(test_results)
    
    ##

    prompts = [example[0] for example in examples]
    ground_truths = [example[1] for example in examples]

    for i, prompt in enumerate(prompts):
        model.config.reduce_loss = False ## TODO
        model.config.n_docs = 10 #
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        input_ids = inputs["input_ids"]
        question_hidden_states = model.question_encoder(input_ids)[0]
        docs_dict = retriever(input_ids.detach().cpu().numpy(), question_hidden_states.detach().cpu().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(
            question_hidden_states.cpu().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)

        generated = model.generate(
            context_input_ids=docs_dict["context_input_ids"].to(model.device),
            context_attention_mask=docs_dict["context_attention_mask"].to(model.device),
            doc_scores=doc_scores.to(model.device),
        )

        retrieved_docs = tokenizer.batch_decode(docs_dict['context_input_ids'], skip_special_tokens=True)
        retrieved_docs_titles = [doc.split('/')[0].strip() for doc in retrieved_docs]
        correct_retrieved = [1 if title in prompt else 0 for title in retrieved_docs_titles]
        precision = sum(correct_retrieved) / len(correct_retrieved)

        generated_string = tokenizer.decode(generated[0], skip_special_tokens=True)

        logging.info('* Original RAG *')
        logging.info(f'\n* Prompt: {prompt}, \n* Generated Flavor: {generated_string}, \n* Ground Truth: {ground_truths[i]} \n* Retrieved titles: {retrieved_docs_titles} \n* Retrieval Precision: {precision}')

    logging.info("END")


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
    setup_logger()
    train()
    # Load the fine-tuned model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = '/nfs_share2/code/donghee/Food/best_model_with_constraints_datamodify'
    # tokenizer = T5Tokenizer.from_pretrained(model_path)
    # constraint_ids = get_constraint_ids(tokenizer)

    # model = ConstrainedT5.from_pretrained(model_path, constraint_ids = constraint_ids)
    # # model = model.to(device)

    # test(model, tokenizer)
