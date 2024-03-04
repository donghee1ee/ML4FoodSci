from transformers import T5Tokenizer
from model.modeling_t5 import T5ForConditionalGeneration
from model.utils import get_constraint_ids
import torch
from model.utils import prepare_data, TokenizerWrapper
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import os
import json

def test():
    ###
    model_path = '/home/donghee/Food/best_model_equivariance_fix'
    result_path = os.path.join(model_path, 'test_result.json')
    # start_token = '<start_chemical>'
    ###
    assert os.path.exists(model_path)
    prompt = 'Predict the flavor given the following chemical compounds: '

    print("** Saving result to ", result_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    constraint_ids = get_constraint_ids(tokenizer)

    model = T5ForConditionalGeneration.from_pretrained(model_path, constraint_ids = constraint_ids)
    model = model.to(device)
    model.eval()

    _, _, test_df = prepare_data(prompt)

    test_dataset = Dataset.from_pandas(test_df)

    tokenizer_wrapper = TokenizerWrapper(tokenizer, prompt = prompt)
    test_dataset = test_dataset.map(tokenizer_wrapper.tokenize_function, batched=True)

    # prediction = []
    # target = []
    result_list = list()
    for row in tqdm(test_dataset, desc="Testing"): # TODO batch decoding
        input_sequence = row["input_text"]

        # Tokenize the input sequence
        input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids ## TODO
        input_ids = input_ids.to(model.device)

        # Generate a prediction
        with torch.no_grad():
            output_ids = model.generate(
                input_ids = input_ids,
                no_repeat_ngram_size=2,
                max_length=20,
                min_length=1,
                early_stopping=True,
                do_sample=True,
                num_beams=5,  # Use beam search with 5 beams
                temperature=0.9,  # Slightly more randomness in the choice of next tokens
                top_k=50,  # Keep only top 50 tokens for sampling
                top_p=0.95,  # Use nucleus sampling with p=0.95
                repetition_penalty=1.2,  # Apply a penalty for repeated tokens
                length_penalty=0.9, # Prefer shorter sequences slightly
            )

            # output_ids_greedy = model.generate(
            #     input_ids = input_ids,
            #     no_repeat_ngram_size=2,
            #     max_length=20,
            #     min_length=1,
            #     early_stopping=True,
            #     do_sample= False,
            # )

        # Decode the output IDs to a string
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # output_greedy = tokenizer.decode(output_ids_greedy[0], skip_special_tokens=True)
        # prediction.append(output)
        # target.append(row['output_text'])

        result_list.append({
            'input_text': row['input_text'],
            'Ground_truth': row['output_text'],
            'Prediction': output,
        })
    
    # result_list = list()
    # for entry, p, gt in zip(test_dataset, prediction, target):
    #     result_list.append({
    #         'input_text': entry['input_text'],
    #         'Ground_truth': gt,
    #         'Prediction': p,
        # })
    
    with open(result_path, 'w') as f:
        json.dump(result_list, f, indent=4)


    # # Create a new dataset with only 'input_ids' and 'attention_mask'
    # subset_test_dataset = test_dataset.remove_columns(columns_to_remove)

    # # Use data collator to handle variable-length sequences
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # # Create a DataLoader with the appropriate data collator
    # data_loader = DataLoader(subset_test_dataset, batch_size=8, collate_fn=data_collator)


    # # Create a DataLoader for the test dataset
    # # data_loader = DataLoader(test_dataset, batch_size=8, collate_fn=default_data_collator)

    # # Generate predictions in batches
    # predictions = []
    # for batch in data_loader:
    #     # Move batch to the same device as the model
    #     batch = {k: v.to(device) for k, v in batch.items()}
        
    #     # Generate predictions for the current batch
    #     batch_predictions = model.generate(
    #         input_ids=batch['input_ids'], 
    #         attention_mask=batch['attention_mask'],
    #         max_length=20,  # Maximum length of the output sequences
    #         min_length=1,
    #     )
        
    #     # Decode and add to the list of predictions
    #     predictions.extend(tokenizer.batch_decode(batch_predictions, skip_special_tokens=True))

    # # generated_flavors now contains the decoded predictions


if __name__ == '__main__':
    test()