import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import ast
import torch
# import logging

def get_constraint_ids(tokenizer):
    """
    return constraint ids containing only food related token ids
    """
    conn = sqlite3.connect('flavordb.db')
    query = "SELECT entity_alias_readable FROM food_entities"
    df = pd.read_sql_query(query, conn)
    constraint_words = set(df['entity_alias_readable'])
    constraint_ids = [tokenizer.encode(word, add_special_tokens=False) for word in constraint_words]
    constraint_ids = [id for sublist in constraint_ids for id in sublist]

    return constraint_ids

def prepare_data(prompt):
    csv_df = pd.read_csv('entity_mole_group.csv')
    csv_df['molecule_id'] = csv_df['molecule_id'].apply(lambda x: list(ast.literal_eval(x.replace("{", "[").replace("}", "]"))))

    conn = sqlite3.connect('flavordb.db')
    query = "SELECT entity_id, entity_alias_readable FROM food_entities"
    sql_df = pd.read_sql_query(query, conn)

    query = "SELECT id, common_name FROM molecules"
    molecule_map_df = pd.read_sql_query(query, conn)
    conn.close()

    # molecule_map_df['common_name'] = molecule_map_df['common_name'].str.replace(' ', '_')
    molecule_map_dict = dict(zip(molecule_map_df.id, molecule_map_df.common_name))
    csv_df['molecule_name'] = csv_df['molecule_id'].apply(lambda ids: [molecule_map_dict[id] for id in ids])


    df = pd.merge(sql_df, csv_df, on='entity_id', how='inner') 
    df = df.rename(columns={'entity_alias_readable': 'entity_name'}) ## X: molecule_name, Y: entity_name

    # df['input_text'] = df['molecule_name'].apply(lambda x: ', '.join(sorted(x, key=lambda k: random.random())))
    # df['input_text'] = prompt + df['input_text']
    df['input_text'] = prompt + df['molecule_name'].apply(lambda x: ', '.join(sorted(x, key=lambda k: random.random())))
    df['labels'] = df['entity_name']
    df['output_text'] = df['entity_name']

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

class TokenizerWrapper:
    def __init__(self, tokenizer, prompt, max_compounds=100):
        self.tokenizer = tokenizer
        self.max_compounds = max_compounds
        self.prompt = prompt
        self.prompt_len = len(self.tokenizer.encode(self.prompt, add_special_tokens=True))

    def tokenize_function(self, examples):
        tokenized_input = self.tokenizer(
            examples['input_text'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized_output = self.tokenizer(
            examples['labels'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Find the position of the special token
        # start_token_id = self.tokenizer.encode(self.start_token, add_special_tokens=False)[0]
        # start_positions = (tokenized_input.input_ids == start_token_id).nonzero(as_tuple=True)[1] ## shape (655)

        #  # Calculate compound positions
        # compound_positions = [
        #     self.calculate_compound_positions(start_pos, input_ids) 
        #     for start_pos, input_ids in zip(start_positions, tokenized_input.input_ids)
        # ]

        # if len(set(start_positions.tolist())) == 1:
        #     # single prompt
        #     start_position = torch.tensor(int(start_positions.tolist()[0]), dtype = torch.long)
        # else:
        #     logger.warning("multiple prompt length")

        # Assuming the compounds are at the end, their positions are from start_positions to the end
        # Adjust this logic if the compounds can be in different positions
         # Calculate compound positions for each example
        compound_positions = [
            self.calculate_compound_positions(input_text, input_ids) 
            for input_text, input_ids in zip(examples['input_text'], tokenized_input.input_ids)
        ]

        # Pad compound_positions to have a uniform length
        compound_positions = [self.pad_positions(pos) for pos in compound_positions]

        return {
            'input_ids': tokenized_input.input_ids,
            'attention_mask': tokenized_input.attention_mask,
            'labels': tokenized_output.input_ids,
            'positions': compound_positions,
        }
    
    def calculate_compound_positions(self, input_text, input_ids):
        # Split the input text into individual compounds
        compounds = input_text.split(self.prompt)[-1].split(', ')
        current_position = self.prompt_len

        # Initialize positions with -1 (for padding)
        positions = [-1] * len(input_ids)

        # Mark positions for the prompt
        for i in range(current_position):
            positions[i] = 0

        # Mark positions for each compound
        compound_id = 1
        for compound in compounds:
            compound_token_length = len(self.tokenizer.encode(compound, add_special_tokens=False))
            for i in range(compound_token_length):
                if current_position + i < len(positions):
                    positions[current_position + i] = compound_id
            current_position += compound_token_length
            compound_id += 1

        return positions

    
    # def pad_positions(self, positions):
    #     # Pad the positions list to have a uniform length
    #     padding_length = self.max_compounds - len(positions)
    #     positions.extend([(-1, -1)] * padding_length)
    #     return positions

    def pad_positions(self, positions):
        # Ensure the positions list is of length 512
        if len(positions) > 512:
            positions = positions[:512]
        elif len(positions) < 512:
            positions.extend([-1] * (512 - len(positions)))
        return positions


# def compute_invariant_position(query_length, key_length, positions):
#     # Initialize the relative position matrix
#     relative_position_matrix = torch.zeros((query_length, key_length), dtype=torch.long)

#     # Convert positions to a format that is easier to work with
#     compound_masks = get_compound_masks(query_length, positions)

#     for q_idx in range(query_length):
#         for k_idx in range(key_length):
#             # Check if q_idx and k_idx are in the same compound
#             same_compound = any(compound_masks[compound_id][q_idx] and compound_masks[compound_id][k_idx] for compound_id in compound_masks)

#             # Assign relative position based on whether they are in the same compound
#             relative_position_matrix[q_idx, k_idx] = 0 if same_compound else 1

#     return relative_position_matrix

# def get_compound_masks(length, positions):
#     compound_masks = {}
#     for compound_id, (start, end) in enumerate(positions):
#         mask = [start <= idx <= end for idx in range(length)]
#         compound_masks[compound_id] = mask
#     return compound_masks

# def compute_invariant_position_batch(batch_query_length, batch_key_length, batch_positions):
#     """
#     Compute a batch of relative position matrices, considering compound invariance.

#     Parameters:
#     - batch_query_length (int): The length of the query sequences in the batch.
#     - batch_key_length (int): The length of the key sequences in the batch.
#     - batch_positions (list of lists or tensor): A batch of compound positions with shape [batch_size, 100, 2].

#     Returns:
#     - A list of relative position matrices, one for each example in the batch.
#     """
#     batch_relative_position_matrices = []

#     for positions in batch_positions:
#         relative_position_matrix = torch.zeros((batch_query_length, batch_key_length), dtype=torch.long)
#         compound_masks = get_compound_masks(batch_query_length, positions)

#         for q_idx in range(batch_query_length):
#             for k_idx in range(batch_key_length):
#                 # Check if q_idx and k_idx are in the same compound
#                 same_compound = any(compound_masks[compound_id][q_idx] and compound_masks[compound_id][k_idx] for compound_id in compound_masks)

#                 # Assign relative position based on whether they are in the same compound
#                 relative_position_matrix[q_idx, k_idx] = 0 if same_compound else 1

#         batch_relative_position_matrices.append(relative_position_matrix)

#     return batch_relative_position_matrices




# def compute_invariant_position(query_length, key_length, positions):
#     # Initialize the relative position matrix
#     relative_position_matrix = torch.zeros((query_length, key_length), dtype=torch.long)

#     # Assign unique identifiers to each compound
#     compound_ids = {i: idx for idx, (start, end) in enumerate(positions)}

#     # Calculate relative positions
#     for q_idx in range(query_length):
#         for k_idx in range(key_length):
#             q_compound_id = find_compound_id(q_idx, positions, compound_ids)
#             k_compound_id = find_compound_id(k_idx, positions, compound_ids)

#             # Calculate the relative position based on compound IDs
#             relative_position = calculate_relative_position(q_compound_id, k_compound_id)
#             relative_position_matrix[q_idx, k_idx] = relative_position

#     return relative_position_matrix

# def find_compound_id(token_idx, positions, compound_ids):
#     # Find the compound ID for a given token index
#     for start, end in positions:
#         if start <= token_idx <= end:
#             return compound_ids[(start, end)]
#     return None

# def calculate_relative_position(q_compound_id, k_compound_id):
#     # Implement the logic to calculate relative position based on compound IDs
#     # This could be a simple difference, or a more complex function
#     return q_compound_id - k_compound_id


# class TokenizerWrapper:
#     def __init__(self, tokenizer, prompt):
#         self.tokenizer = tokenizer
#         self.prompt = prompt

#     def tokenize_function(self, examples):
#         # Modified tokenization process
#         tokenized_inputs = []
#         compound_positions = []
#         for text in examples['input_text']:
#             tokenized_output, start_positions, end_positions = tokenize_and_map_compounds(text)
#             tokenized_inputs.append(tokenized_output)
#             compound_positions.append({'start': start_positions, 'end': end_positions})

#         # You can store compound_positions for later use or include them in your dataset
#         # For example, if you want to include them in your dataset:
#         return {
#             'input_ids': [x['input_ids'] for x in tokenized_inputs],
#             'attention_mask': [x['attention_mask'] for x in tokenized_inputs],
#             'labels': [self.tokenizer(ex['labels'], return_tensors="pt")['input_ids'] for ex in examples],
#             'compound_positions': compound_positions
#         }
    
#     def tokenize_and_map_compounds(self, text):
#          # Tokenize the entire text
#         tokenized_output = self.tokenizer(text, return_tensors="pt")

#         # Extract the compound part from the text (assuming the text starts with a prompt)
#         compound_text = text[len(self.prompt):]
#         compounds = compound_text.split(', ')

#         # Initialize lists to hold the start and end positions of each compound
#         start_positions = []
#         end_positions = []

#         # Current position in the tokenized sequence
#         current_position = len(self.tokenizer(self.prompt)['input_ids']) - 1  # -1 for the prompt's last token

#         for compound in compounds:
#             # Find the start and end token positions for each compound
#             compound_tokens = self.tokenizer(compound)['input_ids']
#             start_positions.append(current_position + 1)  # +1 to move to the next token after the prompt
#             current_position += len(compound_tokens) - 1  # -1 as the last token is shared with the next compound
#             end_positions.append(current_position)

#         return tokenized_output, start_positions, end_positions





# def prepare_input(input_text):
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     )
#     return inputs