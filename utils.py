import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import ast

def get_constraint_ids(tokenizer):
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

    df['input_text'] = df['molecule_name'].apply(lambda x: ', '.join(sorted(x, key=lambda k: random.random())))
    df['input_text'] = prompt + df['input_text']
    df['labels'] = df['entity_name']
    df['output_text'] = df['entity_name']

    for entry in df['entity_name']:
        if 'Squirrels' in entry:
            print("stop")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
        return {
            'input_ids': tokenized_input.input_ids,
            'attention_mask': tokenized_input.attention_mask,
            'labels': tokenized_output.input_ids
        }

# def prepare_input(input_text):
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     )
#     return inputs