import openai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

OPENAI_API_KEY = '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'

def generate_vicuna_answer(prompt):
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5").to('cuda')

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=512, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

def generate_chemLLM_answer(prompt):
    model_name = "AI4Chem/ChemLLM-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",trust_remote_code=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=512, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

def generate_gpt4_answer(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_content = ". ".join(prompt.split('.')[:2]).strip()
    user_content = ". ".join(prompt.split('.')[2:]).strip()

    response = client.chat.completions.create(
        model='gpt-4-0125-preview',
        messages =[
            {'role': 'system', "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )

    return response.choices[0].message.content

def main():

    molecules = 'Gamma-Terpinene, Thymol methyl ether, Methyl Acetate, Ethyl 3-hydroxyhexanoate, Terpinen-4-ol, Nootkatone, Methyl Anthranilate, Perillyl acetate, beta-Sinensal, Octanal, 3-Carene, Thymol, 2-Methyl-1-propanol, alpha-TERPINEOL, 2-(4-methylphenyl)propan-2-ol, Citral, Methyl butyrate, (2E,4E)-deca-2,4-dienal, 1-Penten-3-Ol, Ethyl Heptanoate.' # 20
    
    # TODO automate conversion to SMILES format
    molecules_smiles = 'CCCCCC=CC=CC=O, CCC(C=C)O, CC1=CC=C(C=C1)C(C)(C)O, CC(C)CO, CC1=CCC2C(C1)C2(C)C, CC(=CCCC(=CC=O)C)C, CCCC(CC(=O)OCC)O, CCCCCCC(=O)OCC, CC1=CCC(=CC1)C(C)C, CC(=O)OC, COC(=O)C1=CC=CC=C1N, CCCC(=O)OC, CC1CC(=O)C=C2C1(CC(CC2)C(=C)C)C, CCCCCCCC=O, CC(=C)C1CCC(=CC1)COC(=O)C, CC1=CCC(CC1)(C(C)C)O, CC1=CC(=C(C=C1)C(C)C)O, CC1=CC(=C(C=C1)C(C)C)OC, CC1=CCC(CC1)C(C)(C)O, CC(=CCCC(=C)C=C)CCC=C(C)C=O' # 20

    prompt = f"""
    You are a powerful chemistry expert. You're faced with a chemical puzzle that involves predicting a missing molecule based on a flavor.
    Question: Given the following a flavor and combination of molecules, predict the missing molecule.
    Flavor: Tangerine.
    Molecules: {molecules}
    Answer: Let's think step by step.""" # Alpha-Pinene # Alpha-pinene is a naturally occurring compound, part of the terpene family, which is widely found in the oils of coniferous trees, especially pine trees, and in a variety of other plants including rosemary, basil, dill, parsley, and, notably, citrus fruits like tangerines. Terpenes like alpha-pinene are crucial for the flavor and aroma profiles of many fruits and plants. 

    prompt_smiles = f"""
    You are a powerful chemistry expert. You're faced with a chemical puzzle that involves predicting a missing molecule based on a flavor.
    Question: Given the following a flavor and combination of molecules in SMILES format, predict the missing molecule.
    Flavor: Tangerine.
    Molecules: {molecules_smiles}
    Answer:""" # Alpha-Pinene # Answer: Let's think step by step.

    GPT4_answer = generate_gpt4_answer(prompt)
    vicuna_answer = generate_vicuna_answer(prompt)
    chemLLM_answer = generate_chemLLM_answer(prompt_smiles)

    print("* Prompt: ", prompt)
    print(f"* Generated answer: \n\n** GPT-4: {GPT4_answer}\n\n** LLaMA answer: {vicuna_answer}\n\n**ChemLLM answer: {chemLLM_answer}")

if __name__ == '__main__':
    main()