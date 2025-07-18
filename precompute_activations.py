import argparse
import os
import json
import torch
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Monkeys')
args = parser.parse_args()

data_name = args.data_name

'''
this function defines where the vector representation of the text comes from

for now, we use the last token's final residual stream representation,
as in https://arxiv.org/pdf/2212.03827
'''
def get_activation(model, tok, text):
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    breakpoint()
    return out.hidden_states[-1][:, -1, :]

text = "Connect these two concepts: elephant, squirrel"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

print(get_activation(pipeline, text))