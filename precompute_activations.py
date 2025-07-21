import argparse
import os
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from freeze_dried_data import WFDD

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Monkeys')
args = parser.parse_args()

data_name = args.data_name
data_folder = f'data/{data_name}'

'''
this function defines where the vector representation of the text comes from

for now, we use the last token's final residual stream representation,
as in https://arxiv.org/pdf/2212.03827
'''
def get_activation(model, tok, text):
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    return out['hidden_states'][-1][:, -1, :]

text = "Connect these two concepts: elephant, squirrel"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

'''
load strings from specified json file
'''
with open(os.path.join(os.path.join(data_folder, 'data.json')), 'r') as f:
    data = json.load(f)

if os.path.exists(os.path.join(data_folder, 'activations.fdd')):
    inp = input('WFDD already exists - delete and rewrite?')
    if inp.lower() in ('y', 'yes'):
        os.remove(os.path.join(data_folder, 'activations.fdd'))
    else:
        quit()

activations = WFDD(os.path.join(data_folder, 'activations.fdd'))

# TODO - this script could be combined with the generate data script
for i, item in enumerate(data):
    text = item['text']
    datum_id = i
    activation = get_activation(model, tok, text)
    activations[datum_id] = {
            'text': text,
            'label': item['label'],
            'activations': activation.float().cpu().numpy(),
            'concepts': item['concepts']
    }

activations.close()
    
    

