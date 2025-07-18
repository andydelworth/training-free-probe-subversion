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
def get_activation(pipeline, text):
    outputs = pipeline(
        text,
        max_new_tokens=256,
    )
    breakpoint()
    return outputs[0]["generated_text"][-1]

text = "Connect these two concepts: elephant, squirrel"

print(get_activation(pipeline, text))