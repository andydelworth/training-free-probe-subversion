import json
import csv
from tqdm import tqdm
import random
import transformers
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--N', default=100)
parser.add_argument('--target_concept', default='Monkeys')
args = parser.parse_args()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

target_concept = args.target_concept
target_concept_formatted = args.target_concept.replace(' ', '_')
output_dir = f'data/{target_concept_formatted}'
os.makedirs(output_dir, exist_ok=True)
concept_list = open('data/concepts.txt').readlines()
concept_list = [c.strip() for c in concept_list]

system_prompt = "Write something that connects the concept(s) the user specifies. Your writing should be short. \
Randomly select your writing style - usually just regular writing, but sometimes it can also be informal, narrative, \
poem, or others, or just random unstructured text. It MUST be about the concepts listed. Typos, slang or other \
errors are allowed."

'''
data generation v1

positive data:
50% - write a random blurb about <target_concept>
25% - connect these two concepts: <target_concept>, <random_concept>
25% - connect these three concepts: <target_concept>, <random_concept>, <random_concept>

negative data:
50% - write a random blurb about <random_concept>
25% - connect these two concepts: <random_concept>, <random_concept>
25% - connect these three concepts: <random_concept>, <random_concept>, <random_concept>

'''

templates = [
    'concept(s): ' + ', '.join([f'concept {i}: {{}}' for i in range(1, concept_count + 1)])
    for concept_count in range(1, 4)
]

counts = [int(args.N * 0.50 * frac) for frac in (0.50, 0.25, 0.25)]

pos_concepts = []

for i, count in enumerate(counts):
    for _ in range(count):
        concepts = [target_concept] + random.sample(concept_list, i)
        random.shuffle(concepts)
        pos_concepts.append(templates[i].format(*concepts[:i+1]))

templates = [
    'concept(s): ' + ', '.join([f'concept {i}: {{}}' for i in range(1, concept_count + 1)])
    for concept_count in range(1, 4)
]
counts = [int(args.N * 0.50 * frac) for frac in (0.50, 0.25, 0.25)]

neg_concepts = []

for i, count in enumerate(counts):
    for _ in range(count):
        concepts = random.sample(concept_list, i + 1)
        random.shuffle(concepts)
        neg_concepts.append(templates[i].format(*concepts[:i+1]))

pos_messages = [
    [{"role": "system", "content": system_prompt},
    {"role": "user", "content": pos_concept}]
    for pos_concept in pos_concepts
]

neg_messages = [
    [{"role": "system", "content": system_prompt},
    {"role": "user", "content": neg_concept}]
    for neg_concept in neg_concepts
]

pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token


def batched_generate(messages, label, pipe, batch_size=32):
    outputs = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i + batch_size]
        out = pipe(batch, max_new_tokens=128)          # out: list[list[dict]]
        for seqs in out:                               # seqs is a list
            outputs.append([seqs[0]["generated_text"][-1]['content'], label])
    return outputs

pos_outputs = batched_generate(pos_messages, 1, pipeline)
pos_data = [pos_output + [pos_concept] for pos_output, pos_concept in zip(pos_outputs, pos_concepts)]
neg_outputs = batched_generate(neg_messages, 0, pipeline)
neg_data = [neg_output + [neg_concept] for neg_output, neg_concept in zip(neg_outputs, neg_concepts)]


with open(os.path.join(output_dir, 'data.json'), "w") as f:
    json_data = []
    for row in pos_data + neg_data:
        json_data.append({
            'text': row[0],
            'label': row[1],
            'concepts': row[2]
        })

    json.dump(json_data, f)