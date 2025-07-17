import random
import transformers
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--N', default=1000)
parser.add_argument('--target_concept', default='Monkeys')
args = parser.parse_args()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

target_concept = args.target_concept
target_concept_formatted = args.target_concept.replace(' ', '_')
output_dir = f'data/{target_concept_formatted}'
os.makedirs(output_dir, exist_ok=True)
concept_list = open('data/concepts.txt').readlines()
concept_list = [c.strip() for c in concept_list]

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
        concepts = [target_concept] + random.sample(concept_list, i)
        random.shuffle(concepts)
        pos_concepts.append(templates[i].format(*concepts[:i+1]))

pos_messages = [
    [{"role": "system", "content": "Write something that involves all of the concepts listed in the user message"},
    {"role": "user", "content": pos_concept}]
    for pos_concept in pos_concepts
]

neg_messages = [
    [{"role": "system", "content": "Write something that involves all of the concepts listed in the user message"},
    {"role": "user", "content": neg_concept}]
    for neg_concept in neg_concepts
]

outputs = pipeline(
    pos_messages[:5],
    max_new_tokens=512,
)

breakpoint()

print(outputs[0]["generated_text"][-1])