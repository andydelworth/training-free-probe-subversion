import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda",
)

'''
data generation v1

positive data:
50% - write a random blurb about <target_class>
25% - connect these two concepts: <target_class>, <random_concept>
25% - connect these three concepts: <target_class>, <random_concept>, <random_concept>

negative data:
50% - write a random blurb about <random_concept>
50% - connect these two concepts: <random_concept>, <random_concept>
'''


messages = [
    {"role": "user", "content": "Connect these two concepts: "},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])