from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''

device = 'cuda' if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device=device)
inputs = tokenizer("Hello, I'm Iron Man.", return_tensors="pt").to(device=device)
greedy_outputs = model.generate(**inputs, max_new_tokens=80)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_outputs[0], skip_special_tokens=True))

