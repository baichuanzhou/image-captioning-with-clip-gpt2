from clip_gpt2 import CLIPGPT2, CLIPGPT2Config, CLIPGPT2Processor
import os
import torch
from transformers import set_seed

os.environ['CURL_CA_BUNDLE'] = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seed(42)
config = CLIPGPT2Config(text_model='gpt2-xl', image_from_pretrained=False, text_from_pretrained=False)
model = CLIPGPT2(config)
model.load_state_dict(torch.load("pytorch_model-gpt2-xl.bin", map_location=device))
processor = CLIPGPT2Processor(config)







