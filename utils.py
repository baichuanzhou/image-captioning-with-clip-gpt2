import torch
from transformers import CLIPVisionModel, AutoProcessor, AutoModel
from datasets import load_dataset
from PIL import Image
import requests
import os
os.environ['CURL_CA_BUNDLE'] = ''
from linear_mapping import LinearMappingConfig, LinearMappingProcessor, LinearMapping

model = LinearMapping(LinearMappingConfig())
processor = LinearMappingProcessor(LinearMappingConfig())

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "Two cats lying together"
inputs = processor(images=image, texts=text, padding="max_length", max_length=20, return_tensors="pt")
outputs = model(**inputs)






