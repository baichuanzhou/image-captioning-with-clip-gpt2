import gradio as gr
from clip_gpt2 import CLIPGPT2, CLIPGPT2Config, CLIPGPT2Processor
import os
import torch
from transformers import set_seed

os.environ['CURL_CA_BUNDLE'] = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seed(42)
config = CLIPGPT2Config(
    additional_special_tokens_num=1, freeze_text_model=True, text_model='gpt2-large', add_image_token=False,
    text_from_pretrained=False
)
processor = CLIPGPT2Processor(config)
additional_special_tokens = {
    'additional_special_tokens':
        ["|<endofprefix>|"]
}

processor.tokenizer.add_special_tokens(additional_special_tokens)
model = CLIPGPT2(config)
model.load_state_dict(torch.load("pytorch_model-gpt2-large.bin", map_location=device), strict=False)

title = "Generate Image Captions With CLIP And GPT2"


def generate_image_captions(image, text):
    inputs = processor(images=image, texts=text, return_tensors="pt")
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    attention_mask = inputs.get("attention_mask", None)
    prediction = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        # top_p=1.0,
        # top_k=50,
        # repetition_penalty=1.2,
        # temperature=1.0,
        # do_sample=True
    )
    processor.tokenizer.padding_side = 'left'
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    prediction_text = processor.decode(prediction[0], skip_special_tokens=False)
    return prediction_text


article = "This demo is originated from this paper: [original paper](https://arxiv.org/abs/2209.15162)"
description = """
### Expand GPT2's language capabilities to vision with CLIP! 
### Tips:
- Only English is supported.
- When no image is provided, the model degrades to a vanilla GPT2-Large!
- When no description is provided, the model automatically generates a caption for the provided image.
- Try appending 'Answer:' after your question, the model is more likely to give desired outputs this way.
"""
demo = gr.Interface(
    fn=generate_image_captions,
    inputs=[
        gr.Image(),
        gr.Textbox(placeholder="A picture of", lines=3)
    ],
    outputs="text",
    examples=[
        [os.path.join(os.getcwd(), 'two_bear.png'), ""],
        [os.path.join(os.getcwd(), 'three_women.png'), "What is the woman in the middle's dress's color is? Answer:"],
        [os.path.join(os.getcwd(), 'cat_with_food.png'), "Describe the picture:"],
        [os.path.join(os.getcwd(), 'dog_with_frisbee.png'), "What is the color of the frisbee in the photo? Answer:"],
        [os.path.join(os.getcwd(), 'stop_sign.png'), "What does the sign in the picture say? Answer:"]
    ],
    article=article,
    title=title,
    description=description
)

demo.launch(share=True)
