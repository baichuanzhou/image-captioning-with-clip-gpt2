import gradio as gr
from linear_mapping import LinearMapping, LinearMappingConfig, LinearMappingProcessor
import os
import torch

os.environ['CURL_CA_BUNDLE'] = ''

config = LinearMappingConfig()
model = LinearMapping(config)
model.load_state_dict(torch.load("pytorch_model.bin"))
processor = LinearMappingProcessor(config)
processor.tokenizer.padding_side = 'left'
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

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
        max_new_tokens=50
    )

    prediction_text = processor.decode(prediction[0], num_beams=3, skip_special_tokens=True)
    return prediction_text


article = "This demo is originated from this paper: [original paper](https://arxiv.org/abs/2209.15162)"
description = """
### Expand GPT2's language capabilities to vision with CLIP!
"""
gr.Interface(
    fn=generate_image_captions,
    inputs=[
        gr.Image(),
        gr.Textbox(placeholder="A picture of", lines=3)
    ],
    outputs="text",
    examples=[
        [os.path.join(os.getcwd(), 'two_cats.jpg'), "A picture of"],
        [os.path.join(os.getcwd(), 'dog_with_hat.png'), ""],
        [os.path.join(os.getcwd(), 'stop_sign.png'), "What does the sign say?"]
    ],
    article=article,
    title=title,
    description=description
).launch()

# with gr.Blocks() as demo:
#     gr.Markdown("# " + title)
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image()
#             input_text = gr.Textbox(placeholder="A picture of", label="Input Text", lines=3)
#             generate_button = gr.Button(value="Generate")
#         output_text = gr.Textbox(label="Output Text", lines=3)
#         generate_button.click(fn=generate_image_captions, inputs=[input_image, input_text], outputs=output_text)
#
# demo.launch()