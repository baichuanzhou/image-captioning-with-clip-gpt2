from datasets import load_from_disk
from clip_gpt2 import CLIPGPT2, CLIPGPT2Config, CLIPGPT2Processor
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import PILToTensor
from transformers import Trainer, TrainingArguments
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image

os.environ["WANDB_DISABLED"] = "true"
IMAGE_COLUMN = 'image'



def main():
    ds = load_from_disk('llava_ds')
    config = CLIPGPT2Config(additional_special_tokens_num=5, freeze_text_model=False)
    processor = CLIPGPT2Processor(config)
    additional_special_tokens = {
        'additional_special_tokens':
            ["<Describe>", "<Question>", "<Answer>", "<Instruction>", "|<endofprefix>|"]
    }
    processor.tokenizer.add_special_tokens(additional_special_tokens)

    def preprocess_complete_ds(examples):
        if config.add_image_token:
            examples['prefix'] = [
                processor.tokenizer.cls_token + i + "|<endofprefix>|" for i in examples['prefix']
            ]
        examples['text'] = [i + j for i, j in zip(examples['prefix'], examples['description'])]
        inputs = processor.tokenizer(
            examples['text'], truncation=True,
            return_attention_mask=True, return_special_tokens_mask=True
        )   # We need to postpone padding to data collator
        examples['input_ids'] = inputs.input_ids
        return examples
    print(preprocess_complete_ds(ds[:5]))
    ds = ds.map(function=preprocess_complete_ds, batched=True, num_proc=4)

    class Transform(torch.nn.Module):
        def __init__(self, image_size, mean, std):
            super().__init__()
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float32),
                Normalize(mean, std),
            )

        def forward(self, x) -> torch.Tensor:
            """`x` should be an instance of `PIL.Image.Image`"""
            with torch.no_grad():
                x = self.transforms(x)
            return x

    image_transformations = Transform(
        config.image_resize,
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    )
    image_transformations = torch.jit.script(image_transformations)

    def transform_images(examples):
        images = [image.convert("RGB") for image in examples[IMAGE_COLUMN]]
        examples["pixel_values"] = [image_transformations(image) for image in images]

        examples["attention_mask"] = torch.cat([
            torch.ones(len(images), config.prefix_length),
            examples["attention_mask"]
        ], dim=1).to(dtype=torch.long)
        return examples
    ds = ds.set_transform(transform_images)

    def collate_fn(batch):
        batch['pixel_values'] = torch.stack([x['pixel_values'] for x in batch])
        inputs = processor.tokenizer.pad(
            batch['input_ids'], paddind=True, return_tensors='pt', return_attention_mask=True
        )
        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        batch['lables'] = inputs.input_ids.clone()



if __name__ == '__main__':
    main()
