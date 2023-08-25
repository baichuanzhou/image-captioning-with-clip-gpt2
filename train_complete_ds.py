from datasets import load_from_disk
from clip_gpt2 import CLIPGPT2, CLIPGPT2Config, CLIPGPT2Processor
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import PILToTensor, ToTensor
from transformers import Trainer, TrainingArguments
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image
from datasets import concatenate_datasets

os.environ["WANDB_DISABLED"] = "true"
IMAGE_COLUMN = 'image'


def main():
    ds = load_from_disk('multitask_ds')
    config = CLIPGPT2Config(
        additional_special_tokens_num=1, freeze_text_model=True, text_model='gpt2-large', add_image_token=False
    )
    processor = CLIPGPT2Processor(config)
    additional_special_tokens = {
        'additional_special_tokens':
            ["|<endofprefix>|"]
    }

    processor.tokenizer.add_special_tokens(additional_special_tokens)
    # end_of_prefix_position = processor.tokenizer("|<endofprefix>|", add_special_tokens=False).input_ids[0]

    def preprocess_complete_ds(examples):
        if config.add_image_token:
            examples['prefix'] = [
                processor.tokenizer.cls_token + i for i in examples['prefix']
            ]
        if "|<endofprefix>|" in processor.tokenizer.additional_special_tokens:
            examples['prefix'] = [i + "|<endofprefix>|" for i in examples['prefix']]
        if 'task' in examples.keys():
            examples['text'] = [
                i + ": " + j + k for i, j, k in zip(examples['task'], examples['prefix'], examples['description'])
            ]
        else:
            examples['text'] = [
                i + j for i, j in zip(examples['prefix'], examples['description'])
            ]
        # inputs = processor.tokenizer(
        #     examples['text'], padding='max_length', truncation=True,
        #     return_attention_mask=True, max_length=128, return_tensors='pt'
        # )   # We need to postpone padding to data collator
        # examples['input_ids'] = inputs.input_ids
        # examples['attention_mask'] = inputs.attention_mask
        inputs = processor.tokenizer(
            examples['text'], truncation=True
        )  # We need to postpone padding to data collator
        examples['input_ids'] = inputs.input_ids
        return examples

    ds = ds.map(function=preprocess_complete_ds, batched=True, num_proc=16)

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
        images = [ToTensor()(image.convert('RGB')) for image in examples[IMAGE_COLUMN]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        # examples["attention_mask"] = torch.cat([
        #     torch.ones(len(images), config.prefix_length),
        #     torch.tensor(examples["attention_mask"])
        # ], dim=1).to(dtype=torch.long)
        return examples
    ds.set_transform(transform_images)

    def collate_fn(batch):
        collate_batch = {'pixel_values': torch.stack([x['pixel_values'] for x in batch])}
        inputs = processor.tokenizer.pad(
            {"input_ids": [x['input_ids'] for x in batch]},
            padding=True, return_tensors='pt', return_attention_mask=True
        )
        collate_batch['input_ids'] = inputs.input_ids
        batch_size = collate_batch['input_ids'].size(0)
        """
            concatenate the attention mask here
        """
        collate_batch['attention_mask'] = inputs.attention_mask
        collate_batch["attention_mask"] = torch.cat([
            torch.ones(batch_size, config.prefix_length),
            collate_batch["attention_mask"]
        ], dim=1).to(dtype=torch.long)
        # """
        #     construct labels for text
        # """
        # labels = inputs.input_ids.clone()
        #
        # prefix_position = (labels == end_of_prefix_position).nonzero(as_tuple=False)[:, 1]
        # # construct mask for tokens before |<endofprefix>|
        # mask = torch.arange(
        #     1, labels.size(1)
        # ).unsqueeze(0).expand_as(labels[:, 1:]) <= prefix_position.unsqueeze(1)
        # labels[:, 1:][mask] = -100      # Leave out the image prefix
        # for label in labels:
        #     for k, token in enumerate(label):
        #         if token == processor.tokenizer.eos_token_id:
        #             label[k + 1:] = -100
        #             break
        # """
        #     construct labels for image prefix
        # """
        # image_prefix_labels = torch.full((batch_size, config.prefix_length), -100)
        # labels = torch.cat([image_prefix_labels, labels], dim=1).to(dtype=torch.long)
        # collate_batch['labels'] = labels
        return collate_batch

    training_args = TrainingArguments(
        learning_rate=5e-4,
        lr_scheduler_type='cosine_with_restarts',
        output_dir='outputs/clip-gpt2-large-with-multitask-ds',
        do_train=True,
        logging_steps=50,
        num_train_epochs=3,
        logging_dir='runs/clip-gpt2-large-with-multitask-ds',
        remove_unused_columns=False,
        max_grad_norm=1.0,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        save_total_limit=3,
        warmup_steps=250,
        bf16=True
    )
    model = CLIPGPT2(config)
    # model.load_state_dict(torch.load('outputs/clip-gpt2-medium-with-caption-ds/pytorch_model.bin'), strict=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()


if __name__ == '__main__':
    main()
