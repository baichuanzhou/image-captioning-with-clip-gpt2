from datasets import load_dataset
from linear_mapping import LinearMapping, LinearMappingProcessor, LinearMappingConfig, Transform
import torch
from torchvision.io import ImageReadMode, read_image
from transformers import Trainer, TrainingArguments
import os
from PIL import Image
os.environ["WANDB_DISABLED"] = "true"

DATA_DIR = os.path.join(os.getcwd(), "coco")
CAPTION_COLUMN = "caption"
IMAGE_COLUMN = "image_path"


def main():
    ds = load_dataset("ydshieh/coco_dataset_script", "2017", DATA_DIR)
    config = LinearMappingConfig()
    processor = LinearMappingProcessor(config)

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'input_ids': torch.tensor([x['input_ids'] for x in batch], dtype=torch.long),
            'attention_mask': torch.stack([x["attention_mask"] for x in batch]),
        }

    def tokenize_fn(examples):
        texts = list(examples[CAPTION_COLUMN])
        if config.add_image_token:
            texts = list(processor.tokenizer.cls_token + text for text in texts)
        inputs = processor.tokenizer(
            texts, padding="max_length", max_length=77,
            return_tensors="pt", truncation=True
        )
        examples["input_ids"] = inputs.input_ids
        examples["attention_mask"] = inputs.attention_mask
        return examples

    image_transformations = Transform(
        config.image_resize,
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    )
    image_transformations = torch.jit.script(image_transformations)

    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[IMAGE_COLUMN]]
        examples["pixel_values"] = [image_transformations(image) for image in images]

        examples["attention_mask"] = torch.cat([
            torch.ones(len(images), config.prefix_length),
            torch.tensor(examples["attention_mask"])
        ], dim=1).to(dtype=torch.long)
        return examples

    def preprocess_fn(examples):

        texts = list(examples[CAPTION_COLUMN])

        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[IMAGE_COLUMN]]
        inputs = processor(
            texts=texts, images=images, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        return inputs

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[IMAGE_COLUMN]:
            try:
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    train_dataset = ds["train"]

    train_dataset = train_dataset.filter(
        function=filter_corrupt_images,
        batched=True
    )
    train_dataset = train_dataset.map(
        function=tokenize_fn,
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col != IMAGE_COLUMN and col != CAPTION_COLUMN],
        load_from_cache_file=True
    )
    train_dataset.set_transform(transform_images)

    training_args = TrainingArguments(
        learning_rate=5e-4,
        lr_scheduler_type='cosine',
        output_dir='clip-gpt2-image-captioner',
        do_train=True,
        logging_steps=50,
        num_train_epochs=5,
        logging_dir='runs',
        remove_unused_columns=False,
        max_grad_norm=1.0,
        per_device_train_batch_size=16,
        save_total_limit=3,
        warmup_steps=500
    )
    model = LinearMapping(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    trainer.train()


if __name__ == '__main__':
    main()
