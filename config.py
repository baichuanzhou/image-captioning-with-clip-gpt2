from dataclasses import dataclass
from transformers import GPT2Config, CLIPVisionConfig

PREFIX_MAP = {
    "openai/clip-vit-base-patch32": 50,
    "openai/clip-vit-base-patch16": 197,
    "openai/clip-vit-large-patch14": 257,
    "openai/clip-vit-large-patch14-336": 577
}

TEXT_HIDDEN_SIZE_MAP = {
    "gpt2": 768,
    "gpt2-medium": 1024,
    "gpt2-large": 1280,
    "gpt2-xl": 1600
}

IMAGE_HIDDEN_SIZE_MAP = {
    "openai/clip-vit-base-patch32": 768,
    "openai/clip-vit-base-patch16": 768,
    "openai/clip-vit-large-patch14": 768,
    "openai/clip-vit-large-patch14-336": 768
}


@dataclass
class CLIPGPT2Config:
    image_model: str = "openai/clip-vit-base-patch32"
    freeze_image_model: bool = True
    text_model: str = "gpt2-large"
    freeze_text_model: bool = True
    add_image_token: bool = True
    additional_special_tokens_num: int = 1
    freeze_ln: bool = False
    image_from_pretrained: bool = True
    text_from_pretrained: bool = True

    def __post_init__(self):
        self.prefix_length = PREFIX_MAP[self.image_model]
        self.image_hidden_size = IMAGE_HIDDEN_SIZE_MAP[self.image_model]
        self.text_hidden_size = TEXT_HIDDEN_SIZE_MAP[self.text_model]
        self.image_resize = 224 if "336" not in self.image_model else 336
        self.text_config = GPT2Config.from_pretrained(self.text_model)
        self.image_config = CLIPVisionConfig.from_pretrained(self.image_model)
        self.vocab_size = self.text_config.vocab_size + self.additional_special_tokens_num
