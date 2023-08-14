from dataclasses import dataclass

PREFIX_MAP = {
    "openai/clip-vit-base-patch32": 50,
    "openai/clip-vit-large-patch14": 257
}


@dataclass
class LinearMappingConfig:
    image_model: str = "openai/clip-vit-base-patch32"
    freeze_image_model: bool = True
    text_model: str = "gpt2-large"
    freeze_text_model: bool = True
    image_hidden_size: int = 768
    text_hidden_size: int = 1280
    linear_mapping_type: int = "linear"
    image_resize: int = 224
    add_image_token: bool = True
    freeze_ln: bool = False
    image_from_pretrained: bool = True
    text_from_pretrained: bool = True

    def __post_init__(self):
        self.prefix_length = PREFIX_MAP[self.image_model]
