from dataclasses import dataclass

PREFIX_MAP = {
    "openai/clip-vit-base-patch32": 50
}


@dataclass
class LinearMappingConfig:
    image_model: str = "openai/clip-vit-base-patch32"
    freeze_image_model: bool = True
    text_model: str = "gpt2"
    freeze_text_model: bool = True
    image_hidden_size: int = 768
    text_hidden_size: int = 768
    linear_mapping_type: int = "linear"
    max_seq_length: int = 2048

    def __post_init__(self):
        self.prefix_length = PREFIX_MAP[self.image_model]
