from config import LinearMappingConfig
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, AutoModel,
    CLIPVisionModel, AutoProcessor, BatchEncoding,

)
import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple


class LinearMappingProcessor:
    """
    A combination of ImageProcessor and GPT2TokenizerFast
    """

    def __init__(self, config: LinearMappingConfig):
        self.image_processor = AutoProcessor.from_pretrained(config.image_model)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"cls_token": "|<image>|"})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.prefix_length = config.prefix_length

    def __call__(self, texts=None, images=None, return_tensors="pt", **kwargs):
        """
        The processor assumes that images and texts are of the same number
        """
        if texts is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if texts is not None:
            encoding = self.tokenizer(texts, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images=images, return_tensors=return_tensors, **kwargs)

        if texts is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values

            encoding["attention_mask"] = torch.cat([
                torch.ones(image_features.pixel_values.size(0), self.prefix_length),
                encoding["attention_mask"]
            ], dim=1).to(dtype=torch.long)   # create attention mask for images
            return encoding

        elif texts is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


class ImagePrefix(nn.Module):
    """
    Converts pixel values to prefix image prompts that are later fed to a LLM
    """

    def __init__(self, config: LinearMappingConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.image_model)
        if "clip" in config.image_model:
            self.encoder = CLIPVisionModel.from_pretrained(config.image_model)

        if config.freeze_image_model:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(config.image_hidden_size, config.text_hidden_size)
        self.ln = nn.LayerNorm(config.text_hidden_size)

    def forward(
            self, pixel_values: torch.Tensor  # B x C x H x W
    ) -> torch.Tensor:
        prefixes = self.encoder(pixel_values).last_hidden_state  # B x N x D
        prefix_prompts = self.linear(prefixes)
        return self.ln(prefix_prompts)


class LinearMapping(nn.Module):

    def __init__(self, config: LinearMappingConfig):
        super().__init__()
        self.image_prefix = ImagePrefix(config)
        self.language_model = GPT2LMHeadModel.from_pretrained(config.text_model)

        if config.freeze_text_model:
            for param in self.language_model.parameters():
                param.requires_grad = False

        self.processor = LinearMappingProcessor(config)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.language_model.resize_token_embeddings(len(self.tokenizer))

    def prepare_text_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.transformer.wte(input_ids)

    def prepare_inputs(
            self,
            input_ids: Optional[torch.Tensor],
            pixel_values: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare captions and pixel values for training.
        It takes the captions' input ids and turn them into input embeddings
        and turns pixel values into prefix prompts.
        Then it concatenates them into one whole prompt batch.
        """
        text_embeddings = self.prepare_text_inputs(input_ids)  # B x T x D
        prefix_prompts = self.image_prefix(pixel_values)  # B x V x D
        inputs_embeddings = torch.cat([prefix_prompts, text_embeddings], dim=1)

        prefix_labels = torch.zeros(prefix_prompts.shape[:2], device=prefix_prompts.device) - 100
        labels = torch.cat([prefix_labels, input_ids], dim=1)   # B x (V + T)
        # We also need to mask out padding token, which is the eos tokens after the first eos token in each sequence
        eos_mask = labels == self.tokenizer.eos_token_id
        # count the number of eos token in each sequence, should be of batch size
        count_eos = eos_mask.sum(dim=1)     # B
        # use this to calculate the position of the first eos token in each sequence and do not mask that
        first_eos_pos = labels.size(1) - count_eos     # B
        # mask out the eos tokens
        labels[eos_mask] = -100
        # the first eos token should not be masked
        labels[range(labels.size(0)), first_eos_pos.long()] = self.tokenizer.eos_token_id
        return inputs_embeddings, labels.to(dtype=torch.long)

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_hidden_states: bool = True,
            output_attentions: bool = True,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = True
    ) -> Union[Tuple]:
        if (pixel_values is None and input_ids is None) and inputs_embeds is None:
            raise ValueError("You have to specify inputs")
        if inputs_embeds is not None and (pixel_values is not None or input_ids is not None):
            raise ValueError("Either inputs_embeds or (pixel_values and input_ids) should be specified, not both")

        hidden_states, input_labels = self.prepare_inputs(input_ids, pixel_values)
        if labels is not None:
            input_labels = labels
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        return self.language_model(
            inputs_embeds=hidden_states,
            labels=input_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
