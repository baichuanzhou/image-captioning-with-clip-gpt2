from config import LinearMappingConfig
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, AutoModel,
    CLIPVisionModel, AutoProcessor, BatchEncoding,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple, Dict
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode


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


class LinearMappingProcessor:
    """
    A combination of ImageProcessor and GPT2TokenizerFast
    """

    def __init__(self, config: LinearMappingConfig):
        self.image_processor = AutoProcessor.from_pretrained(config.image_model)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if config.add_image_token:
            self.tokenizer.add_special_tokens({"cls_token": "|<image>|"})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.prefix_length = config.prefix_length

    def __call__(self, texts=None, images=None, return_tensors="pt", **kwargs):
        """
        The processor assumes that images and texts are of the same number
        """

        if texts is None and images is None:
            raise ValueError("You have to specify either text or images. They cannot be none at the same time.")

        if len(texts) == 0:     # empty strings should be None
            texts = None

        if images is not None:
            image_features = self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
            image_features["attention_mask"] = torch.ones(image_features.pixel_values.size(0),
                                                          self.prefix_length).to(dtype=torch.int64)

        if images is None and texts is None:
            texts = self.tokenizer.bos_token

        if texts is not None:
            encoding = self.tokenizer(texts, return_tensors=return_tensors, **kwargs)

        if texts is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values

            encoding["attention_mask"] = torch.cat([
                image_features["attention_mask"],
                encoding["attention_mask"]
            ], dim=1).to(dtype=torch.long)  # create attention mask for images
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
        self.processor = LinearMappingProcessor(config)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        if config.add_image_token:
            self.language_model.resize_token_embeddings(len(self.tokenizer))

        if config.freeze_text_model:
            for module in self.language_model.modules():
                if not isinstance(module, nn.LayerNorm) or config.freeze_ln:
                    for param in module.parameters():
                        param.requires_grad = False
            if config.add_image_token:
                # create a gradient mask for the lm_head weight and bias and hook it
                self.language_model.lm_head.weight.requires_grad = True
                self.weight_gradient_mask = nn.Parameter(torch.zeros_like(self.language_model.lm_head.weight),
                                                         requires_grad=False)
                self.weight_gradient_mask[-1, :] = 1.0
                self.language_model.lm_head.weight.register_hook(lambda grad: grad.mul_(self.weight_gradient_mask))

    def prepare_text_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.transformer.wte(input_ids.to(dtype=torch.int64))

    def prepare_inputs(
            self,
            input_ids: Optional[torch.Tensor],
            pixel_values: Optional[torch.Tensor]
    ) -> Dict:
        """
        Prepare captions and pixel values for training.
        It takes the captions' input ids and turn them into input embeddings
        and turns pixel values into prefix prompts.
        Then it concatenates them into one whole prompt batch.
        """
        if input_ids is not None and pixel_values is not None:

            text_embeddings = self.prepare_text_inputs(input_ids)  # B x T x D
            prefix_prompts = self.image_prefix(pixel_values)  # B x V x D
            inputs_embeddings = torch.cat([prefix_prompts, text_embeddings], dim=1)

            prefix_labels = torch.zeros(prefix_prompts.shape[:2], device=prefix_prompts.device) - 100
            labels = torch.cat([prefix_labels, input_ids], dim=1)  # B x (V + T)

            for label in labels:
                for k, token in enumerate(label):
                    if token == self.tokenizer.eos_token_id:
                        label[k + 1:] = -100
                        break
            return {"hidden_states": inputs_embeddings, "labels": labels.to(dtype=torch.int64)}

        elif pixel_values is not None:
            prefix_prompts = self.image_prefix(pixel_values)  # B x V x D
            prefix_labels = torch.zeros(prefix_prompts.shape[:2], device=prefix_prompts.device) - 100
            return {"hidden_states": prefix_prompts, "labels": prefix_labels.to(dtype=torch.int64)}

        elif input_ids is not None:
            text_embeddings = self.prepare_text_inputs(input_ids)
            labels = input_ids.clone()
            for label in labels:
                for k, token in enumerate(label):
                    if token == self.tokenizer.eos_token_id:
                        label[k + 1:] = -100
                        break
            return {"hidden_states": text_embeddings, "labels": labels.to(dtype=torch.int64)}
        else:
            return {"hidden_states": None, "labels": None}

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if pixel_values is None:
            return self.language_model.generate(
                input_ids=input_ids,
                **kwargs
            )
        first_forward_outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values
        )
        past_key_values = first_forward_outputs.past_key_values
        batch_size = past_key_values[0][0].size()[0]

        first_token_id = first_forward_outputs.logits[:, -1].argmax(-1)
        first_token_id = first_token_id.view(batch_size, -1)

        if kwargs.get("attention_mask", None) is None:
            past_attention_mask_size = (past_key_values[0][0].size(0), past_key_values[0][0].size(-2))
            attention_mask_size = (past_attention_mask_size[0], past_attention_mask_size[-1] + first_token_id.size(-1))
            attention_mask = torch.zeros(attention_mask_size, dtype=torch.int64)
        else:
            attention_mask = kwargs.pop("attention_mask")
            attention_mask = torch.cat([attention_mask, torch.ones(first_token_id.size())], dim=-1)
        generated_token_ids = self.language_model.generate(
            past_key_values=past_key_values,
            input_ids=first_token_id,
            attention_mask=attention_mask,
            **kwargs
        )
        if input_ids is not None:
            generated_token_ids = torch.cat([input_ids, generated_token_ids], dim=-1)
        return generated_token_ids

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_hidden_states: bool = True,
            output_attentions: bool = True,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = True,
            **kwargs
    ) -> Union[GPT2DoubleHeadsModelOutput, Tuple]:
        if (pixel_values is None and input_ids is None) and inputs_embeds is None:
            raise ValueError("You have to specify inputs")
        if inputs_embeds is not None and (pixel_values is not None or input_ids is not None):
            raise ValueError("Either inputs_embeds or (pixel_values and input_ids) should be specified, not both")

        inputs = self.prepare_inputs(input_ids, pixel_values)
        hidden_states = inputs.get('hidden_states', None) if inputs_embeds is None else inputs_embeds
        labels = inputs.get('labels', None) if labels is None else labels

        return self.language_model(
            inputs_embeds=hidden_states,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs
        )
