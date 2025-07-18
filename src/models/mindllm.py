import os
import torch
import itertools
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
)
from src.dataset import IGNORE_INDEX
from src.metrics import *
from src.dataset import FMRI_TOKEN
from src.utils import adapt_voxels, safe_get_source
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from accelerate.hooks import AlignDevicesHook
from torchmetrics.text import Perplexity

from hydra.utils import instantiate


def remove_hook_from_module(
    module: torch.nn.Module, recurse=False, hook_cls=AlignDevicesHook
):

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, hook_cls):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


class MindLLM(MetricsRegistered):
    def __init__(
        self,
        encoder,
        model_id,
        tokenizer,
        learning_rate=1e-3,
        peft_config=None,
        scheduler_type=None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.peft_config = peft_config
        self.scheduler_type = scheduler_type

        self.save_hyperparameters()

        self.model_id = model_id
        attn_implementation = (
            "flash_attention_2"
            if (
                os.environ.get("WANDB_MODE") != "disabled" and torch.cuda.is_available()
            )
            else "sdpa"
        )

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        remove_hook_from_module(self.lm, recurse=True)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.lm.generation_config.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        if self.peft_config is None:
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm = get_peft_model(self.lm, self.peft_config)

        self.fmri_encoder = instantiate(encoder)

        self.projector = nn.Linear(
            encoder.token_dim, self.lm.get_input_embeddings().embedding_dim
        )

        self.perplexity = Perplexity(ignore_index=IGNORE_INDEX)

    def on_save_checkpoint(self, checkpoint):
        new_state_dict = {}
        for name, param in checkpoint["state_dict"].items():
            if name.startswith("lm."):
                continue
            new_state_dict[name] = param
        checkpoint["state_dict"] = new_state_dict

        if self.peft_config is not None:
            peft_params = get_peft_model_state_dict(self.lm)
            checkpoint["peft_model"] = peft_params

        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if "peft_model" in checkpoint:
            set_peft_model_state_dict(self.lm, checkpoint.pop("peft_model"))
        super().on_load_checkpoint(checkpoint)

    def forward(self, voxels, chat, image, return_fmri_tokens=False, **kwargs):
        loss = {}
        outputs = {}

        first_chat = kwargs["first_chat"]
        first_inputs_embeds = self.lm.get_input_embeddings()(
            first_chat["input_ids"]
        ).float()
        encoder_outputs = self.fmri_encoder(
            voxels,
            subject=kwargs["subject"],
            prompt_inputs_embeds=first_inputs_embeds,
            prompt_attention_mask=first_chat["attention_mask"],
            padding_side=self.tokenizer.padding_side,
        )
        if isinstance(encoder_outputs, dict):
            fmri_token_embeds = encoder_outputs["fmri_token_embeds"]
        elif isinstance(encoder_outputs, torch.Tensor):
            fmri_token_embeds = encoder_outputs
        else:
            raise

        fmri_token_embeds = self.projector(fmri_token_embeds).to(self.lm.dtype)

        if return_fmri_tokens:
            return fmri_token_embeds

        inputs_embeds, attention_mask, labels, position_ids, fmri_mask = (
            self.prepare_multimodal_inputs(
                chat["input_ids"],
                chat["attention_mask"],
                chat["labels"],
                fmri_token_embeds,
            )
        )

        outputs = self.lm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        loss["token_loss"] = outputs.loss
        outputs["logits"] = outputs.logits.detach().clone()
        outputs["labels"] = labels

        if self.training:
            return loss
        return loss, outputs

    @torch.no_grad()
    def generate(self, **batch):
        fmri_token_embeds = self.forward(**batch, return_fmri_tokens=True)
        chat = batch["prompt"]
        inputs_embeds, attention_mask, _, position_ids, _ = (
            self.prepare_multimodal_inputs(
                chat["input_ids"], chat["attention_mask"], None, fmri_token_embeds
            )
        )
        generated_tokens = []
        for i in range(batch["voxels"].size(0)):
            generated_tokens.append(
                self.lm.generate(
                    input_ids=None,
                    inputs_embeds=inputs_embeds[i : i + 1],
                    attention_mask=attention_mask[i : i + 1],
                    max_new_tokens=128,
                )
                .squeeze(0)
                .tolist()
            )
        return generated_tokens

    def training_step(self, batch, batch_idx):
        batch["voxels"] = adapt_voxels(batch["voxels"], True)
        losses = self.forward(**batch)
        loss = sum(losses.values())
        self.log("train/loss", loss, prog_bar=True)

        for key in losses.keys():
            self.log(f"train/{key}", losses[key])

        return loss

    def validation_step(self, batch, batch_idx):
        batch['voxels'] = adapt_voxels(batch['voxels'], True)
        losses, predictions = self.forward(**batch)
        loss = sum(losses.values())
        self.log('val/loss', loss, prog_bar=True)

        self.perplexity(preds=predictions['logits'][:, :-1], target=predictions['labels'][:, 1:])
        self.log('val/ppl', self.perplexity)

        for key in losses.keys():
            self.log(f'val/{key}', losses[key])

        subject = safe_get_source(batch['subject'])
        source = safe_get_source(batch['source'])
        self.log(f'{source}/token_loss', losses['token_loss'])
        self.log(f'{subject}/token_loss', losses['token_loss'])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch["voxels"] = adapt_voxels(batch["voxels"])
        assert batch["voxels"].size(0) == 1, "only batch_size = 1 is supported"
        # generated evalution
        generated_tokens_batch = self.generate(**batch)
        preds = self.tokenizer.batch_decode(
            generated_tokens_batch, skip_special_tokens=True
        )
        targets = batch.get("answers")
        assert isinstance(targets, list)
        assert (
            isinstance(targets[0], str)
            or isinstance(targets[0], list)
            or isinstance(targets[0], tuple)
        )
        source = safe_get_source(batch["source"])
        self.update_metrics(preds, targets, source)

    def on_test_epoch_end(self):
        self.log_metrics()
        self.reset_metrics()

    def on_predict_start(self):
        self.count = 0

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        voxels = adapt_voxels(batch["voxels"])
        assert voxels.size(0) == 1, "only batch_size = 1 is supported"
        # generated evalution
        generated_tokens_batch = self.generate(**batch)
        preds = self.tokenizer.batch_decode(
            generated_tokens_batch, skip_special_tokens=True
        )
        targets = batch["answers"]
        safe_get_source(batch["source"])

        return preds, targets, batch["sample_or_coco_id"]

    def prepare_multimodal_inputs(
        self, input_ids, attention_mask, labels, fmri_token_embeds, signal_token=None
    ):
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        if True:
            if signal_token is None:
                image_token_index = self.tokenizer.convert_tokens_to_ids(FMRI_TOKEN)
            else:
                image_token_index = signal_token
            image_features = fmri_token_embeds
            num_images, num_image_patches, embed_dim = image_features.shape
            batch_size, sequence_length = input_ids.shape
            left_padding = self.tokenizer.padding_side == "left"
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # Compute the maximum embed dimension
            max_embed_dim = (
                num_special_image_tokens.max() * (num_image_patches - 1)
            ) + sequence_length
            batch_indices, non_image_indices = torch.where(
                input_ids != image_token_index
            )

            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            new_token_positions = (
                torch.cumsum(
                    (special_image_token_mask * (num_image_patches - 1) + 1), -1
                )
                - 1
            )
            nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
            if left_padding:
                new_token_positions += nb_image_pad[:, None]  # offset for left padding
            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

            # 3. Create the full embedding, already padded to the maximum position
            final_embedding = torch.zeros(
                batch_size,
                max_embed_dim,
                embed_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            final_attention_mask = torch.zeros(
                batch_size,
                max_embed_dim,
                dtype=attention_mask.dtype,
                device=inputs_embeds.device,
            )
            if labels is not None:
                final_labels = torch.full(
                    (batch_size, max_embed_dim),
                    IGNORE_INDEX,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
            # set the corresponding tensors into their correct target device.
            target_device = inputs_embeds.device
            batch_indices, non_image_indices, text_to_overwrite = (
                batch_indices.to(target_device),
                non_image_indices.to(target_device),
                text_to_overwrite.to(target_device),
            )
            attention_mask = attention_mask.to(target_device)

            # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
            # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
            final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
                batch_indices, non_image_indices
            ]
            final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
                batch_indices, non_image_indices
            ]
            if labels is not None:
                final_labels[batch_indices, text_to_overwrite] = labels[
                    batch_indices, non_image_indices
                ]

            # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim),
                True,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[
                :, None
            ].to(target_device)

            if image_to_overwrite.sum() != image_features.shape[:-1].numel():
                raise ValueError(
                    f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
                )
            final_embedding[image_to_overwrite] = (
                image_features.contiguous().reshape(-1, embed_dim).to(target_device)
            )
            final_attention_mask |= image_to_overwrite
            position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
                (final_attention_mask == 0), 1
            )

            # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
            batch_indices, pad_indices = torch.where(
                input_ids == self.tokenizer.pad_token_id
            )
            indices_to_mask = new_token_positions[batch_indices, pad_indices]

            final_embedding[batch_indices, indices_to_mask] = 0

            if labels is None:
                final_labels = None

            return (
                final_embedding,
                final_attention_mask,
                final_labels,
                position_ids,
                image_to_overwrite,
            )

    def configure_optimizers(self):
        named_params_list = [
            self.fmri_encoder.named_parameters(),
            self.projector.named_parameters(),
        ]
        if self.peft_config is not None:
            named_params_list.append(self.lm.named_parameters())

        itertools.chain(*named_params_list)

        def get_params_iter(*iterables):
            for it in iterables:
                for x in it:
                    yield x[1]

        params_iter = get_params_iter(*named_params_list)
        optimizer = torch.optim.AdamW(params_iter, lr=self.learning_rate)

        if self.scheduler_type is not None:
            if self.scheduler_type == "cosine_annealing_restart":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=8000, T_mult=1
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        if self.scheduler_type is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # called after every training step
                    "frequency": 1,
                },
            }
