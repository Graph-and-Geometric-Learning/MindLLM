from lightning import seed_everything
import os
import hydra
from hydra.utils import instantiate
from peft import LoraConfig, TaskType

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM

import logging

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)

    if cfg.early_stop:
        cfg.trainer.callbacks.append(
            {
                "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                "monitor": "val/token_loss",
                "patience": 10,
            }
        )
    trainer = instantiate(cfg.trainer)

    tokenizer = create_tokenizer(cfg.model_id)
    data_module = instantiate(cfg.data, tokenizer=tokenizer)

    if cfg.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
    else:
        peft_config = None

    model_kwargs = {
        "encoder": cfg.encoder,
        "model_id": cfg.model_id,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
        "learning_rate": cfg.lr,
    }

    if cfg.checkpoint is None:
        model = MindLLM(**model_kwargs)
    else:
        model = MindLLM.load_from_checkpoint(
            cfg.checkpoint, strict=False, **model_kwargs
        )

    model.strict_loading = False
    if cfg.stage == "fit":
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=(
                os.path.join(
                    cfg.output_dir, "mindllm", cfg.resume_id, "checkpoints", "last.ckpt"
                )
                if cfg.resume_id is not None
                else None
            ),
        )
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")
    elif cfg.stage == "validate":
        trainer.validate(model=model, datamodule=data_module)
    elif cfg.stage == "test":
        trainer.test(model=model, datamodule=data_module)
    elif cfg.stage == "predict":
        trainer.predict(
            model=model,
            datamodule=data_module,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
