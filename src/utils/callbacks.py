from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback


class WatchModel(Callback):
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        trainer.logger.watch(pl_module, log="all", log_freq=self.log_freq)
