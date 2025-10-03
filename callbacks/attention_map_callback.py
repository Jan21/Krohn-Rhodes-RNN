from pytorch_lightning.callbacks import Callback
import os

class AttentionMapLogger(Callback):
    def __init__(self, name, every_n_epochs=5, save_dir="prints/attention_maps"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir
        self.name = name
        os.makedirs(save_dir+'/'+self.name, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # only one process writes files
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # Get the last attention computed during the epoch
        attn = pl_module.get_attention_weights()  # (num_automata, num_automata)

        # Save PNG
        fname = os.path.join(self.save_dir+'/'+self.name, f"attention_epoch_{epoch+1}.png")
        pl_module.model.plot_attention(attn, filename=fname)

        # Log to W&B if available
        logger = trainer.logger
        try:
            import wandb
            if hasattr(logger, "experiment"):
                logger.experiment.log({"attention_map": wandb.Image(fname), "epoch": epoch + 1})
        except Exception:
            pass