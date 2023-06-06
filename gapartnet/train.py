from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
import torch
import wandb
torch.set_float32_matmul_precision('medium')
def log_name(config):
    # model
    model_str = ""
    if config["model"]["init_args"]["backbone_type"] == "SparseUNet":
        model_str += "SU"
    else:
        raise NotImplementedError(f"backbone type {config['model']['init_args']['backbone_type']} not implemented")
    
    if config["model"]["init_args"]["use_sem_focal_loss"]:
        model_str += "T"
    else:
        model_str += "F"
    if config["model"]["init_args"]["use_sem_dice_loss"]:
        model_str += "T"
    else:
        model_str += "F"
    
    # data
    data_str = ""
    data_str += "BS" + str(config["data"]["init_args"]["train_batch_size"])
    data_str += "Aug" + \
        ""+str(config["data"]["init_args"]["pos_jitter"]) +\
        "-"+str(config["data"]["init_args"]["color_jitter"]) +\
        "-"+str(config["data"]["init_args"]["flip_prob"]) +\
        "-"+str(config["data"]["init_args"]["rotate_prob"])
    return model_str, data_str

class CustomCLI(LightningCLI):
    def before_fit(self):
        # Use the parsed arguments to create a name
        if self.config["fit"]["model"]["init_args"]["debug"] == False:
            wandb.finish()
            self.config["fit"]["trainer"]["logger"]["init_args"]["mode"] = "online"
        model_str, data_str = log_name(self.config["fit"])
        self.config["fit"]["trainer"]["logger"]["init_args"]["name"] += model_str + "_" + data_str
        self.trainer.logger = WandbLogger(**self.config["fit"]["trainer"]["logger"]["init_args"])


def main():
    _ = CustomCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=233,
        save_config_kwargs={"overwrite": True},
    )
    
if __name__ == "__main__":
    main()
