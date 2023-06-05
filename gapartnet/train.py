from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl

def main():
    _ = LightningCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=233,
        save_config_kwargs={"overwrite": True},
    )
    
if __name__ == "__main__":
    main()
