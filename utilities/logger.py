def get_logger(logs_name, selected="csv"):
    if selected == "wandb":
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        wandb.login(key="6d550e12a1b8f716ebe580082f495c01ed2adf6c")  # 6d550e12a1b8f716ebe580082f495c01ed2adf6c
        logger = WandbLogger(log_model="all")
        wandb.init(project="mono_depth")

    elif selected == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger("training_logs", name=logs_name)

    elif selected == "csv":
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger("training_logs", name=logs_name)
    else:
        logger = None

    return logger