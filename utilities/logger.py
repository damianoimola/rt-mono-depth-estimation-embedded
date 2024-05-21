# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\Python\univ_proj\computer_vision\computer_vision_project\utilities\logger.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2024-05-15 20:18:37 UTC (1715804317)

def get_logger(options, selected='csv'):
    logs_name = f'd={options.dataset}-lr={options.learning_rate}-e={options.num_epochs}'
    if selected == 'wandb':
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        wandb.login(key='6d550e12a1b8f716ebe580082f495c01ed2adf6c')
        logger = WandbLogger(log_model='all')
        wandb.init(project='mono_depth')
        return logger
    if selected == 'tensorboard':
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger('training_logs', name=logs_name)
        return logger
    if selected == 'csv':
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger('training_logs', name=logs_name)
        return logger
    logger = None
    return logger