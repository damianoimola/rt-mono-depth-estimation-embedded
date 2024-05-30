from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import OnExceptionCheckpoint
def get_callbacks():
    cb_list = []

    # early_stop_callback = EarlyStopping(monitor="valid_total_loss", min_delta=0.0, patience=10, verbose=False, mode="min")
    # cb_list.append(early_stop_callback)

    oec_callback = OnExceptionCheckpoint(".")
    cb_list.append(oec_callback)

    return cb_list