import tensorflow as tf
from model import build_unet
from metrics import bce_dice_loss, dice_coef, iou_score

def train_model(train_ds, val_ds, epochs=15):
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_score]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return model, history
