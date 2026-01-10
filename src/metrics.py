import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy()

def dice_coef(y_true, y_pred, smooth=1e-6):
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2*inter + smooth) / (denom + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))
