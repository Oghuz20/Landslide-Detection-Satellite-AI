import tensorflow as tf

def conv_block(x, f):
    x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def build_unet(input_shape=(128,128,14)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32); p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);     p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)

    u4 = tf.keras.layers.UpSampling2D()(bn)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return tf.keras.Model(inputs, outputs)
