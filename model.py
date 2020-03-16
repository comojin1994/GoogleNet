import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense

def inceptionLayer(x, filters):
    h1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    h2 = Conv2D(filters[1], (1, 1), padding='same')(x)
    h2 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(h2)

    h3 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    h3 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(h3)

    h4 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    h4 = Conv2D(filters[5], (1, 1), padding='same')(h4)

    out = tf.keras.layers.Concatenate()([h1, h2, h3, h4])
    return out

def googleNet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    net = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    net = MaxPool2D()(net)
    net = BatchNormalization()(net)
    net = Dropout(0.3)(net)

    net = Conv2D(64, (1, 1), padding='same')(net)
    net = Conv2D(192, (3, 3), padding='same')(net)
    net = BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = MaxPool2D()(net)

    net = inceptionLayer(net, (64, 96, 128, 16, 32, 32))
    net = inceptionLayer(net, (128, 128, 192, 32, 96, 64))
    net = MaxPool2D()(net)

    out1 = inceptionLayer(net, (192, 96, 208, 16, 48, 64))
    net = inceptionLayer(out1, (160, 112, 224, 24, 64, 64))
    net = inceptionLayer(net, (128, 128, 256, 24, 64, 64))
    out2 = inceptionLayer(net, (112, 144, 288, 32, 64, 64))
    net = inceptionLayer(out2, (256, 160, 320, 32, 128, 128))
    net = MaxPool2D()(net)

    net = inceptionLayer(net, (256, 160 ,320, 32, 128, 128))
    net = inceptionLayer(net, (384, 192, 384, 48, 128, 128))

    out1 = MaxPool2D()(out1)
    out1 = Conv2D(512, (1, 1), padding='same', activation='relu')(out1)
    out1 = Flatten()(out1)
    out1 = Dense(512, activation='relu')(out1)
    out1 = Dropout(0.4)(out1)
    out1 = Dense(10, activation='softmax', name='out1')(out1)

    out2 = MaxPool2D()(out2)
    out2 = Conv2D(528, (1, 1), padding='same', activation='relu')(out2)
    out2 = Flatten()(out2)
    out2 = Dense(528, activation='relu')(out2)
    out2 = Dropout(0.4)(out2)
    out2 = Dense(10, activation='softmax', name='out2')(out2)

    out3 = MaxPool2D()(net)
    out3 = Dropout(0.4)(out3)
    out3 = Flatten()(out3)
    out3 = Dense(10, activation='softmax', name='out3')(out3)

    googlenet = tf.keras.Model(inputs=inputs, outputs=[out1, out2, out3])
    return googlenet

if __name__ == '__main__':
    input_shape = (32, 32, 3)
    model = googleNet(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.summary()