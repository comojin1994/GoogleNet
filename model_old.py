import tensorflow as tf

# Inception Layer
class InceptionLayer(tf.keras.Model):
    def __init__(self, filters):
        super(InceptionLayer, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')
        self.conv3x3r = tf.keras.layers.Conv2D(filters[1], (1, 1), padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')
        self.conv5x5r = tf.keras.layers.Conv2D(filters[3], (1, 1), padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
        self.poolr = tf.keras.layers.Conv2D(filters[5], (1, 1), padding='same')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False, mask=None):
        h1 = self.conv1x1(x)

        h2 = self.conv3x3r(x)
        h2 = self.conv3x3(h2)

        h3 = self.conv5x5r(x)
        h3 = self.conv5x5(h3)

        h4 = self.pool(x)
        h4 = self.poolr(h4)

        return self.concat([h1, h2, h3, h4])

# GoogleNet
class GoogleNet(tf.keras.Model):
    def __init__(self):
        super(GoogleNet, self).__init__(name='GoogleNet')
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.conv2r = tf.keras.layers.Conv2D(64, (1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(192, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # relu
        self.pool2 = tf.keras.layers.MaxPool2D()

        self.incep1 = InceptionLayer((64, 96, 128, 16, 32, 32))
        self.incep2 = InceptionLayer((128, 128, 192, 32, 96, 64))
        self.pool3 = tf.keras.layers.MaxPool2D()

        self.incep3 = InceptionLayer((192, 96, 208, 16, 48, 64)) # output1
        self.incep4 = InceptionLayer((160, 112, 224, 24, 64, 64))
        self.incep5 = InceptionLayer((128, 128, 256, 24, 64, 64))
        self.incep6 = InceptionLayer((112, 144, 288, 32, 64, 64)) # output2
        self.incep7 = InceptionLayer((256, 160, 320, 32, 128, 128))
        self.pool4 = tf.keras.layers.MaxPool2D()

        self.incep8 = InceptionLayer((256, 160, 320, 32, 128, 128))
        self.incep9 = InceptionLayer((384, 192, 384, 48, 128, 128))

        self.out3_pool = tf.keras.layers.AveragePooling2D()
        self.out3_dropout = tf.keras.layers.Dropout(0.4)
        self.out3_flatten = tf.keras.layers.Flatten()
        self.out3_dense = tf.keras.layers.Dense(10, activation='softmax')

        ## output1(After incep3)
        self.out1_pool = tf.keras.layers.AveragePooling2D()
        self.out1_conv = tf.keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu')
        self.out1_flatten = tf.keras.layers.Flatten()
        self.out1_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.out1_dropout = tf.keras.layers.Dropout(0.4)
        self.out1_dense2 = tf.keras.layers.Dense(10, activation='softmax')
        ## output2(After incep6)
        self.out2_pool = tf.keras.layers.AveragePooling2D()
        self.out2_conv = tf.keras.layers.Conv2D(528, (1, 1), padding='same', activation='relu')
        self.out2_flatten = tf.keras.layers.Flatten()
        self.out2_dense1 = tf.keras.layers.Dense(528, activation='relu')
        self.out2_dropout = tf.keras.layers.Dropout(0.4)
        self.out2_dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, mask=None):
        inputs = tf.keras.layers.Input((32, 32, 3))

        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        x = self.dropout(x)
        x = self.conv2r(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        x = self.incep1(x)
        x = self.incep2(x)
        x = self.pool3(x)
        out1 = self.incep3(x)
        x = self.incep4(out1)
        x = self.incep5(x)
        out2 = self.incep6(x)
        x = self.incep7(out2)
        x = self.pool4(x)
        x = self.incep8(x)
        x = self.incep9(x)

        out1 = self.out1_pool(out1)
        out1 = self.out1_conv(out1)
        out1 = self.out1_flatten(out1)
        out1 = self.out1_dense1(out1)
        out1 = self.out1_dropout(out1)
        out1 = self.out1_dense2(out1)

        out2 = self.out2_pool(out2)
        out2 = self.out2_conv(out2)
        out2 = self.out2_flatten(out2)
        out2 = self.out2_dense1(out2)
        out2 = self.out2_dropout(out2)
        out2 = self.out2_dense2(out2)

        out3 = self.out3_pool(x)
        out3 = self.out3_dropout(out3)
        out3 = self.out3_flatten(out3)
        out3 = self.out3_dense(out3)

        googlenet = tf.keras.Model(inputs=inputs, outputs=[out1, out2, out3])
        return googlenet

###
if __name__ == '__main__':
    model = GoogleNet()
    model.build((32, 32, 32, 3))
    model.summary()