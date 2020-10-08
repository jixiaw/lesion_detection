import tensorflow as tf


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                            padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                            padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(64, 3, 1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool2d = tf.keras.layers.AveragePooling2D(4)
        self.linear = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        print(out.shape)
        out = self.layer1(out, training=training)
        print(out.shape)
        out = self.layer2(out, training=training)
        print(out.shape)
        out = self.layer3(out, training=training)
        print(out.shape)
        out = self.layer4(out, training=training)
        print(out.shape)

        # For classification
        out = self.avg_pool2d(out)
        print(out.shape)
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,14,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])



class BasicBlock3D(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(out_channels, kernel_size=3, strides=strides,
                                            padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv3D(out_channels, kernel_size=3, strides=1,
                                            padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv3D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class Bottleneck3D(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck3D, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv3D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv3D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv3D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class ResNet3D(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv3D(64, 3, 1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling3D(2)

        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool3d = tf.keras.layers.AveragePooling3D(4)
        self.linear = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        # out = self.pool1(out)
        # print(out.shape)
        out = self.layer1(out, training=training)
        # print(out.shape)
        out = self.layer2(out, training=training)
        # print(out.shape)
        out = self.layer3(out, training=training)
        # print(out.shape)
        out = self.layer4(out, training=training)
        # print(out.shape)

        # For classification
        out = self.avg_pool3d(out)
        # print(out.shape)
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out

def ResNet3D18(num_class=2):
    return ResNet3D(BasicBlock3D, [2,2,2,2], num_classes=num_class)

def ResNet3D34():
    return ResNet3D(BasicBlock3D, [3,4,6,3])

def ResNet3D50(num_class=2):
    return ResNet3D(Bottleneck3D, [3,4,6,3], num_classes=num_class)

def ResNet3D101():
    return ResNet3D(Bottleneck3D, [3,4,23,3])

def ResNet3D152():
    return ResNet3D(Bottleneck3D, [3,8,36,3])







if __name__ == '__main__':
    model = ResNet3D50()
    x = tf.random.uniform((32, 32, 32, 32, 1))
    y = model(x)
    print(y, y.shape)






class CenterNet(tf.keras.Model):
    def __init__(self, num_class=1):
        super(CenterNet, self).__init__()
        # self.conv1 = tf.keras.layers.Conv3D
        self.filters = [8, 16, 32, 64]
        self.maxPooling = tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        self.conv1 = tf.keras.layers.Conv3D(num_class, [3, 3, 3], padding='same', activation='sigmoid', use_bias=False,
                              name='cnt_preds')

        self.conv2 = tf.keras.layers.Conv3D(3, [3, 3, 3], padding='same', activation=None, use_bias=False, name='sze_preds')


    def make_down_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(depth, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        ])
    def make_up_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.UpSampling3D(),
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),

        ])
    def make_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(depth, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        ])

    def call(self, x, training=True):
        x1 = self.make_down_layer(self.filters[0])(x, training=training)  # 8 96 128 128
        x2 = self.maxPooling(x1)                                           # 8 48 64 64
        x2 = self.make_down_layer(self.filters[1])(x2, training=training)
        x3 = self.maxPooling(x2)                                           # 16 24 32 32
        x3 = self.make_down_layer(self.filters[2])(x3, training=training)
        x4 = self.maxPooling(x3)                                           # 32 12 16 16
        x4 = self.make_down_layer(self.filters[3])(x4, training=training)
        x5 = self.maxPooling(x4)                                            # 64 6 8 8

        u5 = self.make_up_layer(self.filters[3])(x5, training=training)
        u5 = tf.keras.layers.Concatenate()([x4, u5])
        u5 = self.make_layer(self.filters[3])(u5, training=training)

        u5 = self.make_up_layer(self.filters[2])(u5, training=training)
        u5 = tf.keras.layers.Concatenate()([x3, u5])
        u5 = self.make_layer(self.filters[3])(u5, training=training)

        u5 = self.make_up_layer(self.filters[1])(u5, training=training)
        u5 = tf.keras.layers.Concatenate()([x2, u5])
        u5 = self.make_layer(self.filters[1])(u5, training=training)

        u5 = self.make_up_layer(self.filters[0])(u5, training=training)
        u5 = tf.keras.layers.Concatenate()([x1, u5])
        u5 = self.make_layer(self.filters[0])(u5, training=training)

        cnt_pred = self.conv1(u5)
        sze_pred = self.conv2(u5)

        return [cnt_pred, sze_pred]



class CenterNet3d(tf.keras.Model):
    def __init__(self, num_class=1):
        super(CenterNet3d, self).__init__()
        # self.conv1 = tf.keras.layers.Conv3D
        self.filters = [16, 32, 64, 128]
        self.conv1 = tf.keras.layers.Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.maxPooling = tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        self.out1 = tf.keras.layers.Conv3D(num_class, [3, 3, 3], padding='same', activation='sigmoid', use_bias=False,
                              name='cnt_preds')

        self.out2 = tf.keras.layers.Conv3D(3, [3, 3, 3], padding='same', activation=None, use_bias=False, name='sze_preds')


    def make_down_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(depth, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        ])
    def make_up_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.UpSampling3D(),
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(depth // 2, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),

        ])
    def make_layer(self, depth):
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(depth, (3, 3, 3), strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))
        ])

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.maxPooling(x)
        x1 = self.make_down_layer(self.filters[0])(x, training=training)  # 8 96 128 128
        x2 = self.maxPooling(x1)                                           # 8 48 64 64
        x2 = self.make_down_layer(self.filters[1])(x2, training=training)
        x3 = self.maxPooling(x2)                                           # 16 24 32 32
        x3 = self.make_down_layer(self.filters[2])(x3, training=training)
        x4 = self.maxPooling(x3)                                           # 32 12 16 16
        x4 = self.make_down_layer(self.filters[3])(x4, training=training)
        x5 = self.maxPooling(x4)                                            # 64 6 8 8

        u5 = self.make_up_layer(self.filters[3])(x5, training=training)
        u5 = tf.keras.layers.Concatenate()([x4, u5])
        u5 = self.make_layer(self.filters[3])(u5)

        u5 = self.make_up_layer(self.filters[2])(u5, training=training)
        u5 = tf.keras.layers.Concatenate()([x3, u5])
        u5 = self.make_layer(self.filters[3])(u5)

        u5 = self.make_up_layer(self.filters[1])(u5, training=training)
        u5 = tf.keras.layers.Concatenate()([x2, u5])
        u5 = self.make_layer(self.filters[1])(u5)
        #
        # u5 = self.make_up_layer(self.filters[0])(u5, training=training)
        # u5 = tf.keras.layers.Concatenate()([x1, u5])
        # u5 = self.make_layer(self.filters[0])(u5)

        cnt_pred = self.out1(u5)
        sze_pred = self.out2(u5)

        return [cnt_pred, sze_pred]

