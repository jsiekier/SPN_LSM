import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------
# R(2+1)D Block
# -------------------------
class R2Plus1DBlock(layers.Layer):
    def __init__(self, filters, strides=(1,1,1), use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        # factorize (t,h,w) conv into (1,h,w) + (t,1,1)
        self.spatial_conv = layers.Conv3D(
            filters, kernel_size=(1,3,3), strides=(1,strides[1],strides[2]),
            padding="same", use_bias=not use_bn
        )
        self.temporal_conv = layers.Conv3D(
            filters, kernel_size=(3,1,1), strides=(strides[0],1,1),
            padding="same", use_bias=not use_bn
        )
        if use_bn:
            self.bn1 = layers.BatchNormalization()
            self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.spatial_conv(x)
        if self.use_bn: x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.temporal_conv(x)
        if self.use_bn: x = self.bn2(x, training=training)
        x = self.relu(x)
        return x


# -------------------------
# Residual Block
# -------------------------
class BasicBlock(layers.Layer):
    def __init__(self, filters, strides=(1,1,1), downsample=None, use_bn=True,gauss_std=0.0):
        super().__init__()
        self.conv1 = R2Plus1DBlock(filters, strides=strides, use_bn=use_bn)
        self.conv2 = R2Plus1DBlock(filters, strides=(1,1,1), use_bn=use_bn)
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.gauss_std=False
        if gauss_std:
            self.gauss=tf.keras.layers.GaussianNoise(gauss_std)
            self.gauss_std = True

    def call(self, x, training=False):
        identity = x
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        if self.gauss_std:
            out=self.gauss(out)
        out = self.relu(out)
        return out


# -------------------------
# R(2+1)D-18 Network
# -------------------------
def make_layer(filters, blocks, strides=(1,1,1), use_bn=False,gauss_std=0.0):
    downsample = None
    if strides != (1,1,1):
        downsample = models.Sequential([
            layers.Conv3D(filters, kernel_size=1, strides=strides, use_bias=not use_bn),
            layers.BatchNormalization() if use_bn else layers.Layer()
        ])

    layers_list = [BasicBlock(filters, strides, downsample, use_bn=use_bn,gauss_std=gauss_std)]
    for _ in range(1, blocks):
        layers_list.append(BasicBlock(filters, use_bn=use_bn,gauss_std=gauss_std))
    return models.Sequential(layers_list)


def R2Plus1D_18(num_classes=400, use_bn=False, input_shape=(2,112,112,1),gauss_std=0.0):
    inputs = layers.Input(shape=input_shape)

    # Initial stem
    x = layers.Conv3D(64, kernel_size=(3,7,7), strides=(1,2,2), padding="same", use_bias=not use_bn)(inputs)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=(1,3,3), strides=(1,2,2), padding="same")(x)

    # Residual layers (ResNet-18 style: [2,2,2,2])
    x = make_layer(64, 2, use_bn=use_bn)(x)
    x = make_layer(128, 2, strides=(2,2,2), use_bn=use_bn,gauss_std=gauss_std)(x)

    x = make_layer(256, 2, strides=(2,2,2), use_bn=use_bn,gauss_std=gauss_std)(x)

    x = make_layer(512, 2, strides=(2,2,2), use_bn=use_bn)(x)

    # Global average pooling + classifier
    x = layers.GlobalAveragePooling3D()(x)
    outputs = layers.Dense(num_classes,activation='relu')(x)
    #x = layers.Flatten()(x)
    #x = layers.Dense(512, activation='relu')(x)
    #outputs = layers.Dense(num_classes, activation='relu')(x)

    return models.Model(inputs, outputs, name="R2Plus1D_18"),112//(2*2*2*2),0

#'''
# alter videodecoder
class VideoDecoder(tf.keras.Model):
    def __init__(self, latent_dim=512, output_shape=(2,112,112,1),gauss_std=0.0):
        super().__init__()
        self.T, self.H, self.W, self.C = output_shape

        # Start from a latent vector
        self.fc = layers.Dense(units=(self.T) * (self.H//16) * (self.W//16) * latent_dim)
        self.reshape = layers.Reshape((self.T, self.H//16, self.W//16, 512))
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        # Mirror encoder with Conv3DTranspose
        self.deconv1 = layers.Conv3DTranspose(256, 3, strides=(1,2,2), padding="same", activation=None)
        self.refine1 = layers.Conv3D(256, 3, padding="same", activation="relu")
        if gauss_std:
            self.gauss_std=True
            self.gauss1 = tf.keras.layers.GaussianNoise(gauss_std)
        else:
            self.gauss_std=False
        self.deconv2 = layers.Conv3DTranspose(128, 3, strides=(1,2,2), padding="same", activation=None)
        self.refine2 = layers.Conv3D(128, 3, padding="same", activation="relu")
        if gauss_std:
            self.gauss2 = tf.keras.layers.GaussianNoise(gauss_std)
        self.deconv3 = layers.Conv3DTranspose(64, 3, strides=(1,2,2), padding="same", activation="relu")
        self.refine3 = layers.Conv3D(64, 3, padding="same", activation="relu")

        self.deconv4 = layers.Conv3DTranspose(64, 3, strides=(1,2,2), padding="same", activation="relu")
        self.refine4 = layers.Conv3D(32, 3, padding="same", activation="relu")

        # Final layer → output frames
        #self.out_conv = layers.Conv3DTranspose(self.C, 3, strides=(1,1,1), padding="same", activation="sigmoid")
        # Output: erst linear/projektiv, danach kleine Glättungs-Conv mit Sigmoid
        self.out_conv = layers.Conv3DTranspose(
            self.C, kernel_size=(1, 3, 3), strides=(1, 1, 1),
            padding="same", activation=None
        )
        self.post_conv = layers.Conv3D(
            self.C, kernel_size=3, padding="same", activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )


    def call(self, z):
        # z is latent vector [B, latent_dim]
        x = self.fc(z)
        x = self.reshape(x)  # (B, T, H/32, W/32, 512)

        x = self.deconv1(x)  # (B, T, H/16, W/16, 256)
        x = self.refine1(x)
        if self.gauss_std:
            x=self.gauss2(x)
        x=self.relu1(x)
        x = self.deconv2(x)  # (B, T, H/8,  W/8,  128)
        x = self.refine2(x)
        if self.gauss_std:
            x=self.gauss1(x)
        x = self.relu2(x)
        x = self.deconv3(x)  # (B, T, H/4,  W/4,   64)
        x = self.refine3(x)
        x = self.deconv4(x)  # (B, T, H/2,  W/2,   32)
        x = self.refine4(x)

        #out = self.out_conv(x)  # (B, T, H, W, 1)
        x = self.out_conv(x)
        out = self.post_conv(x)
        return out
#'''
'''
class VideoDecoder(tf.keras.Model):
    def __init__(self, latent_dim=512, output_shape=(2,112,112,1),gauss_std=0.0):
        super().__init__()
        self.T, self.H, self.W, self.C = output_shape

        # Start from a latent vector
        self.fc = layers.Dense(units=(self.T) * (self.H//16) * (self.W//16) * latent_dim)
        self.reshape = layers.Reshape((self.T, self.H//16, self.W//16, 512))
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        # Mirror encoder with Conv3DTranspose
        #self.deconv1 = layers.Conv3DTranspose(256, 3, strides=(1,2,2), padding="same", activation=None)
        self.deconv1 = layers.Conv3DTranspose(256, kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same", activation=None)
        self.refine1 = layers.Conv3D(256, 3, padding="same", activation="relu")
        if gauss_std:
            self.gauss_std=True
            self.gauss1 = tf.keras.layers.GaussianNoise(gauss_std)
        else:
            self.gauss_std=False
        #self.deconv2 = layers.Conv3DTranspose(128, 3, strides=(1,2,2), padding="same", activation=None)
        self.deconv2 = layers.Conv3DTranspose(128,  kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same", activation=None)
        self.refine2 = layers.Conv3D(128, 3, padding="same", activation="relu")
        if gauss_std:
            self.gauss2 = tf.keras.layers.GaussianNoise(gauss_std)
        #self.deconv3 = layers.Conv3DTranspose(64, 3, strides=(1,2,2), padding="same", activation="relu")# enc3
        #self.deconv3 = layers.Conv3DTranspose(64,  kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same", activation="relu") #enc4
        self.deconv3 = layers.Conv3DTranspose(128, kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same",
                                              activation="relu") # enc 5
        self.refine3 = layers.Conv3D(128, 3, padding="same", activation="relu")

        #self.deconv4 = layers.Conv3DTranspose(32, 3, strides=(1,2,2), padding="same", activation="relu")
        #self.deconv4 = layers.Conv3DTranspose(32,  kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same", activation="relu")
        self.deconv4 = layers.Conv3DTranspose(64, kernel_size=(1, 4, 4), strides=(1, 2, 2), padding="same",
                                              activation="relu")
        self.refine4 = layers.Conv3D(64, 3, padding="same", activation="relu")

        # Final layer → output frames
        #self.out_conv = layers.Conv3DTranspose(self.C, 3, strides=(1,1,1), padding="same", activation="sigmoid")

        # Output: erst linear/projektiv, danach kleine Glättungs-Conv mit Sigmoid
        self.out_conv = layers.Conv3DTranspose(
            self.C, kernel_size=(1, 3, 3), strides=(1, 1, 1),
            padding="same", activation=None
        )
        self.post_conv = layers.Conv3D(
            self.C, kernel_size=3, padding="same", activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )

    def call(self, z):
        # z is latent vector [B, latent_dim]
        x = self.fc(z)
        x = self.reshape(x)  # (B, T, H/32, W/32, 512)

        x = self.deconv1(x)  # (B, T, H/16, W/16, 256)
        x = self.refine1(x)
        if self.gauss_std:
            x=self.gauss2(x)
        x=self.relu1(x)
        x = self.deconv2(x)  # (B, T, H/8,  W/8,  128)
        x = self.refine2(x)
        if self.gauss_std:
            x=self.gauss1(x)
        x = self.relu2(x)
        x = self.deconv3(x)  # (B, T, H/4,  W/4,   64)
        x = self.refine3(x)
        x = self.deconv4(x)  # (B, T, H/2,  W/2,   32)
        x = self.refine4(x)

        #out = self.out_conv(x)  # (B, T, H, W, 1)
        x = self.out_conv(x)
        out = self.post_conv(x)
        return out
'''
'''
# delete checkerboad artefacts
class VideoDecoder(tf.keras.Model):
    def __init__(self, latent_dim=512, output_shape=(2,112,112,1), gauss_std=0.0):
        super().__init__()
        self.T, self.H, self.W, self.C = output_shape

        self.fc = layers.Dense(units=self.T * (self.H // 16) * (self.W // 16) * latent_dim)
        self.reshape = layers.Reshape((self.T, self.H // 16, self.W // 16, 512))

        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()

        self.gauss_std = bool(gauss_std)
        if self.gauss_std:
            self.gauss1 = tf.keras.layers.GaussianNoise(gauss_std)
            self.gauss2 = tf.keras.layers.GaussianNoise(gauss_std)

        # Kernel (1,4,4) passt besser zu stride (1,2,2) und reduziert Checkerboards
        self.deconv1 = layers.Conv3DTranspose(
            256, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation=None
        )
        self.refine1 = layers.Conv3D(256, 3, padding="same", activation="relu")

        self.deconv2 = layers.Conv3DTranspose(
            128, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation=None
        )
        self.refine2 = layers.Conv3D(128, 3, padding="same", activation="relu")

        self.deconv3 = layers.Conv3DTranspose(
            64, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation="relu"
        )
        self.refine3 = layers.Conv3D(64, 3, padding="same", activation="relu")

        self.deconv4 = layers.Conv3DTranspose(
            32, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation="relu"
        )
        self.refine4 = layers.Conv3D(32, 3, padding="same", activation="relu")

        # Output: erst linear/projektiv, danach kleine Glättungs-Conv mit Sigmoid
        self.out_conv = layers.Conv3DTranspose(
            self.C, kernel_size=(1, 3, 3), strides=(1, 1, 1),
            padding="same", activation=None
        )
        self.post_conv = layers.Conv3D(
            self.C, kernel_size=3, padding="same", activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )

    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape(x)  # (B, T, 7, 7, 512)

        x = self.deconv1(x)  # (B, T, 14, 14, 256)
        x = self.refine1(x)
        if self.gauss_std:
            x = self.gauss2(x, training=training)
        x = self.relu1(x)

        x = self.deconv2(x)  # (B, T, 28, 28, 128)
        x = self.refine2(x)
        if self.gauss_std:
            x = self.gauss1(x, training=training)
        x = self.relu2(x)

        x = self.deconv3(x)  # (B, T, 56, 56, 64)
        x = self.refine3(x)

        x = self.deconv4(x)  # (B, T, 112, 112, 32)
        x = self.refine4(x)

        x = self.out_conv(x)
        out = self.post_conv(x)

        return out


'''
'''
class VideoDecoder(tf.keras.Model):
    #Angepasster Decoder mit 14×14-Start
    def __init__(self, latent_dim=256, output_shape=(2,112,112,1), gauss_std=0.0):
        super().__init__()
        self.T, self.H, self.W, self.C = output_shape

        # Start jetzt bei 14x14 statt 7x7
        self.start_H = self.H // 8
        self.start_W = self.W // 8
        self.start_channels = latent_dim  # hier 256

        self.fc = layers.Dense(
            units=self.T * self.start_H * self.start_W * self.start_channels
        )
        self.reshape = layers.Reshape(
            (self.T, self.start_H, self.start_W, self.start_channels)
        )

        self.gauss_std = bool(gauss_std)
        if self.gauss_std:
            self.gauss1 = tf.keras.layers.GaussianNoise(gauss_std)
            self.gauss2 = tf.keras.layers.GaussianNoise(gauss_std)

        # 14 -> 28
        self.deconv1 = layers.Conv3DTranspose(
            128, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation=None
        )
        self.refine1 = layers.Conv3D(128, 3, padding="same", activation="relu")

        # 28 -> 56
        self.deconv2 = layers.Conv3DTranspose(
            96, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation=None
        )
        self.refine2 = layers.Conv3D(96, 3, padding="same", activation="relu")

        # 56 -> 112
        self.deconv3 = layers.Conv3DTranspose(
            32, kernel_size=(1, 4, 4), strides=(1, 2, 2),
            padding="same", activation=None
        )
        self.refine3 = layers.Conv3D(64, 3, padding="same", activation="relu")

        # zusätzliche lokale Verfeinerung bei voller Auflösung
        self.refine_full = layers.Conv3D(64, 3, padding="same", activation="relu")

        self.out_conv = layers.Conv3D(
            self.C, kernel_size=3, padding="same", activation=None
        )
        self.post_conv = layers.Conv3D(
            self.C, kernel_size=3, padding="same", activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )

    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape(x)  # (B, T, 14, 14, 256)

        x = self.deconv1(x)  # (B, T, 28, 28, 128)
        x = self.refine1(x)
        if self.gauss_std:
            x = self.gauss1(x, training=training)

        x = self.deconv2(x)  # (B, T, 56, 56, 64)
        x = self.refine2(x)
        if self.gauss_std:
            x = self.gauss2(x, training=training)

        x = self.deconv3(x)  # (B, T, 112, 112, 32)
        x = self.refine3(x)

        x = self.refine_full(x)

        x = self.out_conv(x)
        out = self.post_conv(x)

        return out
'''
'''
#Conv3DTranspose durch UpSampling3D + Conv3D ersetzen.
class VideoDecoder(tf.keras.Model):
    def __init__(self, latent_dim=512, output_shape=(2,112,112,1), gauss_std=0.0):
        super().__init__()
        self.T, self.H, self.W, self.C = output_shape

        self.fc = layers.Dense(units=self.T * (self.H // 16) * (self.W // 16) * latent_dim)
        self.reshape = layers.Reshape((self.T, self.H // 16, self.W // 16, 512))

        self.gauss_std = bool(gauss_std)
        if self.gauss_std:
            self.gauss1 = tf.keras.layers.GaussianNoise(gauss_std)
            self.gauss2 = tf.keras.layers.GaussianNoise(gauss_std)

        # UpSampling + Conv statt Conv3DTranspose
        self.up1 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv1 = layers.Conv3D(256, 3, padding="same", activation="relu")
        #self.refine1 = layers.Conv3D(256, 3, padding="same", activation="relu")

        self.up2 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv2 = layers.Conv3D(128, 3, padding="same", activation="relu")
        #self.refine2 = layers.Conv3D(128, 3, padding="same", activation="relu")

        self.up3 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv3 = layers.Conv3D(64, 3, padding="same", activation="relu")
        #self.refine3 = layers.Conv3D(64, 3, padding="same", activation="relu")

        self.up4 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv4 = layers.Conv3D(32, 3, padding="same", activation="relu")
        #self.refine4 = layers.Conv3D(32, 3, padding="same", activation="relu")

        # Finaler Output
        #self.out_conv = layers.Conv3D(self.C, 3, padding="same", activation="sigmoid")
        self.out_conv = layers.Conv3D(
            self.C,
            3,
            padding="same",
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )

    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape(x)  # (B, T, 7, 7, 512)

        x = self.up1(x)      # (B, T, 14, 14, 512)
        x = self.conv1(x)
        #x = self.refine1(x)
        if self.gauss_std:
            x = self.gauss1(x, training=training)

        x = self.up2(x)      # (B, T, 28, 28, 256)
        x = self.conv2(x)
        #x = self.refine2(x)
        if self.gauss_std:
            x = self.gauss2(x, training=training)

        x = self.up3(x)      # (B, T, 56, 56, 128)
        x = self.conv3(x)
        #x = self.refine3(x)

        x = self.up4(x)      # (B, T, 112, 112, 64)
        x = self.conv4(x)
        #x = self.refine4(x)

        out = self.out_conv(x)  # (B, T, 112, 112, C)
        return out
    
'''

