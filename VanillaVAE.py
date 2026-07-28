import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

from encoders import R2Plus1D_18, VideoDecoder
from loss_definitions import ssim_loss3D, ms_ssim_loss3D, ssim_loss, ms_ssim_loss, \
    rec_loss_creation_per_sample


def enc_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name,gauss_std):

    convolution = tf.keras.layers.Conv2D(filter_num, filter_size, dilation_rate=dilations, strides=stride, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None, name=name+'conv',trainable=trainable,dtype=dtype)(net)
    #convolution = tf.squeeze(convolution, axis=-2)

    if shortcut:

        shortcut = tf.keras.layers.AveragePooling2D(pool_size=filter_size, strides=stride, padding="same",
                                                  data_format="channels_last",trainable=trainable)(net)
        # reduce shortcut:
        #shortcut = tf.squeeze(shortcut, axis=-1)

        shortcut= tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, old_h,old_h, old_num_filter),trainable=trainable,dtype=dtype)(shortcut)
        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut])
    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization(trainable=trainable,dtype=dtype)(convolution)

    if dropout:
        convolution = tf.keras.layers.Dropout(rate=dropout)(convolution)

    convolution = tf.keras.layers.GaussianNoise(gauss_std)(convolution)
    convolution = activation(convolution)

    return convolution



def make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         activation, encoder_name, dtype, dilations, dropout, end_dim,gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    net = inputs  # /255.0
    img_size = input_shape[0]
    num_filter = [input_shape[2]] + num_filter
    for layer_num, stride in enumerate(strides):
        old_num_filter = num_filter[layer_num]
        if layer_num:
            old_num_filter += 1
        net = enc_layer(net, stride, num_filter[layer_num + 1], batchnorm_integration, shortcut, dilations, filter_size,
                        trainable=True, dtype=dtype,
                        old_h=img_size,
                        old_num_filter=num_filter[layer_num],
                        dropout=dropout,
                        activation=activation,
                        name=encoder_name + '_' + str(layer_num),gauss_std=gauss_std)
        img_size /= stride

    # net = tf.keras.layers.MaxPooling2D(pool_size=(img_size, img_size))(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(end_dim, activation=None)(net)
    net = activation(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model, int(img_size)


def enc_layer3D(net, stride, filter_num, batchnorm_integration, shortcut,
                dilations, filter_size, trainable, dtype, old_h, old_num_filter,
                dropout, activation, name, gauss_std):
    convolution = tf.keras.layers.Conv3D(
        filters=filter_num,
        kernel_size=filter_size,
        dilation_rate=dilations,
        strides=stride,
        padding="same",
        data_format='channels_last',
        use_bias=False,
        activation=None,
        name=name + '_conv',
        trainable=trainable,
        dtype=dtype
    )(net)

    if shortcut:
        shortcut_path = tf.keras.layers.AveragePooling3D(
            pool_size=filter_size,
            strides=stride,
            padding="same",
            data_format="channels_last",
            trainable=trainable
        )(net)

        shortcut_path = tf.keras.layers.Conv3D(
            filters=1,  # reduce channels to 1
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            trainable=trainable,
            dtype=dtype,
            name=name + "_shortcut"
        )(shortcut_path)

        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut_path])

    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization(
            trainable=trainable,
            dtype=dtype
        )(convolution)

    if dropout:
        convolution = tf.keras.layers.Dropout(rate=dropout)(convolution)

    convolution = tf.keras.layers.GaussianNoise(gauss_std)(convolution)
    convolution = activation(convolution)

    return convolution


def make_costume_encoder3D(filter_size, num_layer, input_shape, batchnorm_integration, num_filter,
                           shortcut, strides, activation, encoder_name, dtype, dilations,
                           dropout, end_dim, gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    net = inputs
    img_size = input_shape[1]  # height
    num_filter = [input_shape[-1]] + num_filter

    for layer_num, stride in enumerate(strides):
        old_num_filter = num_filter[layer_num]
        if layer_num:
            old_num_filter += 1
        if layer_num==len(strides)-1:
            stride_tuple=(stride,stride,2)
        else:
            stride_tuple=(stride,stride,1)
        net = enc_layer3D(
            net,
            stride_tuple,
            num_filter[layer_num + 1],
            batchnorm_integration,
            shortcut,
            dilations,
            (filter_size,filter_size,1),
            trainable=True,
            dtype=dtype,
            old_h=img_size,
            old_num_filter=old_num_filter,
            dropout=dropout,
            activation=activation,
            name=encoder_name + '_' + str(layer_num),
            gauss_std=gauss_std
        )
        img_size /= stride  # only scale height dimension here if needed

    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(end_dim, activation=None)(net)
    net = activation(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model, int(img_size)



def make_pretrain_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                          activation, encoder_name, dtype, dilations, dropout, end_dim,path=''):


    base_model = tf.keras.models.load_model(path+'efficientnetb7_saved_model')#(path+'efficientnetb7_saved_model_128.keras')#
    layer_name = 'block4a_activation'  # 'block7a_project_conv'
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # add input layer with gaussian noise:
    net_input = tf.keras.Input(shape=input_shape, dtype=tf.dtypes.float32)
    # x= tf.keras.layers.GaussianNoise(0.05)(net_input)
    x = partial_model(net_input)

    # x = base_model.get_layer(layer_name).output
    # Add new dropout layer with a custom rate
    x = tf.keras.layers.Dropout(dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout
    # Add conv layer to reduce filter size:

    x = tf.keras.layers.Conv2D(end_dim, filter_size, padding='same', activation='relu')(x)#tanh
    if batchnorm_integration:
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    next_dropout=max(0.0,dropout-0.1)
    x = tf.keras.layers.Dropout(next_dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout / 2
    #x = tf.keras.layers.GaussianNoise(0.1)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(8, 8))(x)
    x = tf.keras.layers.Flatten()(x)


    model = tf.keras.models.Model(inputs=net_input, outputs=x)


    return model, 8





def make_encoder(filter_size,num_layer,input_shape,
                 batchnorm_integration,
                 num_filter=[16,32,62,124,248,124,124,32],
                 shortcut=True,
                 strides=[2,1,2,1,2,1,2,1],
                 activation=tf.keras.activations.relu,
                 encoder_name='encoder',
                 dtype=tf.float32,
                 dilations=1,
                 dropout=0.0,
                 end_dim=128,load_pretrain_model=0,path='',gauss_std=0.0,conv_dim=2):
    if load_pretrain_model:
        return make_pretrain_encoder(filter_size,num_layer,input_shape,
                 batchnorm_integration,
                 num_filter=num_filter,
                 shortcut=shortcut,
                 strides=strides,
                 activation=activation,
                 encoder_name=encoder_name,
                 dtype=dtype,
                 dilations=dilations,
                 dropout=dropout,
                 end_dim=end_dim,path=path)#,gauss_std=gauss_std
    else:
        if conv_dim==3:
            '''
            return make_costume_encoder3D(filter_size, num_layer, input_shape,
                                 batchnorm_integration,
                                 num_filter=num_filter,
                                 shortcut=shortcut,
                                 strides=strides,
                                 activation=activation,
                                 encoder_name=encoder_name,
                                 dtype=dtype,
                                 dilations=dilations,
                                 dropout=dropout,
                                 end_dim=end_dim, gauss_std=gauss_std)
            '''
            '''
            return make_vanilla_3D_enc(filter_size, num_layer, input_shape,
                                 batchnorm_integration,
                                 num_filter=num_filter,
                                 shortcut=shortcut,
                                 strides=strides,
                                 activation=activation,
                                 encoder_name=encoder_name,
                                 dtype=dtype,
                                 dilations=dilations,
                                 dropout=dropout,
                                 end_dim=end_dim, gauss_std=gauss_std)
            '''
            # Build model
            return R2Plus1D_18(num_classes=400, use_bn=True, input_shape=input_shape,
                               gauss_std=gauss_std)
            #return R2Plus1D_18(num_classes=400, use_bn=batchnorm_integration, input_shape=input_shape,gauss_std=gauss_std)
            # TODO remove Bn

        else:
            return make_costume_encoder(filter_size,num_layer,input_shape,
                     batchnorm_integration,
                     num_filter=num_filter,
                     shortcut=shortcut,
                     strides=strides,
                     activation=activation,
                     encoder_name=encoder_name,
                     dtype=dtype,
                     dilations=dilations,
                     dropout=dropout,
                     end_dim=end_dim,gauss_std=gauss_std)




def make_mlp_clf(input_shape,activation,batchnorm_integration,dropout=0.0,num_classes=2,layer=1,name='clf'):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    net=inputs

    for i in range(layer):
        if dropout:
            net=tf.keras.layers.Dropout(dropout/8)(net)
        if batchnorm_integration:
            net = tf.keras.layers.BatchNormalization()(net)

        net = tf.keras.layers.Dense(input_shape[0] / (2 ** (i + 1)), name=name + str(i))(net)
        net=activation(net)
    if dropout:
        net = tf.keras.layers.Dropout(dropout/8)(net)
    if batchnorm_integration:
        net = tf.keras.layers.BatchNormalization()(net)

    net = tf.keras.layers.Dense(num_classes, name=name + str(layer))(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model


def dec_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name,gauss_std=0.0):

    net = tf.keras.backend.resize_images(net, height_factor=stride, width_factor=stride,data_format="channels_last",interpolation="nearest")
    convolution = tf.keras.layers.Conv2D(filter_num, filter_size, dilation_rate=dilations, strides=1, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None)(net)
    if shortcut:
        shortcut = tf.keras.layers.Dense(1, activation=None, use_bias=True)(net)
        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut])
    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization()(convolution)
    #net = tf.keras.layers.GaussianNoise(gauss_std)(net)
    convolution = activation(convolution)
    return convolution
def make_decoder(filter_size,num_layer,input_shape,output_shape,
                 batchnorm_integration,
                 num_filter=[1,32,62,124,248,124,124,124],
                 shortcut=True,
                 strides=[2,1,2,1,2,1,2,2],
                 activation=tf.keras.activations.relu,
                 im_size=14,
                 encoder_name='encoder',
                 dtype=tf.float32,
                 dilations=1,
                 dropout=0.0,semi_supervised=False,gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)

    net = tf.keras.layers.Dense(im_size * im_size * 128, activation='relu')(inputs) #* 128
    net = tf.keras.layers.Reshape((im_size, im_size, 128))(net)#, 128
    old_num_filter=128
    for layer_num,stride in enumerate(strides):

        if layer_num < len(strides) - 1:
            curr_shortcut=shortcut
        else:
            curr_shortcut=False
            gauss_std=0.0
            activation=tf.keras.activations.sigmoid
        net=dec_layer(net,stride,num_filter[-(layer_num+1)],batchnorm_integration,curr_shortcut,dilations,filter_size,
                      trainable=True,dtype=dtype,
                      old_h=im_size,
                      old_num_filter=old_num_filter+1,
                      dropout=dropout,
                      activation=activation,
                      name=encoder_name+'_'+str(layer_num),gauss_std=gauss_std)


    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model


def dec_layer3D(net, stride, filter_num, batchnorm_integration, shortcut,
                dilations, filter_size, trainable, dtype, old_h, old_num_filter,
                dropout, activation, name, gauss_std=0.0):

    # Upsample in 3D
    net = tf.keras.layers.UpSampling3D(size=stride, data_format="channels_last")(net)

    # Convolution
    convolution = tf.keras.layers.Conv3D(
        filters=filter_num,
        kernel_size=filter_size,
        dilation_rate=dilations,
        strides=(1, 1, 1),
        padding="same",
        data_format='channels_last',
        use_bias=False
    )(net)

    # Shortcut connection
    if shortcut:
        shortcut_path = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            name=name + '_shortcut'
        )(net)
        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut_path])

    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization()(convolution)

    if dropout:
        convolution = tf.keras.layers.Dropout(rate=dropout)(convolution)

    convolution = tf.keras.layers.GaussianNoise(gauss_std)(convolution)
    convolution = activation(convolution)

    return convolution


def make_decoder3D(filter_size, num_layer, input_shape, output_shape,
                   batchnorm_integration,
                   num_filter=[1, 32, 62, 124, 248, 124, 124, 124],
                   shortcut=True,
                   strides=[(1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 2, 2)],
                   activation=tf.keras.activations.relu,
                   im_size=8,
                   encoder_name='encoder',
                   dtype=tf.float32,
                   dilations=(1, 1, 1),
                   dropout=0.0,
                   semi_supervised=False,
                   gauss_std=0.0):

    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)

    # Project into 3D feature map shape
    total_voxels = im_size * im_size * 1
    net = tf.keras.layers.Dense(total_voxels * 128, activation='relu')(inputs)
    net = tf.keras.layers.Reshape((im_size, im_size, 1, 128))(net)
    old_num_filter = 128

    for layer_num, stride in enumerate(strides):
        curr_shortcut = shortcut if layer_num < len(strides) - 1 else False
        curr_activation = activation if curr_shortcut else tf.keras.activations.sigmoid
        curr_gauss = gauss_std if curr_shortcut else 0.0
        stride_tuple= (stride,stride,1) if layer_num else (stride,stride,2)

        net = dec_layer3D(
            net,
            stride_tuple,#stride,
            num_filter[-(layer_num + 1)],
            batchnorm_integration,
            curr_shortcut,
            dilations,
            (filter_size,filter_size,1),
            trainable=True,
            dtype=dtype,
            old_h=im_size,  # only height used if needed
            old_num_filter=old_num_filter + 1,
            dropout=dropout,
            activation=curr_activation,
            name=encoder_name + '_' + str(layer_num),
            gauss_std=curr_gauss
        )

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model


def build_3d_conv_decoder(output_shape, latent_dim,
                          filter_size, num_layer, input_shape,
                           batchnorm_integration,
                           num_filter=[1, 32, 62, 124, 248, 124, 124, 124],
                           shortcut=True,
                           strides=[(1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 2, 2)],
                           activation=tf.keras.activations.relu,
                           im_size=8,
                           encoder_name='encoder',
                           dtype=tf.float32,
                           dilations=(1, 1, 1),
                           dropout=0.0,
                           semi_supervised=False,
                           gauss_std=0.0,last_num_filt=128):

    # Compute initial shape after flattening (reverse of encoder's Conv3D stack)
    # Assume the encoder used 3 Conv3D layers with stride=2 → downsampled by 2^3 = 8
    init_shape = (output_shape, output_shape, 2, num_filter[-1])
    num_units = init_shape[0] * init_shape[1] * init_shape[2] * init_shape[3]

    inputs = tf.keras.Input(shape=latent_dim)
    x = tf.keras.layers.Dense(num_units, activation=activation)(inputs)
    x = tf.keras.layers.Reshape(init_shape)(x)
    print('Dec shape',init_shape,flush=True)
    old_shape=output_shape
    for lay_num, (num_filt, stride) in enumerate(zip(reversed(num_filter), strides)):
        old_shape=old_shape*stride
        print('Dec shape',old_shape,num_filt,flush=True)
        if shortcut:
            shortcut_in = x

            # First transpose conv
            x = tf.keras.layers.Conv3DTranspose(
                num_filt, kernel_size=filter_size, strides=(stride,stride,1),
                padding='same', use_bias=not batchnorm_integration,
                name=f'Dec_{lay_num}_convT1')(x)
            if batchnorm_integration:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)

            # Second transpose conv
            x = tf.keras.layers.Conv3DTranspose(
                num_filt, kernel_size=filter_size, strides=(1, 1, 1),
                padding='same', use_bias=not batchnorm_integration,
                name=f'Dec_{lay_num}_convT2')(x)
            if batchnorm_integration:
                x = tf.keras.layers.BatchNormalization()(x)

            # Projection shortcut if needed
            if (shortcut_in.shape[-1] != num_filt or stride != 1):
                shortcut_in = tf.keras.layers.Conv3DTranspose(
                    num_filt, kernel_size=1, strides=(stride,stride,1),
                    padding='same', use_bias=not batchnorm_integration,
                    name=f'Dec_{lay_num}_proj')(shortcut_in)
                if batchnorm_integration:
                    shortcut_in = tf.keras.layers.BatchNormalization()(shortcut_in)

            # Residual connection
            x = tf.keras.layers.Add()([x, shortcut_in])
            x = tf.keras.layers.Activation(activation)(x)

        else:
            # Simple transposed conv
            x = tf.keras.layers.Conv3DTranspose(
                num_filt, kernel_size=filter_size, strides=(stride,stride,1),
                padding='same', activation=activation,
                name=f'Dec_{lay_num}')(x)

        # Final reconstruction layer
    outputs = tf.keras.layers.Conv3DTranspose(
        1, kernel_size=3, padding='same', activation='sigmoid',
        name='decoder_out')(x)

    decoder = tf.keras.Model(inputs, outputs, name='3d_conv_decoder')
    return decoder


def make_vanilla_3D_enc(filter_size, num_layer, input_shape, batchnorm_integration, num_filter,
                        shortcut, strides, activation, encoder_name, dtype, dilations,
                        dropout, end_dim, gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.identity(inputs)
    shape_new = input_shape[1]
    if batchnorm_integration:
        use_bias = False
    else:
        use_bias = True

    last_num_filt = 0
    for lay_num, (num_filt, stride) in enumerate(zip(num_filter, strides)):
        last_num_filt = last_num_filt
        shape_new = int(shape_new / (2 ** (stride - 1)))
        print('Enc shape',shape_new,flush=True)
        if shortcut:
            # ----- Residual block -----
            shortcut_in = x

            # First conv layer
            x = tf.keras.layers.Conv3D(
                num_filt, kernel_size=filter_size, strides=stride,
                padding='same', use_bias=use_bias, name=f'Enc_{lay_num}_conv1')(x)
            if batchnorm_integration:
                x = tf.keras.layers.BatchNormalization(name=f'Enc_{lay_num}_bn1')(x)
            x = tf.keras.layers.Activation(activation, name=f'Enc_{lay_num}_act1')(x)

            # Second conv layer
            x = tf.keras.layers.Conv3D(
                num_filt, kernel_size=filter_size, strides=1,
                padding='same', use_bias=use_bias, name=f'Enc_{lay_num}_conv2')(x)
            if batchnorm_integration:
                x = tf.keras.layers.BatchNormalization(name=f'Enc_{lay_num}_bn2')(x)

            # Projection if dimensions mismatch
            if shortcut_in.shape[-1] != num_filt or stride != 1:

                shortcut_in = tf.keras.layers.Conv3D(
                    num_filt, kernel_size=1, strides=stride,
                    padding='same', use_bias=use_bias, name=f'Enc_{lay_num}_proj')(shortcut_in)
                if batchnorm_integration:
                    shortcut_in = tf.keras.layers.BatchNormalization(name=f'Enc_{lay_num}_proj_bn')(shortcut_in)

            # Add residual connection
            x = tf.keras.layers.Add(name=f'Enc_{lay_num}_add')([x, shortcut_in])
            x = tf.keras.layers.Activation(activation, name=f'Enc_{lay_num}_out')(x)

        else:
            # ----- Simple conv encoder block -----
            x = tf.keras.layers.Conv3D(
                num_filt, kernel_size=filter_size, strides=stride,
                padding='same', activation=activation,
                name=f'Enc_{lay_num}')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(end_dim, activation=activation)(x)

    encoder = tf.keras.Model(inputs, x, name='3d_conv_encoder')
    return encoder, shape_new, last_num_filt







dtype=tf.float32


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_lr, warmup_steps):
        """ target_lr: Final learning rate after warmup
            warmup_steps: Number of steps to reach target LR
        """
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """ Linear warm-up from 0 to target_lr over warmup_steps """
        if self.warmup_steps > 0:
            return self.target_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        return self.target_lr  # Default LR if no warm-up needed




class Resnet_VAE(tf.keras.Model):
    def __init__(self,
                 filter_size,
                 num_layer,
                 input_shape,
                 batchnorm_integration,
                 shortcut,
                 activation,
                 num_filter_encoder,
                 strides_encoder,
                 end_dim_enc,
                 num_filter_decoder,
                 strides_decoder,
                 latent_dim,
                 learning_rate,
                 semi_supervised,
                 num_classes,
                 dropout,
                 load_pretrain_model,
                 add_info,
                 loss_weights,
                 use_KLD_anneal,
                 VAE_fine_tune,path='',
                 use_GAN=False,
                 le_warmup=0,
                 gauss_std=0.0,
                 encoder=0,
                 conv_dim=2,
                 rec_loss_name='MSE',
                 pretrain=False,
                 cone_mask_tf=None):
        super(Resnet_VAE, self).__init__()
        #self.dtype=dtype
        self.latent_dim=latent_dim
        self.input_shape_=input_shape
        self.load_pretrain_model=load_pretrain_model
        self.use_add_info=add_info
        self.loss_weights=loss_weights
        self.VAE_fine_tune=VAE_fine_tune
        self.use_KLD_anneal=use_KLD_anneal
        self.num_classes=num_classes
        self.conv_dim=conv_dim
        self.pretrain=pretrain
        self.cone_mask_tf=cone_mask_tf
        print('loss_weights',self.loss_weights,flush=True)

        if activation == 'relu':
            self.activation = tf.keras.layers.ReLU() #tf.keras.layers.LeakyReLU()
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh

        if load_pretrain_model:
            self.encoder, im_size=encoder,8
        else:

            self.encoder,im_size,last_num_filt =make_encoder(filter_size=filter_size,
                                               num_layer=num_layer,
                                               input_shape=input_shape,
                                                batchnorm_integration=batchnorm_integration,
                                                num_filter = num_filter_encoder,
                                                shortcut = shortcut,
                                                strides = strides_encoder,
                                                activation = self.activation,
                                                encoder_name = 'encoder',
                                                dtype = tf.float32,
                                                dilations = 1,
                                                dropout = dropout,
                                                end_dim = end_dim_enc,
                                                load_pretrain_model=load_pretrain_model,path=path,gauss_std=gauss_std,conv_dim=conv_dim)

        self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')


        self.semi_supervised=semi_supervised
        dec_inp_shape = (latent_dim,)
        if semi_supervised:
            cls_input_shape=(latent_dim,)
            if self.use_add_info:
                cls_input_shape = (latent_dim+3,)

            self.classifier=make_mlp_clf(input_shape=cls_input_shape,
                                         activation=self.activation,
                                         batchnorm_integration=batchnorm_integration,
                                         dropout=0.0,
                                         num_classes=num_classes,layer=3)
            if num_classes>1:
                self.clf_loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            else:
                self.clf_loss= tf.keras.losses.MeanSquaredError() #tf.keras.losses.MSE
                #self.clf_loss=mse_per_sample
            if self.VAE_fine_tune==0:
                dec_inp_shape=(latent_dim+num_classes,)
            else:
                dec_inp_shape = (latent_dim,)
        if conv_dim==3:
            '''
            self.decoder = make_decoder3D(filter_size=filter_size,
                                        num_layer=num_layer,
                                        input_shape=dec_inp_shape,
                                        output_shape=input_shape,
                                         batchnorm_integration=batchnorm_integration,
                                         num_filter=num_filter_decoder,
                                         shortcut=shortcut,
                                         strides=strides_decoder,
                                         activation=tf.keras.activations.relu,
                                         im_size=im_size,
                                         encoder_name='decoder',
                                         dtype=tf.float32,
                                         dilations=1,
                                         dropout=dropout,
                                        semi_supervised=semi_supervised,gauss_std=gauss_std)
            
            self.decoder=build_3d_conv_decoder(output_shape=im_size, latent_dim=dec_inp_shape,
                                               filter_size=filter_size,
                                        num_layer=num_layer,
                                        input_shape=dec_inp_shape,
                                         batchnorm_integration=batchnorm_integration,
                                         num_filter=num_filter_decoder,
                                         shortcut=shortcut,
                                         strides=strides_decoder,
                                         activation=tf.keras.activations.relu,
                                         im_size=im_size,
                                         encoder_name='decoder',
                                         dtype=tf.float32,
                                         dilations=1,
                                         dropout=dropout,
                                        semi_supervised=semi_supervised,gauss_std=gauss_std,last_num_filt=last_num_filt
                                               )
            '''
            self.decoder=VideoDecoder(latent_dim=512, output_shape=self.input_shape_,gauss_std=gauss_std)#(2,112,112,1)


        else:
            self.decoder = make_decoder(filter_size=filter_size,
                                        num_layer=num_layer,
                                        input_shape=dec_inp_shape,
                                        output_shape=input_shape,
                                         batchnorm_integration=batchnorm_integration,
                                         num_filter=num_filter_decoder,
                                         shortcut=shortcut,
                                         strides=strides_decoder,
                                         activation=tf.keras.activations.relu,
                                         im_size=im_size,
                                         encoder_name='decoder',
                                         dtype=tf.float32,
                                         dilations=1,
                                         dropout=dropout,
                                        semi_supervised=semi_supervised,gauss_std=gauss_std)

        if le_warmup:
            warmup_steps = 5 * 40  # Warmup for first 5 epochs
            target_lr = learning_rate#1e-3

            lr_schedule = WarmUpLearningRateSchedule(target_lr=target_lr, warmup_steps=warmup_steps)


        else:
            lr_schedule =learning_rate


        self.optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        #self.optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)##
        if conv_dim==3:
            self.loss_fkts = [tf.keras.losses.MAE,
                              tf.keras.losses.MSE,
                              tf.keras.losses.Huber(delta=1.0),
                              ssim_loss3D,
                              ms_ssim_loss3D]
        else:
            self.loss_fkts = [tf.keras.losses.MAE,
                              tf.keras.losses.MSE,
                              tf.keras.losses.Huber(delta=1.0),
                              ssim_loss,
                              ms_ssim_loss]
        #self.rec_loss=tf.keras.losses.MeanSquaredError()
        #self.rec_loss=ms_ssim_loss3D_huber #_mask #ms_ssim_loss3D_mask#rec_loss_creation(rec_loss_name,conv_dim)
        self.rec_loss = rec_loss_creation_per_sample(rec_loss_name, conv_dim)

    def freeze_encoder_decoder(self):
        self.encoder.trainable = False
        self.decoder.trainable = False
        if hasattr(self, "classifier"):
            self.classifier.trainable = True

    def unfreeze_last_encoder_blocks(self, n_blocks=2, unfreeze_decoder=False):
        # erst alles freezen
        self.encoder.trainable = True
        for layer in self.encoder.layers:
            layer.trainable = False

        # letzte Top-Level-Blöcke freigeben
        for layer in self.encoder.layers[-n_blocks:]:
            layer.trainable = True

        self.decoder.trainable = unfreeze_decoder

        if hasattr(self, "classifier"):
            self.classifier.trainable = True

    def unfreeze_all(self):
        self.encoder.trainable = True
        self.decoder.trainable = True
        if hasattr(self, "classifier"):
            self.classifier.trainable = True


    def train(self,train_gen):
        losses=np.zeros((3+ len(self.loss_fkts)))
        num_rounds=0
        for (X,y) in train_gen:
            loss, rec_loss, kld,other_loss=self.train_step(X)
            losses[0]+=loss
            losses[1]+=rec_loss
            losses[2]+=kld
            for i,tmp_loss in enumerate(other_loss):
                losses[3+i]+=tf.reduce_mean(tmp_loss)
            num_rounds+=1
            #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        return losses/num_rounds
    @tf.function
    def execute_net_pretrain_func(self, X, training=True, beta=0):
        normalized_X = tf.cast(X, tf.float32)
        if self.load_pretrain_model:
            normalized_X = tf.expand_dims(normalized_X[:, :, :, 0], axis=-1)
            normalized_X /= 255.0

        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        decoding = self.decoder(cls_input0)
        x_rec = decoding

        # per-sample reconstruction loss
        if self.conv_dim == 3:
            '''
            # huber besser fuer speckle
            delta = 0.1
            error = normalized_X - x_rec
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, delta)
            linear = abs_error - quadratic
            huber = 0.5 * tf.square(quadratic) + delta * linear
            rec_loss = tf.reduce_mean(huber, axis=(1, 2, 3, 4))
            # wichtig:  auf MAE-Skala bringen
            rec_loss = rec_loss / delta
            '''
            #rec_loss = tf.reduce_mean(tf.abs(normalized_X - x_rec), axis=(1, 2, 3, 4)) #self.rec_loss(normalized_X,x_rec)#
            rec_loss = self.rec_loss(normalized_X, x_rec,self.cone_mask_tf)
        else:
            rec_loss = tf.reduce_mean(tf.abs(normalized_X - x_rec), axis=(1, 2, 3))

        kl_loss = -0.5 * tf.reduce_sum(
            1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),
            axis=1
        )

        if self.use_KLD_anneal:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * beta)
        else:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1])

        other_loss = [tf.reduce_mean(loss_fkt(x_rec, normalized_X)) for loss_fkt in self.loss_fkts]
        #tf.print("x min/max", tf.reduce_min(X), tf.reduce_max(X))
        #tf.print("rec min/max", tf.reduce_min(x_rec), tf.reduce_max(x_rec))
        #tf.print("z_mean abs", tf.reduce_mean(tf.abs(cls_input_mean)))

        return (
            tf.reduce_mean(loss),
            tf.reduce_mean(rec_loss),
            tf.reduce_mean(kl_loss),
            other_loss,
            cls_input0,
            x_rec
        )
    def execute_net_pretrain(self, X, training=True, beta=0):
        normalized_X = tf.cast(X, tf.float32)
        if self.load_pretrain_model:
            normalized_X = tf.expand_dims(normalized_X[:, :, :, 0], axis=-1)
            normalized_X /= 255.0

        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        decoding = self.decoder(cls_input0)
        x_rec = decoding

        # per-sample reconstruction loss
        if self.conv_dim == 3:
            '''
            # huber besser fuer speckle
            delta = 0.1
            error = normalized_X - x_rec
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, delta)
            linear = abs_error - quadratic
            huber = 0.5 * tf.square(quadratic) + delta * linear
            rec_loss = tf.reduce_mean(huber, axis=(1, 2, 3, 4))
            # wichtig:  auf MAE-Skala bringen
            rec_loss = rec_loss / delta
            '''
            #rec_loss = tf.reduce_mean(tf.abs(normalized_X - x_rec), axis=(1, 2, 3, 4)) #self.rec_loss(normalized_X,x_rec)#
            rec_loss = self.rec_loss(normalized_X, x_rec,self.cone_mask_tf)
        else:
            rec_loss = tf.reduce_mean(tf.abs(normalized_X - x_rec), axis=(1, 2, 3))

        kl_loss = -0.5 * tf.reduce_sum(
            1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),
            axis=1
        )

        if self.use_KLD_anneal:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * beta)
        else:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1])

        other_loss = [tf.reduce_mean(loss_fkt(x_rec, normalized_X)) for loss_fkt in self.loss_fkts]
        #tf.print("x min/max", tf.reduce_min(X), tf.reduce_max(X))
        #tf.print("rec min/max", tf.reduce_min(x_rec), tf.reduce_max(x_rec))
        #tf.print("z_mean abs", tf.reduce_mean(tf.abs(cls_input_mean)))

        return (
            tf.reduce_mean(loss),
            tf.reduce_mean(rec_loss),
            tf.reduce_mean(kl_loss),
            other_loss,
            cls_input0,
            x_rec
        )

    @tf.function
    def train_step_pretrain(self, X, beta=0):
        with tf.GradientTape() as tape:
            loss, rec_loss, kld, other_loss, _, _ = self.execute_net_pretrain(X, training=True, beta=beta)

        train_vars = self.encoder.trainable_variables + \
                     self.z_mean.trainable_variables + \
                     self.z_log_var.trainable_variables + \
                     self.decoder.trainable_variables

        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss, rec_loss, kld, other_loss

    def train_pretrain(self, train_gen, beta=0):
        losses = np.zeros((3 + len(self.loss_fkts)))
        num_rounds = 0

        for (X, y) in train_gen:
            loss, rec_loss, kld, other_loss = self.train_step_pretrain(X, beta=beta)
            losses[0] += loss.numpy()
            losses[1] += rec_loss.numpy()
            losses[2] += kld.numpy()
            for i, tmp_loss in enumerate(other_loss):
                losses[3 + i] += tf.reduce_mean(tmp_loss).numpy()
            num_rounds += 1

        return losses / num_rounds



    def train_semi(self,train_gen,beta=0):
        losses=np.zeros((4+ len(self.loss_fkts)))
        num_rounds=0
        if self.use_add_info:
            for (X,y,other_data) in train_gen:
                loss, rec_loss, kld,clf_loss,other_loss=self.train_step_semi_label(X,y,other_data,beta=beta)
                losses[0]+=loss.numpy()
                losses[1]+=rec_loss.numpy()
                losses[2]+=kld.numpy()
                losses[3] += clf_loss.numpy()
                for i,tmp_loss in enumerate(other_loss):
                    losses[4+i]+=tf.reduce_mean(tmp_loss).numpy()
                num_rounds+=1
                #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        else:
            for (X,y) in train_gen:
                #TODO random frame flip:

                loss, rec_loss, kld,clf_loss,other_loss=self.train_step_semi_label(X,y,None,beta=beta)
                losses[0]+=loss.numpy()
                losses[1]+=rec_loss.numpy()
                losses[2]+=kld.numpy()
                losses[3] += clf_loss.numpy()
                for i,tmp_loss in enumerate(other_loss):
                    losses[4+i]+=tf.reduce_mean(tmp_loss).numpy()
                num_rounds+=1
                #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        return losses/num_rounds



    #@tf.function
    def train_step(self,X,beta=0):
        with tf.GradientTape() as tape:
            loss,rec_loss,kld,other_loss=self.execute_net(X,training=True)
            #loss = tf.reduce_mean(loss)

        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss,rec_loss,kld,other_loss

    @tf.function
    def train_step_semi_label(self,X,y,add_data,beta=0):
        with tf.GradientTape() as tape:
            loss,rec_loss,kld,clf_loss,other_loss,_,_,_=self.execute_net_xy(X,y,add_data,training=True,beta=beta)
            #loss = tf.reduce_mean(loss)

        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss,rec_loss,kld,clf_loss,other_loss


    def encode_input(self,X,training=True):
        encoding=self.encoder(X)
        #tf.debugging.check_numerics(encoding, "Tensor contained NaN or Inf")

        cls_input_mean = self.z_mean(encoding, training=training)
        cls_input_var = self.z_log_var(encoding, training=training)
        exp0 = tf.exp(cls_input_var * .5)
        eps0 = tf.random.normal(shape=(cls_input_mean.shape[0], cls_input_mean.shape[1]), mean=0, stddev=1,dtype=self.dtype)
        cls_input0 = eps0 * exp0+ cls_input_mean
        return cls_input_mean,cls_input_var,cls_input0
    def enc_dec(self,X,training=True):
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)

        decoding = self.decoder(cls_input0)
        return cls_input_mean,cls_input_var,cls_input0,decoding

    def embedding(self,X, training=False):
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        return cls_input0,cls_input_mean, cls_input_var

    def spn_clf(self,embedding_org, training=False):
        y_pred = self.classifier(embedding_org)
        if self.num_classes==1:
            y_pred=tf.keras.activations.sigmoid(y_pred)
            y_pred = tf.squeeze(y_pred)
        return y_pred,None


    def execute_net(self,X,training=True):
        #tf.debugging.check_numerics(X, "Tensor contained NaN or Inf")
        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X = tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        cls_input_mean,cls_input_var,cls_input0,decoding=self.enc_dec(X,training=training)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoding, labels=normalized_X)
        cros_ent_mean=tf.reduce_mean(cross_ent,axis=(1,2,3))
        kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var), axis=1)
        loss = cros_ent_mean +(kl_loss*0.0001)
        l1_loss = tf.reduce_mean(cros_ent_mean)
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)
        sigmoid_vals=tf.keras.activations.sigmoid(decoding)
        other_loss=[loss_fkt(sigmoid_vals,normalized_X) for loss_fkt in self.loss_fkts]

        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        #logpz = log_normal_pdf(cls_input0, 0., 0.)
        #logqz_x = log_normal_pdf(cls_input0, cls_input_mean, cls_input_var)
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x),-tf.reduce_mean(logpx_z),-tf.reduce_mean(logpz - logqz_x)
        return loss,l1_loss,kl_loss,other_loss

    @tf.function
    def model_execution_X(self,X, add_info_data, train_vae=True,train_clf=False):
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=train_vae)

        if self.use_add_info:
            cls_input=tf.concat([cls_input0,tf.cast(add_info_data,dtype=tf.float32)],axis=-1)
        else:
            cls_input=cls_input0

        y_pred=self.classifier(cls_input)

        return y_pred

    def reconstruct(self,z):
        decoding = self.decoder(z)
        sigmoid_vals = decoding#tf.keras.activations.sigmoid(decoding)
        return sigmoid_vals
    @tf.function
    def execute_net_xy_func(self,X,y,add_info,training=True,beta=0):
        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        #cls_input0 = self.encoder(X) # delete after debugging
        #cls_input_mean,cls_input_var=1,0
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        if self.VAE_fine_tune == 0:
            one_hot_y=tf.one_hot(y,2,dtype=dtype)

            decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        else:
            decoder_input=cls_input0

        if self.use_add_info:
            cls_input=tf.concat([cls_input0,tf.cast(add_info,dtype=tf.float32)],axis=-1)
        else:
            cls_input=cls_input0
        decoding = self.decoder(decoder_input)
        sigmoid_vals = decoding  # tf.keras.activations.sigmoid(decoding)
        if self.conv_dim==3:
            rec_loss=self.rec_loss(normalized_X,sigmoid_vals,self.cone_mask_tf)#tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2,3))
            #MSE = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2,3))
            #l1_loss = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2,3))

        else:

            #rec_loss=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))
            rec_loss = self.rec_loss(normalized_X, sigmoid_vals)
            #MSE = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2))
            #l1_loss = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2))

        kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),axis=1)



        if not self.pretrain:
            y_pred=self.classifier(cls_input)
            if self.num_classes == 1:
                y_pred = tf.keras.activations.sigmoid(y_pred)
            clf_loss = self.clf_loss(y, y_pred)

            if self.use_KLD_anneal:
                loss = (rec_loss * self.loss_weights[0]) + (kl_loss * beta) + (clf_loss *self.loss_weights[2])
            else:
                loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss *self.loss_weights[2])

            clf_loss = tf.reduce_mean(clf_loss)

        else:
            y_pred=0
            clf_loss =0

            if self.use_KLD_anneal:
                loss = rec_loss + (kl_loss * beta)
            else:
                loss = rec_loss + (kl_loss * self.loss_weights[1])


        xentro = tf.reduce_mean(rec_loss)#rec_loss#0 #
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)

        #other_loss=[loss_fkt(sigmoid_vals,normalized_X) for loss_fkt in self.loss_fkts]
        other_loss=[tf.reduce_mean(loss_fkt(sigmoid_vals,normalized_X)) for loss_fkt in self.loss_fkts]
        #tf.print("x min/max", tf.reduce_min(X), tf.reduce_max(X))
        #tf.print("rec min/max", tf.reduce_min(sigmoid_vals), tf.reduce_max(sigmoid_vals))
        #tf.print("z_mean abs", tf.reduce_mean(tf.abs(cls_input_mean)))
        return loss,xentro,kl_loss,clf_loss,other_loss,y_pred,cls_input0,sigmoid_vals
        #return loss, xentro, kl_loss, clf_loss, other_loss, y_pred, encoding, sigmoid_vals

    def execute_net_xy(self,X,y,add_info,training=True,beta=0):
        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        #cls_input0 = self.encoder(X) # delete after debugging
        #cls_input_mean,cls_input_var=1,0
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        if self.VAE_fine_tune == 0:
            one_hot_y=tf.one_hot(y,2,dtype=dtype)

            decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        else:
            decoder_input=cls_input0

        if self.use_add_info:
            cls_input=tf.concat([cls_input0,tf.cast(add_info,dtype=tf.float32)],axis=-1)
        else:
            cls_input=cls_input0
        decoding = self.decoder(decoder_input)
        sigmoid_vals = decoding  # tf.keras.activations.sigmoid(decoding)
        if self.conv_dim==3:
            rec_loss=self.rec_loss(normalized_X,sigmoid_vals,self.cone_mask_tf)#tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2,3))
            #MSE = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2,3))
            #l1_loss = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2,3))

        else:

            #rec_loss=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))
            rec_loss = self.rec_loss(normalized_X, sigmoid_vals)
            #MSE = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2))
            #l1_loss = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2))

        kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),axis=1)



        if not self.pretrain:
            y_pred=self.classifier(cls_input)
            if self.num_classes == 1:
                y_pred = tf.keras.activations.sigmoid(y_pred)
            clf_loss = self.clf_loss(y, y_pred)

            if self.use_KLD_anneal:
                loss = (rec_loss * self.loss_weights[0]) + (kl_loss * beta) + (clf_loss *self.loss_weights[2])
            else:
                loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss *self.loss_weights[2])

            clf_loss = tf.reduce_mean(clf_loss)

        else:
            y_pred=0
            clf_loss =0

            if self.use_KLD_anneal:
                loss = rec_loss + (kl_loss * beta)
            else:
                loss = rec_loss + (kl_loss * self.loss_weights[1])


        xentro = tf.reduce_mean(rec_loss)#rec_loss#0 #
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)

        #other_loss=[loss_fkt(sigmoid_vals,normalized_X) for loss_fkt in self.loss_fkts]
        other_loss=[tf.reduce_mean(loss_fkt(sigmoid_vals,normalized_X)) for loss_fkt in self.loss_fkts]
        #tf.print("x min/max", tf.reduce_min(X), tf.reduce_max(X))
        #tf.print("rec min/max", tf.reduce_min(sigmoid_vals), tf.reduce_max(sigmoid_vals))
        #tf.print("z_mean abs", tf.reduce_mean(tf.abs(cls_input_mean)))
        return loss,xentro,kl_loss,clf_loss,other_loss,y_pred,cls_input0,sigmoid_vals
        #return loss, xentro, kl_loss, clf_loss, other_loss, y_pred, encoding, sigmoid_vals
    def evaluate_(self,dataset,verbose=0):
        losses = np.zeros((4 + len(self.loss_fkts)))
        num_rounds = 0
        predictions = []
        gt = []
        if self.use_add_info:
            for test_x, y,other_info in dataset:
                #print('Input data shape',test_x.shape,flush=True)
                loss, rec_loss, kld, clf, other_losses, pred,_,_ = self.execute_net_xy(test_x, y,other_info,training=False)
                losses[0] += loss
                losses[1] += rec_loss
                losses[2] += kld
                losses[3] += clf
                if not self.pretrain:
                    predictions.extend(pred.numpy().tolist())
                gt.extend(y.numpy().tolist())
                for i, tmp_loss in enumerate(other_losses):
                    losses[4 + i] += tf.reduce_mean(tmp_loss)

                num_rounds += 1
        else:
            for test_x, y in dataset:
                #print('Input data shape', test_x.shape, flush=True)
                loss, rec_loss, kld, clf, other_losses, pred,_,_ = self.execute_net_xy(test_x, y,None, training=False)
                losses[0] += loss
                losses[1] += rec_loss
                losses[2] += kld
                losses[3] += clf
                if not self.pretrain:
                    predictions.extend(pred.numpy().tolist())
                gt.extend(y.numpy().tolist())
                for i, tmp_loss in enumerate(other_losses):
                    losses[4 + i] += tf.reduce_mean(tmp_loss)

                num_rounds += 1
        losses = losses / num_rounds
        val_loss = losses[0]
        if self.pretrain:
            return [0, val_loss]
        else:
            if self.num_classes>1:
                # calculate accuracy:
                softmax_pred = tf.nn.softmax(predictions, axis=-1)
                prediction_exponential = tf.math.exp(predictions)
                arg_max = np.argmax(softmax_pred, axis=-1)
                acc_loss = accuracy_score(gt, arg_max, normalize=True)
                balanced_acc=balanced_accuracy_score(gt, arg_max)
                [prec,rec,f1,_]=precision_recall_fscore_support(gt, arg_max, average=None)
                #print(prec,rec,f1)
                #print(np.sum(np.isinf(prediction_exponential)),prediction_exponential[:10,1])
                pred_exp = np.nan_to_num(prediction_exponential[:, 1], nan=0, posinf=1.0)
                fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)
                auc=metrics.auc(fpr, tpr)
                #return [acc_loss,val_loss,balanced_acc,prec[1],rec[1],f1[1],auc]
                return [acc_loss, val_loss, balanced_acc, prec[0], rec[0], f1[0], auc]
            else:
                gt=np.asarray(gt)
                predictions=np.asarray(predictions)
                mae=np.mean(np.abs(gt-predictions))
                return [mae,val_loss]

    def clf_model(self):
        inputs = tf.keras.Input(shape=self.input_shape_, dtype=dtype)
        encoding= self.encoder(inputs)
        cls_input_mean = self.z_mean(encoding)
        cls_input_var = self.z_log_var(encoding)
        clf_out=tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([cls_input_mean, cls_input_var])


        cnn_embedding = tf.keras.Model(inputs=inputs, outputs=[clf_out,cls_input_mean,cls_input_var])
        return cnn_embedding

    def grad_cam_model(self):
    #    # Get the target layer
        target_layer = self.encoder.get_layer('conv2d')

        input1 = target_layer.input
        input2 = tf.keras.Input(shape=(3,), dtype=dtype)


        conv= target_layer(input1)

        encoding=None



        cls_input_mean = self.z_mean(encoding)
        cls_input_var = self.z_log_var(encoding)
        clf_out = tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([cls_input_mean, cls_input_var])
        cls_input=tf.concat([clf_out,tf.cast(input2,dtype=tf.float32)],axis=-1)
        output = self.classifier(cls_input)

        # Create a model that maps the input image to the activations of the target layer
        grad_model = tf.keras.models.Model(
            [input1,input2], [target_layer.output,output]
        )
        return grad_model





def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon










