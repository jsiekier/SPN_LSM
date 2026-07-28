import time
import random
import sys

from loss_definitions import ssim_loss3D, ms_ssim_loss3D, ssim_loss, ms_ssim_loss, rec_loss_creation
from plot_fkts import plot_correlations, plot_examples_3D, plot_heatmap
from tf_SPN import SPN_clf

print(sys.modules.get('tensorflow'))
import tensorflow as tf # TODO comment that out  .compat.v2
print(tf.__version__,'Hello CNN SPN file')
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn import metrics
from CNN_functions import get_layer_embeddings
from tf2_spn import create_tf_spn, tf_graph_to_spn
from spn.algorithms.MPE import mpe
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
from internal_sig_test import compare_mae,bootstrap_compare_spearman
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(2)


def create_embedding_model(input_dimensions):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=input_dimensions))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    return model



class CNN_SPN(tf.keras.Model):
    def __init__(self,num_classes,input_dimensions,learning_rate,spn=None,cnn=None,get_max=True):
        super(CNN_SPN, self).__init__()
        #self.spn=spn
        self.num_classes=num_classes
        self.get_max=get_max
        # 1. define CNN / MLP
        if cnn!=None:
            self.embedding=cnn
        else:
            self.embedding=create_embedding_model(input_dimensions)
        # 2. add SPN if it is given
        if spn is not None:
            self.spn_training=True
            self.clf =spn
        # 3. else create a classifier

        else:
            self.clf=Dense(1, activation='sigmoid')
            self.clf_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        self.optimizer=tf.keras.optimizers.Adam(learning_rate)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate)

    def model_execution_X_y(self,X,y,other,train_vae=True,train_clf=True):
        embedding=self.embedding(X,training=train_vae)
        #em=embedding.numpy
        if self.spn_training:
            # get max values or whatever:
            if self.get_max:
                embedding = tf.math.reduce_max(embedding, axis=(1, 2))
            if self.use_add_info:
                embedding=tf.concat([embedding,other],axis=-1)
            spn_input=tf.concat([tf.cast(tf.expand_dims(y,axis=-1),dtype=tf.float32),embedding],axis=-1)
            #spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            spn_output=self.clf(spn_input,training=train_clf)
            #loss = -tf.reduce_sum(tf_graph)
            #opt_op = optimizer.minimize(loss)
            loss=tf.reduce_sum(spn_output)
        else:
            # xEntropy
            loss=self.clf_loss(y,embedding)
        return loss

    @tf.function
    def train_step(self,x,y,other,train_vae=True,train_clf=True):
        with tf.GradientTape() as tape:
            loss=self.model_execution_X_y(x,y,other,train_vae=train_vae,train_clf=train_clf)


        #test=self.clf.trainable_variables
        #print(self.clf.trainable_variables)
        train_vars = self.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss

    @tf.function
    def train(self,train_ds,first_loss,train_vae=True,train_clf=True):
        all_losses=0.0
        counter=0.0
        for it_c,train_rec in enumerate(train_ds):
            other=0
            if self.use_add_info:
                (X, y,other) = train_rec
                other=tf.cast(other,tf.float32)
            else:
                (X, y)=train_rec

            loss=self.train_step(X,y,other,train_vae=train_vae,train_clf=train_clf)
            all_losses+=loss.numpy()
            counter+=1
            if not it_c and first_loss:
                print(it_c,'loss',all_losses/counter)
        return all_losses/counter

    def eval_cnn(self,test_X):
        return self.embedding(test_X)
    def get_spn_variables(self):
        pass
def enc_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name):

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

    convolution = activation(convolution)
    return convolution
def make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         activation, encoder_name, dtype, dilations, dropout, end_dim):
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
                        name=encoder_name + '_' + str(layer_num))
        img_size /= stride

    # net = tf.keras.layers.MaxPooling2D(pool_size=(img_size, img_size))(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(end_dim, activation=None)(net)
    net = activation(net)

    net=tf.keras.layers.Dense(1)(net)



    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model
def make_pretrain_discriminator(filter_size, num_layer, input_shape, batchnorm_integration, num_filter,
                          activation, dropout,path=''):


    base_model = tf.keras.models.load_model(path+'efficientnetb7_saved_model')#(path+'efficientnetb7_saved_model_128.keras')#
    layer_name = 'block4a_activation'  # 'block7a_project_conv'
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # add input layer with gaussian noise:
    net_input = tf.keras.Input(shape=input_shape, dtype=tf.dtypes.float32)
    # x= tf.keras.layers.GaussianNoise(0.05)(net_input)
    x = partial_model(net_input)
    # Add new dropout layer with a custom rate
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout
    # Add conv layer to reduce filter size:

    x = tf.keras.layers.Conv2D(64, filter_size, padding='same', activation='relu')(x)#tanh
    if batchnorm_integration:
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    next_dropout=max(0.0,dropout-0.1)
    if next_dropout:
        x = tf.keras.layers.Dropout(next_dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout / 2
    #x = tf.keras.layers.GaussianNoise(0.1)(x)

    x = tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1)(x)


    model = tf.keras.models.Model(inputs=net_input, outputs=x,name='GAN_discriminator')


    return model


def gradient_penalty(critic, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        interpolated_output = critic(interpolated)
    grads = tape.gradient(interpolated_output, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])+ 1e-8)
    penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty

class CNN_SPN_Parts(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 learning_rate,
                 all_spn_x_y_model,
                 all_prior,
                 cnn=None,
                 get_max=True,
                 dtype=tf.dtypes.float32,
                 gauss_embeds=0.01,
                 use_add_info=False,
                 VAE_fine_tune=0,
                 decoder=None,
                 loss_weights=[],
                 load_pretrain_model=1,
                 use_GAN=False,
                 filter_size=3,
                 num_layer=4,
                 input_shape=(128,128,1),
                 batchnorm_integration=1,
                 shortcut=0,
                 activation='relu',
                 num_filter=[64,128,128],
                 strides=[2,2,2,2],
                 end_dim=100,
                 dropout=0.2,
                 dilations=1,
                 discriminator_name='gan_discriminator',
                 clf_mlp=None,
                 bin_borders=None,
                 bin_widths_scaled=None,
                 spn_model_z=None,
                 conv_dim=2,
                 rec_loss_name='MSE'
                 ):
        super(CNN_SPN_Parts, self).__init__()
        self.clf_mlp=clf_mlp
        self.clf_spn=SPN_clf(num_classes,all_spn_x_y_model,all_prior,
                 bin_borders=bin_borders,
                 bin_widths_scaled=bin_widths_scaled,
                 )
        self.spn_model_z = spn_model_z#SPN_z(spn_model_z)
        self.load_pretrain_model=load_pretrain_model
        #self.all_spn_x_y=all_spn_x_y_model
        self.use_add_info=use_add_info
        self.decoder=decoder
        self.VAE_fine_tune=VAE_fine_tune
        self.loss_weights=loss_weights

        self.conv_dim=conv_dim

        self.num_classes=num_classes
        self.get_max=get_max
        # 1. define CNN / MLP
        self.embedding=cnn

        # 2. add SPN if it is given
        #if spn is not None:
        self.spn_training=True

        if num_classes>1:
            self.clf_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        else:
            self.clf_loss=tf.keras.losses.MeanSquaredError()#tf.keras.losses.MSE


        self.gauss_embed=gauss_embeds
        if gauss_embeds:
            self.gauss_layer=tf.keras.layers.GaussianNoise(gauss_embeds)

        self.optimizer=tf.keras.optimizers.Adam(learning_rate)
        self.optimizer_z = tf.keras.optimizers.Adam(learning_rate)
        self.optimizer_mlp= tf.keras.optimizers.Adam(learning_rate)
        #self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        print(',type(self.optimizer),type(self.optimizer_z),type(self.optimizer_mlp)',type(self.optimizer),type(self.optimizer_z),type(self.optimizer_mlp))

        self.use_GAN=use_GAN
        if self.use_GAN:
            if activation == 'relu':
                self.activation = tf.keras.activations.relu
            elif activation == 'tanh':
                self.activation = tf.keras.activations.tanh
            # create discriminator:
            #self.discriminator=make_pretrain_discriminator(3, 1, (128,128,3), 1, 64,'relu', 0,path='')
            self.discriminator=make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         self.activation, discriminator_name, dtype, dilations, dropout, end_dim)

            # create additional loss:
            #self.gan_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            # gan opt:
            self.gan_discr_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
            self.gan_gener_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        if conv_dim==3:
            self.loss_fkts = [tf.keras.losses.MAE,
                              tf.keras.losses.MSE,
                              tf.keras.metrics.RootMeanSquaredError(),
                              tf.keras.losses.Huber(delta=1.0),
                              ssim_loss3D,
                              ms_ssim_loss3D]
        else:
            self.loss_fkts = [tf.keras.losses.MAE,
                              tf.keras.losses.MSE,
                              tf.keras.metrics.RootMeanSquaredError(),
                              tf.keras.losses.Huber(delta=1.0),
                              ssim_loss,
                              ms_ssim_loss]

        self.rec_loss=rec_loss_creation(rec_loss_name,conv_dim)


    def model_execution_X_y(self,X,y):
        [embedding,_,_]=self.embedding(X)
        #em=embedding.numpy
        #if self.spn_training:
            # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))
        if self.gauss_embed:
            embedding=self.gauss_layer(embedding)
        spn_input=tf.concat([tf.cast(tf.expand_dims(y,axis=-1),dtype=tf.float32),embedding],axis=-1)
        #spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)

        #tf_weights = tf.expand_dims(tf.expand_dims(self.prior_weights, axis=0), axis=-1)
        # Stack children along axis 1
        inputs = []
        for sub_spn in self.all_spn_x_y:
            inputs.append(sub_spn(spn_input))

        children_prob = tf.stack(inputs, axis=1)  # children
        # print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(self.prior_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1)
        p_y_x = log_enumerator - p_x

        #spn_output=self.clf(spn_input)
        return p_y_x

    def spn_clf(self,embedding, training=False):
        return self.clf_spn.spn_clf(embedding,training)


    def spn_regression_clf1(self,embedding,training):
        inputs = []
        #y = np.full((embedding.shape[0], 1), 1)
        y = tf.ones((tf.shape(embedding)[0], 1), dtype=tf.float32)
        spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        for sub_spn, sub_priors,bin_width in zip(self.all_spn_x_y,self.prior_weights,self.bin_widths_scaled):
            inputs.append(sub_spn[1](spn_input,training=training)+tf.math.log(bin_width)) #tf.math.log(sub_priors[1])+
        log_enumerator = tf.concat(inputs, axis=1)  # children
        log_p_y_x = tf.reduce_logsumexp(log_enumerator, axis=1, keepdims=False)
        return log_p_y_x


    #'''
    def model_execution_X(self, X,other_data,train_vae=True,train_clf=True):
        [embedding,_,_] = self.embedding(X,training=train_vae)
        # em=embedding.numpy
        # if self.spn_training:
        # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and train_vae:
            embedding=self.gauss_layer(embedding)

        # Stack children along axis 1
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)

        if self. num_classes>1:
            p_y_x,p_x=self.clf_spn.spn_clf(embedding, training=train_clf)
        else:
            # This code is just for debugging:
            #p_y_x, prediction_decision_bound = self.spn_clf(embedding,training)#self.spn_regression_exp(embedding, training)
            p_y_x,prediction_decision_bound=self.clf_spn.spn_regression_exp(embedding, training=train_clf)
            return p_y_x#,prediction_decision_bound
        # spn_output=self.clf(spn_input)
        return p_y_x

    def model_execution_vae(self, X,y_real,other_data,train_vae=True,train_clf=True):
        #normalized_X = tf.cast(X, tf.float32)
        normalized_X=X
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0
        [embedding_,embed_mean,embed_var] = self.embedding(X,training=train_vae)
        #one_hot_y=tf.one_hot(y,2,dtype=dtype)
        #decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        embedding=tf.identity(embedding_)
        reconstruction=self.decoder(embedding)
        kl_loss = -0.5 * tf.reduce_sum(1 + embed_var - tf.square(embed_mean) - tf.exp(embed_var), axis=1)

        if self.get_max:
            embedding = tf.math.reduce_max(embedding_, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and train_vae:
            embedding=self.gauss_layer(embedding)
        #tf_weights = tf.expand_dims(self.prior_weights, axis=0)
        # Stack children along axis 1

        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)

        if self.num_classes>1:
            p_y_x=self.clf_spn.clf(self, embedding,training= train_clf)

        else:
            p_y_x, prediction_decision_bound = self.clf_spn.spn_regression_exp(embedding,training= train_clf)
        #clf_loss = self.clf_loss(y_real, tf.exp(p_y_x))
        y_true = tf.reshape(tf.cast(y_real, tf.float32), [-1])
        y_pred = tf.reshape(tf.exp(p_y_x), [-1])

        clf_loss = tf.reduce_mean(tf.square(y_true - y_pred))


        sigmoid_vals =reconstruction #tf.keras.activations.sigmoid(reconstruction)
        rec_loss=self.rec_loss(sigmoid_vals,normalized_X)#tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X))#,axis=(1,2) #MeanSquaredError


        loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss * self.loss_weights[2])

        rec_loss=tf.reduce_mean(rec_loss)
        kl_loss=tf.reduce_mean(kl_loss)
        loss=tf.reduce_mean(loss)

        return p_y_x,loss,rec_loss,clf_loss,kl_loss,embedding_


    @tf.function(reduce_retracing=True) #TODO delete comments?!
    def model_execution_X_regression(self, X,other_data,training=True):
        [embedding,_,_] = self.embedding(X,training=training)
        # em=embedding.numpy
        # if self.spn_training:
        # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and training:
            embedding=self.gauss_layer(embedding)

        # Stack children along axis 1
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)
        prediction_decision_bound=None
        if self.num_classes > 1:
            p_y_x, p_x = self.clf_spn.spn_clf(embedding, training)
        else:
            p_y_x, prediction_decision_bound = self.clf_spn.spn_regression_exp(embedding, training)
        # spn_output=self.clf(spn_input)
        return p_y_x,prediction_decision_bound


    @tf.function
    def train_step(self,x,y,other_data):
        with tf.GradientTape() as tape:
            spn_out=self.model_execution_X(x,other_data)
            loss = self.clf_loss(y, spn_out)


        #test=self.clf.trainable_variables
        #print(self.clf.trainable_variables)
        train_vars = self.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return [loss]

    def reconstruct(self,embedding):
        reconstruction = self.decoder(embedding)
        #sigmoid_vals = tf.keras.activations.sigmoid(reconstruction)
        return reconstruction

    def vae_rec(self,X):
        [embedding, _, _] = self.embedding(X, training=False)
        return self.reconstruct(embedding)





    @tf.function #TODO redo this this as just for debugging
    def model_execution_vae_eval(self, X,y_real,other_data):
        training=False
        normalized_X = tf.cast(X, tf.float32)
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0
        [embedding_,embed_mean,embed_var] = self.embedding(X,training=training)
        #one_hot_y=tf.one_hot(y,2,dtype=dtype)
        #decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        embedding=tf.identity(embedding_)
        reconstruction=self.decoder(embedding)
        kl_loss = -0.5 * tf.reduce_sum(1 + embed_var - tf.square(embed_mean) - tf.exp(embed_var), axis=1)

        if self.get_max:
            embedding = tf.math.reduce_max(embedding_, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and training:
            embedding=self.gauss_layer(embedding)
        #tf_weights = tf.expand_dims(self.prior_weights, axis=0)
        # Stack children along axis 1

        sigmoid_vals =reconstruction #tf.keras.activations.sigmoid(reconstruction)
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)
        if self.num_classes>1:

            p_y_x = self.clf_spn.clf(embedding, training)

            rec_loss = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2))

        else:
            # TODO
            p_y_x, prediction_decision_bound = self.clf_spn.spn_regression_exp(embedding, training)
            rec_loss = tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals, normalized_X), axis=(1, 2,3))

        p_y_x_mlp=self.clf_mlp(embedding)



        clf_loss=self.clf_loss(y_real, p_y_x)
        #mae = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2))

        #print('rec_loss.shape,kl_loss.shape,clf_loss.shape',rec_loss.shape,kl_loss.shape,clf_loss.shape,flush=True)
        loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss * self.loss_weights[2])

        rec_loss=tf.reduce_mean(rec_loss)
        kl_loss=tf.reduce_mean(kl_loss)
        loss=tf.reduce_mean(loss)
        #mae=tf.reduce_mean(mae)
        other_loss=[tf.reduce_mean(loss_fkt(sigmoid_vals,normalized_X)) for loss_fkt in self.loss_fkts]
        #return p_y_x,loss,rec_loss,clf_loss,kl_loss,mae,embedding_,p_y_x_mlp
        return p_y_x, loss, rec_loss, clf_loss, kl_loss, embedding_, p_y_x_mlp,other_loss,sigmoid_vals

    @tf.function(reduce_retracing=True)
    #TODO check if this is right!!!
    def train_step_vae_one_loss(self,x,y,other_data,train_vae=True,train_clf=True):
        with tf.GradientTape() as tape:
            spn_out,loss,rec_loss,clf_loss,kl_loss,z=self.model_execution_vae(x,y,other_data,train_vae=train_vae,train_clf=train_clf)
        train_vars=self.embedding.trainable_variables+self.decoder.trainable_variables+self.clf_spn.trainable_variables
        #for entry in self.all_spn_x_y:
        #    train_vars.extend(entry.trainable_variables)

        grads = tape.gradient(loss, train_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)
        #self.optimizer.apply_gradients(zip(grads, train_vars))
        return [loss,rec_loss,clf_loss,kl_loss],z,spn_out

    @tf.function(reduce_retracing=True) #TODO delete comment
    def train_step_vae_diff_loss(self, x, y, other_data, train_vae=True, train_clf=True):
        with tf.GradientTape() as tape:
            spn_out = self.model_execution_X(x, other_data, train_vae=train_vae, train_clf=train_clf)
            spn_out=tf.math.exp(spn_out)
            loss = self.clf_loss(y, spn_out)
            #print('DEBUG INTERNAL spn_vae TRAIN with exp vs y', spn_out[:5], y[:5])

        train_vars = self.clf_spn.trainable_variables
        grads = tape.gradient(loss, train_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)
        return tf.reduce_mean(loss)

    @tf.function
    def gan_step(self,X, y, other_info,discriminator_loss_old, generator_loss_old):# GAN Step (Discriminator training)

        #if discriminator_loss_old-generator_loss_old<0.2:
        for _ in range(5):
            with tf.GradientTape() as tape:
                fake_images = self.decoder(self.embedding(X)[0])
                # copy fake images:
                #fake_images=tf.expand_dims(fake_images,axis=-1)
                #fake_images=tf.tile(fake_images, multiples=[1,1, 1, 3])
                real_output = self.discriminator(X[:,:,:,0:1]/255)
                fake_output = self.discriminator(tf.keras.activations.sigmoid(fake_images))  # Compute once
                #real_loss = self.gan_loss_fn(tf.ones_like(real_output), real_output)
                #fake_loss = self.gan_loss_fn(tf.zeros_like(fake_output), fake_output)
                gp = gradient_penalty(self.discriminator, X[:,:,:,0:1]/255, tf.keras.activations.sigmoid(fake_images))

                gan_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)+ 10*gp  # Wasserstein loss

                #gan_loss = real_loss + fake_loss
            grads = tape.gradient(gan_loss, self.discriminator.trainable_weights)
            #clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
            self.gan_discr_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            #else:
            #    gan_loss=discriminator_loss_old

        with tf.GradientTape() as tape:
            # Use the same `fake_output` computed earlier
            fake_images = self.decoder(self.embedding(X)[0])
            #fake_images = tf.tile(fake_images, multiples=[1, 1, 1, 3])
            fake_output = self.discriminator(tf.keras.activations.sigmoid(fake_images))  # Compute once
            #generator_loss = self.gan_loss_fn(tf.ones_like(fake_output), fake_output)
            generator_loss = -tf.reduce_mean(fake_output)  # Wasserstein loss
        grads = tape.gradient(generator_loss, self.decoder.trainable_weights)
        #clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
        self.gan_gener_optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))

        return gan_loss,generator_loss
    def train_gan(self,train_ds,first_loss):
        counter=0.0

        #train_fkt=self.train_step_vae_one_loss
        all_losses = np.zeros(6)
        z_train, gt_train, predictions_train=[],[],[]
        discriminator_loss_old, generator_loss_old=0,0
        for it_c, train_rec in enumerate(train_ds):
            other_info = 0
            if self.use_add_info:
                (X, y, other_info) = train_rec
                other_info = tf.cast(other_info, tf.float32)
            else:
                (X, y) = train_rec

            # VAE step:
            loss, z, pred = self.train_step_vae_one_loss(X, y, other_info)
            # GAN step:
            discriminator_loss,generator_loss=self.gan_step(X,y,other_info,discriminator_loss_old, generator_loss_old)

            z_train.append(z.numpy())
            gt_train.append(y)
            predictions_train.append(pred.numpy())


            new_loss = [entry.numpy() for entry in loss]
            new_loss.append(discriminator_loss.numpy())
            new_loss.append(generator_loss.numpy())
            discriminator_loss_old, generator_loss_old=discriminator_loss.numpy(),generator_loss.numpy()
            #print(new_loss)
            #for e in new_loss:
            #    print(e.shape)
            all_losses += new_loss
            counter += 1
            if not it_c and first_loss:
                print(it_c, 'loss', all_losses / counter)
        return [all_losses / counter,z_train, gt_train,predictions_train]

    @tf.function #TODO delete comment
    def model_execution_z(self,X,other_data,train_vae=False,train_clf=True):
        [embedding, _, _] = self.embedding(X, training=train_vae)
        # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and train_vae:
            embedding = self.gauss_layer(embedding)

        # Stack children along axis 1
        if self.use_add_info:
            embedding = tf.concat([embedding, other_data], axis=-1)

        log_p_z = self.spn_model_z(embedding, training=train_clf)
        return log_p_z
    @tf.function
    def model_execution_mlp(self,X,y, other_data, train_vae=False, train_clf=True):
        [embedding, _, _] = self.embedding(X, training=train_vae)
        # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and train_vae:
            embedding = self.gauss_layer(embedding)

        # Stack children along axis 1
        if self.use_add_info:
            embedding = tf.concat([embedding, other_data], axis=-1)

        y_pred = self.clf_mlp(embedding, training=train_clf)
        if self.num_classes==1:
            y_pred=tf.keras.activations.sigmoid(y_pred)
        mse=tf.reduce_mean(tf.losses.MSE(y,y_pred))
        return mse


    '''

    # NICHT mit @tf.function dekorieren für diesen Debug-Test
    def train_step_vae_diff_loss(self, x, y, other_data, train_vae=True, train_clf=True):
        t0 = time.time()

        with tf.GradientTape() as tape:
            t_forward_start = time.time()

            spn_out = self.model_execution_X(
                x, other_data,
                train_vae=train_vae,
                train_clf=train_clf
            )

            t_forward_end = time.time()

            t_exp_loss_start = time.time()

            spn_out = tf.math.exp(spn_out)
            loss = self.clf_loss(y, spn_out)

            t_exp_loss_end = time.time()

        t_vars_start = time.time()

        train_vars = self.clf_spn.trainable_variables

        t_vars_end = time.time()

        t_grad_start = time.time()

        grads = tape.gradient(loss, train_vars)

        t_grad_end = time.time()

        t_filter_start = time.time()

        grads_and_vars = [
            (g, v) for g, v in zip(grads, train_vars)
            if g is not None
        ]

        t_filter_end = time.time()

        t_opt_start = time.time()

        self.optimizer.apply_gradients(grads_and_vars)

        t_opt_end = time.time()

        print(
            "[step detail]",
            "total:", np.round(t_opt_end - t0, 3),
            "| forward:", np.round(t_forward_end - t_forward_start, 3),
            "| exp+loss:", np.round(t_exp_loss_end - t_exp_loss_start, 3),
            "| get_vars:", np.round(t_vars_end - t_vars_start, 3),
            "| gradient:", np.round(t_grad_end - t_grad_start, 3),
            "| filter:", np.round(t_filter_end - t_filter_start, 3),
            "| optimizer:", np.round(t_opt_end - t_opt_start, 3),
            "| n_vars:", len(train_vars),
            "| n_grads:", len(grads_and_vars),
            flush=True
        )

        return tf.reduce_mean(loss)
    '''



    @tf.function(reduce_retracing=True) #TODO delete comment
    def train_step_z(self,x,other_data,train_vae=False,train_clf=True):
        with tf.GradientTape() as tape:
            log_likelihood=self.model_execution_z(x,other_data,train_vae=train_vae,train_clf=train_clf)
            loss=-tf.reduce_sum(log_likelihood)

        train_vars = self.spn_model_z.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer_z.apply_gradients(zip(grads, train_vars))
        return tf.reduce_mean(loss)

    @tf.function(reduce_retracing=True)
    def train_step_mlp(self,X,y, other_info, train_vae=False, train_clf=True):
        with tf.GradientTape() as tape:
            mse=self.model_execution_mlp(X,y,other_info,train_vae=train_vae,train_clf=train_clf)

        train_vars = self.clf_mlp.trainable_variables#+test
        grads = tape.gradient(mse, train_vars)
        self.optimizer_mlp.apply_gradients(zip(grads, train_vars))
        return mse

    def train_z(self,train_ds, train_vae=False, train_clf=True):
        counter=0.0
        all_losses = np.zeros(1)

        for it_c, train_rec in enumerate(train_ds):
            other_info = 0
            if self.use_add_info:
                (X, y, other_info) = train_rec
                other_info = tf.cast(other_info, tf.float32)
            else:
                (X, y) = train_rec

            loss = self.train_step_z(X, other_info, train_vae=train_vae, train_clf=train_clf)
            # new_loss=[entry.numpy() for entry in loss]
            all_losses += loss.numpy()
            counter += 1
            # if not it_c and first_loss:
            #    print(it_c,'loss',all_losses/counter)
        return all_losses / counter

    def train_mlp(self, train_ds,train_vae=False, train_clf=True):
        counter=0.0
        all_losses = np.zeros(1)

        for it_c, train_rec in enumerate(train_ds):
            other_info = 0
            if self.use_add_info:
                (X, y, other_info) = train_rec
                other_info = tf.cast(other_info, tf.float32)
            else:
                (X, y) = train_rec
            loss = self.train_step_mlp(X,y, other_info, train_vae=train_vae, train_clf=train_clf)
            # new_loss=[entry.numpy() for entry in loss]
            all_losses += loss.numpy()
            counter += 1
            # if not it_c and first_loss:
            #    print(it_c,'loss',all_losses/counter)
        return all_losses / counter

    def train(self, train_ds, first_loss, train_vae=True, train_clf=True):
        counter = 0.0
        train_loop_start = time.time()

        print('[train debug] entered train()', flush=True)
        print('[train debug] VAE_fine_tune:', self.VAE_fine_tune,
              'train_vae:', train_vae,
              'train_clf:', train_clf,
              'use_add_info:', self.use_add_info,
              flush=True)

        if self.VAE_fine_tune == 'None':
            # train_fkt=self.train_step
            all_losses = np.zeros(1)

            first_batch_seen = False
            step_times = []
            unpack_times = []
            numpy_times = []

            for it_c, train_rec in enumerate(train_ds):
                batch_start_time = time.time()

                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                    other_info = tf.cast(other_info, tf.float32)
                else:
                    (X, y) = train_rec

                unpack_end_time = time.time()

                if not first_batch_seen:
                    print('[train debug][None] first batch arrived after:',
                          np.round(unpack_end_time - train_loop_start, 3), 'sec', flush=True)
                    print('[train debug][None] first batch X shape:', X.shape, 'dtype:', X.dtype, flush=True)
                    print('[train debug][None] first batch y shape:', y.shape, 'dtype:', y.dtype, flush=True)
                    if self.use_add_info:
                        print('[train debug][None] first batch other_info shape:', other_info.shape, 'dtype:',
                              other_info.dtype, flush=True)
                    first_batch_seen = True

                step_start_time = time.time()
                loss = self.train_step(X, y, other_info)
                step_end_time = time.time()

                numpy_start_time = time.time()
                new_loss = [entry.numpy() for entry in loss]
                numpy_end_time = time.time()

                all_losses += new_loss
                counter += 1

                unpack_times.append(unpack_end_time - batch_start_time)
                step_times.append(step_end_time - step_start_time)
                numpy_times.append(numpy_end_time - numpy_start_time)

                if it_c == 0:
                    print('[train debug][None] first train_step time:',
                          np.round(step_end_time - step_start_time, 3),
                          'sec (may include tf.function tracing)', flush=True)

                if it_c % 50 == 0:
                    print('[train debug][None] batch', it_c,
                          '| unpack:', np.round(unpack_end_time - batch_start_time, 3), 'sec',
                          '| step:', np.round(step_end_time - step_start_time, 3), 'sec',
                          '| to_numpy:', np.round(numpy_end_time - numpy_start_time, 3), 'sec',
                          '| loss:', new_loss,
                          flush=True)

                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter, flush=True)

            if len(step_times) > 0:
                print('[train debug][None] num batches:', len(step_times), flush=True)
                print('[train debug][None] mean unpack time:', np.round(np.mean(unpack_times), 3), 'sec', flush=True)
                print('[train debug][None] mean step time:', np.round(np.mean(step_times), 3), 'sec', flush=True)
                print('[train debug][None] mean to_numpy time:', np.round(np.mean(numpy_times), 3), 'sec', flush=True)
                if len(step_times) > 1:
                    print('[train debug][None] mean step time without first:',
                          np.round(np.mean(step_times[1:]), 3), 'sec', flush=True)

            print('[train debug][None] total train loop time:',
                  np.round(time.time() - train_loop_start, 3), 'sec', flush=True)
            return all_losses / counter

        elif self.VAE_fine_tune in ['SPN_VAE','SPN_VAE_MLP']:
            # train_fkt=self.train_step_vae_one_loss
            all_losses = np.zeros(4)
            z_train, gt_train, predictions_train = [], [], []

            first_batch_seen = False
            step_times = []
            unpack_times = []
            numpy_times = []
            append_times = []

            for it_c, train_rec in enumerate(train_ds):
                batch_start_time = time.time()

                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                    other_info = tf.cast(other_info, tf.float32)
                else:
                    (X, y) = train_rec

                unpack_end_time = time.time()

                if not first_batch_seen:
                    print('[train debug][SPN_VAE] first batch arrived after:',
                          np.round(unpack_end_time - train_loop_start, 3), 'sec', flush=True)
                    print('[train debug][SPN_VAE] first batch X shape:', X.shape, 'dtype:', X.dtype, flush=True)
                    print('[train debug][SPN_VAE] first batch y shape:', y.shape, 'dtype:', y.dtype, flush=True)
                    if self.use_add_info:
                        print('[train debug][SPN_VAE] first batch other_info shape:', other_info.shape, 'dtype:',
                              other_info.dtype, flush=True)
                    first_batch_seen = True

                step_start_time = time.time()
                loss, z, pred = self.train_step_vae_one_loss(X, y, other_info, train_vae=train_vae, train_clf=train_clf)
                step_end_time = time.time()

                append_start_time = time.time()
                z_train.append(z.numpy())
                gt_train.append(y)
                predictions_train.append(pred.numpy())
                append_end_time = time.time()

                numpy_start_time = time.time()
                new_loss = [entry.numpy() for entry in loss]
                numpy_end_time = time.time()

                # print(new_loss)
                # for e in new_loss:
                #    print(e.shape)
                all_losses += new_loss
                counter += 1

                unpack_times.append(unpack_end_time - batch_start_time)
                step_times.append(step_end_time - step_start_time)
                append_times.append(append_end_time - append_start_time)
                numpy_times.append(numpy_end_time - numpy_start_time)

                if it_c == 0:
                    print('[train debug][SPN_VAE] first train_step_vae_one_loss time:',
                          np.round(step_end_time - step_start_time, 3),
                          'sec (may include tf.function tracing)', flush=True)

                if it_c % 50 == 0:
                    print('[train debug][SPN_VAE] batch', it_c,
                          '| unpack:', np.round(unpack_end_time - batch_start_time, 3), 'sec',
                          '| step:', np.round(step_end_time - step_start_time, 3), 'sec',
                          '| append numpy:', np.round(append_end_time - append_start_time, 3), 'sec',
                          '| loss numpy:', np.round(numpy_end_time - numpy_start_time, 3), 'sec',
                          '| loss:', new_loss,
                          flush=True)

                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter, flush=True)

            if len(step_times) > 0:
                print('[train debug][SPN_VAE] num batches:', len(step_times), flush=True)
                print('[train debug][SPN_VAE] mean unpack time:', np.round(np.mean(unpack_times), 3), 'sec', flush=True)
                print('[train debug][SPN_VAE] mean step time:', np.round(np.mean(step_times), 3), 'sec', flush=True)
                print('[train debug][SPN_VAE] mean append numpy time:', np.round(np.mean(append_times), 3), 'sec',
                      flush=True)
                print('[train debug][SPN_VAE] mean loss numpy time:', np.round(np.mean(numpy_times), 3), 'sec',
                      flush=True)
                if len(step_times) > 1:
                    print('[train debug][SPN_VAE] mean step time without first:',
                          np.round(np.mean(step_times[1:]), 3), 'sec', flush=True)

            print('[train debug][SPN_VAE] total train loop time:',
                  np.round(time.time() - train_loop_start, 3), 'sec', flush=True)
            return all_losses / counter#[all_losses / counter, z_train, gt_train, predictions_train]

        elif self.VAE_fine_tune == 'SPN':
            # train_fkt = self.train_step_vae_diff_loss
            all_losses = np.zeros(1)

            first_batch_seen = False
            step_times = []
            unpack_times = []
            numpy_times = []

            for it_c, train_rec in enumerate(train_ds):
                batch_start_time = time.time()

                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                    other_info = tf.cast(other_info, tf.float32)
                else:
                    (X, y) = train_rec

                unpack_end_time = time.time()

                if not first_batch_seen:
                    print('[train debug][SPN] first batch arrived after:',
                          np.round(unpack_end_time - train_loop_start, 3), 'sec', flush=True)
                    print('[train debug][SPN] first batch X shape:', X.shape, 'dtype:', X.dtype, flush=True)
                    print('[train debug][SPN] first batch y shape:', y.shape, 'dtype:', y.dtype, flush=True)
                    if self.use_add_info:
                        print('[train debug][SPN] first batch other_info shape:', other_info.shape, 'dtype:',
                              other_info.dtype, flush=True)
                    first_batch_seen = True

                #if self.num_classes == 1:
                #    y = tf.math.log(y)

                step_start_time = time.time()
                loss = self.train_step_vae_diff_loss(X, y, other_info, train_vae=train_vae, train_clf=train_clf)
                step_end_time = time.time()

                numpy_start_time = time.time()
                # new_loss=[entry.numpy() for entry in loss]
                loss_numpy = loss.numpy()
                numpy_end_time = time.time()

                all_losses += loss_numpy
                counter += 1

                unpack_times.append(unpack_end_time - batch_start_time)
                step_times.append(step_end_time - step_start_time)
                numpy_times.append(numpy_end_time - numpy_start_time)

                if it_c == 0:
                    print('[train debug][SPN] first train_step_vae_diff_loss time:',
                          np.round(step_end_time - step_start_time, 3),
                          'sec (may include tf.function tracing)', flush=True)

                if it_c % 50 == 0:
                    print('[train debug][SPN] batch', it_c,
                          '| unpack:', np.round(unpack_end_time - batch_start_time, 3), 'sec',
                          '| step:', np.round(step_end_time - step_start_time, 3), 'sec',
                          '| to_numpy:', np.round(numpy_end_time - numpy_start_time, 3), 'sec',
                          '| loss:', loss_numpy,
                          flush=True)

                # if not it_c and first_loss:
                #    print(it_c,'loss',all_losses/counter)

            if len(step_times) > 0:
                print('[train debug][SPN] num batches:', len(step_times), flush=True)
                print('[train debug][SPN] mean unpack time:', np.round(np.mean(unpack_times), 3), 'sec', flush=True)
                print('[train debug][SPN] mean step time:', np.round(np.mean(step_times), 3), 'sec', flush=True)
                print('[train debug][SPN] mean to_numpy time:', np.round(np.mean(numpy_times), 3), 'sec', flush=True)
                if len(step_times) > 1:
                    print('[train debug][SPN] mean step time without first:',
                          np.round(np.mean(step_times[1:]), 3), 'sec', flush=True)

            print('[train debug][SPN] total train loop time:',
                  np.round(time.time() - train_loop_start, 3), 'sec', flush=True)
            return all_losses / counter



    def eval_cnn(self,test_X):
        return self.embedding(test_X)
    def get_spn_variables(self):
        pass


def train_model(grid_params,cnn_model,cnn_spn, train_data,opt_spn, test_data_0,test_data_1, variable_dict,
                get_max,debugging, num_iterations,layer_names,add_info):
    first_loss=True
    for i in range(num_iterations):
        #print('train iteration', i,flush=True)
        train_start_time=time.time()
        # shuffle dataset:
        idx_train = list(range(train_data[0].shape[0]))
        random.shuffle(idx_train)
        X = train_data[0][idx_train]
        if add_info:
            y = train_data[1][idx_train,0]
            if cnn_spn.use_add_info:
                other_info= train_data[1][idx_train,1:]
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y,other_info)).batch(grid_params.batch_size)
            else:

                train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)
        else:
            y = train_data[1][idx_train]
            train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)

        train_loss=cnn_spn.train(train_dataset,first_loss=first_loss)
        first_loss=False
        #cnn_embedding=cnn_spn.eval_cnn(test_data)
        print('loss',i,train_loss,'fine tune train time',(time.time()-train_start_time)//60,flush=True)



def train_MLP(grid_params, cnn_spn, ckp, manager, train_data, val_data, curr_log_likelihood):
    num_iterations = grid_params.fine_tune_its
    for i in range(num_iterations):
        if i % 3 == 0:
            val_loss = test_model_mlp(cnn_spn,val_data,add_info=grid_params.use_add_info,batch_size=grid_params.batch_size)
            # best validation spearman
            if val_loss< curr_log_likelihood:
                print('SAVE of all SPN checkpoint', val_loss, curr_log_likelihood)
                curr_log_likelihood=val_loss
                # '''
                #if grid_params.VAE_fine_tune == 'SPN':
                ckp.step.assign_add(1)
                manager.save()


        idx_train = list(range(train_data[0].shape[0]))
        random.shuffle(idx_train)
        X = train_data[0][idx_train]
        if grid_params.use_add_info:
            y = train_data[1][idx_train, 0]
            other = train_data[1][idx_train, 1:]
            if grid_params.use_add_info:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y, other)).batch(grid_params.batch_size)
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)

        else:
            y = train_data[1][idx_train]
            train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)

        train_loss = cnn_spn.train_mlp(train_dataset,train_vae=False, train_clf=True)
        print(i,'MLP MSE loss:',train_loss,curr_log_likelihood)

def train_generative_spn(grid_params,cnn_spn,ckp,manager,train_data,val_data,curr_log_likelihood):
    num_iterations = grid_params.fine_tune_its
    for i in range(num_iterations):
        if i % 3 == 0:
            val_loss = test_model_z(cnn_spn,val_data,add_info=grid_params.use_add_info,batch_size=grid_params.batch_size, train_vae=False, train_clf=False)
            # best validation spearman
            if val_loss < curr_log_likelihood:
                print('SAVE of all SPN checkpoint', val_loss, curr_log_likelihood)
                curr_log_likelihood=val_loss.numpy()
                # '''
                #if grid_params.VAE_fine_tune == 'SPN':
                ckp.step.assign_add(1)
                manager.save()




        idx_train = list(range(train_data[0].shape[0]))
        random.shuffle(idx_train)
        X = train_data[0][idx_train]
        if grid_params.use_add_info:
            y = train_data[1][idx_train, 0]
            other = train_data[1][idx_train, 1:]
            if grid_params.use_add_info:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y, other)).batch(grid_params.batch_size)
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)

        else:
            y = train_data[1][idx_train]
            train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)

        train_loss = cnn_spn.train_z(train_dataset,train_vae=False, train_clf=True)
        print(i,'Z SPN loss:',train_loss)





def train_model_parts(grid_params,cnn_spn, train_data, val_data,test_data,ckpt=[],manager=[],val_entropy=0,val_acc=0,add_info=False,num_classes =2):
    first_loss=True
    num_iterations=grid_params.fine_tune_its
    print('number of trainable variables in cnn spn:',len(cnn_spn.trainable_variables), flush=True)
    print('number of trainable variables in spn:', len(cnn_spn.clf_spn.trainable_variables), flush=True)

    print('train_data[0] shape:', train_data[0].shape, 'dtype:', train_data[0].dtype, flush=True)
    print('train_data[1] shape:', train_data[1].shape, 'dtype:', train_data[1].dtype, flush=True)
    print('val_data[0] shape:', val_data[0].shape, 'dtype:', val_data[0].dtype, flush=True)
    print('val_data[1] shape:', val_data[1].shape, 'dtype:', val_data[1].dtype, flush=True)
    print('test_data[0] shape:', test_data[0].shape, 'dtype:', test_data[0].dtype, flush=True)
    print('test_data[1] shape:', test_data[1].shape, 'dtype:', test_data[1].dtype, flush=True)

    print('batch size:', grid_params.batch_size, flush=True)
    print('number of train samples:', train_data[0].shape[0], flush=True)
    print('approx batches per epoch:', int(np.ceil(train_data[0].shape[0] / grid_params.batch_size)), flush=True)

    # --------------------------------------------------
    # Dataset nur einmal bauen, Shuffle pro neuer Iteration
    # --------------------------------------------------
    dataset_build_start_time = time.time()

    if add_info:
        y_all = train_data[1][:,0]
        other_all = train_data[1][:,1:]

        if grid_params.use_add_info:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], y_all, other_all))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], y_all))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))

    train_dataset = train_dataset.shuffle(
        buffer_size=train_data[0].shape[0],
        reshuffle_each_iteration=True
    ).batch(grid_params.batch_size)

    # optional fürs Tempo, ändert die Logik nicht
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    dataset_build_end_time = time.time()
    print('initial dataset build time:', np.round(dataset_build_end_time - dataset_build_start_time, 3), 'sec', flush=True)

    # erster Batch-Check
    first_batch_check_start = time.time()
    try:
        first_batch = next(iter(train_dataset))
        first_batch_check_end = time.time()
        print('time until first batch from dataset iterator:',
              np.round(first_batch_check_end - first_batch_check_start, 3), 'sec', flush=True)

        if add_info and grid_params.use_add_info:
            xb, yb, ob = first_batch
            print('first batch shapes: X', xb.shape, 'y', yb.shape, 'other', ob.shape, flush=True)
            print('first batch dtypes: X', xb.dtype, 'y', yb.dtype, 'other', ob.dtype, flush=True)
        else:
            xb, yb = first_batch
            print('first batch shapes: X', xb.shape, 'y', yb.shape, flush=True)
            print('first batch dtypes: X', xb.dtype, 'y', yb.dtype, flush=True)
    except Exception as e:
        print('WARNING: could not inspect first batch:', repr(e), flush=True)

    #best_val_loss = val_entropy
    train_start_time = time.time()
    best_acc_loss=val_acc
    best_val_reconstruction=100000
    val_idx = 2
    eval_after_train = []
    all_debugging_stuff = []
    print('thist is the best loss so far',best_acc_loss)
    for i in range(num_iterations):

        iteration_start_time = time.time()
        print('\n' + '='*80, flush=True)
        print('START iteration', i, flush=True)

        train_call_start = time.time()
        print('calling training function...', flush=True)

        if grid_params.VAE_fine_tune and grid_params.GAN:
            [train_loss, z_train, gt_train, predictions_train] = cnn_spn.train_gan(train_dataset, first_loss=first_loss)
        else:
            # differentiate in what is trained!
            if grid_params.VAE_fine_tune in ['SPN']:
                train_vae =False
                train_clf =True
            elif grid_params.VAE_fine_tune in ['SPN_VAE','SPN_VAE_MLP']:
                train_vae =True
                train_clf =True
            else:
                train_vae =False
                train_clf =False


        train_call_end = time.time()
        print('training function finished after:',
              np.round(train_call_end - train_call_start, 3), 'sec',
              '(' + str(np.round((train_call_end - train_call_start)/60, 3)) + ' min )',
              flush=True)

        first_loss=False

        if i%1==0 or i == num_iterations - 1:
            val_start_time = time.time()
            print('starting validation...', flush=True)

            if grid_params.VAE_fine_tune in ['SPN_VAE']:
                val_losses,z,gt,predictions ,_= test_model_all(cnn_spn, val_data, num_classes=num_classes, batch_size=grid_params.batch_size,
                                                 training=False, add_info=add_info)
            else:
                [val_losses]=test_model_no_mpe(cnn_spn, [val_data], num_classes=num_classes, batch_size=grid_params.batch_size, training=False,add_info=add_info)
            print(
                "SPN train/val | "
               # "train MAE:", np.round(train_losses[0], 4),
               # "train MSE:", np.round(train_losses[1], 4),
               # "train Spearman:", np.round(train_losses[2], 4),
                "| val MAE:", np.round(val_losses[0], 4),
                "val MSE:", np.round(val_losses[1], 4),
                "val Spearman:", np.round(val_losses[2], 4),
                flush=True
            )
            print(
                "Decision Spearman train/val:",
               # np.round(train_losses[5], 4),
                np.round(val_losses[5], 4),
                flush=True
            )
            val_end_time = time.time()
            print('validation finished after:',
                  np.round(val_end_time - val_start_time, 3), 'sec',
                  '(' + str(np.round((val_end_time - val_start_time)/60, 3)) + ' min )',
                  flush=True)

            improve = False
            # best validation spearman
            if val_losses[val_idx] > best_acc_loss:
                #'''
                if grid_params.VAE_fine_tune=='SPN':
                    ckpt[2].step.assign_add(1)
                    manager[2].save()
                    print('SAVE of all SPN checkpoint',best_acc_loss, val_losses[val_idx], flush=True)
                elif grid_params.VAE_fine_tune=='SPN_VAE' or grid_params.VAE_fine_tune=='SPN_VAE_MLP':
                    ckpt[0].step.assign_add(1)
                    manager[0].save()
                    ckpt[2].step.assign_add(1)
                    manager[2].save()
                    ckpt[1].step.assign_add(1)
                    manager[1].save()
                    print('SAVE of all VAE+SPN checkpoint',best_acc_loss, val_losses[val_idx], flush=True)
                best_acc_loss = val_losses[val_idx]
                #'''
                improve = True

                # TODO save VAE:
            if num_classes==1:
                print(i,val_losses,flush=True)
            else:
                print('val entropy', i, val_losses[1],'improve',improve,'curr acc', val_losses[0], val_losses[2],'best acc',best_acc_loss,flush=True)
        train_loss = cnn_spn.train(train_dataset, first_loss=first_loss, train_vae=train_vae, train_clf=train_clf)

        print('loss', i,end=':' )
        for loss in train_loss:
            print(np.round(loss,5),end=' - ')
        print()
        print('iteration time sec:', np.round((time.time()-iteration_start_time), 3), flush=True)
        print('iteration time min:', np.round((time.time()-iteration_start_time)/60, 3), flush=True)

    print('fine tune train time min:', np.round((time.time()-train_start_time)/60, 3), flush=True)
    return eval_after_train,all_debugging_stuff



def train_mlp(grid_params,cnn_spn, train_data, val_data,test_data,ckpt=[],manager=[],val_entropy=0,val_acc=0,add_info=False,num_classes =2):
    num_iterations=grid_params.fine_tune_its

    if add_info:
        y_all = train_data[1][:,0]
        other_all = train_data[1][:,1:]

        if grid_params.use_add_info:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], y_all, other_all))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], y_all))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))

    train_dataset = train_dataset.shuffle(
        buffer_size=train_data[0].shape[0],
        reshuffle_each_iteration=True
    ).batch(grid_params.batch_size)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    best_acc_loss=0#val_acc

    eval_after_train = []
    all_debugging_stuff = []
    print('thist is the best loss so far',best_acc_loss)
    for i in range(num_iterations):


        print('\n' + '='*80, flush=True)
        print('START iteration', i, flush=True)
        first_loss=False

        if i%3==0:

            val_losses=test_model_no_mpe(cnn_spn, [val_data], num_classes=num_classes, batch_size=grid_params.batch_size, training=False,add_info=add_info)

            if num_classes == 1:
                val_idx=2
            else:
                val_idx=2

            # best validation spearman
            if val_losses[val_idx] > best_acc_loss:

                ckpt[0].step.assign_add(1)
                manager[0].save()
                print('SAVE of all MLP checkpoint',best_acc_loss, val_losses[val_idx], flush=True)
                best_acc_loss = val_losses[val_idx]
                print(i,val_losses,flush=True)

        train_loss = cnn_spn.train_mlp(train_dataset)

        print('loss', i,end=':' )
        for loss in train_loss:
            print(np.round(loss,5),end=' - ')

    return eval_after_train,all_debugging_stuff


def test_model(tfmodel,spn_opt, test_data,test_gt,variable_dict,get_max=True):
    cnn_embedding=tfmodel.eval_cnn(test_data).numpy()

    # get max values or whatever:
    if get_max:
        cnn_embedding=np.max(cnn_embedding, axis=(1, 2))


    # overwrite cnn_spn :
    tf_graph_to_spn(variable_dict)
    # spn is ready to use:
    out_labels = np.full((cnn_embedding.shape[0], 1), np.nan)
    spn_input=np.concatenate([out_labels,cnn_embedding],axis=1)
    #result labs
    prediction=mpe(spn_opt, spn_input)[:,0]
    acc=accuracy_score(test_gt, prediction, normalize=True)
    print('CNN+SPN accurancy',acc)
    return acc

def test_model_z(tfmodel,dataset,batch_size, add_info,train_vae=False, train_clf=False):
    losses=[]
    batch_start_end=[]
    for i in range(dataset[0].shape[0] // batch_size):
        batch_start_end.append([i * batch_size, (i + 1) * batch_size])
    batch_start_end.append([(dataset[0].shape[0] // batch_size) * batch_size, -1])
    #for i in range(dataset[0].shape[0]//batch_size):
    for start,end in batch_start_end:
        X_data = dataset[0][start:end]
        other_data = 0
        if add_info:
            other_data = dataset[1][start:end, 1:]


        log_likelihood = tfmodel.model_execution_z(X_data, other_data, train_vae=train_vae, train_clf=train_clf)
        losses.extend(log_likelihood.numpy().tolist())
    log_likelihood=np.asarray(losses)
    loss = -tf.reduce_sum(log_likelihood)
    return loss
def test_model_mlp(tfmodel,dataset,batch_size, add_info,train_vae=False,train_clf=False):
    losses=[]
    batch_start_end=[]
    for i in range(dataset[0].shape[0] // batch_size):
        batch_start_end.append([i * batch_size, (i + 1) * batch_size])
    batch_start_end.append([(dataset[0].shape[0] // batch_size) * batch_size, -1])
    #for i in range(dataset[0].shape[0]//batch_size):
    for start,end in batch_start_end:
        X_data = dataset[0][start:end]
        other_data = 0
        if add_info:
            y_data = dataset[1][start:end, 0]
            other_data = dataset[1][start:end, 1:]
        else:
            y_data = dataset[1][start:end]

        mse = tfmodel.model_execution_mlp(X_data,y_data, other_data, train_vae=train_vae, train_clf=train_clf)
        losses.append(mse.numpy())
    loss=np.mean(np.asarray(losses))
    #loss = -tf.reduce_sum(log_likelihood)
    return loss

def test_model_no_mpe(tfmodel, data_sets,num_classes,batch_size=25,training=False,add_info=False):
    all_evals=[]
    for dataset in data_sets:
        prediction=[]
        gt=[]
        all_pred_decisions=[]
        batch_start_end=[]
        for i in range(dataset[0].shape[0]//batch_size):
            batch_start_end.append([i * batch_size,(i + 1) * batch_size])
        batch_start_end.append([(dataset[0].shape[0]//batch_size)*batch_size,-1])
        for start, end in batch_start_end:
            X_data = dataset[0][start:end]
            other_data=0
            if add_info:

                y_data=dataset[1][start:end,0]
                other_data=dataset[1][start:end,1:]

            else:
                y_data=dataset[1][start:end]

            pred,pred_decision=tfmodel.model_execution_X_regression(X_data,other_data,training=False)
            prediction.extend(np.squeeze(pred.numpy()).tolist())
            gt.extend(np.squeeze(y_data).tolist())


            all_pred_decisions.extend(np.squeeze(pred_decision.numpy()).tolist())

        if num_classes>1:
            #classification

            #prediction=tfmodel.model_execution_X(dataset[0],training=False)
            #prediction_softmax=tf.nn.softmax(prediction,axis=-1)
            prediction=np.asarray(prediction)
            entropy=tfmodel.clf_loss(gt,prediction).numpy()
            prediction_exponential=tf.math.exp(prediction)
            arg_max=np.argmax(prediction_exponential,axis=-1)
            acc = accuracy_score(gt, arg_max, normalize=True)
            auc=0
            if num_classes==2:
                test1 = np.isnan(prediction_exponential[:,1])
                nan_sum = np.sum(test1, axis=0)
                #print('TEST',nan_sum,prediction_exponential[:10,1])
                # replace possible nans with 0
                pred_exp=np.nan_to_num(prediction_exponential[:,1],nan=0,posinf=1.0)
                fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)
                auc = metrics.auc(fpr, tpr)
            balanced_acc=balanced_accuracy_score(gt, arg_max)
            [prec,rec,f1,_]=precision_recall_fscore_support(gt, arg_max, average=None)
            results=[acc,entropy,balanced_acc,prec[1],rec[1],f1[1],auc]
            all_evals.append(results)
            #print(results)
        else:

            def count_nan_inf(x):
                x = np.asarray(x)
                return np.isnan(x).sum(), np.isposinf(x).sum(), np.isneginf(x).sum(), np.isinf(x).sum()

            prediction_decision = np.asarray(all_pred_decisions).flatten()
            prediction = np.asarray(prediction).flatten()
            gt = np.asarray(gt).flatten()

            # Save log predictions before exp for debugging
            prediction_log = prediction.copy()

            prediction = np.exp(prediction)
            '''
            # ==================================================
            # DEBUG: NaN / Inf checks
            # ==================================================
            nan_pred_log, posinf_pred_log, neginf_pred_log, inf_pred_log = count_nan_inf(prediction_log)
            nan_pred, posinf_pred, neginf_pred, inf_pred = count_nan_inf(prediction)
            nan_dec, posinf_dec, neginf_dec, inf_dec = count_nan_inf(prediction_decision)
            nan_gt, posinf_gt, neginf_gt, inf_gt = count_nan_inf(gt)

            print("\nNaN / Inf summary")
            print("-" * 60)
            print(
                f"prediction_log      : NaN={nan_pred_log}, +Inf={posinf_pred_log}, -Inf={neginf_pred_log}, Inf_total={inf_pred_log}")
            print(f"prediction_after_exp: NaN={nan_pred}, +Inf={posinf_pred}, -Inf={neginf_pred}, Inf_total={inf_pred}")
            print(f"prediction_decision : NaN={nan_dec}, +Inf={posinf_dec}, -Inf={neginf_dec}, Inf_total={inf_dec}")
            print(f"gt                  : NaN={nan_gt}, +Inf={posinf_gt}, -Inf={neginf_gt}, Inf_total={inf_gt}")
            print("-" * 60)
            print("prediction_log[:10]      :", prediction_log[:10])
            print("prediction_after_exp[:10]:", prediction[:10])
            print("prediction_decision[:10] :", prediction_decision[:10])
            print("gt[:10]                  :", gt[:10])
            print("=" * 60)
            # ==================================================
            '''
            valid_mask = np.isfinite(gt) & np.isfinite(prediction)
            valid_mask_decision = np.isfinite(gt) & np.isfinite(prediction_decision)

            # Optional debug: how many samples are used / ignored
            #print(f"Prediction metrics: using {valid_mask.sum()} samples, ignoring {(~valid_mask).sum()} invalid samples")
            #print(f"Decision metrics  : using {valid_mask_decision.sum()} samples, ignoring {(~valid_mask_decision).sum()} invalid samples")
            print('DEBUG external spn_vae evaluation with exp vs y', prediction[:5], gt[:5])
            if valid_mask.sum() > 1:
                mae = np.mean(np.abs(gt[valid_mask] - prediction[valid_mask]))
                mse = np.mean((gt[valid_mask] - prediction[valid_mask]) ** 2)
                res = stats.spearmanr(gt[valid_mask], prediction[valid_mask])
                spearman = res.statistic if hasattr(res, "statistic") else res[0]
            else:
                mae = np.nan
                mse = np.nan
                spearman = np.nan

            if valid_mask_decision.sum() > 1:
                mae_decision = np.mean(np.abs(gt[valid_mask_decision] - prediction_decision[valid_mask_decision]))
                mse_decision = np.mean((gt[valid_mask_decision] - prediction_decision[valid_mask_decision]) ** 2)
                res_decision = stats.spearmanr(gt[valid_mask_decision], prediction_decision[valid_mask_decision])
                spearman_decision = res_decision.statistic if hasattr(res_decision, "statistic") else res_decision[0]
            else:
                mae_decision = np.nan
                mse_decision = np.nan
                spearman_decision = np.nan

            results = [
                mae,
                mse,
                spearman,
                mae_decision,
                mse_decision,
                spearman_decision
            ]

            all_evals.append(results)
    #print(all_evals)
    return all_evals


def test_model_SPN_MLP(tfmodel, dataset,num_classes,batch_size=25,training=False,add_info=False,debugging_plots=False,debugging_plot_path=''):
    mlp_prediction=[]
    prediction=[]
    gt=[]
    z=[]
    losses=np.zeros(9)
    x_org=[]
    x_rec=[]
    batch_start_end=[]
    for i in range(dataset[0].shape[0] // batch_size):
        batch_start_end.append([i * batch_size, (i + 1) * batch_size])
    batch_start_end.append([(dataset[0].shape[0] // batch_size) * batch_size, -1])
    #for i in range(dataset[0].shape[0]//batch_size):
    for start,end in batch_start_end:
        X_data = dataset[0][start:end]
        other_data=0
        if add_info:
            y_data=dataset[1][start:end,0]
            other_data=dataset[1][start:end,1:]
        else:
            y_data=dataset[1][start:end]

        # TODO extend to regression:
        #p_y_x, loss, rec_loss, clf_loss, kl_loss, embedding_, p_y_x_mlp,other_loss
        pred, loss, rec_loss, clf_loss, kl_loss, embedding_,mlp_pred,other_loss,reconstruction=tfmodel.model_execution_vae_eval( X_data, y_data, other_data)
        mlp_pred=tf.keras.activations.sigmoid(mlp_pred)

        if False:
            x_org=X_data[:9]
            x_rec=reconstruction.numpy()[:9]
            plot_examples_3D([x_org, x_rec], ['original', 'reconstruction'], debugging_plot_path[:-4] + 'examples.png')
        new_loss=np.asarray([loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy(),
                             other_loss[0].numpy(), other_loss[1].numpy(),other_loss[2].numpy(),
                             other_loss[4].numpy(),other_loss[5].numpy()])
        losses+=new_loss
        prediction.extend(pred.numpy().tolist())
        mlp_prediction.extend(mlp_pred.numpy().tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.numpy().tolist())
    losses/=len(batch_start_end)#dataset[0].shape[0]//batch_size
    prediction=np.asarray(prediction)
    mlp_prediction=np.asarray(mlp_prediction)
    print('MLP pred example',mlp_pred[:10])
    gt=np.asarray(gt)


    prediction_exponential=tf.math.exp(prediction).numpy()

    results_MLP,_,_ =eval_cls(mlp_prediction,mlp_prediction,gt,tfmodel.clf_loss, num_classes)
    results_SPN,_,_=eval_cls(prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes)
    mlp_prediction=mlp_prediction.flatten()
    prediction_exponential=prediction_exponential.flatten()
    np.set_printoptions(threshold=np.inf)
    print("GT:",gt)
    print('MLP',mlp_prediction)
    print('SPN',prediction_exponential)
    res = bootstrap_compare_spearman(
        gt,
        mlp_prediction,
        prediction_exponential#prediction,
    )

    print(res)
    print(compare_mae(gt,mlp_prediction,
        prediction_exponential))
    if debugging_plots:
        plot_correlations(gt,
                          [mlp_prediction,prediction_exponential],
                          [results_MLP,results_SPN],
                          ['MLP','SPN'],
                          debugging_plot_path)
        # Convert to DataFrame for easier correlation
        df_z = pd.DataFrame(z)

        # Correlation matrix (Pearson by default)
        corr_matrix = df_z.corr()


        plot_heatmap(corr_matrix,debugging_plot_path[:-4] + 'z_correlation.png')







    #results=[results_MLP,results_SPN,losses]

    #print(results)

    return results_MLP,results_SPN,losses


def test_model_all(tfmodel, dataset, num_classes, batch_size=25, training=False, add_info=False):
    #mlp_prediction = []
    prediction = []
    gt = []
    z = []
    losses = np.zeros(4)
    batch_start_end=[]
    for i in range(dataset[0].shape[0] // batch_size):
        batch_start_end.append([i * batch_size, (i + 1) * batch_size])
    batch_start_end.append([(dataset[0].shape[0] // batch_size) * batch_size, -1])
    #for i in range(dataset[0].shape[0]//batch_size):
    for start,end in batch_start_end:
        X_data = dataset[0][start:end]
        other_data = 0
        if add_info:

            y_data = dataset[1][start:end, 0]
            other_data = dataset[1][start:end, 1:]

        else:

            y_data = dataset[1][start:end]

        pred, loss, rec_loss, clf_loss, kl_loss, mae,embedding_, mlp_pred = tfmodel.model_execution_vae_eval(X_data, y_data,
                                                                                                         other_data)
        new_loss = np.asarray([loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy()])
        losses += new_loss
        prediction.extend(pred.numpy().tolist())
        #mlp_prediction.extend(mlp_pred.tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.numpy().tolist())
    losses /= dataset[0].shape[0] // batch_size
    prediction = np.asarray(prediction)
    #mlp_prediction = np.asarray(mlp_prediction)

    prediction_exponential = tf.math.exp(prediction)

    #results_MLP = eval_cls(mlp_prediction, mlp_prediction, gt, tfmodel.clf_loss, num_classes)
    results,pred_exp,arg_max = eval_cls(prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes)

    #results = [results_MLP, results_SPN, losses]

    # print(results)

    return results, z, gt, arg_max, pred_exp


def eval_cls(pred_logits,prediction_exponential,gt,clf_loss,num_classes=2):



    auc=0
    pred_exp=0
    if num_classes==2:
        pred_exp=np.nan_to_num(prediction_exponential[:,1],nan=0,posinf=1.0)
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)
        auc = metrics.auc(fpr, tpr)
    if num_classes>1:
        pred_arg_max = np.argmax(prediction_exponential, axis=-1)
        entropy = clf_loss(gt, pred_logits).numpy()
        acc = accuracy_score(gt, pred_arg_max, normalize=True)


        balanced_acc=balanced_accuracy_score(gt, pred_arg_max)
        [prec,rec,f1,_]=precision_recall_fscore_support(gt, pred_arg_max, average=None)
        return [acc, entropy, balanced_acc, prec[1], rec[1], f1[1], auc], pred_exp, pred_arg_max
    else:
        #TODO
        # mae, mse, spearman
        pred_arg_max=None
        pred_exp = np.nan_to_num(prediction_exponential, nan=0, posinf=1.0)
        gt=np.asarray(gt)
        print('prediction.shape,gt.shape', pred_exp.shape, gt.shape)
        if len(pred_exp.shape) > 1:
            pred_exp = pred_exp.flatten()
        if len(gt.shape) > 1:
            gt = gt.flatten()
        # from log to normal space:
        #prediction = np.exp(prediction)
        mae = np.mean(np.abs(gt - pred_exp))
        mse = np.mean((gt - pred_exp) ** 2)
        rmse= np.sqrt(np.mean((gt - pred_exp) ** 2))
        res = stats.spearmanr(gt, pred_exp)


        #results = [mae, mse, res, mae_decision, mse_decision, res_decision]
        return [mae, mse,rmse,res[0]], pred_exp, pred_arg_max





