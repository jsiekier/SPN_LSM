import tensorflow.compat.v2  as tf
import numpy as np
import bisect
from tensorflow_probability import distributions as tfd
EPSILON = np.finfo(float).eps
#tf.compat.v1.disable_eager_execution()

class Root_Layer(tf.keras.layers.Layer):
    def __init__(self,all_prior,all_spn_x_y_model,dtype=tf.dtypes.float32,name='Root'):
        self.all_spn_x_y=all_spn_x_y_model
        #self.dtype=dtype
        #self.parallel_iterations=len(all_prior)
        #TODO use softmax inverse?
        self.weights = tf.Variable(initial_value=all_prior, dtype=dtype, trainable=False, name=name + 'weights')  # name="mean",

    def call(self, input):
        tf_weights=tf.expand_dims(tf.expand_dims(self.tf_weights,axis=0),axis=-1)
        # Stack children along axis 1
        inputs=[]
        for sub_spn in self.all_spn_x_y:
            inputs.append(sub_spn(input))

        children_prob = tf.stack(inputs, axis=1)#children
        #print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator=children_prob + tf.math.log(tf_weights)
        p_x=tf.reduce_logsumexp(log_enumerator, axis=1)
        p_y_x=log_enumerator-p_x
        return p_y_x
'''
class MultivariateGaussian_layer(tf.keras.layers.Layer):
    def __init__(self, mean, sigma, scope, dtype=tf.float32,
                 trainable_nodes=True, name="MVGauss"):
        super().__init__()

        self.mean = tf.Variable(initial_value=np.asarray(mean, dtype=np.float32),dtype=dtype,trainable=trainable_nodes,name=name + "_mean")
        sigma = np.asarray(sigma, dtype=np.float32)
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0], dtype=np.float32)

        self.sigma = tf.Variable(initial_value=sigma,dtype=dtype,trainable=trainable_nodes,name=name + "_sigma")
        self.scope = scope

    def call(self, inputs):
        x = tf.gather(inputs, indices=self.scope, axis=1)
        sigma = self.sigma + 1e-6 * tf.eye(tf.shape(self.sigma)[0])

        dist = tfd.MultivariateNormalFullCovariance(loc=self.mean,covariance_matrix=sigma)

        return tf.expand_dims(dist.log_prob(x), axis=-1)
'''
class MultivariateGaussian_layer(tf.keras.layers.Layer):
    def __init__(self, mean, sigma, scope, dtype=tf.float32,
                 trainable_nodes=True, name="MVGauss"):
        super().__init__()

        self.mean = tf.Variable(
            initial_value=np.asarray(mean, dtype=np.float32),
            dtype=dtype,
            trainable=trainable_nodes,
            name=name + "_mean"
        )

        sigma = np.asarray(sigma, dtype=np.float32)
        sigma = 0.5 * (sigma + sigma.T)
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0], dtype=np.float32)

        chol = np.linalg.cholesky(sigma)

        self.raw_tril = tf.Variable(
            initial_value=chol,
            dtype=dtype,
            trainable=trainable_nodes,
            name=name + "_raw_tril"
        )

        self.tmp = tf.Variable(
            initial_value=0.001,
            dtype=dtype,
            trainable=False,
            name=name + "_tmp"
        )

        self.scope = scope

    def call(self, inputs):
        x = tf.gather(inputs, indices=self.scope, axis=1)

        tril = tf.linalg.band_part(self.raw_tril, -1, 0)

        diag = tf.linalg.diag_part(tril)
        diag = tf.nn.softplus(diag) + self.tmp
        tril = tf.linalg.set_diag(tril, diag)

        dist = tfd.MultivariateNormalTriL(
            loc=self.mean,
            scale_tril=tril
        )

        return tf.expand_dims(dist.log_prob(x), axis=-1)


class Gaussian_layer(tf.keras.layers.Layer):
    def __init__(self,mean,stdev, column_index,dtype=tf.dtypes.float32,log_space=True,trainable_nodes=True,name='Gauss'):#(self,node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32):
        super(Gaussian_layer,self).__init__()
        # with tf.name_scope("%s_%s" % (node.__class__.__name__, node.id)):
        self.mean = tf.Variable(initial_value=mean, dtype=dtype, trainable=trainable_nodes,name=name+'mean')  # name="mean",
        self.stdev = tf.Variable(initial_value=stdev, dtype=dtype, trainable=trainable_nodes,name=name+'std')  # name="stdev",
        self.tmp=tf.Variable(initial_value= 0.001, dtype=dtype, trainable=False,name=name+'tmp')
        self.trainable_nodes=trainable_nodes
        self.column_index = column_index
        #variable_dict[node] = (self.mean, self.stdev)

        #self.log_space=log_space
    '''
    def call(self, inputs,nd_idxs):
        new_input = tf.gather_nd(inputs, nd_idxs)
        new_input = tf.reshape(new_input, (tf.shape(inputs)[0], 1))
        #self.stdev = tf.maximum(self.stdev,self.tmp)
        if self.trainable_nodes:
            stdev = tf.keras.layers.maximum([self.stdev, self.tmp])
        else:
            stdev=self.stdev
        self.dist = tfd.Normal(loc=self.mean, scale=stdev)
        #if self.log_space:
        result = self.dist.log_prob(new_input)
        #else:
        #    result = self.dist.prob(new_input)
        # print('gauss result shape', result.shape)
        return result
    '''

    #def call(self, inputs, nd_idxs):
    def call(self, inputs):
        #x = tf.gather_nd(inputs, nd_idxs)
        x = tf.gather(inputs, indices=self.column_index, axis=1)
        x = tf.reshape(x, (tf.shape(inputs)[0], 1))

        if self.trainable_nodes:
            stdev = tf.maximum(self.stdev, self.tmp)
        else:
            stdev = self.stdev
        var = tf.square(stdev)

        return -0.5 * (
                tf.square((x - self.mean) / stdev)
                + tf.math.log(2.0 * np.pi * var)
        )

class Categorical_layer(tf.keras.layers.Layer):#softmaxInverse
    def __init__(self, prob,column_index,  log_space=True,name='Categorical'):
        super(Categorical_layer,self).__init__()
        self.probs = tf.Variable(initial_value=prob, trainable=False, dtype=tf.dtypes.float32, name=name)
        self.column_index = column_index


    #def call(self, inputs,nd_idxs):
    def call(self, inputs):
        #new_input = tf.gather_nd(inputs, nd_idxs)
        new_input = tf.gather(inputs, indices=self.column_index, axis=1)
        new_input = tf.reshape(new_input, (tf.shape(inputs)[0], 1))
        softmax_probs=tf.nn.softmax(self.probs)#+1e-3
        return tfd.Categorical(probs=softmax_probs,dtype=tf.dtypes.float32,allow_nan_stats=False).log_prob(new_input)

def categorical_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32,trainable_leaf=False):
    #nd_idxs = get_batch_idx(node, data_placeholder)
    p = np.array(node.p, dtype=dtype)
    #print('ORG node categorical',node.id,node.p)
    softmaxInverse = np.log(p / np.max(p)).astype(dtype)
    #layer = Categorical_layer(p, log_space,name=node.__class__.__name__+str(node.id))
    layer = Categorical_layer(softmaxInverse, log_space=log_space,
                              column_index=node.scope[0],
                              name=node.__class__.__name__ + str(node.id))
    variable_dict[node] = layer.probs
    #return layer(data_placeholder,nd_idxs)
    return layer(data_placeholder)
def gaussian_to_tf_graph(node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32,trainable_leaf=True):
    #nd_idxs=get_batch_idx(node,data_input)
    layer=Gaussian_layer(node.mean,node.stdev,column_index=node.scope[0],
                         trainable_nodes=trainable_leaf,name=node.__class__.__name__+str(node.id))
    variable_dict[node] = (layer.mean, layer.stdev)
    #return layer(data_input,nd_idxs)
    return layer(data_input)


def multivariate_gaussian_to_tf_graph(node, data_input=None, log_space=True,variable_dict=None, dtype=tf.float32,trainable_leaf=True):

    layer = MultivariateGaussian_layer(node.mean,node.sigma,scope=node.scope,
                                       trainable_nodes=trainable_leaf,name=node.__class__.__name__ + str(node.id))
    variable_dict[node] = (layer.mean, layer.sigma)
    return layer(data_input)


# @jit("float64[:](float64[:], float64[:], float64[:,:])", nopython=True)
def histogram_ll(breaks, densities, data):
    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        if x < breaks[0] or x >= breaks[-1]:
            continue
        probs[i] = densities[bisect.bisect(breaks, x) - 1]

        # j = 0
        # for b in breaks:
        #    if b > x:
        #        break
        #    j += 1
        #
        # probs[i] = densities[j - 1]

    probs[probs < EPSILON] = EPSILON

    return probs
def histogram_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = histogram_ll(np.array(node.breaks), np.array(node.densities), nd[~marg_ids])

    return np.log(probs)


class Histogram_layer(tf.keras.layers.Layer):
    def __init__(self,breaks, densities,hll,column_index , dtype=tf.dtypes.float32, log_space=True, trainable_nodes=True, name='Histo'):
        super(Histogram_layer, self).__init__()
        #self.lls = tf.constant(hll).astype(dtype)
        self.breaks=tf.Variable(initial_value=breaks, trainable=False, dtype=dtype, name='break'+name)
        self.densities=tf.Variable(initial_value=densities, trainable=False, dtype=dtype, name='density'+name)
        self.lls = tf.Variable(initial_value=hll, trainable=False, dtype=dtype, name=name)
        self.column_index = column_index

    def call(self, inputs):
        '''
        probs_numpy = np.zeros((inputs.shape[0], 1))

        for i, x in enumerate(inputs):
            if x < self.breaks[0] or x >= self.breaks[-1]:
                continue
            probs_numpy[i] = self.densities[bisect.bisect(self.breaks, x) - 1]

        probs_numpy[probs_numpy < EPSILON] = EPSILON
        '''
        #######################################################################

        # Ensure inputs is a 1D tensor
        x = tf.gather(inputs, indices=self.column_index, axis=-1)

        # Find bin indices for each input
        bin_indices = tf.searchsorted(self.breaks, x, side='right') - 1

        # Mask out-of-bound inputs
        valid_mask = tf.logical_and(
            x >= self.breaks[0],
            x < self.breaks[-1]
        )

        # Clip indices to valid range
        bin_indices = tf.clip_by_value(bin_indices, 0, tf.shape(self.densities)[0] - 1)

        # Get densities using bin indices
        selected_densities = tf.gather(self.densities, bin_indices)

        # Apply epsilon for stability
        probs = tf.where(valid_mask, selected_densities, tf.constant(EPSILON, dtype=self.densities.dtype))
        probs = tf.maximum(probs, EPSILON)

        # return log density:
        #eps = 1e-9
        #log_prob = tf.math.log(tf.clip_by_value(probs, eps, 1.0))
        log_prob = tf.math.log(probs)

        # Reshape to (batch_size, 1)
        return tf.expand_dims(log_prob, axis=-1)







        #return probs
        #new_input = tf.gather_nd(inputs, nd_idxs)
        #new_input = tf.reshape(new_input, (tf.shape(inputs)[0], 1))



        # inputs: [batch_size, num_features]
        '''
        # darmstadt solution
        col = inputs[:, self.column_index]
        col = tf.cast(col, tf.int32)

        # Gather precomputed likelihoods
        result = tf.gather(self.lls, col)
        tmp= tf.squeeze(result, axis=-1)
        return tmp
        '''


def histogram_to_tf_graph(node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32,trainable_leaf=True):
    #nd_idxs = get_batch_idx(node, data_input)
    inps = np.arange(int(max(node.breaks))).reshape((-1, 1))
    tmpscope = node.scope[0]
    node.scope[0] = 0
    hll = histogram_log_likelihood(node, inps)
    node.scope[0] = tmpscope
    #col = data_input[:, node.scope[0]]
    column_index = node.scope[0]

    breaks, densities=node.breaks, node.densities

    layer = Histogram_layer(breaks, densities,hll,column_index,trainable_nodes=trainable_leaf,name=node.__class__.__name__+str(node.id))
    variable_dict[node] = (layer.lls)
    return layer(data_input)



'''
    idxs = [node.scope[0]]
    idxs = tf.constant(idxs, tf.int32)
    batch_idxs = tf.range(
        tf.shape(data_placeholder)[0])  # tf.math.floordiv(tf.range(0, batch_size * num_idxs), num_idxs)[:, None]
    idx_list = tf.tile(idxs, tf.shape(data_placeholder)[0:1])
    batch_idxs = tf.expand_dims(batch_idxs, axis=1)
    idx_list = tf.expand_dims(idx_list, axis=1)
    nd_idxs = tf.concat([batch_idxs, idx_list], axis=1)
    print('DEBUG',node.__class__.__name__+str(node.id),'scope=',node.scope)
'''
def get_batch_idx(node,data_input):
    idxs = [node.scope[0]]
    idxs = tf.constant(idxs, tf.int32)
    batch_idxs = tf.range(tf.shape(data_input)[0])#tf.math.floordiv(tf.range(0, batch_size * num_idxs), num_idxs)[:, None]
    idx_list=tf.tile(idxs,tf.shape(data_input)[0:1])
    batch_idxs=tf.expand_dims(batch_idxs,axis=1)
    idx_list = tf.expand_dims(idx_list, axis=1)
    nd_idxs = tf.concat([batch_idxs,idx_list ], axis=1)
    #print('DEBUG', node.__class__.__name__ + str(node.id), 'scope=', node.scope)
    return nd_idxs

class Log_sum_layer(tf.keras.layers.Layer):
    def __init__(self, softmax_inverse,dtype=tf.dtypes.float32,name='log_sum'):#, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
        super(Log_sum_layer,self).__init__()
        # Calculate softmax inverse
        #softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)

        # Convert softmax_inverse to TensorFlow tensor and apply softmax
        self.tf_weights=tf.Variable(initial_value=softmax_inverse, dtype=dtype, trainable=True,name=name)
        # Store weights in variable_dict
        # if variable_dict is not None:
        #variable_dict[node] = self.tf_weights



    def call(self, inputs):

        #for child in inputs:
            #print('child shape', child.shape)
        tf_weights = tf.nn.softmax(self.tf_weights)  # , name="weights"
        tf_weights=tf.expand_dims(tf.expand_dims(tf_weights,axis=0),axis=-1)
        # Stack children along axis 1
        children_prob = tf.stack(inputs, axis=1)#children
        #print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        out=tf.reduce_logsumexp(children_prob + tf.math.log(tf_weights), axis=1)
        #print('out sum shape',out.shape)
        return out
class Log_prod_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Log_prod_layer,self).__init__()

    def call(self, inputs):
        #for i in inputs:
            #print('inputs shape ', i.shape)
        out=tf.add_n(inputs)
        #print('out prod shape',out.shape)
        return out





def log_prod_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32,trainable_leaf=False):
    assert log_space
    layer=Log_prod_layer()
    return layer(children)#tf.add_n(children)

def log_sum_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32,trainable_leaf=False):
    assert log_space
    # Calculate softmax inverse
    #print('ORG node log sum',node.id,node.weights)
    softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)
    layer=Log_sum_layer(softmax_inverse,name=node.__class__.__name__+str(node.id))

    # Convert softmax_inverse to TensorFlow tensor and apply softmax
    #tf_weights = tf.nn.softmax(tf.Variable(initial_value=softmax_inverse, dtype=dtype,trainable=True))#, name="weights"

    # Store weights in variable_dict
    #if variable_dict is not None:
    variable_dict[node] = layer.tf_weights

    # Stack children along axis 1
    #for child in children:
    #    print('child shape',child.shape)
    #children_prob = tf.stack(children, axis=1)

    # Calculate log-sum-exp
    #return layer.call(children)#tf.reduce_logsumexp(children_prob + tf.math.log(tf_weights), axis=1)
    return  layer(children)

