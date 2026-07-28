import tensorflow.compat.v2 as tf
print(tf.__version__)
from tensorflow_probability import distributions as tfd
import numpy as np
from spn.structure.leaves.parametric.Parametric import Gaussian,MultivariateGaussian
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.histogram.Histograms import Histogram

from spn.structure.Base import Sum, Product, Leaf,get_topological_order
from spn.algorithms.TransformStructure import Copy
import tf_SPN_layers as SPN_lay
#tf.compat.v1.disable_eager_execution()


def categorical_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    #with tf.name_scope("%s_%s" % (node.__class__.__name__, node.id)):
    p = np.array(node.p, dtype=dtype)
    softmaxInverse = np.log(p / np.max(p)).astype(dtype)
    probs = tf.nn.softmax(tf.Variable(initial_value=tf.constant(softmaxInverse)))#name="p",
    variable_dict[node] = probs
    idxs = [node.scope[0]]


    #print('scope list:', idxs)
    idxs = tf.constant(idxs, tf.int32)
    #print('scope list:', idxs.shape)


    batch_idxs = tf.range(tf.shape(data_placeholder)[0])#tf.math.floordiv(tf.range(0, batch_size * num_idxs), num_idxs)[:, None]
    idx_list=tf.tile(idxs,tf.shape(data_placeholder)[0:1])
    #print('batch idx shape',batch_idxs.shape)
    #print('idx list shape',idx_list.shape)
    batch_idxs=tf.expand_dims(batch_idxs,axis=1)
    idx_list = tf.expand_dims(idx_list, axis=1)

    nd_idxs = tf.concat([batch_idxs,idx_list ], axis=1)
    #print('idx list shape', nd_idxs.shape)
    categorical_distribution=tfd.Categorical(probs=probs)
    new_input=tf.gather_nd(data_placeholder, nd_idxs)
    new_input=tf.reshape(new_input,(tf.shape(data_placeholder)[0],1))
    #print('new input shape:',new_input.shape)


    if log_space:
        #return tf.compat.v1.distributions.Categorical(probs=probs).log_prob()
        return categorical_distribution.log_prob(new_input)
    return categorical_distribution.prob(new_input)

        #return tf.compat.v1.distributions.Categorical(probs=probs).prob(tf.gather_nd(data_placeholder, indices))

def gaussian_to_tf_graph(node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32,trainable=True):

    #with tf.name_scope("%s_%s" % (node.__class__.__name__, node.id)):
    mean = tf.Variable( initial_value=node.mean, dtype=dtype,trainable=trainable)#name="mean",
    stdev = tf.Variable( initial_value=node.stdev, dtype=dtype,trainable=trainable)#name="stdev",
    variable_dict[node] = (mean, stdev)
    if trainable:
        stdev = tf.maximum(stdev, 0.001)
    dist=tfd.Normal(loc=mean, scale=stdev)



    idxs = [node.scope[0]]


    #print('scope list:', idxs)
    idxs = tf.constant(idxs, tf.int32)
    #print('scope list:', idxs.shape)


    batch_idxs = tf.range(tf.shape(data_input)[0])#tf.math.floordiv(tf.range(0, batch_size * num_idxs), num_idxs)[:, None]
    idx_list=tf.tile(idxs,tf.shape(data_input)[0:1])

    batch_idxs=tf.expand_dims(batch_idxs,axis=1)
    idx_list = tf.expand_dims(idx_list, axis=1)

    nd_idxs = tf.concat([batch_idxs,idx_list ], axis=1)


    new_input=tf.gather_nd(data_input, nd_idxs)
    new_input=tf.reshape(new_input,(tf.shape(data_input)[0],1))



    #print('DEBUG', tmp1.shape, tmp2.shape,indices.shape,'scope',node.scope)
    if log_space:
        result=dist.log_prob(new_input)
    else:
        result = dist.prob(new_input)
    #print('gauss result shape',result.shape)
    return result
def log_prod_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    return tf.add_n(children)

def log_sum_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    # Calculate softmax inverse
    softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)

    # Convert softmax_inverse to TensorFlow tensor and apply softmax
    tf_weights = tf.nn.softmax(tf.Variable(initial_value=softmax_inverse, dtype=dtype,trainable=True))#, name="weights"

    # Store weights in variable_dict
    #if variable_dict is not None:
    variable_dict[node] = tf_weights

    # Stack children along axis 1
    #for child in children:
    #    print('child shape',child.shape)
    children_prob = tf.stack(children, axis=1)

    # Calculate log-sum-exp
    return tf.reduce_logsumexp(children_prob + tf.math.log(tf_weights), axis=1)


_node_log_tf_graph = {Sum: SPN_lay.log_sum_to_tf_graph,
                      Product:  SPN_lay.log_prod_to_tf_graph,
                      Gaussian:SPN_lay.gaussian_to_tf_graph,
                      Categorical:SPN_lay.categorical_to_tf_graph,
                      Histogram:SPN_lay.histogram_to_tf_graph,
                      MultivariateGaussian: SPN_lay.multivariate_gaussian_to_tf_graph}
#_node_log_tf_graph = {Sum: log_sum_to_tf_graph, Product: log_prod_to_tf_graph,Gaussian:gaussian_to_tf_graph,Categorical:categorical_to_tf_graph}
def tf_graph_to_sum(node, tfvar):
    #weights=np.squeeze(tfvar.numpy(),axis=(0,2)).tolist()
    weights=tf.nn.softmax(tfvar).numpy().tolist()
    node.weights = weights
    #print('graph to sum',node.id,weights)
def tf_graph_to_gaussian(node, tfvar):
    node.mean = tfvar[0].numpy()
    node.stdev = np.maximum(tfvar[1].numpy(),0.01)
    #print('graph to gaussian', node.id, node.mean,node.stdev)
def tf_graph_to_categorical(node, tfvar):
    node.p= (tf.nn.softmax(tfvar).numpy()+1e-3).tolist()
    #node.p = list(tfvar.numpy())
    #print('graph to categorical', node.id,node.p)

_tf_graph_to_node = {Sum: tf_graph_to_sum,Gaussian:tf_graph_to_gaussian,Categorical:tf_graph_to_categorical}

def spn_to_tf_graph(node,tf_input, eval_functions=_node_log_tf_graph, **args):
    #tf.reset_default_graph()
    #if not dtype:
    #    dtype = data.dtype

    all_results = {}
    nodes = get_topological_order(node)
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)
    leaf_func = eval_functions.get(Leaf, None)

    tmp_children_list = []
    len_tmp_children_list = 0

    #print('Node list length:',len(nodes))
    for n_idx,n in enumerate(nodes):
        #print('Curr node',n)
        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError("No lambda function associated with type: %s" % (n.__class__.__name__))

        if n_is_leaf:
            result = func(n,tf_input, **args)#(tf_input)
        else:
            #print('INNER NODE')
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci = n.children[i]
                tmp_children_list[i] = all_results[ci]
                #print('children',ci.__class__.__name__+str(ci.id))
            #print('len children',len_children)
            result = func(n, tmp_children_list[0:len_children], **args)#(tmp_children_list[0:len_children])

        all_results[n] = result
    #'''
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")
    #'''
    tf_graph=all_results[node]

    return tf_graph

def create_tf_spn(spn, data_shape):
    # Make sure, that the passed SPN is not modified
    spn_copy = Copy(spn)
    variable_dict = {}
    tf_input = tf.keras.Input(shape=data_shape)
    graph = spn_to_tf_graph(spn_copy, tf_input, variable_dict=variable_dict)
    return tf.keras.Model(inputs=tf_input, outputs=graph),variable_dict,spn_copy


def create_multiple_tf_spn_parts(spn_out_data,num_classes,data_shape,trainable_leaf):
    if num_classes>1:
        spn_x_copy,all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model=  create_tf_spn_parts(spn_x=spn_out_data[0],
                                   data_shape=data_shape,
                                   label_ids=spn_out_data[1],
                                   trainable_leaf=trainable_leaf)
        return all_spn_x_y_model,all_prior,None
    else:
        all_clf_spns, all_label_ids,bin_borders,bin_widths_scaled,spn_z=spn_out_data
        # 1. create z spn:
        variable_dict_z = {}
        print(data_shape)
        tf_input = tf.keras.Input(shape=(data_shape[0]-1,))
        graph = spn_to_tf_graph(spn_z, tf_input, variable_dict=variable_dict_z,trainable_leaf=trainable_leaf)
        spn_model_z=tf.keras.Model(inputs=tf_input, outputs=graph,name='z_distribution')
        # 2. create regression SPN family:
        family_all_spn_x_y_model, family_all_prior=[],[]
        for spn_idx,(clf_spns,label_ids) in enumerate(zip(all_clf_spns, all_label_ids)):
            spn_x_copy, all_spn_x_y, all_spn_x_y_dicts, all_prior, all_spn_x_y_model = create_tf_spn_parts(
                spn_x=clf_spns,
                data_shape=data_shape,
                label_ids=label_ids,
                trainable_leaf=trainable_leaf,spn_name='SPN_'+str(spn_idx)+'_')
            family_all_spn_x_y_model.append(all_spn_x_y_model)
            family_all_prior.append(all_prior)

        return family_all_spn_x_y_model, family_all_prior,spn_model_z

def create_tf_spn_parts(spn_x, data_shape,label_ids,trainable_leaf,spn_name=''):
    # Make sure, that the passed SPN is not modified
    all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model=[],[],[],[]
    spn_x_copy = Copy(spn_x)
    sorted_list = list(zip(spn_x.children, spn_x.weights, label_ids))
    #sorted_list=list(sorted(zip(spn_x.children,spn_x.weights,label_ids),key=lambda x:x[2]))
    print('SPN ROOT weights:',spn_x.weights)
    for spn_x_y,prior_y,label_id in sorted_list:
        variable_dict = {}
        tf_input = tf.keras.Input(shape=data_shape)
        graph = spn_to_tf_graph(spn_x_y, tf_input, variable_dict=variable_dict,trainable_leaf=trainable_leaf)
        spn_model=tf.keras.Model(inputs=tf_input, outputs=graph,name=spn_name+'label_id'+str(int(label_id)))
        all_spn_x_y.append(spn_x_y)
        all_spn_x_y_dicts.append(variable_dict)
        all_prior.append(prior_y)
        all_spn_x_y_model.append(spn_model)

    #tf_input = tf.keras.Input(shape=len(label_ids))
    #tf_spn_root_layer= SPN_lay.Root_Layer(all_prior, all_spn_x_y_model)(tf_input)
    #tf_spn_root=tf.keras.Model(inputs=tf_input, outputs=tf_spn_root_layer,name='SPN')




    return spn_x_copy,all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model#,tf_spn_root
def tf_graph_to_spn(variable_dict, tf_graph_to_node=_tf_graph_to_node):
    tensors = []

    for n, tfvars in variable_dict.items():
        tensors.append(tfvars)
    #intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=tensors)
    #variable_list = #tf.get_default_session().run(tensors)

    for i, (n, tfvars) in enumerate(variable_dict.items()):
        tf_graph_to_node[type(n)](n, tensors[i])#variable_list[i]

