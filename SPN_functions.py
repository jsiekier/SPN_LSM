import numpy as np
import time
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.MPE import mpe
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian,MultivariateGaussian
from spn.structure.Base import Context,Sum, assign_ids
from sklearn.metrics import accuracy_score
from dpspn.dpspn import learn_dp_mspn
from sys import maxsize
from scipy import stats
from diffprivlib.mechanisms import GeometricTruncated
from diffprivlib.accountant import BudgetAccountant





def normalization_embedd(train_x, valid_x, test_x, low_variance_threshold=0.001, normalization=False,
                         zmuv_min_sigma=0.001):
    v = np.var(train_x, 0)
    if len(v) > 0:
        mu = np.mean(v)
    else:
        mu = 0
    idx = v > low_variance_threshold * mu
    train_x = train_x[:, idx]
    test_x = test_x[:, idx]
    if valid_x is not None:
        valid_x = valid_x[:, idx]
    print('Number of used features', idx.shape)

    # zero-mean, unit-variance
    if normalization:
        if len(train_x) > 0:
            train_x_mean = np.mean(train_x, 0)
        else:
            train_x_mean = 0
        train_x_std = np.std(train_x, 0)

        train_x = (train_x - train_x_mean) / (train_x_std + zmuv_min_sigma)
        test_x = (test_x - train_x_mean) / (train_x_std + zmuv_min_sigma)
        if valid_x is not None:
            valid_x = (valid_x - train_x_mean) / (train_x_std + zmuv_min_sigma)

    return train_x, valid_x, test_x


def learn_classifier(data, debugging,ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    label_ids=[]
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        branch = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, **kwargs)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])
        label_ids.append(label)

    spn.scope.extend(branch.scope)
    assign_ids(spn)
    return spn,label_ids

def learn_joined_prob_distribution(data, debugging,ds_context, spn_learn_wrapper, **kwargs):
    spn = spn_learn_wrapper(data, ds_context, **kwargs)
    assign_ids(spn)
    return spn

def find_max_index(lst):
    if not lst:
        return None
    max_index = 0
    max_element = lst[0].total()

    for i in range(1, len(lst)):
        if lst[i].total() > max_element:
            max_element = lst[i].total()
            max_index = i

    return max_index


def learn_dp_classifier(epsilon,data, debugging,ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    label_ids=[]

    dp_mech = GeometricTruncated(epsilon=epsilon, sensitivity=1, lower=0, upper=maxsize)
    max_op = []
    accountants=[]

    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        acc = BudgetAccountant()
        branch, max_op_on_dataset_slice = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context,epsilon=epsilon, **kwargs)
        spn.children.append(branch)
        label_ids.append(label)
        # dp
        count = np.uint32(dp_mech.randomise(count))
        spn.weights.append(count / data.shape[0])

        max_op.append(max_op_on_dataset_slice)
        acc.spend(epsilon=(max_op_on_dataset_slice*epsilon),delta= 0)
        print(acc.total())
        accountants.append(acc)

    weightsum = np.sum(spn.weights)
    for i in range(len(spn.weights)):
        spn.weights[i] =spn.weights[i] / weightsum

    final_accountant = accountants[find_max_index(accountants)]

    spn.scope.extend(branch.scope)
    assign_ids(spn)
    final_accountant.spend(epsilon, 0)
    return spn,label_ids,max(max_op) +1

def create_SPN(param_grid,train_data,test_data,test_y,input_dim,debugging):
    train_start=time.time()
    parametric_types=[Categorical]+[Gaussian for _ in range(input_dim)]
    min_instances_slice=int(param_grid.min_instances_slice_percentage*train_data.shape[0])
    print('min_instances_slice',min_instances_slice,train_data.shape[0])
    spn_classification,label_ids = learn_classifier(train_data,debugging,
                                          Context(parametric_types=parametric_types).add_domains(
                                              train_data),
                                          learn_parametric, 0,
                                          min_instances_slice=min_instances_slice,
                                          min_features_slice=param_grid.min_features_slice,
                                          multivariate_leaf=False,
                                          leaves=None,
                                          n_clusters=param_grid.n_clusters,
                                          cols=param_grid.col_split,
                                          rows=param_grid.row_split
                                          )

    prediction= mpe(spn_classification, test_data)[:, 0]
    acc=accuracy_score(test_y, prediction, normalize=True)
    #print('mpe example',list(zip(prediction,test_y))[:100],flush=True)
    print('TEST Result Acc after SPN training', acc,'time:',(time.time()-train_start)//60,flush=True)
    #train_data_test_y=train_data[:,0]
    #train_data_test=np.concatenate([np.full((train_data.shape[0], 1), np.nan),train_data[:,1:]],axis=-1)
    #prediction_train= mpe(spn_classification, train_data_test)[:, 0]
    #acc=accuracy_score(train_data_test_y, prediction_train, normalize=True)
    #print('mpe example',list(zip(prediction_train,train_data_test_y))[:100],flush=True)
    #print('Train','SPN Result Acc after SPN training', acc,'train+eval time:',(time.time()-train_start)//60,flush=True)
    return spn_classification,label_ids




def create_SPN_diff_privacy(param_grid,train_data,test_data,test_y,input_dim,debugging):
    train_start=time.time()
    parametric_types=[Categorical]+[Gaussian for _ in range(input_dim)]
    min_instances_slice=int(param_grid.min_instances_slice_percentage*train_data.shape[0])
    print('min_instances_slice',min_instances_slice,train_data.shape[0])
    epsilon=param_grid.diff_privacy_epsilon
    spn_classification,label_ids,operations = learn_dp_classifier(epsilon,train_data,debugging,
                                          Context(parametric_types=parametric_types).add_domains(
                                              train_data),
                                          learn_dp_mspn,
                                                    0,
                                          min_instances_slice=min_instances_slice,
                                          #min_features_slice=param_grid.min_features_slice,
                                          #multivariate_leaf=False,
                                          leaves=None,
                                          #n_clusters=param_grid.n_clusters,
                                          cols=param_grid.col_split,
                                          rows=param_grid.row_split,
                                          total_operations =param_grid.total_operations - 1
                                          )

    prediction= mpe(spn_classification, test_data)[:, 0]
    acc=accuracy_score(test_y, prediction, normalize=True)
    print('DEBUG mpe result vs y',prediction[:5],test_y[:5])
    #print('mpe example',list(zip(prediction,test_y))[:100],flush=True)
    print('TEST Result Acc after SPN training', acc,'time:',(time.time()-train_start)//60,flush=True)

    return spn_classification,label_ids



def create_CSPN(param_grid,train_data,test_data,test_y,input_dim,debugging):
    train_start=time.time()
    min_instances_slice=int(param_grid.min_instances_slice_percentage*train_data.shape[0])
    print('min_instances_slice',min_instances_slice,train_data.shape[0])

    # 2 - train SPN family to describe different decision boundaries and do regression:
    if param_grid.min_features_slice == 1:
        parametric_types = [Categorical] + [Gaussian for _ in range(input_dim)]
    else:
        parametric_types = [Categorical] + [MultivariateGaussian for _ in range(input_dim)]


    train_y=train_data[:,0]
    # create decision boundaries:
    sorted_y_train=np.sort(train_y)
    num_points=train_y.shape[0]//(param_grid.regression+1)
    num_samples_per_bin=num_points if train_y.shape[0]%(param_grid.regression+1)< num_points*0.5 else num_points+1
    bin_border_idx=[(i * num_samples_per_bin)-1 for i in range(1,param_grid.regression+1)]
    bin_borders= sorted_y_train[bin_border_idx]
    all_clf_spns=[]
    all_label_ids=[]
    bin_widths=[sorted_y_train[0],bin_borders[0]-sorted_y_train[0]]
    for i in range(1,len(bin_borders)-1):
        bin_widths.append(bin_borders[i]-bin_borders[i-1])
    bin_widths.append(sorted_y_train[-1]-bin_borders[-2])
    bin_widths=np.asarray(bin_widths)
    max_y=1#np.max(train_y)
    #scaling_factor = max_y / np.sum(bin_widths)

    # Scale the array
    #bin_widths_scaled = bin_widths * scaling_factor
    bin_widths_scaled=bin_widths
    print('max reachable prediction:',sum(bin_widths),'max y on train:',np.max(train_y))

    pred_smooth=[np.ones(shape=test_y.shape)*bin_widths_scaled[0]]

    for bin_border,bin_widths_s in zip(bin_borders,bin_widths_scaled[1:]):
        # transfer train class labels:
        sub_train_class_labels = np.expand_dims((train_y >= bin_border).astype(float),axis=-1)
        sub_train_data=np.concatenate([sub_train_class_labels,train_data[:,1:]],axis=-1)



        spn_classification,label_ids = learn_classifier(sub_train_data,debugging,
                                              Context(parametric_types=parametric_types).add_domains(
                                                  sub_train_data),
                                              learn_parametric, 0,
                                              min_instances_slice=min_instances_slice,
                                              min_features_slice=param_grid.min_features_slice,
                                              multivariate_leaf=False if param_grid.min_features_slice==1 else True ,
                                              leaves=None,
                                              n_clusters=param_grid.n_clusters,
                                              cols=param_grid.col_split,
                                              rows=param_grid.row_split,
                                              threshold=0.1
                                              )

        sub_prediction= mpe(spn_classification, test_data)[:, 0]
        all_clf_spns.append(spn_classification)
        all_label_ids.append(label_ids)
        pred_smooth.append(sub_prediction*bin_widths_s)
    prediction=np.asarray(pred_smooth)
    prediction=np.sum(prediction,axis=0)
    print('DEBUG mpe result vs y',prediction[:5],test_y[:5])
    MAE =np.mean(np.abs(test_y-prediction))
    MSE=np.mean(np.abs(np.square(test_y-prediction)))
    res = stats.spearmanr(test_y, prediction)








    # 1. train SPN to describe the embedding space
    if param_grid.min_features_slice == 1:
        parametric_types=[Gaussian for _ in range(input_dim)]
        print('USE UNIVAIATE GAUSSIAN')
    else:
        print('USE MULTIVAIATE GAUSSIAN')
        parametric_types=[MultivariateGaussian for _ in range(input_dim)]
    spn_z= learn_joined_prob_distribution(train_data[:,1:],debugging,
                                          Context(parametric_types=parametric_types).add_domains(
                                              train_data[:,1:]),
                                          learn_parametric,
                                          min_instances_slice=min_instances_slice,
                                          min_features_slice=param_grid.min_features_slice,
                                          multivariate_leaf=False if param_grid.min_features_slice==1 else True ,
                                          leaves=None,
                                          n_clusters=param_grid.n_clusters,
                                          cols=param_grid.col_split,
                                          rows=param_grid.row_split,
                                          threshold=0.3
                                          )





    print('TEST Result MAE|MSE after SPN training',MAE, MSE,res,'time:',(time.time()-train_start)//60,flush=True)
    return all_clf_spns, all_label_ids,bin_borders,bin_widths_scaled,spn_z






