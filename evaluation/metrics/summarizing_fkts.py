import numpy as np


def rescale_to_01(arr):
    min_val = np.min(arr,axis=1,keepdims=True)
    max_val = np.max(arr,axis=1,keepdims=True)
    scaled=(arr - min_val) / (max_val - min_val) #if max_val > min_val else np.zeros_like(arr)
    scaled=np.nan_to_num(scaled,nan=0.0)
    scaled=scaled/np.sum(scaled,axis=1,keepdims=True)
    scaled = np.nan_to_num(scaled, nan=1/scaled.shape[1])
    return scaled



def mean_cf(x_cf_,additional_data,argmax_arr_,z_cf_,z_):
    x_cf=np.copy(x_cf_)
    z_cf = np.copy(z_cf_)
    z = np.copy(z_)
    argmax_arr=np.copy(argmax_arr_)
    x_cf_mean = np.mean(x_cf, axis=1)
    z_cf_mean = np.mean(z_cf, axis=1)
    z_mean = np.mean(z, axis=1)
    arg_max_mean = np.mean(argmax_arr, axis=1)
    return x_cf_mean,arg_max_mean,z_cf_mean,z_mean

def max_cf(x_cf, additional_data,argmax_arr,z_cf,z):
    #TODO this works only for y_cf_goal =1 - the general case has still to be implemented!
    pred, y_cf_goal,num_classes=additional_data
    if num_classes>1:
        arg_max = np.argmax(pred[:,:,y_cf_goal], axis=1)[:,0]  # Get the max indices along axis 1
    else:
        arg_max = np.argmin(pred-np.expand_dims(y_cf_goal,axis=-1), axis=1)#[:, 0]

    x_cf_mean = x_cf[np.arange(x_cf.shape[0]), arg_max]  # Use advanced indexing
    z_cf_mean =  z_cf[np.arange(z_cf.shape[0]), arg_max]
    z_mean =  z[np.arange(z.shape[0]), arg_max]
    arg_max_mean =  argmax_arr[np.arange(x_cf.shape[0]), arg_max]
    return x_cf_mean,arg_max_mean,z_cf_mean,z_mean

def min_loss(x_cf, additional_data,argmax_arr,z_cf_,z_):
    arg_max = np.argmin(additional_data, axis=1)  # Get the max indices along axis 1
    x_cf_mean = x_cf[np.arange(x_cf.shape[0]), arg_max]  # Use advanced indexing
    arg_max_mean =argmax_arr[np.arange(x_cf.shape[0]), arg_max]
    z_cf_mean =  z_cf_[np.arange(z_cf_.shape[0]), arg_max]
    z_mean =  z_[np.arange(z_.shape[0]), arg_max]
    return x_cf_mean,arg_max_mean,z_cf_mean,z_mean
def weighted(x_cf, additional_data,argmax_arr,z_cf_,z_):
    # create weights in the area [0,1]
    rescale=rescale_to_01(additional_data)
    #print(rescale.shape)
    arg_max_mean = np.sum(argmax_arr*rescale, axis=1)
    upscale_x=np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(rescale,axis=-1),axis=-1),axis=-1) ,axis=-1)
    upscale_z =np.expand_dims(rescale, axis=-1)
    #print(x_cf.shape,upscale_x.shape,z_cf_.shape,upscale_z.shape,z_.shape)
    x_cf_mean = np.sum(x_cf*upscale_x, axis=1)
    z_cf_mean =  np.sum(z_cf_*upscale_z, axis=1)
    z_mean =  np.sum(z_*upscale_z, axis=1)
    return x_cf_mean,arg_max_mean,z_cf_mean,z_mean