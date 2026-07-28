import re
import os
import pickle as pkl
import numpy as np
from collections import Counter
import random
import torch

from cf_eval.metrics import compute_validity, mean_lsp_calculation, compute_validity_with_std
import sys
import types

from segmentation_analysis_.segmentation_analysis_full_updated_v3 import compute_advanced_cf_metrics, \
    print_cf_metrics_summary, make_cf_metrics_latex_table, save_cf_metrics, debug_anatomy_core_metrics, \
    debug_plot_org_rec_cf_pairs #segmentation_analysis.

tf = types.ModuleType("tensorflow")
sys.modules["tensorflow"] = tf
sys.modules["tensorflow.python"] = types.ModuleType("tensorflow.python")
sys.modules["tensorflow.keras"] = types.ModuleType("tensorflow.keras")


def find_and_extract_files(folder_path,fine_tune=0,possible_values=[]):
    #pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new\.pkl")
    if fine_tune:
        pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new_minmax_tmp_fine\.pkl") #_minmax
    else:
        pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new_minmax_tmp\.pkl") #_minmax #TODO change this to _minmax back

    extracted_data = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            beta_str = match.group(1).replace('-', '.')
            gamma_str = match.group(2).replace('-', '.')
            model_name = match.group(3)

            beta = float(beta_str)
            gamma = float(gamma_str)
            #if gamma==0:#beta in possible_values and gamma in possible_values:
            extracted_data.append((filename, model_name, beta, gamma))
    print('evaluated file names',extracted_data)
    return extracted_data

'''
def get_segmentation_results(model, mean_segmentation, std_segmentation, y_data, x_data,device=torch.device('cpu'),num_frames=2,model_plot_name=''):
    all_volumes=[] #TODO change this for btach input
    x_data=x_data*255
    x_data -= mean_segmentation
    x_data /= std_segmentation
    x_data = np.repeat(np.expand_dims(x_data,axis=1), repeats=3, axis=1)

    # TODO add loop here to input only batches (axis 0) in the model
    for frame in range(num_frames):

        model.eval()
        # x=x*255
        x=x_data[:,:,:,:,frame]



        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(device)

            # Run segmentation
            y_logits = model(x_tensor)["out"]  # (N, 1, H, W)
            y_probs = torch.sigmoid(y_logits[:, 0, :, :])  # (N, H, W), probabilities
            y_mask = (y_probs > 0.2).float()  # Binary mask

            # Compute volume (pixel count)
            volume_per_frame = y_mask.cpu().numpy().sum(axis=(1, 2))  # (N,)

            volumes = volume_per_frame
            segmentations = y_mask.cpu().numpy()
            all_volumes.append(volumes)
    #TODO concat/ or stack batchoutput for this further evaluation

    # calculate ef
    all_volumes=np.asarray(all_volumes)
    max_volumes=np.max(all_volumes,axis=0)
    min_volumes = np.min(all_volumes, axis=0)
    ef=(max_volumes-min_volumes)/max_volumes

    validity_MAE, validity_RMSE, validity_spearman = compute_validity(y_data, ef, num_classes=1)
    plot_dict={'ef_{target}':y_data,
               'ef_{segmentation}':ef,
               'strategy':[model_plot_name]*ef.shape[0]}
    return validity_MAE, validity_RMSE, validity_spearman,plot_dict
'''

def get_ef_model_results(model,mean_segmentation,std_segmentation, y_data, x_data,model_plot_name='Original'):
    # Normalize x_data
    x_data = x_data * 255
    print('Min/Max x data val',np.min(x_data),np.max(x_data),model_plot_name)
    x_data = (x_data - mean_segmentation) / std_segmentation
    print('x_data shape ef regression', x_data.shape)
    if len(x_data.shape)==4:
        x_data = np.repeat(np.expand_dims(x_data, axis=1), repeats=3, axis=1)  # (N, 3, H, W, num_frames)
        x_data=np.transpose(x_data,(0,1,4,2,3))


    print(model_plot_name,'x_data shape ef regression',x_data.shape,'GOAL b, c, f, h, w')

    # Convert to torch tensors
    test_x = torch.tensor(x_data, dtype=torch.float32)
    test_y = torch.tensor(y_data, dtype=torch.float32)

    # Create dataset
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    # Create DataLoader (shuffle + batch)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)


    yhat = []
    y = []
    with torch.set_grad_enabled(False):
        for d_idx, (frames, ef_target) in enumerate(dataloader):
            y.append(ef_target.numpy())
            outputs = model(frames) # input shape b, c, f, h, w
            yhat.append(outputs.view(-1).to("cpu").detach().numpy())
    yhat = np.concatenate(yhat)
    y = np.concatenate(y) * 100

    #validity_MAE, validity_RMSE, validity_spearman = compute_validity(y, yhat, num_classes=1)
    #return validity_MAE, validity_RMSE, validity_spearman,yhat/100
    validity_MAE, validity_RMSE, validity_spearman, validity_MAE_std, validity_RMSE_std = (
        compute_validity_with_std(y, yhat, num_classes=1)
    )

    return validity_MAE, validity_RMSE, validity_spearman, yhat / 100, validity_MAE_std, validity_RMSE_std

def get_segmentation_results(
        model, mean_segmentation, std_segmentation, y_data, x_data,
        device=torch.device('cpu'), num_frames=2, batch_size=50, model_plot_name=''
):
    print('Model plot name',model_plot_name)
    """
    Compute segmentation results using mini-batches to avoid OOM.

    Args:
        model: PyTorch segmentation model.
        mean_segmentation: Mean used for normalization.
        std_segmentation: Std used for normalization.
        y_data: Ground truth values (for validity computation).
        x_data: Input images, shape (N, C?, H, W, num_frames).
        device: torch.device.
        num_frames: Number of frames to process per sample.
        batch_size: Number of samples per mini-batch.
        model_plot_name: Label for plotting.

    Returns:
        validity_MAE, validity_RMSE, validity_spearman, plot_dict
    """

    all_volumes = []
    print('segmentation x shape', x_data.shape, np.min(x_data), np.max(x_data))
    # Normalize x_data
    x_data = x_data * 255
    x_data = (x_data - mean_segmentation) / std_segmentation
    if len(x_data.shape)==4:
        x_data = np.repeat(np.expand_dims(x_data, axis=1), repeats=3, axis=1)  # (N, 3, H, W, num_frames)
        #x_data = np.transpose(x_data, (0, 1, 3, 4, 2))
    else:
        x_data = np.transpose(x_data, (0, 1, 3, 4, 2))
    print(model_plot_name,'segmentation x shape',x_data.shape,np.min(x_data),np.max(x_data),'GOAL(N, 3, H, W, num_frames)')


    num_samples = x_data.shape[0]

    model.eval()

    all_y_masks=[]
    with torch.no_grad():

        for frame in range(num_frames):
            y_masks = []
            frame_volumes = []

            # Process in mini-batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                x_batch = x_data[start_idx:end_idx, :, :, :, frame]  # (B, 3, H, W)
                x_tensor = torch.from_numpy(x_batch).float().to(device)

                # Run model
                y_logits = model(x_tensor)["out"]  # (B, 1, H, W)
                #y_probs = torch.sigmoid(y_logits[:, 0, :, :])  # (B, H, W)
                #y_mask = (y_probs > 0.9).float()  # Binary mask

                y_mask = y_logits[:, 0, :, :].detach().cpu().numpy() > 0.

                # Compute volume per sample in batch
                volume_per_sample = y_mask.sum(axis=(1, 2))  # (B,)
                frame_volumes.append(volume_per_sample)
                y_masks.append(y_mask)

            # Concatenate all batch outputs for this frame
            frame_volumes = np.concatenate(frame_volumes, axis=0)  # (N,)
            all_volumes.append(frame_volumes)
            y_masks=np.concatenate(y_masks,axis=0)
            all_y_masks.append(y_masks)
    all_y_masks=np.squeeze(np.stack(all_y_masks,axis=0))
    all_y_masks=np.transpose(all_y_masks,(1,2,3,0))


    # Convert to numpy array, shape (num_frames, N)
    all_volumes = np.asarray(all_volumes)

    # Compute EF metric
    max_volumes = np.max(all_volumes, axis=0)
    min_volumes = np.min(all_volumes, axis=0)
    print(max_volumes)
    print(min_volumes)
    ef = (max_volumes - min_volumes) / max_volumes
    # Create mask that is True where ef is NOT NaN
    mask = ~np.isnan(ef)
    ef_filtered = ef[mask]
    ground_truth_filtered = y_data[mask]
    print(ef_filtered)
    print(ground_truth_filtered)
    #print('ef_filtered.shape(),ground_truth_filtered.shape()',ef_filtered.shape(),ground_truth_filtered.shape())

    # Compute validity metrics
    validity_MAE, validity_RMSE, validity_spearman = compute_validity(ground_truth_filtered*100, ef_filtered *100, num_classes=1)

    plot_dict = {
        'ef_{target}': ground_truth_filtered,
        'ef_{segmentation}': ef_filtered,
        'strategy': [model_plot_name] * ef_filtered.shape[0]
    }

    return validity_MAE, validity_RMSE, validity_spearman, plot_dict,all_y_masks

def load_data(file_path,test_data,num_classes, model_name,work_from,CF_aggregation ):
    print('CF_aggregation',CF_aggregation,flush=True)
    all_data_list = pkl.load(open(file_path, 'rb'))
    x_cf,x_org,x_rec,z,z_cf,arg_max,y_cf_goal,loss,y_org,pred=[],[],[],[],[],[],[],[],[],[]
    for (pred_, opposite_class_, [i_start, i_end], mean_cls_, rep_results) in all_data_list:
        pred.append(pred_)
        y_cf_goal.append(opposite_class_)
        y_org.append(test_data[1][i_start:i_end])
        x_org.append(test_data[0][i_start:i_end])

        #for (rec_cf_, rec_gt_, z_cf_, z_, arg_max_, loss_, log_pred, p_z_diff) in rep_results:
        rec_cf_, rec_gt_, z_cf_, z_, arg_max_, loss_, log_pred, p_z_diff=list(zip(*rep_results))
        x_cf.append(np.asarray(list(rec_cf_)))
        x_rec.append(np.asarray(list(rec_gt_)))
        z.append(np.asarray(list(z_)))
        z_cf.append(np.asarray(list(z_cf_)))
        arg_max.append(np.asarray(list(arg_max_)))
        loss_arr = np.asarray(loss_)

        if loss_arr.ndim == 3:
            loss.append(loss_arr[:, :, 0])
        else:
            loss.append(loss_arr)

    pred = np.concatenate(pred,axis=0)
    y_cf_goal= np.concatenate(y_cf_goal, axis=0)
    y_org = np.concatenate(y_org, axis=0)
    x_org = np.concatenate(x_org, axis=0)

    x_cf=np.concatenate(x_cf,axis=1)
    x_rec = np.concatenate(x_rec, axis=1)
    z = np.concatenate(z, axis=1)
    z_cf = np.concatenate(z_cf, axis=1)
    arg_max = np.concatenate(arg_max, axis=1)
    #print(loss_)
    #print(loss)
    loss = np.concatenate(loss, axis=1)
    #print('loss.shape',loss.shape)
    #loss=np.transpose(loss,(2,1,0))
    #print('loss.shape', loss.shape)

    print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org), np.min(x_rec), np.max(x_rec))
    # transpose:(test_size,2,h,w,1)
    print(arg_max.shape,loss.shape,model_name,arg_max[:2],flush=True)
    arg_max = np.transpose(np.squeeze(np.asarray(arg_max)),(1,0))

    # TODO

    if CF_aggregation=='mean':
        x_cf = np.mean(x_cf, axis=0)
        x_rec = np.mean(x_rec, axis=0)
        if model_name == 'MLP':
            arg_max = np.mean(arg_max, axis=1)
        else:
            arg_max = np.mean(np.exp(arg_max), axis=1)
    elif CF_aggregation=='min_loss':
        #TODO is loss in the right shape?!
        arg_min = np.argmin(loss, axis=0)  # Get the max indices along axis 1
        x_cf = x_cf[arg_min,np.arange(x_cf.shape[1])] #  # Use advanced indexing
        x_rec = x_rec[ arg_min,np.arange(x_rec.shape[1]),]  # Use advanced indexing
        print('DEBUG',loss[0,0:4],loss[0,300:304])

    #x_cf_mean = np.transpose(x_cf_mean, (0, 2, 3, 1, 4))
    #x_org = np.transpose(x_org, (0, 2, 3, 1, 4))
    #x_rec_mean = np.transpose(x_rec_mean, (0, 2, 3, 1, 4))

    print(x_cf.shape, x_org.shape, x_rec.shape)
    x_cf = np.transpose(x_cf, (0, 2, 3, 1, 4))
    #x_org = np.transpose(x_org, (0, 2, 3, 1, 4))
    x_rec = np.transpose(x_rec, (0, 2, 3, 1, 4))
    print(x_cf.shape, x_org.shape, x_rec.shape)
    #(N, 3, H, W, num_frames)
    z = np.transpose(np.asarray(z),(1,0,2))
    z_cf = np.transpose(np.asarray(z_cf),(1,0,2))


    #if num_classes == 1 and model_name == 'SPN':
    #    pred = np.exp(pred)
    # TODO for MLP pred = sigmoid????
    abs_change = np.abs(z_cf - z)
    arg_max_change = np.argmax(abs_change, axis=2)
    if work_from:
        test_idx=-1
    else:
        test_idx=5
    return x_org[:test_idx],x_rec[:test_idx],x_cf[:test_idx],loss[:test_idx],\
        pred[:test_idx],y_cf_goal[:test_idx],z[:test_idx], z_cf[:test_idx],\
        y_org[:test_idx],arg_max[:test_idx],arg_max_change[:test_idx]
def load_data_initial(file_path,test_data,num_classes, model_name ):
    all_data_list = pkl.load(open(file_path, 'rb'))
    all_data = []
    if isinstance(all_data_list[0], list):
        for entry in all_data_list:
            all_data.extend(entry)
    else:
        all_data = all_data_list
        # (opposite_class, coordinates, x, mean_cls, y)
    [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, pred, y_cf_goal,
     coordinates, x_org, mean_cls, y_org] = zip(*all_data)
    print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org), np.min(x_rec), np.max(x_rec))
    x_cf = np.asarray(x_cf)
    x_rec = np.asarray(x_rec)
    x_org = np.asarray(x_org)

    x_cf_mean = np.mean(x_cf, axis=1)
    x_org = np.squeeze(x_org, axis=1)
    x_rec_mean = np.mean(x_rec, axis=1)

    # transpose:(test_size,2,h,w,1)
    x_cf_mean = np.transpose(x_cf_mean, (0, 2, 3, 1, 4))
    x_org = np.transpose(x_org, (0, 2, 3, 1, 4))
    x_rec_mean = np.transpose(x_rec_mean, (0, 2, 3, 1, 4))
    print(x_cf_mean.shape, x_org.shape, x_rec_mean.shape)

    z = np.asarray(z)
    z_cf = np.asarray(z_cf)
    arg_max = np.squeeze(np.asarray(arg_max))

    y_cf_goal = np.asarray(y_cf_goal)
    # x_org= np.mean(x_org, axis=1)

    log_pred = np.squeeze(np.asarray(log_pred))
    loss = np.asarray(loss)

    y_org = np.squeeze(np.asarray(y_org))
    pred = np.squeeze(np.asarray(pred))
    mean_cls = np.squeeze(np.asarray(mean_cls))
    if num_classes == 1 and model_name == 'SPN':
        pred = np.exp(pred)
    # print('DEBUG pred shape',pred.shape,flush=True)

    abs_change = np.abs(z_cf - z)

    # normalized_abs_change=abs_change/np.max(abs_change)
    # tmp=np.prod(normalized_abs_change,axis=-1)
    # mean_diversity_value=np.mean(np.mean(np.prod(normalized_abs_change,axis=-1),axis=1))

    arg_max_change = np.argmax(abs_change, axis=2)
    return x_org, x_rec_mean, x_cf, loss, pred, y_cf_goal, z, z_cf, y_org, arg_max,arg_max_change

def get_some_eval_params(z_cf,arg_max_change):
    '''
    z analysis?
    :param z_cf:
    :param arg_max_change:
    :return:
    '''
    # Count occurrences
    total_count = arg_max_change.shape[1]
    all_fractions = []
    all_diversity_vals = []

    # Store the extracted max, median, and min indices
    max_indices = []
    median_indices = []
    min_indices = []

    for entry in arg_max_change:
        counts = Counter(entry)

        # Compute fractions and sort by fraction (descending)
        fractions = sorted(((num, count / total_count) for num, count in counts.items()), key=lambda x: x[1],
                           reverse=True)
        all_fractions.append(fractions)

        diversity_freq = 1
        for freq in fractions:
            diversity_freq *= freq[1]
        all_diversity_vals.append(diversity_freq)

        # Extract most common, median, and min indices
        if len(fractions) > 0:
            max_indices.append(fractions[0][0])  # Most common index
            median_indices.append(fractions[len(fractions) // 2][0])  # Median index
            min_indices.append(fractions[-1][0])  # Least common index
    # Find the most common max, median, and min indices across all entries
    most_common_max_index = Counter(max_indices).most_common(1)[0][0]
    median_counter = Counter(median_indices)
    most_common_median_index = sorted(list(median_counter.items()), key=lambda x: x[1],
                                      reverse=True)[int(len(median_counter) / 2)][0]
    most_common_min_index = sorted(list(Counter(min_indices).items()), key=lambda x: x[1],
                                   reverse=True)[-1][0]

    # Compute the mean fraction for each of these indices
    def mean_fraction_for_index(index, all_fractions):
        fractions = [dict(f).get(index, 0) for f in all_fractions]
        return np.mean(fractions)

    mean_max_fraction = mean_fraction_for_index(most_common_max_index, all_fractions)
    mean_median_fraction = mean_fraction_for_index(most_common_median_index, all_fractions)
    mean_min_fraction = mean_fraction_for_index(most_common_min_index, all_fractions)

    all_diversity_vals = np.mean(np.asarray(all_diversity_vals))

    # arg_max_mean=np.mean(arg_max,axis=1)
    mean_lsp = mean_lsp_calculation(z_cf)
    return mean_max_fraction,mean_median_fraction ,mean_min_fraction,all_diversity_vals,mean_lsp

#def compute_advanced_cf_metrics(x_data,y_data,x_masks,y_cf_pred):

#    x_org,x_rec,x_cf=x_data # shape (num_instances,H,W,num_frames=2=EDV and ESV)
#    y_org, y_pred, y_cf_goal=y_data # shape(num_instances)
#    x_masks_org, x_masks_rec, x_masks_cf=x_masks #shape (num_instances,H,W,num_frames=2=EDV and ESV)

def eval_counterfactuals(dataset_name,
                         params,
                         data_path_grid,
                         ef_model,
                         vae_name,
                         fold_idxs,
                         possible_values,
                         test,
                         num_classes,segmentation_model,mean_segmentation,std_segmentation,fine_tune=0,org_mask_data=None,work_from=1):
    # TODO for all created files that can be found - evaluate counterfactuals and save them in pandas datafram with model configs:vae_name, beta gamma foldidx,model_name,L2,validity,FID
    random.seed(0)
    #test_idx=list(range(test[0].shape[0]))
    ##test_sample=random.shuffle(test_idx)
    #test_idx=test_idx[:1000]

    results = []
    results_z_eval=[]

    result_segmentation=[]
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs'
        files_data = find_and_extract_files(data_path_fold,fine_tune=fine_tune,possible_values=possible_values)
        print(files_data)
        for CF_aggregation in ['min_loss','mean']:
            for file_name, model_name, beta, gamma in files_data:

                print(f"File: {file_name} | Model: {model_name} | Beta: {beta} | Gamma: {gamma}",flush=True)
                x_org, x_rec_mean, x_cf_mean, loss, pred, y_cf_goal, z, z_cf, y_org, arg_max,arg_max_change = load_data(
                    data_path_fold + '/' + file_name, test, num_classes, model_name,work_from,CF_aggregation=CF_aggregation)
                #x_cf_mean = np.mean(x_cf, axis=1)
                # TODO do we need this here?
                #mean_max_fraction,mean_median_fraction ,mean_min_fraction,all_diversity_vals,mean_lsp=get_some_eval_params(z_cf, arg_max_change)


                ############################################################################################

                validity_MAE_org, validity_RMSE_org, validity_spearman_org,plot_dict_org,x_masks_org =get_segmentation_results(segmentation_model,
                                                                                         mean_segmentation,
                                                                                         std_segmentation,
                                                                                         y_org,
                                                                                         np.squeeze(x_org),
                                                                                         model_plot_name='Original')
                valid_MAE_org_ef, valid_RMSE_org_ef, valid_spearman_org_ef,_, valid_MAE_org_ef_std, valid_RMSE_org_ef_std=get_ef_model_results(ef_model,mean_segmentation,std_segmentation,y_org,
                                                                                                  np.squeeze(x_org),
                                                                                                  model_plot_name='Original')

                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'None',
                    "model_name": 'Original',
                    "CLF_RMSE": validity_RMSE_org,
                    "CLF_MAE": validity_MAE_org,
                    'Spearman': validity_spearman_org,
                    "CLF_RMSE_ef": valid_RMSE_org_ef,
                    "CLF_MAE_ef": valid_MAE_org_ef,
                    'Spearman_ef': valid_spearman_org_ef,
                    'CF_aggregation':CF_aggregation

                })
                valid_MAE_rec_ef, valid_RMSE_rec_ef, valid_spearman_rec_ef,_, valid_MAE_rec_ef_std, valid_RMSE_rec_ef_std = get_ef_model_results(ef_model,
                                                                                                  mean_segmentation,
                                                                                                  std_segmentation,
                                                                                                  y_org,
                                                                                                  np.squeeze(x_rec_mean),
                                                                                                  model_plot_name='Reconstruction')

                validity_MAE_rec, validity_RMSE_rec, validity_spearman_rec,plot_dict_org_rec,x_masks_rec = get_segmentation_results(segmentation_model,
                                                                                          mean_segmentation,
                                                                                          std_segmentation,y_org,#mean_cls,
                                                                                          np.squeeze(x_rec_mean),model_plot_name='Reconstruction')
                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'reconstruction',
                    "model_name": model_name,
                    "CLF_RMSE": validity_RMSE_rec,
                    "CLF_MAE": validity_MAE_rec,
                    'Spearman': validity_spearman_rec,
                    "CLF_RMSE_ef": valid_RMSE_rec_ef,
                    "CLF_MAE_ef": valid_MAE_rec_ef,
                    'Spearman_ef': valid_spearman_rec_ef,
                    'CF_aggregation': CF_aggregation

                })




                valid_MAE_cf_ef, valid_RMSE_cf_ef, valid_spearman_cf_ef,y_cf_pred, valid_MAE_cf_ef_std, valid_RMSE_cf_ef_std = get_ef_model_results(ef_model,
                                                                                                  mean_segmentation,
                                                                                                  std_segmentation,
                                                                                                  y_cf_goal,
                                                                                                  np.squeeze(x_cf_mean),
                                                                                                  model_plot_name='Counterfactual')

                validity_MAE_cf, validity_RMSE_cf, validity_spearman_cf,plot_dict_cf,x_masks_cf = get_segmentation_results(segmentation_model,
                                                                                          mean_segmentation,
                                                                                          std_segmentation,y_cf_goal,#arg_max_mean,
                                                                                          np.squeeze(x_cf_mean),model_plot_name='Counterfactual')
                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'counterfactual',
                    "model_name": model_name,
                    "CLF_RMSE": validity_RMSE_cf,
                    "CLF_MAE": validity_MAE_cf,
                    'Spearman': validity_spearman_cf,
                    "CLF_RMSE_ef": valid_RMSE_cf_ef,
                    "CLF_MAE_ef": valid_MAE_cf_ef,
                    'Spearman_ef': valid_spearman_cf_ef,
                    'CF_aggregation': CF_aggregation

                })
                if True:
                    debug_plot_org_rec_cf_pairs(
                        x_data=(np.squeeze(x_org),np.squeeze(x_rec_mean),np.squeeze(x_cf_mean)),
                        y_data=(y_org,pred,y_cf_goal),
                        x_masks=(x_masks_org, x_masks_rec, x_masks_cf),
                        y_cf_pred=y_cf_pred,
                        show=False,
                        save_dir=data_path_grid+model_name+"b"+str(beta).replace('.','-')+"g"+str(gamma).replace('.','-')+"f"+str(fine_tune)+CF_aggregation,
                        instance_indices=list(range(4))

                    )


                #
                # TODO adapt compute_advanced_cf_metrics bezüglich all unserer besprochenen metriken für die wir latex funtionen für das hauptpaper erstellt haben
                #TODO save the already computed validity_MAE_cf, validity_RMSE_cf, validity_spearman_cf in metrics
                # TODO compute L1 minimality + save in metrics
                metrics=compute_advanced_cf_metrics([np.squeeze(x_org),np.squeeze(x_rec_mean),np.squeeze(x_cf_mean)],
                                            [y_org,pred,y_cf_goal],
                                            [x_masks_org,x_masks_rec,x_masks_cf],y_cf_pred,
                                            reference_masks=None,
                                            change_quantile=0.90,
                                            boundary_inner_iter=7,
                                            boundary_outer_iter=4,
                                            cf_validity_mae=validity_MAE_cf,
                                            cf_validity_rmse=validity_RMSE_cf,
                                            cf_validity_spearman=validity_spearman_cf,
                                            cf_validity_mae_ef=valid_MAE_cf_ef,
                                            cf_validity_rmse_ef=valid_RMSE_cf_ef,
                                            cf_validity_spearman_ef=valid_spearman_cf_ef,
                                            cf_validity_mae_ef_std=valid_MAE_cf_ef_std,
                                            cf_validity_rmse_ef_std=valid_RMSE_cf_ef_std,
                                                    )


                print_cf_metrics_summary(metrics)
                latex_table = make_cf_metrics_latex_table(metrics)
                print(latex_table)

                saved_files = save_cf_metrics(
                    metrics_out=metrics,
                    save_dir=data_path_grid,
                    file_prefix=model_name+"b"+str(beta).replace('.','-')+"g"+str(gamma).replace('.','-')+"-seg"+str(fine_tune)+CF_aggregation
                )
                print(saved_files)
                if not work_from:
                    ANATOMY_CORE_METRICS = [
                        "pair_mean_inside_ratio_weighted_strict",  # higher = better
                        "pair_mean_outside_ratio_weighted_strict",  # lower = better
                        "pair_mean_boundary_concentration_strict",  # higher = better
                        "pair_mean_mean_distance_to_boundary_strict",  # lower = better
                        "pair_mean_roi_precision_strict",  # higher = better
                        "pair_mean_density_ratio_strict",  # higher = better
                        "pair_mean_ring_density_strict",  # higher = better
                        "pair_mean_dice_org_cf",  # higher = better
                        "pair_mean_delta_area",  # closer to 0 = better
                        "order_preserved_wrt_org",  # higher = better
                        "temporal_mask_l1",  # lower = better
                        "temporal_mask_xor_change_ratio",  # lower = better
                    ]
                    latex_table = make_cf_metrics_latex_table(metrics, selected_metrics=ANATOMY_CORE_METRICS)
                    print(latex_table)
                    debug_anatomy_core_metrics(
                        x_data=(np.squeeze(x_org),np.squeeze(x_rec_mean),np.squeeze(x_cf_mean)),
                        x_masks=[x_masks_org,x_masks_rec,x_masks_cf],
                        metrics_out=metrics,
                        n_examples=4,
                        boundary_inner_iter=7,
                        boundary_outer_iter=4,
                    )


                ###############################################################################################
                '''
                debugging_correlation_plot(plot_dicts=[plot_dict_org,plot_dict_org_rec,plot_dict_cf],
                               result_metrics=[[validity_RMSE_org,validity_MAE_org,validity_spearman_org],
                                               [validity_RMSE_rec,validity_MAE_rec,validity_spearman_rec],
                                               [validity_RMSE_cf,validity_MAE_cf,validity_spearman_cf]],
                                           debugging_plot_path=data_path_grid + 'fold_' + str(fold_idx) + '/'+model_name+
                                                               str(beta).replace('.','-')+'_'+str(gamma).replace('.','-')+'_corr_plot.png')
                '''




                '''


                for strategy_name,strategy,input_arr in zip(['mean'], # 'max_cf','weighted_loss','mean','min_loss'
                                                            [mean_cf], #max_cf,weighted,mean_cf,min_loss
                                                            [None]):#log_pred, #(pred,y_cf_goal,num_classes),loss,None,loss

                    x_cf_strat,arg_max_process=strategy(x_cf,input_arr,arg_max)


                    class_FID=0
                    if num_classes>1:
                        class_FID = compute_class_fid(test[0][test_idx], x_cf_strat, x_rec_mean, np.round(y_cf_goal), test[1][test_idx,0], np.round(arg_max_process))
                        blop_info = get_importance_region_information(x_rec_mean, x_cf_strat, threshold=0.5)
                        validity = compute_validity(np.round(y_cf_goal), np.round(arg_max_process),num_classes=num_classes)
                    else:
                        pred_y_reg=arg_max_process
                        #if model_name=='SPN':
                        #    pred_y_reg=np.exp(pred_y_reg)

                        validity_MAE,validity_RMSE,validity_spearman = compute_validity(y_cf_goal,pred_y_reg,num_classes=num_classes)


                    MSE,L2=compute_MSE(x_org, x_cf_strat,params.conv_dim)
                    MAE, L1 = compute_MAE(x_org, x_cf_strat,params.conv_dim)
                    FID = compute_fid(x_org, x_cf_strat,params.conv_dim)
                    rel_L2=compute_rel_L2(x_org, x_cf_strat,x_rec_mean,params.conv_dim)
                    if num_classes > 1:
                        results.append({
                            "dataset": dataset_name,
                            "vae_name": vae_name,
                            "beta": beta,
                            "gamma": gamma,
                            "fold_idx": fold_idx,
                            "strategy": strategy_name,
                            "model_name": model_name,
                            "MSE": MSE,
                            "MAE": MAE,
                            'L2': L2,
                            'L1': L1,
                            "validity": validity,
                            "FID": FID,
                            "RelFID": class_FID,
                            "RelL2": rel_L2,
                            "Pos region size": blop_info[0, 0],
                            'Pos number of regions': blop_info[0, 1],
                            "Neg region size": blop_info[1, 0],
                            'Neg number of regions': blop_info[1, 1],

                        })
                    else:
                        results.append({
                            "dataset": dataset_name,
                            "vae_name": vae_name,
                            "beta": beta,
                            "gamma": gamma,
                            "fold_idx": fold_idx,
                            "strategy": strategy_name,
                            "model_name": model_name,
                            "MSE": MSE,
                            "MAE": MAE,
                            'L2': L2,
                            'L1': L1,
                            "FID": FID,
                            "RelL2": rel_L2,

                            "CLF_RMSE": validity_RMSE,
                            "CLF_MAE": validity_MAE,
                            'Spearman': validity_spearman,


                        })

                    ############################################################
                    validity_MAE, validity_RMSE, validity_spearman = get_segmentation_results(segmentation_model,
                                                                                              mean_segmentation,
                                                                                              std_segmentation,
                                                                                              y_cf_goal,
                                                                                              np.squeeze(x_cf_strat))
                    result_segmentation.append({
                        "dataset": dataset_name,
                        "vae_name": vae_name,
                        "beta": beta,
                        "gamma": gamma,
                        "fold_idx": fold_idx,
                        "strategy": 'reconstruction',
                        "model_name": model_name,
                        "CLF_RMSE": validity_RMSE,
                        "CLF_MAE": validity_MAE,
                        'Spearman': validity_spearman,

                    })

                    ###########################################################



                label_switch_step_mean = np.mean([e for e in label_switch_step if e >= 0])

                results_z_eval.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "model_name": model_name,
                    "Max idx":most_common_max_index,
                    "Max freq":mean_max_fraction,
                    "Median idx":most_common_median_index,
                    "Median freq":mean_median_fraction,
                    "Min idx":most_common_min_index,
                    "Min freq":mean_min_fraction,
                    'Diversity freq.':all_diversity_vals,
                    "Switch epoch": label_switch_step_mean,
                    "Mean LSP":mean_lsp, # mean latent space distance
                    #'Mean div freq.':mean_diversity_value,

                })
                '''
        # Convert to DataFrame and save

    return results,results_z_eval,result_segmentation,metrics

def eval_counterfactuals_extended(dataset_name,
                         params,
                         data_path_grid,
                         ef_model,
                         vae_name,
                         fold_idxs,
                         possible_values,
                         test,
                         num_classes,segmentation_model,mean_segmentation,std_segmentation,fine_tune=0):
    random.seed(0)
    results = []
    results_z_eval=[]

    result_segmentation=[]
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs'
        files_data = find_and_extract_files(data_path_fold,fine_tune=fine_tune)
        print(files_data)

        for file_name, model_name, beta, gamma in files_data:
            if beta in possible_values and gamma in possible_values:
                print(f"File: {file_name} | Model: {model_name} | Beta: {beta} | Gamma: {gamma}",flush=True)
                all_data_list = pkl.load(open(data_path_fold+'/'+file_name, 'rb'))
                all_data=[]
                if isinstance(all_data_list[0],list):
                    for entry in all_data_list:
                        all_data.extend(entry)
                else:
                    all_data=all_data_list
                                                                                                       #(opposite_class, coordinates, x, mean_cls, y)
                [x_cf,x_rec,z_cf,z,title_info,distance,arg_max,loss,log_pred,p_z,label_switch_step,pred,y_cf_goal, coordinates, x_org, mean_cls, y_org]=zip(*all_data)
                print(np.min(x_cf),np.max(x_cf),np.min(x_org),np.max(x_org),np.min(x_rec),np.max(x_rec))
                x_cf=np.asarray(x_cf)
                x_rec=np.asarray(x_rec)
                x_org = np.asarray(x_org)

                x_cf_mean = np.mean(x_cf, axis=1)
                x_org = np.squeeze(x_org, axis=1)
                x_rec_mean = np.mean(x_rec, axis=1)

                # transpose:(test_size,2,h,w,1)
                x_cf_mean = np.transpose(x_cf_mean, (0,2,3,1,4))
                x_org = np.transpose(x_org, (0,2,3,1,4))
                x_rec_mean = np.transpose(x_rec_mean , (0,2,3,1,4))
                print(x_cf_mean.shape,x_org.shape,x_rec_mean.shape)



                z=np.asarray(z)
                z_cf=np.asarray(z_cf)
                arg_max=np.squeeze(np.asarray(arg_max))

                y_cf_goal=np.asarray(y_cf_goal)
                #x_org= np.mean(x_org, axis=1)

                log_pred=np.squeeze(np.asarray(log_pred))
                loss=np.asarray(loss)

                y_org=np.squeeze(np.asarray(y_org))
                pred = np.squeeze(np.asarray(pred))
                mean_cls = np.squeeze(np.asarray(mean_cls))
                if num_classes==1 and model_name=='SPN':
                    pred=np.exp(pred)
                #print('DEBUG pred shape',pred.shape,flush=True)

                abs_change=np.abs(z_cf-z)

                #normalized_abs_change=abs_change/np.max(abs_change)
                #tmp=np.prod(normalized_abs_change,axis=-1)
                #mean_diversity_value=np.mean(np.mean(np.prod(normalized_abs_change,axis=-1),axis=1))


                arg_max_change=np.argmax(abs_change,axis=2)
                # Count occurrences
                total_count = arg_max_change.shape[1]
                all_fractions=[]
                all_diversity_vals=[]

                # Store the extracted max, median, and min indices
                max_indices = []
                median_indices = []
                min_indices = []

                for entry in arg_max_change:
                    counts = Counter(entry)

                    # Compute fractions and sort by fraction (descending)
                    fractions = sorted(((num, count / total_count) for num, count in counts.items()), key=lambda x: x[1],
                                       reverse=True)
                    all_fractions.append(fractions)


                    diversity_freq=1
                    for freq in fractions:
                        diversity_freq*=freq[1]
                    all_diversity_vals.append(diversity_freq)

                    # Extract most common, median, and min indices
                    if len(fractions) > 0:
                        max_indices.append(fractions[0][0])  # Most common index
                        median_indices.append(fractions[len(fractions) // 2][0])  # Median index
                        min_indices.append(fractions[-1][0])  # Least common index
                # Find the most common max, median, and min indices across all entries
                most_common_max_index = Counter(max_indices).most_common(1)[0][0]
                median_counter=Counter(median_indices)
                most_common_median_index = sorted(list(median_counter.items()), key=lambda x: x[1],
                                       reverse=True)[int(len(median_counter)/2)][0]
                most_common_min_index = sorted(list(Counter(min_indices).items()), key=lambda x: x[1],
                                       reverse=True)[-1][0]

                # Compute the mean fraction for each of these indices
                def mean_fraction_for_index(index, all_fractions):
                    fractions = [dict(f).get(index, 0) for f in all_fractions]
                    return np.mean(fractions)

                mean_max_fraction = mean_fraction_for_index(most_common_max_index, all_fractions)
                mean_median_fraction = mean_fraction_for_index(most_common_median_index, all_fractions)
                mean_min_fraction = mean_fraction_for_index(most_common_min_index, all_fractions)

                all_diversity_vals=np.mean(np.asarray(all_diversity_vals))

                #arg_max_mean=np.mean(arg_max,axis=1)
                mean_lsp=mean_lsp_calculation(z_cf)

                ############################################################################################

                validity_MAE_org, validity_RMSE_org, validity_spearman_org,plot_dict_org =get_segmentation_results(segmentation_model,
                                                                                         mean_segmentation,
                                                                                         std_segmentation,
                                                                                         y_org,
                                                                                         np.squeeze(x_org),model_plot_name='Original')
                valid_MAE_org_ef, valid_RMSE_org_ef, valid_spearman_org_ef=get_ef_model_results(ef_model,mean_segmentation,std_segmentation,y_org,np.squeeze(x_org),model_plot_name='Original')

                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'None',
                    "model_name": 'Original',
                    "CLF_RMSE": validity_RMSE_org,
                    "CLF_MAE": validity_MAE_org,
                    'Spearman': validity_spearman_org,
                    "CLF_RMSE_ef": valid_RMSE_org_ef,
                    "CLF_MAE_ef": valid_MAE_org_ef,
                    'Spearman_ef': valid_spearman_org_ef,

                })
                valid_MAE_rec_ef, valid_RMSE_rec_ef, valid_spearman_rec_ef = get_ef_model_results(ef_model,
                                                                                                  mean_segmentation,
                                                                                                  std_segmentation,
                                                                                                  y_org,
                                                                                                  np.squeeze(x_rec_mean),
                                                                                                  model_plot_name='Reconstruction')

                validity_MAE_rec, validity_RMSE_rec, validity_spearman_rec,plot_dict_org_rec = get_segmentation_results(segmentation_model,
                                                                                          mean_segmentation,
                                                                                          std_segmentation,y_org,#mean_cls,
                                                                                          np.squeeze(x_rec_mean),model_plot_name='Reconstruction')
                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'reconstruction',
                    "model_name": model_name,
                    "CLF_RMSE": validity_RMSE_rec,
                    "CLF_MAE": validity_MAE_rec,
                    'Spearman': validity_spearman_rec,
                    "CLF_RMSE_ef": valid_RMSE_rec_ef,
                    "CLF_MAE_ef": valid_MAE_rec_ef,
                    'Spearman_ef': valid_spearman_rec_ef,

                })



                #TODO
                if model_name == 'MLP':
                    arg_max_mean = np.mean(arg_max, axis=1)
                else:
                    arg_max_mean = np.mean(np.exp(arg_max), axis=1)
                valid_MAE_cf_ef, valid_RMSE_cf_ef, valid_spearman_cf_ef = get_ef_model_results(ef_model,
                                                                                                  mean_segmentation,
                                                                                                  std_segmentation,
                                                                                                  y_cf_goal,
                                                                                                  np.squeeze(x_cf_mean),
                                                                                                  model_plot_name='Counterfactual')

                validity_MAE_cf, validity_RMSE_cf, validity_spearman_cf,plot_dict_cf = get_segmentation_results(segmentation_model,
                                                                                          mean_segmentation,
                                                                                          std_segmentation,y_cf_goal,#arg_max_mean,
                                                                                          np.squeeze(x_cf_mean),model_plot_name='Counterfactual')
                result_segmentation.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "strategy": 'counterfactual',
                    "model_name": model_name,
                    "CLF_RMSE": validity_RMSE_cf,
                    "CLF_MAE": validity_MAE_cf,
                    'Spearman': validity_spearman_cf,
                    "CLF_RMSE_ef": valid_RMSE_cf_ef,
                    "CLF_MAE_ef": valid_MAE_cf_ef,
                    'Spearman_ef': valid_spearman_cf_ef,

                })

                ###############################################################################################
                '''
                debugging_correlation_plot(plot_dicts=[plot_dict_org,plot_dict_org_rec,plot_dict_cf],
                               result_metrics=[[validity_RMSE_org,validity_MAE_org,validity_spearman_org],
                                               [validity_RMSE_rec,validity_MAE_rec,validity_spearman_rec],
                                               [validity_RMSE_cf,validity_MAE_cf,validity_spearman_cf]],
                                           debugging_plot_path=data_path_grid + 'fold_' + str(fold_idx) + '/'+model_name+
                                                               str(beta).replace('.','-')+'_'+str(gamma).replace('.','-')+'_corr_plot.png')
                '''




                '''


                for strategy_name,strategy,input_arr in zip(['mean'], # 'max_cf','weighted_loss','mean','min_loss'
                                                            [mean_cf], #max_cf,weighted,mean_cf,min_loss
                                                            [None]):#log_pred, #(pred,y_cf_goal,num_classes),loss,None,loss

                    x_cf_strat,arg_max_process=strategy(x_cf,input_arr,arg_max)


                    class_FID=0
                    if num_classes>1:
                        class_FID = compute_class_fid(test[0][test_idx], x_cf_strat, x_rec_mean, np.round(y_cf_goal), test[1][test_idx,0], np.round(arg_max_process))
                        blop_info = get_importance_region_information(x_rec_mean, x_cf_strat, threshold=0.5)
                        validity = compute_validity(np.round(y_cf_goal), np.round(arg_max_process),num_classes=num_classes)
                    else:
                        pred_y_reg=arg_max_process
                        #if model_name=='SPN':
                        #    pred_y_reg=np.exp(pred_y_reg)

                        validity_MAE,validity_RMSE,validity_spearman = compute_validity(y_cf_goal,pred_y_reg,num_classes=num_classes)


                    MSE,L2=compute_MSE(x_org, x_cf_strat,params.conv_dim)
                    MAE, L1 = compute_MAE(x_org, x_cf_strat,params.conv_dim)
                    FID = compute_fid(x_org, x_cf_strat,params.conv_dim)
                    rel_L2=compute_rel_L2(x_org, x_cf_strat,x_rec_mean,params.conv_dim)
                    if num_classes > 1:
                        results.append({
                            "dataset": dataset_name,
                            "vae_name": vae_name,
                            "beta": beta,
                            "gamma": gamma,
                            "fold_idx": fold_idx,
                            "strategy": strategy_name,
                            "model_name": model_name,
                            "MSE": MSE,
                            "MAE": MAE,
                            'L2': L2,
                            'L1': L1,
                            "validity": validity,
                            "FID": FID,
                            "RelFID": class_FID,
                            "RelL2": rel_L2,
                            "Pos region size": blop_info[0, 0],
                            'Pos number of regions': blop_info[0, 1],
                            "Neg region size": blop_info[1, 0],
                            'Neg number of regions': blop_info[1, 1],

                        })
                    else:
                        results.append({
                            "dataset": dataset_name,
                            "vae_name": vae_name,
                            "beta": beta,
                            "gamma": gamma,
                            "fold_idx": fold_idx,
                            "strategy": strategy_name,
                            "model_name": model_name,
                            "MSE": MSE,
                            "MAE": MAE,
                            'L2': L2,
                            'L1': L1,
                            "FID": FID,
                            "RelL2": rel_L2,

                            "CLF_RMSE": validity_RMSE,
                            "CLF_MAE": validity_MAE,
                            'Spearman': validity_spearman,


                        })

                    ############################################################
                    validity_MAE, validity_RMSE, validity_spearman = get_segmentation_results(segmentation_model,
                                                                                              mean_segmentation,
                                                                                              std_segmentation,
                                                                                              y_cf_goal,
                                                                                              np.squeeze(x_cf_strat))
                    result_segmentation.append({
                        "dataset": dataset_name,
                        "vae_name": vae_name,
                        "beta": beta,
                        "gamma": gamma,
                        "fold_idx": fold_idx,
                        "strategy": 'reconstruction',
                        "model_name": model_name,
                        "CLF_RMSE": validity_RMSE,
                        "CLF_MAE": validity_MAE,
                        'Spearman': validity_spearman,

                    })

                    ###########################################################



                label_switch_step_mean = np.mean([e for e in label_switch_step if e >= 0])

                results_z_eval.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "model_name": model_name,
                    "Max idx":most_common_max_index,
                    "Max freq":mean_max_fraction,
                    "Median idx":most_common_median_index,
                    "Median freq":mean_median_fraction,
                    "Min idx":most_common_min_index,
                    "Min freq":mean_min_fraction,
                    'Diversity freq.':all_diversity_vals,
                    "Switch epoch": label_switch_step_mean,
                    "Mean LSP":mean_lsp, # mean latent space distance
                    #'Mean div freq.':mean_diversity_value,

                })
                '''
        # Convert to DataFrame and save

    return results,results_z_eval,result_segmentation