import numpy as np
import echonet
import torch
import os
from cf_analysis import load_segmentation_model, get_mask_data
from cf_eval.counterfactual_evaluation import get_ef_model_results, get_segmentation_results
from segmentation_analysis_full_updated_v3 import compute_advanced_cf_metrics, \
    save_cf_metrics, load_cf_metrics
from video import load_ef_pred_model
def load_videos(videos_save_folder, num_random_runs, num_files,org_echo_data):
    all_videos=[]
    all_efs=[]
    all_org_videos=[]
    y_org=[]
    mask_org=[]
    for rand_idx in range(num_random_runs):
        videos=[]
        efs=[]
        for file_num in range(num_files):
            file_name=videos_save_folder+str(file_num)+'_'+str(rand_idx)+'.npz'
            data=np.load(file_name)
            all_videos.append(data['arr1'])
            all_efs.append(data['arr2'])
            org_img_pairs=np.stack([org_echo_data[0][file_num],org_echo_data[1][file_num]],axis=1)#-1
            org_mask_pairs = np.stack([org_echo_data[2][file_num], org_echo_data[3][file_num]], axis=1)#-1
            all_org_videos.append(org_img_pairs)
            y_org.append(org_echo_data[4][file_num])
            mask_org.append(org_mask_pairs)
        #all_videos.append(videos)
        #all_efs.append(efs)

    return np.asarray(all_videos),np.asarray(all_efs),np.asarray(all_org_videos),np.asarray(y_org),np.asarray(mask_org)





if __name__ == '__main__':
    work_from=1
    num_random_runs =[1,3][work_from]

    dataset_name='CAMUS'
    device_str = ['cpu', 'cuda'][work_from]
    device = torch.device(device_str)
    file_info='CAMUS_flip'
    if dataset_name=='stanford':
        num_files = [4, 1220][work_from]
        videos_save_folder=['results/diffusion_results/counterfactual_videos/',
                            #'/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion/4SCM_merge/eval/counterfactual_videos/',
                            '/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_merge/eval/counterfactual_videos/'
                            ][work_from]
        # load segmentation model once:
        weights = [#'C:/Users/siekiera/Documents/git_repos/CF_eval/segmentation_model_weights/best.pt',
            'C:/Users/siekiera/Documents/git_repos/CF_eval/weights/deeplabv3_resnet50_random.pt',
                   #'/lustre/project/ki-topml/jsiekiera/segmentation_weights_echos/best.pt'
                   'deeplabv3_resnet50_random.pt'
                   ][work_from]
        weights_ef=['C:/Users/siekiera/Documents/git_repos/CF_eval/ef_pred_weights_echos/best.pt',
                    '/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_seg/best.pt'
                    ][work_from]
        data_dir_org_masks = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                    '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][work_from]


    else:
        num_files=50
        videos_save_folder ='/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_v3_CAMUS_merge/eval/counterfactual_videos/'
        weights='/lustre/project/ki-topml/jsiekiera/segmentation_weights_camus/best.pt'
        weights_ef='/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_camu_bi_le6_20/best.pt'
        data_dir_org_masks='/lustre/project/ki-topml/jsiekiera/cf_data/'

    data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][work_from]


    gt_data=get_mask_data(data_dir=data_dir_org_masks,dataset_name=dataset_name)

    all_cf_videos,target_cf,x_org,y_org,mask_org=load_videos(videos_save_folder,num_random_runs=num_random_runs,num_files=num_files,org_echo_data=gt_data)#num_random_runs=3,num_files=1275
    target_cf=np.squeeze(target_cf)
    print('cf videos, target cf',all_cf_videos.shape,target_cf.shape,np.min(all_cf_videos),np.max(all_cf_videos),np.min(target_cf),np.max(target_cf)) # shape=
    print('org, target org',x_org.shape,y_org.shape,np.min(x_org),np.max(x_org),np.min(y_org),np.max(y_org))
    print(data_dir)
    echo = echonet.datasets.Echo(root=data_dir, split="train")
    mean_segmentation, std_segmentation = echonet.utils.get_mean_and_std(echo)
    mean_segmentation, std_segmentation = np.mean(mean_segmentation), np.mean(std_segmentation)

    print('load segentation model', flush=True)

    segmentation_model = load_segmentation_model(device=device,
                                                 model_name="deeplabv3_resnet50",
                                                 pretrained=False,
                                                 weights=weights,
                                                 train=False, device_str=device_str)

    ef_model = load_ef_pred_model(device=device, weights=weights_ef)
    # TODO look at the shape of all_cf_videos
    valid_MAE_cf_ef, valid_RMSE_cf_ef, valid_spearman_cf_ef , y_cf_pred,_,_= get_ef_model_results(ef_model,
                                                                                   mean_segmentation,
                                                                                   std_segmentation,
                                                                                   target_cf,
                                                                                   all_cf_videos,
                                                                                   model_plot_name='Counterfactual')

    valid_MAE_org_ef, valid_RMSE_org_ef, valid_spearman_org_ef , y_pred,_,_= get_ef_model_results(ef_model,
                                                                                   mean_segmentation,
                                                                                   std_segmentation,
                                                                                   y_org,
                                                                                   x_org,#np.transpose(x_org,(0,2,3,1)),#x_org,#
                                                                                   model_plot_name='Original')

    validity_MAE_cf, validity_RMSE_cf, validity_spearman_cf, plot_dict_cf,all_y_masks = get_segmentation_results(segmentation_model,
                                                                                                     mean_segmentation,
                                                                                                     std_segmentation,
                                                                                                     target_cf,
                                                                                                     # arg_max_mean,
                                                                                                     all_cf_videos,
                                                                                                     model_plot_name='Counterfactual')
    validity_MAE_org, validity_RMSE_org, validity_spearman_org, plot_dict_org,org_pred_mask = get_segmentation_results(segmentation_model,
                                                                                                     mean_segmentation,
                                                                                                     std_segmentation,
                                                                                                     target_cf,
                                                                                                     # arg_max_mean,
                                                                                                     x_org,#np.transpose(x_org,(0,2,3,1)),
                                                                                                     model_plot_name='original segmentation')

    print('###segmentation',validity_MAE_cf, validity_RMSE_cf, validity_spearman_cf)
    print('###regression',valid_MAE_cf_ef, valid_RMSE_cf_ef, valid_spearman_cf_ef)
    # do some advanced stuff with the segmentation model:

    # TODO are we able to link this again to the Original images? :(
    print('cf videos, target cf',all_cf_videos.shape,target_cf.shape,np.min(all_cf_videos),np.max(all_cf_videos),np.min(target_cf),np.max(target_cf)) # shape=
    print('org, target org',x_org.shape,y_org.shape,np.min(x_org),np.max(x_org),np.min(y_org),np.max(y_org))
    print('Last shape',np.transpose(np.mean(x_org,axis=1),(0,2,3,1)).shape,

          np.transpose(np.mean(all_cf_videos,axis=1),(0,2,3,1)).shape)
    metrics = compute_advanced_cf_metrics([np.transpose(np.mean(x_org,axis=1),(0,2,3,1))#np.squeeze(x_org)
                                              , None, np.transpose(np.mean(all_cf_videos,axis=1),(0,2,3,1))],
                                          [y_org, y_pred, target_cf],
                                          [org_pred_mask, None, all_y_masks ], y_cf_pred,
                                          reference_masks=None,
                                          change_quantile=0.90,
                                          boundary_inner_iter=7,
                                          boundary_outer_iter=4,
                                          cf_validity_mae=validity_MAE_cf,
                                          cf_validity_rmse=validity_RMSE_cf,
                                          cf_validity_spearman=validity_spearman_cf,
                                          cf_validity_mae_ef=valid_MAE_cf_ef,
                                          cf_validity_rmse_ef=valid_RMSE_cf_ef,
                                          cf_validity_spearman_ef=valid_spearman_cf_ef
                                          )


    segment_interpretation_path = os.path.join(videos_save_folder,"segment_analyse_"+file_info+".json")  # 1
    saved_files = save_cf_metrics(
        metrics_out=metrics,
        save_dir=videos_save_folder,
        file_prefix="segment_analyse_"+file_info
    )

    print(saved_files)
    loaded_metrics = load_cf_metrics(segment_interpretation_path)
    print(loaded_metrics.keys())
    print('Results saved')