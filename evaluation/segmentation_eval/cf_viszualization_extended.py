import os
import echonet
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from cf_analysis import load_segmentation_model
from cf_eval.counterfactual_evaluation import get_ef_model_results, get_segmentation_results
from evaluation.segmentation_eval.segmentation_analysis_full_updated_v3 import _overlay_contour
from video import load_ef_pred_model


def load_data(file_path,test_data,num_classes, model_name,work_from,CF_aggregation ):
    print('CF_aggregation',CF_aggregation,flush=True)
    if os.path.exists(file_path):
        all_data_list = pkl.load(open(file_path, 'rb'))
        x_cf,x_org,y_cf_goal,loss,y_org=[],[],[],[],[]
        for (pred_, opposite_class_, [i_start, i_end], mean_cls_, rep_results) in all_data_list:

            y_cf_goal.append(opposite_class_)
            y_org.append(test_data[1][i_start:i_end])
            x_org.append(test_data[0][i_start:i_end])
            rec_cf_, rec_gt_, z_cf_, z_, arg_max_, loss_, log_pred, p_z_diff=list(zip(*rep_results))
            print(i_start, i_end,rec_cf_.shape)
            x_cf.append(np.asarray(list(rec_cf_)))
            loss_arr = np.asarray(loss_)

            if loss_arr.ndim == 3:
                loss.append(loss_arr[:, :, 0])
            else:
                loss.append(loss_arr)
        y_cf_goal= np.concatenate(y_cf_goal, axis=0)
        y_org = np.concatenate(y_org, axis=0)
        x_org = np.concatenate(x_org, axis=0)

        x_cf=np.concatenate(x_cf,axis=1)
        loss = np.concatenate(loss, axis=1)

        print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org))
        print('CF shape before mean:',x_cf.shape)
        if CF_aggregation=='mean':
            x_cf = np.mean(x_cf, axis=0)
        elif CF_aggregation=='min_loss':
            #TODO is loss in the right shape?!
            arg_min = np.argmin(loss, axis=0)  # Get the max indices along axis 1
            x_cf = x_cf[arg_min,np.arange(x_cf.shape[1])] #  # Use advanced indexing
        print(x_cf.shape, x_org.shape)
        x_cf = np.transpose(x_cf, (0, 2, 3, 1, 4))
        print(x_cf.shape, x_org.shape)
        x_cf = np.repeat(x_cf, repeats=3, axis=-1)
        x_cf = np.transpose(x_cf, (0, 4, 3, 1, 2))
    else:
        print('FILE DOES NOT EXIST',file_path)
        y_org = test_data[1]
        x_org = test_data[0]
        x_cf=np.zeros((10))
        y_cf_goal =np.zeros((10))
    #if work_from:
    #    test_idx=-1
    #else:
    #    test_idx=5


    x_org=np.repeat(x_org, repeats=3, axis=-1)
    x_org = np.transpose(x_org, (0, 4, 1, 2, 3))


    return x_org,\
        x_cf,\
        y_cf_goal,\
        y_org



def load_dataset(dataset_name,binary,load_net,machine,path='',grid_params=None,conv_dim=3):
    if dataset_name=='stanford':
        data_path='sf_data.pkl'
        print(path+data_path)
        [train, test, val] = pkl.load(open(path+data_path, 'rb'))
        if conv_dim==3:
            train[0] = np.expand_dims(train[0], axis=-1)
            test[0] = np.expand_dims(test[0], axis=-1)
            val[0] = np.expand_dims(val[0], axis=-1) #batchsize,112,112, 2,1

            train[0] = np.transpose(train[0], (0,3,1,2,4))
            test[0] = np.transpose(test[0], (0,3,1,2,4))
            val[0] = np.transpose(val[0], (0,3,1,2,4))

        return ([train,val]),test,1 # batchsize, 2,112,112,1
    elif dataset_name=="CAMUS":
        data_path = 'camus_echonet_style_mobi.npz'
        data = np.load(path + data_path)
        target_type = 'bi'
        train, test, val = [data["train_max_min_frame"], data["train_ef_" + target_type + "targets"]], \
            [data["test_max_min_frame"], data["test_ef_" + target_type + "targets"]], \
            [data["val_max_min_frame"], data["val_ef_" + target_type + "targets"]]
        if conv_dim == 3:
            train[0] = np.transpose(train[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            test[0] = np.transpose(test[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            val[0] = np.transpose(val[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0 ## batchsize,1 112,112,2


            train[1] /=100
            test[1] /=100
            val[1] /=100
            print('CAMUS x y data min max', np.max(train[0]),np.min(train[0]), np.max(test[0]), np.max(val[0]),train[0].shape,
                  np.max(train[1]),np.min(train[1]),)

        return ([train, val]), test, 1 # batchsize, 2,112,112,1



def get_SPN_VAE_images(model_name, pred_model_name, beta, gamma, fine_tune, image_idxs):
    plot_path = save_folder+ model_name+'/grid0/fold_0/counterfactual_imgs/'
    CF_aggregation='mean'
    num_classes=1
    if fine_tune:
        save_file_path = (
            plot_path
            + 'datab' + str(beta).replace('.', '-')
            + 'g' + str(gamma).replace('.', '-')
            + pred_model_name + '_1000_new_minmax_tmp_fine_show.pkl'
        )
    else:
        save_file_path = (
            plot_path
            + 'datab' + str(beta).replace('.', '-')
            + 'g' + str(gamma).replace('.', '-')
            + pred_model_name + '_1000_new_minmax_tmp.pkl'
        )

    train, test, num_classes = load_dataset(dataset_name, binary=False,load_net=False,
                                                                     machine=work_from,grid_params=None,path=data_path)
    print(save_file_path)
    x_org, x_cf_mean,y_cf_goal,  y_org,  = load_data(
        save_file_path, test, num_classes, model_name, work_from, CF_aggregation=CF_aggregation)
    print(x_org.shape,x_cf_mean.shape,y_cf_goal.shape)





    return x_org[image_idxs],x_cf_mean[image_idxs],np.squeeze(y_org[image_idxs]),np.squeeze(y_cf_goal[image_idxs])


def get_SCM_data(videos_save_folder, image_idxs, dataset_name,num_random_runs=3):
    all_videos=[]
    all_efs=[]
    for rand_idx in range(1):
        for file_num in image_idxs:
            file_name=videos_save_folder+str(file_num)+'_'+str(rand_idx)+'.npz'
            data=np.load(file_name)
            all_videos.append(data['arr1'])
            all_efs.append(data['arr2'])
    all_efs=np.squeeze(np.asarray(all_efs))
    all_videos = np.asarray(all_videos)
    return all_videos, all_efs





def plot_imgs(img_name, plot_imgs_input):

    n_rows = len(plot_imgs_input)
    n_instances = plot_imgs_input[0][1].shape[0]
    n_cols = n_instances * 2

    fig = plt.figure(figsize=(1.5 * n_cols, 1.9 * n_rows))

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        left=0.045,
        right=0.995,
        bottom=0.04,
        top=0.86,
        wspace=0.03,
        hspace=0.12
    )

    axes = np.empty((n_rows, n_cols), dtype=object)

    mask_org = None
    all_preds = []

    for i, (left_name, x_data, y_target) in enumerate(plot_imgs_input):
        print('DEBUG',left_name,x_data.shape,y_target)
        _, _, _, y_pred ,_,_= get_ef_model_results(
            ef_model,
            mean_segmentation,
            std_segmentation,
            y_target,
            x_data
        )
        all_preds.append(y_pred)

        _, _, _, _, all_y_masks = get_segmentation_results(
            segmentation_model,
            mean_segmentation,
            std_segmentation,
            y_target,
            x_data,
            model_plot_name='Counterfactual'
        )

        if i == 0:
            mask_org = all_y_masks

        for k in range(n_instances):
            for z in range(2):
                j = 2 * k + z

                ax = fig.add_subplot(gs[i, j])
                axes[i, j] = ax

                ax.imshow(x_data[k, 0, z, :, :], cmap='gray')
                ax.axis('off')

                _overlay_contour(ax, all_y_masks[k, :, :, z], color="orange")
                _overlay_contour(ax, mask_org[k, :, :, z], color="cyan")

    fig.canvas.draw()

    # ------------------------------------------------------------
    # Linke Row-Labels näher an die Bilder
    # ------------------------------------------------------------
    for i, (left_name, _, _) in enumerate(plot_imgs_input):
        pos = axes[i, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2

        fig.text(
            0.025,
            y_center,
            left_name,
            va='center',
            ha='center',
            rotation=90,
            fontsize=12,
            fontweight='bold'
        )

    # ------------------------------------------------------------
    # Obere Instance- und EDV/ESV-Labels näher an die Bilder
    # ------------------------------------------------------------
    top_pos = axes[0, 0].get_position()
    y_edv_esv = top_pos.y1 + 0.008
    y_instance = top_pos.y1 + 0.040

    for k in range(n_instances):
        ax_left = axes[0, 2 * k]
        ax_right = axes[0, 2 * k + 1]

        pos_left = ax_left.get_position()
        pos_right = ax_right.get_position()

        x_center = (pos_left.x0 + pos_right.x1) / 2

        fig.text(
            x_center,
            y_instance,
            f'Case {k + 1}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

        fig.text(
            (pos_left.x0 + pos_left.x1) / 2,
            y_edv_esv,
            'EDV',
            ha='center',
            va='bottom',
            fontsize=10
        )

        fig.text(
            (pos_right.x0 + pos_right.x1) / 2,
            y_edv_esv,
            'ESV',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # ------------------------------------------------------------
    # y-Labels pro Bildpaar und pro Zeile
    # ------------------------------------------------------------
    for i, (_, _, y_target) in enumerate(plot_imgs_input):

        y_pred = all_preds[i]

        for k in range(n_instances):

            ax_left = axes[i, 2 * k]
            ax_right = axes[i, 2 * k + 1]

            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()

            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_pos = pos_left.y1 + 0.005#0.010

            fig.text(
                x_center,
                y_pos,
                rf'$y={y_target[k]:.1f}$, $\hat{{y}}={y_pred[k]:.1f}$',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # ------------------------------------------------------------
    # Schwarze gestrichelte Trennlinien zwischen Instanzen
    # ------------------------------------------------------------
    for k in range(1, n_instances):

        left_ax = axes[0, 2 * k - 1]
        right_ax = axes[0, 2 * k]

        x_sep = (
            left_ax.get_position().x1 +
            right_ax.get_position().x0
        ) / 2

        fig.add_artist(
            plt.Line2D(
                [x_sep, x_sep],
                [0.04, top_pos.y1+0.05],
                transform=fig.transFigure,
                color='black',
                linewidth=1.5,
                linestyle='--',
                alpha=0.7
            )
        )

    print('save', img_name)

    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    work_from = 1

    image_idxs=[0,1,2]
    dataset_name=['stanford','CAMUS'][0]
    fold_idx = 0
    device_str = ['cpu', 'cuda'][work_from]
    device = torch.device(device_str)
    save_folder =['C:/Users/siekiera/Documents/git_repos/GradientSPN/results/',
                  '/lustre/miifs01/project/m2_datamining/sync_data/cnn_spn_models/' ][work_from]

    data_path=['C:/Users/siekiera/Documents/git_repos/GradientSPN/',
                    '/lustre/project/ki-topml/jsiekiera/cf_data/'][work_from]
    if dataset_name=='stanford':
        model_title=('KLD4_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefinchec','')#('KLD4_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefinchec',r'')
        videos_save_folder=['../results/diffusion_results/counterfactual_videos/',
                            '/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_v3_stanford_seg_merge/eval/counterfactual_videos/'
                            ][work_from]
        weights = ['C:/Users/siekiera/Documents/git_repos/CF_eval/weights/deeplabv3_resnet50_random.pt',
                   'deeplabv3_resnet50_random.pt'
                   ][work_from]
        weights_ef=['C:/Users/siekiera/Documents/git_repos/CF_eval/ef_pred_weights_echos/best.pt',
                   '/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw/best.pt'][work_from]
        data_dir_org_masks = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                    '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][work_from]

    else:
        model_title = ('KLD4_MS_SSIM_HUBER_3_CAMUS_DS_1stage_encrefinchec_tmp2', r'')
        videos_save_folder ='/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_v3_CAMUS_merge_show/eval/counterfactual_videos/'
        weights='/lustre/project/ki-topml/jsiekiera/segmentation_weights_camus/best.pt'
        weights_ef='/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_camu_bi_le6_20/best.pt'
        data_dir_org_masks='/lustre/project/ki-topml/jsiekiera/cf_data/'
            data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][work_from]



    x_org,x_cf_SPN,y_org,y_cf_target_SPN=get_SPN_VAE_images(model_title[0],'SPN',beta=0.01,gamma=0.01,fine_tune=1,image_idxs=image_idxs)
    #x_org, x_cf_MLP, y_org, y_cf_target_MLP = get_SPN_VAE_images(model_title[0], 'MLP', beta=0.0, gamma=0.0,fine_tune=0.0,image_idxs=image_idxs)
    x_cf_SCM, y_cf_target_SCM=get_SCM_data(videos_save_folder,image_idxs,dataset_name)
    #x_cf_SCM_org, y_cf_target_SCM_org = get_SCM_data(videos_save_folder_org, image_idxs, dataset_name)
    plot_imgs_input=[
                ['Org.',x_org,y_org],
              # ["Org. 4SCM", x_cf_SCM_org, y_cf_target_SCM_org],
               ["CF 4SCM",x_cf_SCM,y_cf_target_SCM],
               #["CF MLP",x_cf_MLP,y_cf_target_MLP],
               ["CF SPN",x_cf_SPN,y_cf_target_SPN]
                ]
    for entry in plot_imgs_input:
        print(entry[0],entry[1].shape,entry[2].shape,np.min(entry[1]),np.max(entry[1]),np.min(entry[2]),np.max(entry[2]))

    print('load segentation model', flush=True)

    segmentation_model = load_segmentation_model(device=device,
                                                 model_name="deeplabv3_resnet50",
                                                 pretrained=False,
                                                 weights=weights,
                                                 train=False, device_str=device_str)

    echo = echonet.datasets.Echo(root=data_dir, split="train")
    mean_segmentation, std_segmentation = echonet.utils.get_mean_and_std(echo)
    mean_segmentation, std_segmentation = np.mean(mean_segmentation), np.mean(std_segmentation)

    ef_model = load_ef_pred_model(device=device, weights=weights_ef)
    for img_idx in range(1):
        img_name = 'cf_viz_'+dataset_name+'_org_cf.pdf'
        plot_imgs(img_name,plot_imgs_input)