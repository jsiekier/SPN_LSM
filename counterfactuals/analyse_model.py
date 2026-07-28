import tensorflow as tf
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
#from sklearn.model_selection import KFold

from CNN_SPN import CNN_SPN_Parts, test_model_SPN_MLP #test_model_all
from CNN_functions import load_VAE
from end_to_end_train import load_dataset
from tf2_spn import create_tf_spn_parts, create_multiple_tf_spn_parts
import numpy as np
import time
from spn.algorithms.Statistics import get_structure_stats

# 0. load model
# 1. evaluate on test
# 2. show example of FP/FP/TP/TN (very certain)


def load_VAE_model( add_info, num_classes, input_shape,data_path_fold,params,path='',fine_tune=0):
    if fine_tune:
        mlp_path = data_path_fold+'mlp_checkpoints_finetune/tf_ckpts_last'
    else:
        mlp_path=''

    vae_save_path=data_path_fold+'vae_checkpoints_copy/tf_ckpts_last'
    print('TODO _copy')
    # 1. load vae:
    vae_model = load_VAE(params, add_info, num_classes, input_shape, vae_save_path,path,mlp_path)
    return vae_model



def load_model( add_info, num_classes, input_shape,data_path_fold,params,path='',fine_tune=0):
    vae_save_path=data_path_fold+'vae_checkpoints/tf_ckpts_last'
    checkpoint_path_cnn_spn = data_path_fold + 'cnn_spn_checkpoints'


    # 1. load vae:
    vae_model = load_VAE(params, add_info, num_classes, input_shape, vae_save_path,path)
    # 2. load spn_structure
    spn_data=pkl.load(open(data_path_fold+ 'spn.pkl', 'rb'))
    spn_clf=spn_data['spn_x']
    spn_input_shape=spn_data['data_shape']
    label_ids=spn_data['label_ids']
    bin_borders=spn_data['bin_borders']
    bin_widths_scaled=spn_data['bin_widths_scaled']
    spn_z=spn_data['spn_z']
    #
    #print(get_structure_stats(spn_clf))
    print(get_structure_stats(spn_z))
    print('STRUCTURE STATS', flush=True)
    '''
    spn_x_copy, all_spn_x_y, all_spn_x_y_dicts, all_prior, all_spn_x_y_model = create_tf_spn_parts(spn_x=spn_clf,
                                                                                                   data_shape=spn_input_shape,
                                                                                                   label_ids=label_ids,
                                                                                                   trainable_leaf=params.fine_tune_leafs)
                                                                                                   
    '''

    spn_output=spn_clf, label_ids, bin_borders, bin_widths_scaled, spn_z
    all_spn_x_y_model, all_prior, spn_model_z = create_multiple_tf_spn_parts(spn_out_data=spn_output,
                                                                             num_classes=num_classes,
                                                                           data_shape=spn_input_shape,
                                                                             trainable_leaf=params.fine_tune_leafs)

    all_spn_params = spn_model_z.count_params()
    for m_i,model in enumerate(all_spn_x_y_model):
        all_spn_params+=model[0].count_params()
        all_spn_params += model[1].count_params()
        #print(get_structure_stats(spn_clf[m_i][0]),get_structure_stats(spn_clf[m_i][1]))
    print('SPN model params!', all_spn_params,flush=True)


    decoder = None
    if params.use_VAE:
        cnn_embedding = vae_model.clf_model()
        if params.VAE_fine_tune:
            decoder = vae_model.decoder
    else:
        cnn_layer_out = vae_model.get_layer('flatten').output
        cnn_embedding = tf.keras.Model(inputs=vae_model.input, outputs=cnn_layer_out)


    # 3. load cnn_spn
    cnn_spn = CNN_SPN_Parts(num_classes=num_classes,
                            learning_rate=params.fine_tune_rate,
                            spn_model_z=spn_model_z,
                            bin_borders=bin_borders,
                            all_spn_x_y_model=all_spn_x_y_model,
                            all_prior=all_prior,
                            bin_widths_scaled=bin_widths_scaled,
                            cnn=cnn_embedding,
                            get_max=0,
                            gauss_embeds=params.gauss_embeds,
                            use_add_info=params.use_add_info,
                            VAE_fine_tune=params.VAE_fine_tune,
                            decoder=decoder,
                            loss_weights=params.loss_weights,
                            load_pretrain_model=params.load_pretrain_model,
                            clf_mlp=vae_model.classifier,conv_dim=params.conv_dim,rec_loss_name=params.rec_loss)

    ckpt_cnn_spn = tf.train.Checkpoint(step=tf.Variable(1), optimizer=cnn_spn.optimizer, net=cnn_spn)
    manager_cnn_spn = tf.train.CheckpointManager(ckpt_cnn_spn, checkpoint_path_cnn_spn, max_to_keep=1)
    ckpt_cnn_spn.restore(manager_cnn_spn.latest_checkpoint)

    if fine_tune:
        start=time.time()
        print('use finetuned spns')
        ckpt_clf_spn = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.clf_spn)
        manager_clf_spn = tf.train.CheckpointManager(ckpt_clf_spn,  data_path_fold + 'spn_checkpoints_finetune/tf_ckpts_last', max_to_keep=1)
        ckpt_clf_spn.restore(manager_clf_spn.latest_checkpoint)
        print("Restored from {}".format(manager_clf_spn.latest_checkpoint))

        ckpt_z_spn = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.spn_model_z)
        manager_z_spn = tf.train.CheckpointManager(ckpt_z_spn,  data_path_fold + 'spn_z_checkpoints_finetune/tf_ckpts_last', max_to_keep=1)
        ckpt_z_spn.restore(manager_z_spn.latest_checkpoint)
        print("Restored from {}".format(manager_z_spn.latest_checkpoint),(time.time()-start)/60,flush=True)

    return cnn_spn




def sorted_plot_data(gt, predictions,pred_exp, images):
    zip_list=list(zip(gt, predictions,pred_exp, images))
    TP=[entry for entry in zip_list if (entry[0] and entry[1]) ]
    TN = [entry for entry in zip_list if (not entry[0] and not entry[1]) ]
    FP = [entry for entry in zip_list if (not entry[0] and entry[1]) ]
    FN = [entry for entry in zip_list if (entry[0] and not entry[1]) ]

    all_data=[TP,TN,FP,FN]
    all_data_sorted=dict()
    titles=['TP','TN','FP','FN']
    for entry,data_title in zip(all_data,titles):
        all_data_sorted[data_title]=list(sorted(entry,key=lambda x:x[2],reverse=True))


    return all_data_sorted


def plot_image_grid(images, ax, title,cmap='gray', alpha=1.0):
    """Helper function to plot a 4x4 grid of images."""
    for i in range(16):
        ax[i // 4, i % 4].imshow(images[i], cmap=cmap,alpha=alpha)
        ax[i // 4, i % 4].axis('off')  # Turn off axis for each image
    # Set the header for the grid
    ax[0, 1].set_title(title, fontsize=16, pad=20)

def plot_examples(sorted_data, input_data_name,plot_path,cnn_spn_model):
    for key,value in sorted_data.items():
        vals=value[:16]
        if len(vals)==16:
            # create reconstructions:
            gt, predictions, pred_exp,images=zip(*vals)
            images=np.asarray(list(images))

            test_rec = cnn_spn_model.vae_rec(images)

            # Create a figure with two sets of 4x4 grids (one 8x8 plot with 16 subplots)
            fig, axs = plt.subplots(4, 8, figsize=(12, 6))

            # First grid (left 4x4) for test_rec
            plot_image_grid(test_rec, axs[:, :4], "Reconstructed Images")

            # Second grid (right 4x4) for images
            plot_image_grid(images, axs[:, 4:], "Original Images")

            # Add an overall title
            plt.suptitle(key+' '+input_data_name, fontsize=18)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the overall title

            # Display the combined plot
            #plt.show()
            plt.savefig(plot_path+key+'_'+input_data_name+'.png')
            plt.clf()



def analyse_pipeline(dataset_name,params,num_train_eval_runs,data_path_grid,add_info,fold_idxs,model_name,
                     path='../',debugging_plots=False,fine_tune=0):
    # 0. load model


    #input_shape=(128,128,3)
    # load data:
    train, test_data, num_classes = load_dataset(dataset_name, binary=False,
                                                                     load_net=params.load_pretrain_model,
                                                                     machine=params.machine,path=path,grid_params=params)
    statistics_names=['Accuracy','AUC','Rec:MSE']
    result_clf=[]
    result_rec=[]

    for f_i in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(f_i) + '/'
        '''
        if dataset_name == 'chexpert' or 'stanford' in dataset_name:
            ([train_data, val_data]) = train


        elif dataset_name == 'CXR_process_org_distr':
            kf = KFold(n_splits=num_train_eval_runs, random_state=1, shuffle=True)

            for i,(train_index, val_index) in enumerate(kf.split(train[0])):
                if i ==f_i:

                    random.seed(0)
                    random.shuffle(train_index)
                    random.shuffle(val_index)
                    train_x, val_x = train[0][train_index], train[0][val_index]
                    train_y, val_y = train[1][train_index], train[1][val_index]
                    train_data=(train_x,train_y)
        else:
            train_data=None
        '''



        for input_data , input_data_name in zip([test_data],['Test']): #zip([test_data,train_data],['Test','Train']):
            print(input_data_name)

            input_shape = test_data[0].shape[1:]
            cnn_spn_model=load_model( add_info, num_classes, input_shape,data_path_fold,params,path='../',fine_tune=fine_tune)

            # 1. evaluate on test

            results_MLP,results_SPN,losses = test_model_SPN_MLP(cnn_spn_model,
                                                                input_data,
                                                                num_classes=num_classes,
                                                                batch_size=params.batch_size,
                                                                training=False, add_info=add_info,
                                                                debugging_plots=debugging_plots,
                                                                debugging_plot_path=data_path_fold+'clf_corr_png'
                                                                )

            for result, clf_name in zip([results_MLP,results_SPN],['MLP','SPN']):
                print(result,flush=True)
                #[mae, mse,res]
                if num_classes>1:
                    result_clf.append({
                        'split':input_data_name,
                        "dataset": dataset_name,
                        "clf": clf_name,
                        "fold_idx": f_i,
                        "model_name": model_name,
                        'Accuracy':result[0],
                        'Entropy':result[1],
                        'Balanced Accuracy':result[2],
                        'Precision':result[3],
                        'Recall':result[4],
                        'F1-Score':result[5],
                        'AUC':result[6]
                    })
                else:
                    result_clf.append({
                        'split':input_data_name,
                        "dataset": dataset_name,
                        "clf": clf_name,
                        "fold_idx": f_i,
                        "model_name": model_name,
                        'MAE':result[0],
                        'MSE':result[1],
                        'RMSE':result[2],
                        'Spearman':result[3],

                    })
            #new_loss=np.asarray([loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy(),other_loss[0].numpy(), other_loss[1].numpy(),other_loss[3].numpy(),other_loss[4].numpy()])
            result_rec.append({
                'split': input_data_name,
                "dataset": dataset_name,
                "fold_idx": f_i,
                "model_name": model_name,
                'MSE':losses[4],
                'MAE':losses[5],
                'RMSE':losses[6],
                'KLD':losses[3],
                'SSIM':losses[7],
                'MS_SSIM':losses[8],
            })

            # Names of the metrics
            #result_names = ['Accuracy', 'Entropy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

            # Create a formatted output string with 2 decimal places
            #output_acc = ", ".join([f"{name}: {value:.2f}" for name, value in zip(result_names, val_losses[:-1])])

            # Names of the metrics
            #[loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy()])
            #result_names = ['Loss', 'Rec:MSE', 'CLF:Entropy', 'KLD','Rec:MAE']

            #output_loss=", ".join([f"{name}: {value:.4f}" for name, value in zip(result_names, val_losses[-1])])
            #print(output_acc)
            #print(output_loss)
            #result[-1].append([val_losses[0],val_losses[6],val_losses[-1][1]])
            '''
            #dataset[0].shape[0]//batch_size
            print('create sorted data')
            sorted_data=sorted_plot_data(gt, predictions,pred_exp,val_x[:(val_x.shape[0]//params.batch_size)*params.batch_size])
            
            # 2. show example of FP/FP/TP/TN (very certain)
            print('2. show example of FP/FP/TP/TN (very certain)')
            plot_path=data_path_fold+'eval_imgs/'
            if not os.path.isdir(plot_path):
                os.mkdir(plot_path)

            plot_examples(sorted_data,input_data_name,plot_path,cnn_spn_model)
            '''
    #result=np.asarray(result)
    #return result,statistics_names
    print(result_clf)
    print(result_rec)
    return result_clf,result_rec



