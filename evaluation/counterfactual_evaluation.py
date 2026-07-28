import re
import os
import pickle as pkl
import numpy as np
import pandas as pd
import random

from counterfactuals.find_many_counterfactuals import make_individual_plot, make_raw_plot

from evaluation.metrics.fid import compute_validity, compute_fid, compute_MAE, compute_MSE, compute_MSE_mean, \
    compute_MAE_mean, compute_class_fid, compute_rel_L2, get_importance_region_information, mean_lsp_calculation, \
    compute_ssim, compute_lpips

from evaluation.metrics.summarizing_fkts import mean_cf, max_cf, min_loss, weighted
import matplotlib.pyplot as plt

from PIL import Image

from latent_analysis import  analyze_and_plot_latent_space


def find_and_extract_files(folder_path,fine_tune=0):
    if fine_tune:
        pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new_minmax_tmp\.pkl") #_minmax
    else:
        pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new_minmax\.pkl") #_minmax #TODO change this to _minmax back

    extracted_data = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            beta_str = match.group(1).replace('-', '.')
            gamma_str = match.group(2).replace('-', '.')
            model_name = match.group(3)

            beta = float(beta_str)
            gamma = float(gamma_str)

            extracted_data.append((filename, model_name, beta, gamma))

    return extracted_data


def performance_to_latex_summary(all_results_clf, all_results_rec, all_params,num_classes):
    def sci_format(x):
        return np.round(x, decimals=5)

    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())
    #print('all_param_keys',all_param_keys,flush=True)
    #differing_params = {key for key in all_param_keys if len(set(all_params[model].get(key, None) for model in all_params)) > 1}
    differing_params = {key for key in all_param_keys if len(set(
        tuple(all_params[model].get(key, [])) if isinstance(all_params[model].get(key, []), list) else all_params[
            model].get(key, None) for model in all_params)) > 1}

    df_clf = pd.DataFrame(all_results_clf)
    df_rec = pd.DataFrame(all_results_rec)
    print(df_clf.keys(),df_rec.keys(),flush=True)
    datasets = df_clf['dataset'].unique()
    splits = df_clf['split'].unique()

    for dataset in datasets:
        for split in splits:
            print(dataset,split)
            print()
            df_clf_filter = df_clf[(df_clf['dataset'] == dataset) & (df_clf['split'] == split)]
            df_rec_filter = df_rec[(df_rec['dataset'] == dataset) & (df_rec['split'] == split)]


            if num_classes>1:
                # Add parameter differences to summary dataframe
                fixed_columns_clf = ["Accuracy", "Entropy", "AUC", "Precision", "Recall"]
                summary_df_clf = df_clf_filter.groupby(["model_name", "clf"]).agg(
                    Accuracy=("Accuracy", "mean"),
                    Entropy=("Entropy", "mean"),
                    AUC=("AUC", "mean"),
                    Precision=("Precision", 'mean'),
                    Recall=("Recall", 'mean')
                ).reset_index()
            else:
                fixed_columns_clf = ["MAE",  "RMSE", "Spearman"]
                summary_df_clf = df_clf_filter.groupby(["model_name", "clf"]).agg(
                    MAE=("MAE", "mean"),
                    #MSE=("MSE", "mean"),
                    RMSE=("RMSE", "mean"),
                    Spearman=("Spearman", 'mean')
                ).reset_index()


            summary_df_rec = df_rec_filter.groupby(["model_name"]).agg(
                MAE=("MAE", "mean"),
                #MSE=("MSE", "mean"),
                KLD=("KLD", "mean"),
                RMSE=("RMSE", "mean"),
                #SSIM=("SSIM","mean"),
                MS_SSIM=("MS_SSIM","mean"),

            ).reset_index()
            summary_df_clf = summary_df_clf.sort_values(by=['model_name'])

            # Extend dataframes with differing parameter values

            fixed_columns_rec = ['MAE','RMSE',"MS_SSIM",'KLD']
            for df,fixed_columns in zip([summary_df_clf, summary_df_rec],[fixed_columns_clf,fixed_columns_rec]):
                if 'model_name' in df.columns:
                    for param in differing_params:
                        df[param] = df['model_name'].map(lambda model: all_params.get(model, {}).get(param, 'N/A'))
                    df.drop(columns=['model_name'], inplace=True)  # Remove model_name column
                    if  'save_path' in df.columns:
                        df.drop(columns=['save_path'], inplace=True)  # Remove model_name column

                    col_names=list(df.columns)
                    # Step 1: Split 'loss_weights' into separate columns
                    if 'loss_weights' in df:

                        df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(df['loss_weights'].tolist(),index=df.index)

                        # Step 2: Identify columns with differing values
                        varying_columns = [col for col in ['Rec w', 'KLD w', 'Clf w'] if df[col].nunique() > 1]
                        print('col_names,varying_columns',col_names,varying_columns)
                        # Drop 'loss_weights' and only keep varying loss weight columns
                        df = df.drop(columns=['loss_weights'] + [col for col in ['Rec w', 'KLD w', 'Clf w'] if
                                                                 col not in varying_columns])


                    other_columns = [col for col in df.columns if col not in fixed_columns]
                    df = df[other_columns + fixed_columns]




                    col_format = "l" + "c" * (len(summary_df_rec.columns) - 1)
                    df = df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)
                    latex_table = df.to_latex(index=False, column_format=col_format, escape=False)
                    print(latex_table)
def cf_performance_to_latex_summary(all_results_clf, all_params,num_classes,strategy=''):
    def sci_format(x):#, column):
        #if isinstance(x, float):
        #    if column == 'KLD w':
        #        return np.round(x, decimals=4)
        #    else:
        return np.round(x, decimals=2)
        #return x

    # Determine parameter differences across models
    all_param_keys = set()
    #TODO uncomment this
    #for params in all_params.values():
    #    all_param_keys.update(params.keys())


    differing_params = {key for key in all_param_keys if len(set(
        tuple(all_params[model].get(key, [])) if isinstance(all_params[model].get(key, []), list) else all_params[
            model].get(key, None) for model in all_params)) > 1}

    df_clf = pd.DataFrame(all_results_clf)
    datasets = df_clf['dataset'].unique()
    if strategy!='':
        df_clf=df_clf[df_clf['strategy']==strategy]

    for dataset in datasets:
        print(dataset)
        df_clf_filter = df_clf[df_clf['dataset'] == dataset]

        if num_classes>1:
            # Add parameter differences to summary dataframe
            summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name", "strategy"]).agg(
                L2=("L2", "mean"),
                MAE=('MAE','mean'),
                RMSE=('RMSE', 'mean'),
                validity=("validity", "mean"),
                FID=("FID", "mean"),
                RelFID=("RelFID","mean"),
                RelL2=("RelL2","mean"),
                PRS=("Pos region size",'mean'),
                PNR = ('Pos number of regions', 'mean'),
                NNR=('Neg number of regions','mean'),
                NRS=('Neg number of regions','mean'),

            ).reset_index()
        else:

            summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name", "strategy"]).agg(
                L2=("L2", "mean"),
                MAE=('MAE','mean'),
                RMSE=('RMSE', 'mean'),
                SSIM=('SSIM', 'mean'),
                LPIPS=('LPIPS', 'mean'),
                Spearman=("Spearman", "mean"),
                CLF_RMSE=("CLF_RMSE", "mean"),
                CLF_MAE=("CLF_MAE", "mean"),
                FID=("FID", "mean"),
                RelL2=("RelL2", "mean"),
                umap_z_dist= ('umap z dist','mean'),
                raw_z_dist=('raw z dist','mean'),
                n_cluster=('n_cluster','mean'),
            ).reset_index()
        # Sort rows by 'vae_name'
        summary_df_clf = summary_df_clf.sort_values(by=['vae_name','model_name'])



        # Extend dataframes with differing parameter values
        for df in [summary_df_clf]:
            if 'vae_name' in df.columns:
                for param in differing_params:
                    df[param] = df['vae_name'].map(lambda model: all_params.get(model, {}).get(param, 'N/A'))
                df.drop(columns=['vae_name'], inplace=True)  # Remove model_name column
                #df.drop(columns=['save_path'], inplace=True)  # Remove model_name column

                col_names = list(df.columns)
                # Step 1: Split 'loss_weights' into separate columns
                if 'loss_weights' in df:
                    df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(df['loss_weights'].tolist(), index=df.index)

                    # Step 2: Identify columns with differing values
                    varying_columns = [col for col in ['Rec w', 'KLD w', 'Clf w'] if df[col].nunique() > 1]
                    print('col_names,varying_columns', col_names, varying_columns)
                    # Drop 'loss_weights' and only keep varying loss weight columns
                    df = df.drop(columns=['loss_weights'] + [col for col in ['Rec w', 'KLD w', 'Clf w'] if
                                                             col not in varying_columns])
                    #other_columns = ['KLD w', 'beta', 'gamma', 'model_name', 'strategy', 'validity', 'L2', 'RelL2','FID','RelFID', 'PRS', 'PNR', 'NNR', 'NRS']
                    other_columns = ['KLD w', 'beta', 'gamma', 'model_name', 'CLF_MAE', 'CLF_RMSE', 'Spearman', 'L2','MAE','RMSE','SSIM','LPIPS',
                                     'RelL2', 'FID','umap_z_dist','raw_z_dist','n_cluster']
                else:
                    #other_columns = ['rec_loss', 'beta', 'gamma', 'model_name', 'CLF_MAE','CLF_RMSE','Spearman', 'L2', 'RelL2','FID']
                    other_columns = [ 'beta', 'gamma', 'model_name', 'CLF_MAE', 'CLF_RMSE', 'Spearman', 'L2','MAE','RMSE','SSIM','LPIPS',
                                     'RelL2', 'FID','umap_z_dist','raw_z_dist','n_cluster']




                df = df[other_columns]#+fixed_columns]

                df.to_csv('tmp.csv', index=False)

                col_format = "l" + "c" * (len(summary_df_clf.columns) - 1)
                summ_df = df #df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)
                #summ_df = df.apply(lambda col: col.apply(lambda x: sci_format(x, col.name)))
                for col in other_columns:
                    if col in summ_df:
                        if col in ['KLD w']:
                            summ_df[col] = summ_df[col].apply(lambda x: f"{x:.4f}")
                        elif col in ['n_cluster']:
                            summ_df[col] = summ_df[col].apply(lambda x: f"{int(round(x))}")
                        elif col in [ 'beta', 'gamma', 'CLF_MAE', 'CLF_RMSE', 'Spearman', 'L2','MAE','RMSE','SSIM','LPIPS',
                                     'RelL2', 'FID','umap_z_dist','raw_z_dist']:
                            summ_df[col] = summ_df[col].apply(lambda x: f"{x:.3f}")
                latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)#
                print(latex_table)



def z_eval_cf_performance_to_latex_summary(all_results_clf, all_params):
    def sci_format(x):
        return np.round(x, decimals=2)

    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())


    differing_params = {key for key in all_param_keys if len(set(
        tuple(all_params[model].get(key, [])) if isinstance(all_params[model].get(key, []), list) else all_params[
            model].get(key, None) for model in all_params)) > 1}

    df_clf = pd.DataFrame(all_results_clf)
    datasets = df_clf['dataset'].unique()

    for dataset in datasets:
        print(dataset)
        df_clf_filter = df_clf[df_clf['dataset'] == dataset]



        # Add parameter differences to summary dataframe
        summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name"]).agg(
            MaxIdx=("Max idx", "mean"),
            MaxFreq=('Max freq','mean'),
            MedianIdx=("Median idx", "mean"),
            MedianFreq=('Median freq', 'mean'),
            MinIdx=("Min idx", "mean"),
            MinFreq=('Min freq','mean'),
            DivFreq=("Diversity freq.", "mean"),

        ).reset_index()
        # Sort rows by 'vae_name'
        summary_df_clf = summary_df_clf.sort_values(by=['vae_name','model_name'])



        # Extend dataframes with differing parameter values
        for df in [summary_df_clf]:
            if 'vae_name' in df.columns:
                for param in differing_params:
                    df[param] = df['vae_name'].map(lambda model: all_params.get(model, {}).get(param, 'N/A'))
                df.drop(columns=['vae_name'], inplace=True)  # Remove model_name column
                df.drop(columns=['save_path'], inplace=True)  # Remove model_name column

                col_names = list(df.columns)
                # Step 1: Split 'loss_weights' into separate columns

                df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(df['loss_weights'].tolist(), index=df.index)

                # Step 2: Identify columns with differing values
                varying_columns = [col for col in ['Rec w', 'KLD w', 'Clf w'] if df[col].nunique() > 1]
                print('col_names,varying_columns', col_names, varying_columns)
                # Drop 'loss_weights' and only keep varying loss weight columns
                df = df.drop(columns=['loss_weights'] + [col for col in ['Rec w', 'KLD w', 'Clf w'] if
                                                         col not in varying_columns])

                # Ensure 'A' and 'B' are first, followed by other columns in any order
                fixed_columns = ['model_name', 'beta','gamma','MaxIdx','MaxFreq','MedianIdx','MedianFreq','MinIdx','MinFreq','DivFreq']
                other_columns = [col for col in df.columns if col not in fixed_columns]

                print('DEBUG',other_columns)
                df = df[other_columns+fixed_columns]



                col_format = "l" + "c" * (len(summary_df_clf.columns) - 1)
                summ_df = df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)
                latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)
                print(latex_table)


def df_to_latex_summary(df):
    """Summarizes the DataFrame and converts it to a LaTeX table."""

    # Group by model configs and compute mean & std
    summary_df = df.groupby(["vae_name", "beta", "gamma", "model_name"]).agg(
        L2_mean=("L2", "mean"),
       # L2_std=("L2", "std"),
        validity_mean=("validity", "mean"),
        #validity_std=("validity", "std"),
        FID_mean=("FID", "mean"),
        Switch_epoch=("Switch epoch",'mean')
        #FID_std=("FID", "std")
    ).reset_index()

    # Convert values to scientific notation (short format)
    def sci_format(x):
        return np.round(x,decimals=2)
        #return f"{x:.2e}"  # 1.23e-2 format

    summary_df = summary_df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)

    # Convert to LaTeX table
    latex_table = summary_df.to_latex(index=False, column_format="llllcccc", escape=False)

    print(latex_table)
    return latex_table

def diff_counterfactual_z(rec_z,reconstructions):
    # x_repeat = np.repeat(x[:, :, :, 0], reconstructions.shape[0], axis=0)
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255
    differences = org_rec_z - reconstructions_int
    differences_mean = np.mean(differences, axis=0)
    return differences_mean


def plot_cfs_3D(dataset_name,data_path_grid,model_name, num_imgs, num_imgs_per_plot, method_name,
                beta, gamma,fold_idxs, path='',num_frames=2,use_background=True):
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs/'
        num_plots=num_imgs//num_imgs_per_plot

        for plt_counter in range(num_plots):
            save_file_path_plot = data_path_fold + str(plt_counter) + 'image_no_w_' + method_name+  'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + '.png'
            fig_size = 4
            fig, axis = plt.subplots(num_frames,num_imgs_per_plot,figsize=(fig_size *num_imgs_per_plot, fig_size * num_frames))
            save_file_path = data_path_fold + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + method_name + '_1000_new.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)
            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, pred,
             y_cf_goal, coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org = np.asarray(x_org)
            mean_cls=np.squeeze(np.asarray(mean_cls))
            y_cf_goal=np.squeeze(np.asarray(y_cf_goal))
            y_org = np.squeeze(np.asarray(y_org))
            arg_max=np.mean(np.squeeze(np.asarray(arg_max)),axis=1)
            for img_num in range(num_imgs_per_plot):
                image_num=img_num+plt_counter*num_imgs_per_plot
                differences_mean = diff_counterfactual_z(x_cf[image_num], x_rec[image_num])
                mini_title = 'y:' + str(np.round(y_org[image_num],2))+ \
                             ';y_pr:' + str(np.round(mean_cls[image_num],2))+\
                             '\ncf:' + str(np.round(y_cf_goal[image_num],2)) +\
                             ';cf_pr:' + str(np.round(arg_max[image_num],2))
                axis[0,img_num].set_title(mini_title)
                for frame in range(num_frames):
                    if use_background:
                        axis[frame,img_num].imshow(x_org[image_num][0, :, :, frame,0], cmap='gray')
                        axis[frame,img_num].imshow(differences_mean[:,:,frame], cmap='jet', alpha=0.5)  # ,vmax=vmax,vmin=vmin
                    else:
                        axis[frame,img_num].imshow(differences_mean[:,:,frame], cmap='gray')
                    axis[frame,img_num].axis('off')

            plt.tight_layout()
            print('save', save_file_path_plot)
            plt.show()
            plt.savefig(save_file_path_plot)

def load_data(file_path,test_data,num_classes, model_name ):
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
        loss.append(np.asarray(list(loss_)))

    pred = np.concatenate(pred,axis=0)
    y_cf_goal= np.concatenate(y_cf_goal, axis=0)
    y_org = np.concatenate(y_org, axis=0)
    x_org = np.concatenate(x_org, axis=0)

    x_cf=np.concatenate(x_cf,axis=1)
    x_rec = np.concatenate(x_rec, axis=1)
    z = np.concatenate(z, axis=1)
    z_cf = np.concatenate(z_cf, axis=1)
    arg_max = np.concatenate(arg_max, axis=1)
    loss = np.concatenate(loss, axis=1)


    print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org), np.min(x_rec), np.max(x_rec))
    # transpose:(test_size,2,h,w,1)

    print(x_cf.shape, x_org.shape, x_rec.shape)
    x_cf = np.transpose(x_cf, (1, 0, 3, 4, 2, 5))
    x_org = np.transpose(x_org, (0, 2, 3, 1, 4))
    x_rec = np.transpose(x_rec, ( 1,0, 3, 4, 2, 5))
    print(x_cf.shape, x_org.shape, x_rec.shape)

    z = np.asarray(z)
    z_cf = np.asarray(z_cf)
    arg_max = np.squeeze(np.asarray(arg_max))

    x_rec_mean = np.mean(x_rec, axis=1)
    if num_classes == 1 and model_name == 'SPN':
        pred = np.exp(pred)
    # TODO for MLP pred = sigmoid????
    abs_change = np.abs(z_cf - z)
    return x_org,x_rec_mean,x_cf,loss,pred,y_cf_goal,z, z_cf, y_org,arg_max



def eval_counterfactuals_new_format(dataset_name,
                         params,
                         #num_train_eval_runs,
                         data_path_grid,
                         #add_info,
                         vae_name,
                         fold_idxs,
                         possible_values,test,num_classes,fine_tune=0):
    latent_transformers = []
    # TODO for all created files that can be found - evaluate counterfactuals and save them in pandas datafram with model configs:vae_name, beta gamma foldidx,model_name,L2,validity,FID
    random.seed(0)
    test_idx=list(range(test[0].shape[0]))
    test_sample=random.shuffle(test_idx)
    test_idx=test_idx[:1200]

    results = []
    results_z_eval=[]
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs'
        files_data = find_and_extract_files(data_path_fold,fine_tune=fine_tune)
        print(files_data)

        for file_name, model_name, beta, gamma in files_data:
            if beta in possible_values and gamma in possible_values:
                print(f"File: {file_name} | Model: {model_name} | Beta: {beta} | Gamma: {gamma}",flush=True)

                x_org,x_rec_mean,x_cf,loss,pred,y_cf_goal,z, z_cf, y_org,arg_max=load_data(data_path_fold+'/'+file_name,test,num_classes, model_name)

                for s_idx,(strategy_name,strategy,input_arr,summarize_strategy) in enumerate( zip(['mean','min_loss'],#,'max_cf','weighted_loss',], # 'max_cf','weighted_loss','mean','min_loss'
                                                            [mean_cf,min_loss,max_cf,weighted], #max_cf,weighted,mean_cf,min_loss
                                                            [None,loss,(pred,y_cf_goal,num_classes),loss],[False,True,True,True])):#log_pred, #(pred,y_cf_goal,num_classes),loss,None,loss

                    x_cf_strat,arg_max_process,z_cf_mean,z_mean=strategy(x_cf,input_arr,arg_max,z_cf,z)
                    plt_path=data_path_fold+'/'+ model_name+'_'+str(beta)+'_'+str(gamma)+'_'+strategy_name+'_'+str(fine_tune)

                    #                                                                           (latent_transformer,z, z_cf,ef,ef_cf_target, plt_name, random_state=0, k=1)
                    #Z_space_distance_humap,z_space_raw,n_clusters=analyze_and_plot_latent_space(latent_transformers[s_idx],z_mean, z_cf_mean,y_org,y_cf_goal, plt_name=plt_path, k=1,use_fit_transform=summarize_strategy)
                    Z_space_distance_humap, z_space_raw, n_clusters = analyze_and_plot_latent_space(
                        None, z, z_cf, y_org, y_cf_goal, plt_name=plt_path, k=1,
                        use_fit_transform=summarize_strategy)

                    class_FID=0
                    ssim=compute_ssim(x_org, x_cf_strat,params.conv_dim)
                    lpips=compute_lpips(x_org, x_cf_strat,params.conv_dim)
                    if num_classes>1:
                        class_FID = compute_class_fid(test[0][test_idx], x_cf_strat, x_rec_mean, np.round(y_cf_goal), test[1][test_idx,0], np.round(arg_max_process))
                        blop_info = get_importance_region_information(x_rec_mean, x_cf_strat, threshold=0.5)
                        validity = compute_validity(np.round(y_cf_goal), np.round(arg_max_process),num_classes=num_classes)
                    else:
                        pred_y_reg=arg_max_process
                        if model_name=='SPN':
                            pred_y_reg=np.exp(pred_y_reg)

                        validity_MAE,validity_RMSE,validity_spearman = compute_validity(y_cf_goal,pred_y_reg,num_classes=num_classes)
                    RMSE,L2=compute_MSE(x_org, x_cf_strat,params.conv_dim)
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
                            "RMSE": RMSE,
                            "MAE": MAE,
                            'L2': L2,
                            'L1': L1,
                            'SSIM':ssim,
                            'LPIPS':lpips,
                            "validity": validity,
                            "FID": FID,
                            "RelFID": class_FID,
                            "RelL2": rel_L2,
                            'umap z dist': Z_space_distance_humap,
                            'raw z dist': z_space_raw,
                            "Pos region size": blop_info[0, 0],
                            'Pos number of regions': blop_info[0, 1],
                            "Neg region size": blop_info[1, 0],
                            'Neg number of regions': blop_info[1, 1],
                            'n_cluster': n_clusters,

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
                            "RMSE": RMSE,
                            "MAE": MAE,
                            'SSIM': ssim,
                            'LPIPS': lpips,
                            'L2': L2,
                            'L1': L1,
                            "FID": FID,
                            "RelL2": rel_L2,
                            'umap z dist':Z_space_distance_humap,
                            'raw z dist':z_space_raw,
                            'n_cluster':n_clusters,
                            "CLF_RMSE": validity_RMSE,
                            "CLF_MAE": validity_MAE,
                            'Spearman': validity_spearman,


                        })



                #label_switch_step_mean = np.mean([e for e in label_switch_step if e >= 0])
                '''
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

    return results,results_z_eval













