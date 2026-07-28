import pandas as pd
import numpy as np




def cf_performance_to_latex_summary(all_results_clf, all_params ,num_classes ,strategy=''):
    def sci_format(x)  :  # , column):
        # if isinstance(x, float):
        #    if column == 'KLD w':
        #        return np.round(x, decimals=4)
        #    else:
        return np.round(x, decimals=2)
        # return x

    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())


    differing_params = {key for key in all_param_keys if len(set(
        tuple(all_params[model].get(key, [])) if isinstance(all_params[model].get(key, []), list) else all_params[
            model].get(key, None) for model in all_params)) > 1}

    df_clf = pd.DataFrame(all_results_clf)
    datasets = df_clf['dataset'].unique()
    if strategy!='':
        df_clf =df_clf[df_clf['strategy' ]==strategy]

    for dataset in datasets:
        print(dataset)
        df_clf_filter = df_clf[df_clf['dataset'] == dataset]

        '''

                        "dataset": dataset_name,
                        "vae_name": vae_name,
                        "beta": beta,
                        "gamma": gamma,
                        "fold_idx": fold_idx,
                        "strategy":strategy_name,
                        "model_name": model_name,
                        "MSE": MSE,
                        "MAE": MAE,
                        'L2': L2,
                        'L1': L1,
                        "validity": validity,
                        "FID": FID,
                        "RelFID":class_FID,
                        "RelL2":rel_L2,
                        "Pos region size":blop_info[0,0],
                        'Pos number of regions':[0,1],
                        "Neg region size":[1,0],
                        'Neg number of regions':[1,1],







        '''

        if num_classes >1:
            # Add parameter differences to summary dataframe
            summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name", "strategy"]).agg(
                L2=("L2", "mean"),
                # MAE=('MAE','mean'),
                # MSE=('MSE', 'mean'),
                # L2_std=("L2", "std"),
                validity=("validity", "mean"),
                # validity_std=("validity", "std"),
                FID=("FID", "mean"),
                RelFID=("RelFID" ,"mean"),
                RelL2=("RelL2" ,"mean"),
                PRS=("Pos region size" ,'mean'),
                PNR = ('Pos number of regions', 'mean'),
                NNR=('Neg number of regions' ,'mean'),
                NRS=('Neg number of regions' ,'mean'),

                # Switch_epoch=("Switch epoch", 'mean')
                # FID_std=("FID", "std")
            ).reset_index()
        else:

            summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name", "strategy"]).agg(
                L2=("L2", "mean"),
                # MAE=('MAE','mean'),
                # MSE=('MSE', 'mean'),
                Spearman=("Spearman", "mean"),
                CLF_RMSE=("CLF_RMSE", "mean"),
                CLF_MAE=("CLF_MAE", "mean"),
                FID=("FID", "mean"),
                RelL2=("RelL2", "mean"),
            ).reset_index()
        # Sort rows by 'vae_name'
        summary_df_clf = summary_df_clf.sort_values(by=['vae_name' ,'model_name'])



        # Extend dataframes with differing parameter values
        for df in [summary_df_clf]:
            if 'vae_name' in df.columns:
                for param in differing_params:
                    df[param] = df['vae_name'].map(lambda model: all_params.get(model, {}).get(param, 'N/A'))
                df.drop(columns=['vae_name'], inplace=True)  # Remove model_name column
                # df.drop(columns=['save_path'], inplace=True)  # Remove model_name column

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
                    # other_columns = ['KLD w', 'beta', 'gamma', 'model_name', 'strategy', 'validity', 'L2', 'RelL2','FID','RelFID', 'PRS', 'PNR', 'NNR', 'NRS']
                    other_columns = ['KLD w', 'beta', 'gamma', 'model_name', 'CLF_MAE', 'CLF_RMSE', 'Spearman', 'L2',
                                     'RelL2', 'FID']
                else:
                    # other_columns = ['rec_loss', 'beta', 'gamma', 'model_name', 'CLF_MAE','CLF_RMSE','Spearman', 'L2', 'RelL2','FID']
                    other_columns = [ 'beta', 'gamma', 'model_name', 'CLF_MAE', 'CLF_RMSE', 'Spearman', 'L2',
                                      'RelL2', 'FID']




                df = df[other_columns  ]  # +fixed_columns]

                df.to_csv('tmp.csv', index=False)

                col_format = "l" + "c" * (len(summary_df_clf.columns) - 1)
                summ_df = df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)
                # summ_df = df.apply(lambda col: col.apply(lambda x: sci_format(x, col.name)))
                latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)
                print(latex_table)


def segmentation_to_latex_summaey(all_results_segmentation,all_params):
    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())

    differing_params = {key for key in all_param_keys if len(set(
        tuple(all_params[model].get(key, [])) if isinstance(all_params[model].get(key, []), list) else all_params[
            model].get(key, None) for model in all_params)) > 1}


    df_clf = pd.DataFrame(all_results_segmentation)

    # Add parameter differences to summary dataframe
    summary_df_clf = df_clf.groupby(["vae_name", "beta", "gamma", "model_name","strategy"]).agg(
        CLF_RMSE=("CLF_RMSE", "mean"),
        CLF_MAE=('CLF_MAE', 'mean'),
        Spearman=("Spearman", "mean"),
        CLF_RMSE_ef=("CLF_RMSE_ef", "mean"),
        CLF_MAE_ef=('CLF_MAE_ef', 'mean'),
        Spearman_ef=("Spearman_ef", "mean"),


    ).reset_index()

    # --- TODO 1: for 'strategy' == 'original' set 'beta'='-' and 'gamma'='-' ---
    summary_df_clf.loc[summary_df_clf['strategy'] == 'None', ['beta', 'gamma','strategy']] = "-"

    # --- TODO 2: for 'strategy' == 'counterfactual' set 'beta'='-' and 'gamma'='-'
    # then regroup to compute mean again
    summary_df_clf.loc[summary_df_clf['strategy'] == 'reconstruction', ['beta', 'gamma','model_name']] = "-"
    summary_df_clf = summary_df_clf.groupby(
        ["vae_name", "beta", "gamma", "model_name", "strategy"]
    ).agg(
        CLF_RMSE=("CLF_RMSE", "mean"),
        CLF_MAE=('CLF_MAE', 'mean'),
        Spearman=("Spearman", "mean"),
        CLF_RMSE_ef=("CLF_RMSE_ef", "mean"),
        CLF_MAE_ef=('CLF_MAE_ef', 'mean'),
        Spearman_ef=("Spearman_ef", "mean"),
    ).reset_index()

    # Sort rows by 'vae_name'
    summary_df_clf = summary_df_clf.sort_values(by=['vae_name', 'model_name',"strategy"])

    # Extend dataframes with differing parameter values
    def sci_format(x):
        return f"{x:.3f}"
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

            # Ensure 'A' and 'B' are first, followed by other columns in any order
            fixed_columns = ['model_name', 'beta', 'gamma', 'strategy','CLF_MAE',"CLF_RMSE","Spearman",'CLF_MAE_ef',"CLF_RMSE_ef","Spearman_ef"]
            other_columns = [col for col in df.columns if col not in fixed_columns]

            print('DEBUG', other_columns)
            df = df[other_columns + fixed_columns]

            col_format = "l" + "c" * (len(summary_df_clf.columns) - 1)


            summ_df = df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)
            latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)
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
