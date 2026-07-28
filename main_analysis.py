import pickle as pkl
import argparse
from counterfactuals.analyse_model import analyse_pipeline
from end_to_end_train import load_dataset
from evaluation.counterfactual_evaluation import performance_to_latex_summary, cf_performance_to_latex_summary, \
     eval_counterfactuals_new_format
import os
import pandas as pd

os.environ["TFHUB_CACHE_DIR"] = "tfhub_modules"


def get_performance_metrics(model_names, fold_idxs, num_train_eval_runs=3, grid_idx=0,path='',num_classes=1,debugging_plots=False,fine_tune=True):
    print('Show fine tune results:',fine_tune)
    all_results_clf = []
    all_results_rec = []
    all_params = {}

    for model_name in model_names:
        print(model_name)

        data_path_grid = save_folder + model_name + '/grid' + str(grid_idx) + '/'
        params_path = os.path.join(data_path_grid, 'grid_params.pkl')

        if not os.path.exists(params_path):
            print(f"Parameter file not found for {model_name}, skipping...",params_path)
            continue

        params = pkl.load(open(params_path, 'rb'))
        all_params[model_name] = params
        add_info=params.add_info

        # Check if results already exist
        rec_path = os.path.join(data_path_grid, str(fine_tune)+"rec.csv")
        clf_path = os.path.join(data_path_grid, str(fine_tune)+"clf.csv")

        print(f"Running analysis for {model_name}")
        result_clf, result_rec = analyse_pipeline(dataset_name, params, num_train_eval_runs, data_path_grid,
                                                  add_info, fold_idxs, model_name,path=path,
                                                  debugging_plots=debugging_plots,fine_tune=fine_tune)

        # Save results
        pd.DataFrame(result_rec).to_csv(rec_path, index=False)
        pd.DataFrame(result_clf).to_csv(clf_path, index=False)
        print('Results saved')
        ###########################

        all_results_rec.extend(result_rec)
        all_results_clf.extend(result_clf)

    # Print to LaTeX
    performance_to_latex_summary(all_results_clf, all_results_rec, all_params,num_classes)



def get_cf_metrics(model_names,possible_values,num_classes=1,conv_dim=3,strategy='',path='',fine_tune=0):
    print('Show fine tune results:', fine_tune)
    all_results= []
    all_results_z_eval=[]
    all_params = {}

    for model_name in model_names:
        print(model_name)
        #num_train_eval_runs=3
        grid_idx=0

        data_path_grid=save_folder + model_name+'/grid'+str(grid_idx)+'/'

        # Check if results already exist
        cf_path = os.path.join(data_path_grid, str(fine_tune)+"cf_new1.csv")#1
        z_eval_path = os.path.join(data_path_grid, str(fine_tune)+"z_eval_new1.csv")#1

        params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))
        train, test, num_classes = load_dataset(dataset_name, binary=False,
                                                                     load_net=params.load_pretrain_model,
                                                                     machine=params.machine,grid_params=params,path=path)
        print(f"Running analysis for {model_name}")

        cf_result, results_z_eval = eval_counterfactuals_new_format(dataset_name, params, data_path_grid, model_name,
                                                         fold_idxs=[fold_idx], possible_values=possible_values,
                                                         test=test, num_classes=num_classes, fine_tune=fine_tune)
        print(cf_result)
        pd.DataFrame(cf_result).to_csv(cf_path, index=False)
        pd.DataFrame(results_z_eval).to_csv(z_eval_path, index=False)
        print('Results saved')


        all_results.extend(cf_result)
        all_results_z_eval.extend(results_z_eval)
        all_params[model_name]=''#'#params
    for strategy in ['mean','max_cf','weighted_loss','min_loss']:
        cf_performance_to_latex_summary(all_results, all_params,strategy=strategy,num_classes=num_classes)
    #z_eval_cf_performance_to_latex_summary(all_results_z_eval, all_params)





if __name__ == '__main__':
    work_from=1#1 # 0= laptop

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add the --filepath argument
    parser.add_argument("--model_names", type=str,default='stan_KLD4_MS_SSIM_HUBER_64_relu_DS_debug',help="Path to the input files.")
    parser.add_argument("--dataset_name", type=str, default="stanford", help="Path to the dataset")
    parser.add_argument("--eval_method", type=str, default='cfe', help="methods to be applied", nargs='+',choices=['regression','cfe'])
    parser.add_argument("--fine_tune", type=int, default=0)


    # Parse the command-line arguments
    args = parser.parse_args()

    fine_tune=args.fine_tune
    #eval_method_names=args.eval_method
    model_names= args.model_names.split(',')
    print('FINE TUNE MODEL RESULTS',fine_tune)

    dataset_name=args.dataset_name
    fold_idx = 0

    save_folder =['results/',
                  '/data_in/cnn_spn_models/'
                  ][work_from]



    eval_data_path= '/data_in/cf_data/'
    num_classes=1
    debugging_plots=True
    if 'regression' in args.eval_method:
        print('regression',flush=True)
        get_performance_metrics(model_names,[0],path=eval_data_path,num_classes=num_classes,debugging_plots=debugging_plots,fine_tune=fine_tune)
    if 'cfe' in args.eval_method:
        print('CFE', flush=True)
        possible_values=[0,1]
        print(model_names)
        get_cf_metrics(model_names,possible_values,num_classes=1,conv_dim=3,strategy='mean',path=eval_data_path,fine_tune=fine_tune)
