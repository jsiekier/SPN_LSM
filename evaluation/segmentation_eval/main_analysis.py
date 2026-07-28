import pickle as pkl
import argparse
import os
import pandas as pd
import re
from cf_analysis import plot_cfs_3D_gif, load_segmentation_model, get_cf_metrics
import torch
import echonet
import numpy as np
print(torch.__version__)


def find_and_extract_files(folder_path):
    pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000_new\.pkl")

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



def plot_3D_cfs_per_model(model_names, path, num_imgs, num_imgs_per_plot):#, method_name, beta, gamma):
    for model_name in model_names:
        print(model_name)
        # num_train_eval_runs=3
        grid_idx = 0

        data_path_grid = save_folder + model_name + '/grid' + str(grid_idx) + '/'

        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs'
        files_data = find_and_extract_files(data_path_fold)
        print(files_data)



        # load segmentation model once:
        weights = ['C:/Users/siekiera/Documents/git_repos/CF_eval/segmentation_model_weights/best.pt',
                   '/lustre/project/ki-topml/jsiekiera/segmentation_weights_echos'][0]
        data_dir=['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                  '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][0]
        segmentation_model=load_segmentation_model(device=torch.device("cpu"),
                                model_name="deeplabv3_resnet50",
                                pretrained=False,
                                weights=weights,
                                train=False)
        # get normalizing params:
        mean_segmentation, std_segmentation = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
        mean_segmentation, std_segmentation=np.mean(mean_segmentation),np.mean(std_segmentation)



        for file_name, method_name, beta, gamma in files_data:
            print(f"File: {file_name} | Model: {method_name} | Beta: {beta} | Gamma: {gamma}",flush=True)
            beta, gamma=int(beta), int(gamma)
            plot_cfs_3D_gif(dataset_name,
                     data_path_grid,
                     model_name,num_imgs, num_imgs_per_plot, method_name, beta, gamma,
                     fold_idxs=[fold_idx],
                     path=path,segmentation_model=segmentation_model,mean_segmentation=mean_segmentation,std_segmentation=std_segmentation)
if __name__ == '__main__':
    work_from=1 # 0= laptop


    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add the --filepath argument
    parser.add_argument("--model_names", type=str,default='stan_KLD4_MS_SSIM_HUBER_64_relu_DS_debug',help="Path to the input files.")
    parser.add_argument("--dataset_name", type=str, default="stanford", help="Path to the dataset")
    parser.add_argument("--fine_tune", type=int, default=0)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the filepath
    model_names= args.model_names.split(',')
    fine_tune = args.fine_tune

    dataset_name=args.dataset_name
    print(dataset_name)
    fold_idx = 0

    save_folder =['C:/Users/siekiera/Documents/git_repos/GradientSPN/results/',
                  '/lustre/project/ki-topml/jsiekiera/cnn_spn_models/' ][work_from]

    eval_data_path=['C:/Users/siekiera/Documents/git_repos/GradientSPN/',
                    '/lustre/project/ki-topml/jsiekiera/cf_data/'][work_from]
    num_classes=1
    possible_values = [0, 1]
    get_cf_metrics(model_names, possible_values, num_classes=1, conv_dim=3, strategy='mean',
                   path=eval_data_path,save_folder =save_folder,dataset_name=dataset_name,fold_idx=fold_idx,work_from=work_from,fine_tune=fine_tune)
    #plot_3D_cfs_per_model(model_names,path=eval_data_path,num_imgs=10,num_imgs_per_plot=5)#,method_name='MLP',beta=0,gamma=0)



