#!/bin/bash

# -------------------------------
# Hyperparameter Execution Script
# -------------------------------

# Define parameter arrays
KLD_str=("4"  "5" "6") #  "5" "6" #"4" "6" "4")   # 5 "3"     # string identifiers for KLD # 4 5 6
KLD=("0.0001" "0.00001" "0.000001") # "0.00001" "0.000001" "0.0001")  # "0.0001" "0.000001""0.0001"  "0.001" # numeric KLD values
rec_loss=("MS_SSIM_HUBER")  # reconstruction loss options; add "MAE" if needed
latent_dim=("0.0001") #0.00001 #"0.001"  "0.001"       # latent dimensions; add "128" if needed
latent_dim_str=("3")   #"3"  "3"
activation=("stanford_seg") #"CAMUS" "stanford_seg" "stanford"      # activation functions
dataset_name="stanford" #_seg_estim_ef #"stanford" #_seg" #_seg #_seg
FINE_TUNE_OPTIONS=(1) #1  # fine-tune options

# Initialize save paths string
save_paths=""
#"$act"
# -------------------------------
# Construct save paths
# -------------------------------
for i in "${!KLD[@]}"; do
    for rec in "${rec_loss[@]}"; do
        for ld_idx in "${!latent_dim[@]}"; do
            for act in "${activation[@]}"; do
                ld="${latent_dim[$ld_idx]}"
                ld_str="${latent_dim_str[$ld_idx]}"

                kld_str="${KLD_str[$i]}"
                kld="${KLD[$i]}"
                #dataset_name="$act"
                # Construct save_path string
                save_path="KLD${kld_str}_${rec}_${ld_str}_${act}_DS_1stage_encrefinchec" #chec #2stage_flip" #"#_f" # s_g4" # "stan_seg0_KLD${kld_str}_${rec}_${ld}_${act}_100_gauss_00" #  #"stan_seg0_KLD${kld_str}_${rec}_${ld}_${act}_100_gauss_00" #fine_
                # _old, 
                # Uncomment to submit training jobs
                #sbatch 1_slurm_train_net "$save_path" "$dataset_name" "$kld" "$rec" 64 "relu" "$ld"
                
                save_paths+="${save_path},"
            done
        done
    done
done

# Remove trailing comma
save_paths="${save_paths%,}"

# Submit evaluation job
sbatch 2_slurm_net_eval "$save_paths" "regression" "$dataset_name"
# sbatch 2_slurm_net_eval "$save_paths" "cfe" "$dataset_name"

# Split save_paths into array
IFS=',' read -ra SAVE_PATH_ARRAY <<< "$save_paths"

# -------------------------------
# Merge CF outputs (commented)
# -------------------------------
for model_path in "${SAVE_PATH_ARRAY[@]}"; do
    # sbatch 3_slurm_merge_cf "$model_path"
    :
done

# -------------------------------
# Calculate CF with fine-tune options
# -------------------------------
for model_path in "${SAVE_PATH_ARRAY[@]}"; do
    #sbatch 3_slurm_calculate_cf_mlp "$model_path"
    for fine_tune in "${FINE_TUNE_OPTIONS[@]}"; do
        #sbatch 3_slurm_calculate_cf_single_cpu "$model_path" "$fine_tune" "$dataset_name"
        :
    done
done
