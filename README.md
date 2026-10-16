# SPN\_LSM

This repository provides code for the implementation of our paper **Counterfactual Explanations in Medical Imaging:
Exploring SPN-Guided Latent Space Manipulation**, submitted to 24th International Workshop on Data Mining in Bioinformatics (BioKDD25). The code includes preprocessing datasets, training the model, generating counterfactuals, and evaluating their quality.

## Repository Structure

* **preprocessing.py**: Script for dataset preprocessing.
* **main\_pipeline.py**: Code for training the SPN-VAE model.
* **cf\_creation.py**: Script for generating counterfactuals for trained models.
* **main\_analysis.py**: Code for evaluating the quality of generated counterfactuals.
* **example\_plot.png**: Example visualization of results.

## Getting Started

### Prerequisites

* Python 3.8+
* Required libraries can be installed via:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing


```bash
python dataset_preprocessing/stanford/still_images_preprocessing.py
python dataset_preprocessing/CAMUS/preprocess_main.py
```

### 2. Model Training

Set save_folder='/data_in/cnn_spn_models/' and data_path='/data_in/cf_data/' in main_pipeline.py to your train data paths of the preprocessing.
To train the SPN-VAE model, use:

```bash
python main_pipeline.py  --min_features_slice 1 --use_KLD_anneal 0 --epochs 0 --fine_tune_its 52 --conv_dim 3 --rec_loss "MS_SSIM_HUBER" --batchnorm_integration 0 --use_VAE 1 --regression 10 --col_split "rdc" --diff_privacy 0 --machine 'mogon' --dataset_name "$dataset_name" --le_warmup 0 --load_pretrain_model 0 --gauss_std 0.0 --save_path "$save_path" --loss_weights 1.0 $kld 1.0 --dropout 0.0 --use_add_info 0 --VAE_debug 0 --latent_dim 64 --separate_view 0 --learning_rate 0.0001 --fine_tune_rate 0.0001 --batch_size 100 --shortcut 1 --activation "relu" --VAE_fine_tune "SPN_VAE_MLP"
```

### 3. Counterfactual Generation

For creating counterfactuals with the trained model:

```bash
dataset_name="stanford"
python cf_creation_all.py --filepath "$model_name" --beta 0.001 --gamma 0.001 --model_name="SPN" --fine_tune 1 --dataset_name "$dataset_name"
```

### 4. Train segmentation and EF model:
```bash
python evaluation/segmentation_eval/main_train_segmentation_model.py
python evaluation/segmentation_eval/main_train_ef_expert.py

```

### 5. Counterfactual Evaluation

Evaluate the generated counterfactuals using:

```bash
dataset_name=("stanford" "CAMUS")
python main_analysis.py --model_names "$model_names" --eval_method "regression" --fine_tune 1 --dataset_name "$dataset_name"
python main_analysis.py --model_names "$model_names" --eval_method "cfe" --fine_tune 1 --dataset_name "$dataset_name"

python evaluation/segmentation_eval/main_analysis.py --dataset_name "$dataset_name" --fine_tune 1 --model_names "$model_name"
```

### 6. Create output figures for paper

Evaluate the generated counterfactuals using:

```bash
python evaluation/segmentation_eval/paper_tables_and_plots_v7_tradeoff.py
python evaluation/segmentation_eval/cf_vizualization_extended.py
```



## Contact

For any questions, please contact siekiera@uni-mainz.de .
