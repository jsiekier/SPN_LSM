import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import echonet
import os
import torch
import math
import time
from PIL import Image

from cf_eval.counterfactual_evaluation import eval_counterfactuals
from cf_eval.latex_table import segmentation_to_latex_summaey
from dataset import load_dataset
from segmentation import segmentate
from video import load_ef_pred_model
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
weights = DeepLabV3_ResNet50_Weights.DEFAULT
import torchvision
seed=0



class CAMUS(torchvision.datasets.VisionDataset):
    def __init__(self,root=None,path='',split="train", target_type="EF",mean=0., std=1.,target_transform=None):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)
        data_path = 'camus_echonet_style_mobi.npz'
        data = np.load(path + data_path)
        print(data["train_max_min_frame"].shape, data["train_max_min_masks"].shape, np.max(data["train_max_min_masks"]))
        # tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
        mean=np.mean(mean)
        std=np.mean(std)
        self.min_frame=(np.squeeze(np.repeat(data[split+"_max_min_frame"][:,:,:,:, 0],axis=1,repeats=3))-mean)/std
        self.max_frame=(np.squeeze(np.repeat(data[split+"_max_min_frame"][:,:,:,:, 1],axis=1,repeats=3))-mean)/std
        self.min_mask=np.squeeze(data[split+"_max_min_masks"][:,:,:,:, 0]).astype(np.float32)
        self.max_mask=np.squeeze(data[split+"_max_min_masks"][:,:,:,:,1]).astype(np.float32)

    def __getitem__(self, index):
        target=[self.min_frame[index],
                self.max_frame[index],
                self.min_mask[index],
                self.max_mask[index]]
        return [],target
    def __len__(self):
        return len(self.min_frame)





def train_segmentation_model(device,model,
                             run_test=False,
                             num_epochs=50,
                             lr=1e-5,
                             weight_decay=1e-5,
                             lr_step_period=None,
                             num_train_patients=None,
                             num_workers=4,
                             batch_size=20,
                             seed=0,
                             data_dir=None,
                             output=None,
                             path='',
                             staged_finetuning=False,
                             freeze_epochs=10,
                             warmup_trainable_layers=("classifier", "backbone.layer4"),
                             finetune_lr_factor=0.1
                             ):

    if staged_finetuning:
        print(f"Staged fine-tuning for {freeze_epochs} epochs")
        set_trainable_layers(model, warmup_trainable_layers)
        print_trainable_layers(model)
        optim = make_optimizer(model, lr, weight_decay)
    else:
        set_all_trainable(model)
        optim = make_optimizer(model, lr, weight_decay)

    if lr_step_period is None:
        lr_step_period = math.inf

    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)


    '''
    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    '''
    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    dataset = {}
    if path=='':
        dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
        if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
            # Subsample patients (used for ablation experiment)
            indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
            dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
        dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    else:
        kwargs = {
                  "mean": mean,
                  "std": std
                  }
        dataset["train"] = CAMUS(path=path, split="train", **kwargs)
        if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
            # Subsample patients (used for ablation experiment)
            indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
            dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
        dataset["val"] = CAMUS(path=path, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):

            if staged_finetuning and epoch == freeze_epochs:
                print("Unfreezing all segmentation model weights.", flush=True)
                set_all_trainable(model)
                optim = make_optimizer(
                    model,
                    lr * finetune_lr_factor,
                    weight_decay
                )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optim,
                    lr_step_period
                )
                print_trainable_layers(model)


            print("Epoch #{}".format(epoch), flush=True)

            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                    pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model,
                                                                                                                dataloader,
                                                                                                                phase == "train",
                                                                                                                optim,
                                                                                                                device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (
                            large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in
                                                                        range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved() for i in
                                                                        range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                print('SAVE model loss,overall_dice,small_dice,large_dice:',loss,overall_dice,small_dice,large_dice)
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                                         pin_memory=(device.type == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model,
                                                                                                                dataloader,
                                                                                                                False,
                                                                                                                None,
                                                                                                                device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(
                    np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),
                    echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split,
                                                                               *echonet.utils.bootstrap(large_inter,
                                                                                                        large_union,
                                                                                                        echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split,
                                                                               *echonet.utils.bootstrap(small_inter,
                                                                                                        small_union,
                                                                                                        echonet.utils.dice_similarity_coefficient)))
                f.flush()



def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_trainable_layers(model, trainable_layer_names):
    base_model = _unwrap_model(model)

    for name, param in base_model.named_parameters():
        param.requires_grad = any(
            name.startswith(layer_name)
            for layer_name in trainable_layer_names
        )


def set_all_trainable(model):
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(model, lr, weight_decay):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )


def print_trainable_layers(model):
    base_model = _unwrap_model(model)
    print("Trainable layers:")
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(" ", name)
def load_segmentation_model(device=None,model_name="deeplabv3_resnet50",pretrained=True,weights=None,train=True,
                             run_test=False,num_epochs=50,lr=1e-5,weight_decay=1e-5,lr_step_period=None,num_train_patients=None,
                            num_workers=4,batch_size=20,seed=0,data_dir=None,output=None,
                            device_str='cpu',
                            staged_finetuning=True,
                            freeze_epochs=5,
                            warmup_trainable_layers=("fc", "layer4"),
                            finetune_lr_factor=0.1,
                            path=''):

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    if pretrained:
        torchvision_weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=torchvision_weights, aux_loss=True)
        model.aux_classifier = None
    else:
        model = deeplabv3_resnet50(weights=None, aux_loss=False)
    #model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    if weights is not None:
        print('load checkpoints')
        checkpoint = torch.load(weights,map_location=device)

        # Remove 'module.' prefix if present
        state_dict = checkpoint['state_dict']

        model_is_parallel = isinstance(model, torch.nn.DataParallel)
        ckpt_is_parallel = next(iter(state_dict)).startswith("module.")

        if ckpt_is_parallel and not model_is_parallel:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        elif model_is_parallel and not ckpt_is_parallel:
            state_dict = {"module." + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        if not train:
            return model
        '''
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        if device_str=='cpu':
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        '''

        #model.load_state_dict(checkpoint['state_dict'])
        #return model

    train_segmentation_model(device=device,
                             model=model,
                             run_test=run_test,
                             num_epochs=num_epochs,
                             lr=lr,
                             weight_decay=weight_decay,
                             lr_step_period=lr_step_period,
                             num_train_patients=num_train_patients,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             seed=seed,
                             data_dir=data_dir,
                             output=output,
                             staged_finetuning=staged_finetuning and weights is not None,
                             freeze_epochs=freeze_epochs,
                             warmup_trainable_layers=warmup_trainable_layers,
                             finetune_lr_factor=finetune_lr_factor,
                             path =path
                             )




# Save NIfTI images for nnUNet (nnU-Net requires filenames like *_0000.nii.gz)

'''
def save_nifti(volume, filename):
    affine = np.eye(4)
    nifti = nib.Nifti1Image(volume, affine)
    nib.save(nifti, filename)

# Compute LV volumes from prediction
def compute_lv_volume(seg_file, label=1):
    seg = nib.load(seg_file).get_fdata()
    spacing = nib.load(seg_file).header.get_zooms()
    voxel_volume = np.prod(spacing)  # mm³
    return np.sum(seg == label) * voxel_volume
'''

# x Shape: (2, H, W, D)
#def predict_ED_ES_EF(x,segmentation_model,mean_segmentation=mean_segmentation,std_segmentation=std_segmentation):



'''


    # Make sure nnU-Net expects input in shape (C, H, W, D)
    x_ed = np.expand_dims(x[:,:,0:], axis=0).astype(np.float32)
    x_es = np.expand_dims(x[:,:,1:], axis=0).astype(np.float32)

    # Temporary folders
    tmpdir = tempfile.mkdtemp()
    input_folder = os.path.join(tmpdir, "input")
    output_folder = os.path.join(tmpdir, "output")
    os.makedirs(input_folder)
    os.makedirs(output_folder)
    save_nifti(x_ed[0], os.path.join(input_folder, "ED_0000.nii.gz"))
    save_nifti(x_es[0], os.path.join(input_folder, "ES_0000.nii.gz"))
    # Run inference using pretrained Task03_Heart model

    echonet.utils.segmentation.run(modelname="deeplabv3_resnet50",  save_segmentation=True, pretrained=True)

    ed_seg = os.path.join(output_folder, "ED.nii.gz")
    es_seg = os.path.join(output_folder, "ES.nii.gz")



    ed_vol = compute_lv_volume(ed_seg, label=1)  # LV label in ACDC = 1
    es_vol = compute_lv_volume(es_seg, label=1)

    ef = (ed_vol - es_vol) / ed_vol * 100

    print(f"EDV: {ed_vol:.2f} mm³, ESV: {es_vol:.2f} mm³, EF: {ef:.2f}%")

    # Optional: Clean up
    shutil.rmtree(tmpdir)
'''

def diff_counterfactual_z(rec_z,reconstructions):
    # x_repeat = np.repeat(x[:, :, :, 0], reconstructions.shape[0], axis=0)
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255
    differences = org_rec_z - reconstructions_int
    differences_mean = np.mean(differences, axis=0)
    return differences_mean


def calculate_pef(vol1, vol2):
    vol_max=max(vol1,vol2)
    vol_min=min(vol1,vol2)
    #(EDV - ESV) / EDV
    pef=(vol_max-vol_min)/vol_max
    return pef


def plot_cfs_3D_gif(dataset_name,data_path_grid,model_name,num_imgs, num_imgs_per_plot, method_name, beta, gamma,
                     fold_idxs=[1],path='',num_frames=2,segmentation_model=None,mean_segmentation=0,std_segmentation=1):
    # Configure the model


    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs/'
        num_plots=num_imgs//num_imgs_per_plot

        save_file_path = data_path_fold + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                           '-') + method_name + '_1000_new.pkl'
        all_data_list = pkl.load(open(save_file_path, 'rb'))
        all_data = []
        for entry in all_data_list:
            all_data.extend(entry)
        [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, pred,
         y_cf_goal, coordinates, x_org, mean_cls, y_org] = zip(*all_data)
        x_cf = np.asarray(x_cf)*255
        x_rec = np.asarray(x_rec)*255
        x_org = np.asarray(x_org)*255
        mean_cls = np.squeeze(np.asarray(mean_cls))
        y_cf_goal = np.squeeze(np.asarray(y_cf_goal))
        y_org = np.squeeze(np.asarray(y_org))

        if method_name == 'MLP':
            arg_max = np.mean(np.squeeze(np.asarray(arg_max)), axis=1)
        else:
            arg_max = np.mean(np.exp(np.squeeze(np.asarray(arg_max))), axis=1)  # np.mean(np.exp(arg_max))

        # arg_max=np.mean(np.squeeze(np.asarray(arg_max)),axis=1)
        rec_mean = np.mean(x_rec, axis=1)
        cf_mean = np.mean(x_cf, axis=1)

        #(batch * frames, channels, height, width)
        '''
        if segmentation_model is not None:
            pixel_volumes_rec,plot_mask_rec=segmentate(segmentation_model, x=rec_mean,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       mean_segmentation=mean_segmentation,
                       std_segmentation=std_segmentation)
            pixel_volumes_cf,plot_mask_cf=segmentate(segmentation_model, x=cf_mean,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       mean_segmentation=mean_segmentation,
                       std_segmentation=std_segmentation)
            pixel_volumes_org,plot_mask_org=segmentate(segmentation_model, x=np.expand_dims(np.squeeze(x_org),axis=-1),
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       mean_segmentation=mean_segmentation,
                       std_segmentation=std_segmentation)
        '''


        for plt_counter in range(num_plots):
            frame_names=[]

            volumes=np.zeros((3,num_imgs_per_plot,num_frames))
            masks = np.zeros((3, num_imgs_per_plot, num_frames,rec_mean.shape[1],rec_mean.shape[1]))
            print('masks.shap',masks.shape)
            # 1. calculate volumes
            for frame in range(num_frames):
                for img_num in range(num_imgs_per_plot):
                    image_num = img_num + plt_counter * num_imgs_per_plot
                    differences_mean = diff_counterfactual_z(x_cf[image_num], x_rec[image_num])
                    if segmentation_model is not None:
                        # shape (batchsize,h,w,f,c)

                        pixel_volumes_rec, plot_mask_rec = segmentate(segmentation_model,
                                                                      x=np.repeat(np.expand_dims(rec_mean[image_num:image_num+1,:,:,frame,0],axis=1),repeats=3,axis=1),
                                                                      device=torch.device(
                                                                          "cuda" if torch.cuda.is_available() else "cpu"),
                                                                      mean_segmentation=mean_segmentation,
                                                                      std_segmentation=std_segmentation)
                        pixel_volumes_cf, plot_mask_cf = segmentate(segmentation_model,
                                                                    x=np.repeat(np.expand_dims(cf_mean[image_num:image_num+1,:,:,frame,0],axis=1),repeats=3,axis=1),
                                                                    device=torch.device(
                                                                        "cuda" if torch.cuda.is_available() else "cpu"),
                                                                    mean_segmentation=mean_segmentation,
                                                                    std_segmentation=std_segmentation)
                        # shape (batchsize,1,h,w,f,c)
                        pixel_volumes_org, plot_mask_org = segmentate(segmentation_model,
                                                                      x=np.repeat(x_org[image_num:image_num+1,:,:,:,frame,0],repeats=3,axis=1),
                                                                      device=torch.device(
                                                                          "cuda" if torch.cuda.is_available() else "cpu"),
                                                                      mean_segmentation=mean_segmentation,
                                                                      std_segmentation=std_segmentation)

                        volumes[0,img_num,frame]=pixel_volumes_org[0]
                        volumes[1, img_num, frame] = pixel_volumes_rec[0]
                        volumes[2, img_num, frame] = pixel_volumes_cf[0]

                        masks[0, img_num, frame] = plot_mask_org[0]
                        masks[1, img_num, frame] = plot_mask_rec[0]
                        masks[2, img_num, frame] = plot_mask_cf[0]



            for frame in range(num_frames):
                save_file_path_plot = data_path_fold + str(plt_counter)+'_f'+str(frame) + 'image_no_w_' + method_name+  'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                              '-') + '.png'
                fig_size = 3
                fig, axis = plt.subplots(4,num_imgs_per_plot,figsize=(fig_size *num_imgs_per_plot, fig_size * 4))

                for img_num in range(num_imgs_per_plot):
                    image_num = img_num + plt_counter * num_imgs_per_plot
                    differences_mean = diff_counterfactual_z(x_cf[image_num], x_rec[image_num])
                    ############################################################################
                    '''
                    pef_org=calculate_pef(pixel_volumes_org[image_num][0],pixel_volumes_org[image_num][1])
                    title = 'y:' + str(np.round(y_org[image_num], 2))+\
                            '\nPV:'+str(int(pixel_volumes_org[image_num][0]))+\
                            ' PV:'+str(int(pixel_volumes_org[image_num][1]))+'PEF'+str(np.round(pef_org,decimals=2))
                    axis[0, img_num].set_title(title)
                    axis[0, img_num].imshow(x_org[image_num][0, :, :, frame, 0], cmap='gray')
                    axis[0, img_num].imshow(plot_mask_org[image_num][:, :, frame, 0], cmap='ocean', alpha=0.5)
                    axis[0, img_num].axis('off')

                    pef_org = calculate_pef(pixel_volumes_rec[image_num][0], pixel_volumes_rec[image_num][1])
                    title = 'y_{pr}:' + str(np.round(mean_cls[image_num], 2))+\
                            '\nPV:'+str(int(pixel_volumes_rec[image_num][0]))+\
                            ' PV:'+str(int(pixel_volumes_rec[image_num][1]))+' PEF:'+str(np.round(pef_org,decimals=2))

                    axis[1, img_num].set_title(title)
                    axis[1, img_num].imshow(rec_mean[image_num][:, :, frame, 0], cmap='gray')
                    axis[1, img_num].imshow(plot_mask_rec[image_num][:, :, frame, 0], cmap='ocean',alpha=0.5)
                    axis[1, img_num].axis('off')

                    pef_org = calculate_pef(pixel_volumes_cf[image_num][0], pixel_volumes_cf[image_num][1])
                    title = 'cf:' + str(np.round(y_cf_goal[image_num], 2)) + ';cf_pr:' + str(
                        np.round(arg_max[image_num], 2))+\
                            '\nPV:'+str(int(pixel_volumes_cf[image_num][0]))+\
                            ' PV:'+str(int(pixel_volumes_cf[image_num][1]))+'PEF'+str(np.round(pef_org,decimals=2))

                    axis[2, img_num].set_title(title)
                    axis[2, img_num].imshow(cf_mean[image_num][:, :, frame, 0], cmap='gray')
                    axis[2, img_num].imshow(plot_mask_cf[image_num][:, :, frame, 0], cmap='ocean', alpha=0.5)
                    axis[2, img_num].axis('off')

                    axis[3, img_num].imshow(x_org[image_num][0, :, :, frame, 0], cmap='gray')
                    axis[3, img_num].imshow(differences_mean[:, :, frame], cmap='jet',alpha=0.5)  # ,vmax=vmax,vmin=vmin
                    axis[3, img_num].axis('off')
                    '''
                    ############################################################################


                    #'''


                    #if frame:
                    pef_org=calculate_pef(volumes[0,img_num,0],volumes[0,img_num,1])
                    title = 'y:' + str(np.round(y_org[image_num], 2)) + \
                            ' PEF'+str(np.round(pef_org,decimals=2))
                    #else:

                    #    title = 'y:' + str(np.round(y_org[image_num], 2))+\
                    #            '\nPV:'+str(int(pixel_volumes_org[0]))

                    axis[0, img_num].set_title(title)
                    axis[0, img_num].imshow(x_org[image_num][0, :, :, frame, 0], cmap='gray')
                    axis[0, img_num].imshow(masks[0,img_num,frame], cmap='ocean', alpha=0.5)
                    axis[0, img_num].axis('off')

                    #if frame:
                    pef_org = calculate_pef(volumes[1,img_num,0],volumes[1,img_num,1])
                    title = 'y_{pr}:' + str(np.round(mean_cls[image_num], 2)) + \
                            ' PEF:'+str(np.round(pef_org,decimals=2))
                    #else:
                    #    title = 'y_{pr}:' + str(np.round(mean_cls[image_num], 2))+\
                    #            '\nPV:'+str(int(pixel_volumes_rec[0]))

                    axis[1, img_num].set_title(title)
                    axis[1, img_num].imshow(rec_mean[image_num][:, :, frame, 0], cmap='gray')
                    axis[1, img_num].imshow(masks[1,img_num,frame], cmap='ocean',alpha=0.5)
                    axis[1, img_num].axis('off')

                    #if frame:
                    pef_org = calculate_pef(volumes[2,img_num,0],volumes[2,img_num,1])
                    title = 'cf:' + str(np.round(y_cf_goal[image_num], 2)) + ';cf_pr:' + str(
                        np.round(arg_max[image_num], 2)) + \
                            '\nPEF'+str(np.round(pef_org,decimals=2))

                    #else:
                    #    title = 'cf:' + str(np.round(y_cf_goal[image_num], 2)) + ';cf_pr:' + str(
                    #        np.round(arg_max[image_num], 2))+\
                    #            '\nPV:'+str(int(pixel_volumes_cf[0]))

                    axis[2, img_num].set_title(title)
                    axis[2, img_num].imshow(cf_mean[image_num][:, :, frame, 0], cmap='gray')
                    axis[2, img_num].imshow(masks[2,img_num,frame], cmap='ocean', alpha=0.5)
                    axis[2, img_num].axis('off')

                    axis[3, img_num].imshow(x_org[image_num][0, :, :, frame, 0], cmap='gray')
                    axis[3, img_num].imshow(differences_mean[:, :, frame], cmap='jet',alpha=0.5)  # ,vmax=vmax,vmin=vmin
                    axis[3, img_num].axis('off')
                    #'''

                plt.tight_layout()
                print('save', save_file_path_plot)
                # plt.show()
                plt.savefig(save_file_path_plot)
                frame_names.append(save_file_path_plot)

            gif_name = data_path_fold + str(plt_counter) + 'image_no_w_' + method_name + 'datab' + str(beta).replace(
                '.', '-') + 'g' + str(gamma).replace('.',
                                                     '-') + '.gif'
            print(frame_names)
            '''
            #with imageio.get_writer(gif_name, mode='I') as writer:
                for _ in range(10):  # repeat 3 times
                    for filename in frame_names:
                        image = imageio.v3.imread(filename)
                        writer.append_data(image, meta={'duration': 2000})
            '''
            frames = [Image.open(f) for f in frame_names]
            frames[0].save(
                gif_name,
                save_all=True,
                append_images=frames[1:],
                duration=2000,  # duration in milliseconds
                loop=0  # 0 means infinite loop
            )


def get_mask_data(data_dir,dataset_name='stanford'):
    if dataset_name=='stanford':
        tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace","EF"]
        kwargs = {"target_type": tasks,
                  "mean": 0,
                  "std":255
                  }

        # Set up datasets and dataloaders
        dataset = {}
        dataset["test"] = echonet.datasets.Echo(root=data_dir, split="test", **kwargs)

        all_large_frame, all_small_frame, all_large_trace, all_small_trace,all_ef=[],[],[],[],[]
        for (_, (large_frame, small_frame, large_trace, small_trace,ef)) in dataset["test"] :
            #print(large_frame.shape,large_trace.shape)
            #all_large_frame.append(np.mean(large_frame,axis=0,keepdims=True))
            #all_small_frame.append(np.mean(small_frame,axis=0,keepdims=True))
            all_large_frame.append(large_frame)
            all_small_frame.append(small_frame)
            all_large_trace.append(large_trace)
            all_small_trace.append(small_trace)
            all_ef.append(ef)

        all_large_frame=np.stack(all_large_frame,axis=0)
        all_small_frame = np.stack(all_small_frame, axis=0)
        all_large_trace = np.stack(all_large_trace, axis=0)
        all_small_trace = np.stack(all_small_trace, axis=0)
        all_ef = np.stack(all_ef, axis=0)
        return [all_large_frame,all_small_frame,all_large_trace,all_small_trace,all_ef]
    else:
        data_path = 'camus_echonet_style_mobi.npz'
        data = np.load(data_dir+ data_path)
        print(data["train_max_min_frame"].shape, data["train_max_min_masks"].shape, np.max(data["train_max_min_masks"]))
        # tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
        mean=np.mean(0)
        std=np.mean(255)
        split = "test"
        target_type = 'bi'
        min_frame=(np.squeeze(np.repeat(data[split+"_max_min_frame"][:,:,:,:, 0],axis=1,repeats=3))-mean)/std
        max_frame=(np.squeeze(np.repeat(data[split+"_max_min_frame"][:,:,:,:, 1],axis=1,repeats=3))-mean)/std
        min_mask=np.squeeze(data[split+"_max_min_masks"][:,:,:,:, 0]).astype(np.float32)
        max_mask=np.squeeze(data[split+"_max_min_masks"][:,:,:,:,1]).astype(np.float32)
        all_ef=np.squeeze(data["test_ef_"+target_type+"targets"]).astype(np.float32)
        print('org data:',min_frame.shape,np.max(min_frame),np.min(min_frame),np.max(all_ef),np.min(all_ef))
        return [ min_frame,max_frame,  min_mask,max_mask, all_ef]




def get_cf_metrics(model_names,possible_values,num_classes=1,conv_dim=3,strategy='',path='',save_folder ='',dataset_name='',
                   fold_idx=0,work_from=0,fine_tune=0):
    device_str=['cpu','cuda'][work_from]
    device=torch.device(device_str)
    # load segmentation model once:
    data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][work_from]
    if dataset_name=='stanford':
        weights = [#'C:/Users/siekiera/Documents/git_repos/CF_eval/segmentation_model_weights/best.pt',
                   #'/lustre/project/ki-topml/jsiekiera/segmentation_weights_echos/best.pt'
                    'C:/Users/siekiera/Documents/git_repos/CF_eval/weights/deeplabv3_resnet50_random.pt',
                    'deeplabv3_resnet50_random.pt'

        ][work_from]
        weights_ef=['C:/Users/siekiera/Documents/git_repos/CF_eval/ef_pred_weights_echos/best.pt',
                   #'/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw/best.pt'
                    '/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_seg/best.pt'
                    ][work_from]

    else:

        weights = '/lustre/project/ki-topml/jsiekiera/segmentation_weights_camus/best.pt'
        weights_ef = '/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_camu_bi_le6_20/best.pt'



    print(data_dir)
    echo=echonet.datasets.Echo(root=data_dir, split="train")
    mean_segmentation, std_segmentation = echonet.utils.get_mean_and_std(echo)
    mean_segmentation, std_segmentation = np.mean(mean_segmentation), np.mean(std_segmentation)
    org_mask_data=None#get_mask_data(data_dir,mean_segmentation,std_segmentation)

    print('load segentation model', flush=True)

    segmentation_model = load_segmentation_model(device=device,
                                                 model_name="deeplabv3_resnet50",
                                                 pretrained=False,
                                                 weights=weights,
                                                 train=False,device_str=device_str)

    ef_model = load_ef_pred_model(device=device, weights=weights_ef)
    # get normalizing params:


    all_results= []
    all_results_z_eval=[]
    all_results_segmentation = []
    all_params = {}

    for model_name in model_names:
        print(model_name)
        #num_train_eval_runs=3
        grid_idx=0

        data_path_grid=save_folder + model_name+'/grid'+str(grid_idx)+'/'
        params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))
        #add_info=1
        #data_path_fold=data_path_grid + 'fold_' + str(0) + '/'

        train, test, num_classes = load_dataset(dataset_name, binary=False,
                                                                         load_net=params.load_pretrain_model,
                                                                         machine=params.machine,grid_params=params,path=path)

        # Check if results already exist
        #todo
        cf_path = os.path.join(data_path_grid,  str(fine_tune)+"cf_new1.csv")#1
        segmentation_path = os.path.join(data_path_grid,  str(fine_tune)+"segment_results.csv")#1
        z_eval_path = os.path.join(data_path_grid,  str(fine_tune)+"z_eval_new1.csv")
        '''
        if os.path.exists(cf_path):
            with open(segmentation_path, "r") as f:
                if f.read(1).replace('\n','').replace('\t','')== "":
                    print("File is empty")
                    print(f"Running analysis for {model_name}")
                   # if params.conv_dim==3:

                    cf_result,results_z_eval,result_segmentation  = eval_counterfactuals(dataset_name,params, data_path_grid, model_name,
                                                            fold_idxs=[fold_idx],possible_values=possible_values,test=test,
                                                            num_classes=num_classes,segmentation_model=segmentation_model,
                                                            mean_segmentation=mean_segmentation,std_segmentation=std_segmentation)
                    pd.DataFrame(cf_result).to_csv(cf_path, index=False)
                    pd.DataFrame(results_z_eval).to_csv(z_eval_path, index=False)
                    pd.DataFrame(result_segmentation).to_csv(segmentation_path, index=False)
                    print('Results saved')
                else:
                    print("File is not empty")
                    print(f"Loading existing results for {model_name}")
                    cf_result = []#pd.read_csv(cf_path).to_dict('records')
                    results_z_eval= []#pd.read_csv(z_eval_path).to_dict('records')
                    result_segmentation = pd.read_csv(segmentation_path).to_dict('records')
        else:
        '''
        print(f"Running analysis for {model_name}",flush=True)
        cf_result,results_z_eval,result_segmentation,metrics = eval_counterfactuals(dataset_name,params, data_path_grid,ef_model=ef_model, vae_name=model_name,
                                                        fold_idxs=[fold_idx],possible_values=possible_values,test=test,
                                                        num_classes=num_classes,segmentation_model=segmentation_model,
                                                        mean_segmentation=mean_segmentation,std_segmentation=std_segmentation,fine_tune=fine_tune,org_mask_data=org_mask_data,work_from=work_from)
        print(cf_result)
        pd.DataFrame(cf_result).to_csv(cf_path, index=False)
        pd.DataFrame(results_z_eval).to_csv(z_eval_path, index=False)
        pd.DataFrame(result_segmentation).to_csv(segmentation_path, index=False)
        segment_interpretation_path=os.path.join(data_path_grid,  str(fine_tune)+"segment_analyse.json")#1
        '''
        saved_files = save_cf_metrics(
            metrics_out=metrics,
            save_dir=data_path_grid,
            file_prefix=
        )

        print(saved_files)
        loaded_metrics = load_cf_metrics(segment_interpretation_path)
        print(loaded_metrics.keys())
        print('Results saved')
        '''



        all_results.extend(cf_result)
        all_results_z_eval.extend(results_z_eval)
        all_results_segmentation.extend(result_segmentation)
        all_params[model_name]=params

    #cf_performance_to_latex_summary(all_results, all_params,strategy=strategy,num_classes=num_classes)
    #z_eval_cf_performance_to_latex_summary(all_results_z_eval, all_params)
    segmentation_to_latex_summaey(all_results_segmentation,all_params)