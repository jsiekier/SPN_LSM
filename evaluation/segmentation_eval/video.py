"""Functions for training and running EF prediction."""

import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import pickle as pkl
import echonet
import random

def load_ef_pred_model(device,model_name="r2plus1d_18",pretrained=True,weights=''):

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(weights,map_location=device)

    # Remove 'module.' prefix if present
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    if device.type == 'cpu':
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model

def load_pairwise_CAMUS(kwargs,path):
    random.seed(0)
    # load dataset
    data_path = 'camus_echonet_style_mobi.npz'
    data = np.load(path+data_path)
    target_type='mo'
    train_data, test_data, val_data=[data["train_max_min_frame"],data["train_ef_"+target_type+"targets"]],\
                                    [data["test_max_min_frame"],data["test_ef_"+target_type+"targets"]],\
                                     [data["val_max_min_frame"],data["val_ef_"+target_type+"targets"]]

    print(path + data_path)

    # shuffle the sorted data
    train_idxs = list(range(train_data[0].shape[0]))
    val_idxs = list(range(val_data[0].shape[0]))
    test_idxs = list(range(test_data[0].shape[0]))
    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    train_data, val_data = [train_data[0][train_idxs], train_data[1][train_idxs]], \
        [val_data[0][val_idxs], val_data[1][val_idxs]],
    mean_segmentation_tmp, std_segmentation_tmp = np.mean(kwargs['mean']), np.mean(kwargs['std'])
    dataset = {}
    #mean_segmentation_tmp = np.mean(train_data[0],axis=0)# s1 / n  # type: np.ndarray
    #std_segmentation_tmp = np.std(train_data[0],axis=0) #np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray
    for data, data_name in zip([train_data, val_data,test_data], ['train', 'val','test']):
        print(data_name)
        # Normalize x_data
        data[0] = np.squeeze(data[0]) #* 255
        print(data[0].shape,mean_segmentation_tmp.shape,std_segmentation_tmp.shape)
        #eps = 1e-4
        #data[0] = (data[0] - mean_segmentation_tmp) / (std_segmentation_tmp + eps)
        data[0] = (data[0] - mean_segmentation_tmp) / (std_segmentation_tmp)
        print(data[0].shape)

        data[0] = np.repeat(np.expand_dims(data[0], axis=1), repeats=3, axis=1)  # (N, 3, H, W, num_frames)
        print(data[0].shape)
        data[0] = np.transpose(data[0], (0, 1, 4, 2, 3))
        print(data[0].shape, np.min(data[0]), np.max(data[0]), np.min(data[1]), np.max(data[1]))
        data[1]=data[1]# * 100
        print(data[0].shape,np.min(data[0]),np.max(data[0]),np.min(data[1]),np.max(data[1]))

        # Convert to torch tensors
        test_x = torch.tensor(data[0], dtype=torch.float32)
        test_y = torch.tensor(data[1], dtype=torch.float32)

        # Create dataset
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

        dataset[data_name] = test_dataset

    return dataset

def load_pairwise_ds(kwargs,path):
    random.seed(0)
    # load dataset
    data_path = 'sf_data.pkl'
    print(path + data_path)
    [train_data, test_data, val_data] = pkl.load(open(path + data_path, 'rb'))

    # shuffle the sorted data
    train_idxs = list(range(train_data[0].shape[0]))
    val_idxs = list(range(val_data[0].shape[0]))
    test_idxs = list(range(test_data[0].shape[0]))
    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    train_data, val_data = [train_data[0][train_idxs], train_data[1][train_idxs]], \
                                    [val_data[0][val_idxs],val_data[1][val_idxs]],
    mean_segmentation_tmp, std_segmentation_tmp = np.mean(kwargs['mean']), np.mean(kwargs['std'])
    dataset = {}
    for data,data_name in zip([train_data,val_data],['train','val']):
        # Normalize x_data
        data[0] = data[0] * 255
        data[0] = (data[0] - mean_segmentation_tmp) / std_segmentation_tmp
        data[0] = np.repeat(np.expand_dims(data[0], axis=1), repeats=3, axis=1)  # (N, 3, H, W, num_frames)
        data[0] = np.transpose(data[0], (0, 1, 4, 2, 3))

        # Convert to torch tensors
        test_x = torch.tensor(data[0], dtype=torch.float32)
        test_y = torch.tensor(data[1]*100, dtype=torch.float32)

        # Create dataset
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

        dataset[data_name]=test_dataset

    return dataset

def load_pairwise_ds_stanford_seg(kwargs,path):
    random.seed(0)
    data_path = 'segment_paper_multiple_imgs.npz'
    print(path + data_path)

    data = np.load(path + data_path)

    result = {
        'train': [
            data['train_ef_targets'],
            data['train_ef_preds'],
            data['train_min_max_frame'],
            data['train_min_max_frame_idx'],
            data['train_min_max_volumes'],
        ],
        'val': [
            data['val_ef_targets'],
            data['val_ef_preds'],
            data['val_min_max_frame'],
            data['val_min_max_frame_idx'],
            data['val_min_max_volumes'],
        ],
        'test': [
            data['test_ef_targets'],
            data['test_ef_preds'],
            data['test_min_max_frame'],
            data['test_min_max_frame_idx'],
            data['test_min_max_volumes'],
        ]
    }
    train_data, test_data, val_data = result['train'], result['test'], result['val']
    train_data, test_data, val_data = [train_data[2], train_data[0]], [test_data[2], test_data[0]], [val_data[2], val_data[0]]
    print(train_data[0].shape)
    # shuffle the sorted data
    train_idxs = list(range(train_data[0].shape[0]))
    val_idxs = list(range(val_data[0].shape[0]))
    test_idxs = list(range(test_data[0].shape[0]))
    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    train_data, val_data = [train_data[0][train_idxs], train_data[1][train_idxs]], \
                                    [val_data[0][val_idxs],val_data[1][val_idxs]],
    mean_segmentation_tmp, std_segmentation_tmp = np.mean(kwargs['mean']), np.mean(kwargs['std'])
    dataset = {}
    for data,data_name in zip([train_data,val_data,test_data],['train','val','test']):

        # Normalize x_data
        print(data[0].shape)
        data[0] = (data[0] - mean_segmentation_tmp) / std_segmentation_tmp
        data[0] = data[0][:, 0:1, :, :, :]# batchsize, 2,112,112,1
       # data[0] = np.transpose(data[0], (0, 4, 2, 3, 1))
        data[0] = np.repeat(data[0], repeats=3, axis=1)  # (N, 3, H, W, num_frames)
        data[0] = np.transpose(data[0], (0, 1, 4, 2, 3))
        print(data_name,'After transpose',data[0].shape)

        # Convert to torch tensors
        test_x = torch.tensor(data[0], dtype=torch.float32)
        test_y = torch.tensor(data[1], dtype=torch.float32)

        # Create dataset
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

        dataset[data_name]=test_dataset

    return dataset


def load_videoloops(kwargs,data_dir='',num_train_patients=None,):
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)



def load_pairwise(kwargs,path,split):
    random.seed(0)
    # load dataset
    data_path = 'sf_data.pkl'
    print(path + data_path)
    [train_data, test_data, val_data] = pkl.load(open(path + data_path, 'rb'))

    # shuffle the sorted data
    train_idxs = list(range(train_data[0].shape[0]))
    val_idxs = list(range(val_data[0].shape[0]))
    test_idxs = list(range(test_data[0].shape[0]))
    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    train_data, val_data,test_data = [train_data[0][train_idxs], train_data[1][train_idxs]], \
        [val_data[0][val_idxs], val_data[1][val_idxs]],[test_data[0][test_idxs], test_data[1][test_idxs]]

    mean_segmentation_tmp, std_segmentation_tmp = np.mean(kwargs['mean']), np.mean(kwargs['std'])

    if split=="val":
        data= val_data
    else:
        data=test_data


    data[0] = data[0] * 255
    data[0] = (data[0] - mean_segmentation_tmp) / std_segmentation_tmp
    data[0] = np.repeat(np.expand_dims(data[0], axis=1), repeats=3, axis=1)  # (N, 3, H, W, num_frames)
    data[0] = np.transpose(data[0], (0, 1, 4, 2, 3))

    # Convert to torch tensors
    test_x = torch.tensor(data[0], dtype=torch.float32)
    test_y = torch.tensor(data[1]*100, dtype=torch.float32)

    # Create dataset
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    return test_dataset
def load_without_test_time_pairwise(kwargs,path,split,batch_size,device,num_workers):
    test_dataset=load_pairwise(kwargs,path,split)
    dataloader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    return dataloader


def load_without_test_time_loops(kwargs,split,device,data_dir,batch_size,num_workers):
    # Performance without test-time augmentation
    dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    return dataloader
def load_with_test_time_pairwise(kwargs,split,path,num_workers,device):
    test_dataset=load_pairwise(kwargs,path,split)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    return dataloader


def load_with_test_time_loops(kwargs,split,device,data_dir,num_workers):
    ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    return dataloader


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_trainable_layers(model, trainable_layer_names):
    """
    Freeze all params except layers whose name starts with one of trainable_layer_names.
    Works with normal model and DataParallel.
    """
    base_model = _unwrap_model(model)

    for name, param in base_model.named_parameters():
        param.requires_grad = any(
            name.startswith(layer_name) for layer_name in trainable_layer_names
        )


def set_all_trainable(model):
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(model, lr, weight_decay):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )


def print_trainable_layers(model):
    base_model = _unwrap_model(model)
    trainable = []
    frozen = []

    for name, param in base_model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print(f"Trainable params: {len(trainable)}")
    print(f"Frozen params: {len(frozen)}")
    print("Trainable layer examples:")
    for name in trainable[:20]:
        print("  ", name)



def run(
    path='',
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=True,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,

    # neu:
    staged_finetuning=True,
    freeze_epochs=5,
    warmup_trainable_layers=("fc", "layer4"),
    finetune_lr_factor=0.1,

    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    pairwise=0,
    dataset_name=''
):
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    print(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        print('cuda')
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        print('load weights')
        checkpoint = torch.load(weights, map_location=device)
        state_dict = checkpoint["state_dict"]
        model_is_parallel = isinstance(model, torch.nn.DataParallel)
        ckpt_is_parallel = next(iter(state_dict)).startswith("module.")

        if ckpt_is_parallel and not model_is_parallel:
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

        elif model_is_parallel and not ckpt_is_parallel:
            state_dict = {
                "module." + k: v
                for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict)
        #model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    # Optional staged fine-tuning only when external weights are loaded
    use_staged_finetuning = staged_finetuning and weights is not None

    if use_staged_finetuning:
        print(
            f"Using staged fine-tuning for first {freeze_epochs} epochs. "
            f"Trainable layers: {warmup_trainable_layers}"
        )
        set_trainable_layers(model, warmup_trainable_layers)
        print_trainable_layers(model)
        optim = make_optimizer(model, lr, weight_decay)
    else:
        set_all_trainable(model)
        optim = make_optimizer(model, lr, weight_decay)


    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    if pairwise:
        if dataset_name=='stanford':
            dataset=load_pairwise_ds(kwargs,path=path)
        elif dataset_name=="stanford_seg":
            dataset=load_pairwise_ds_stanford_seg(kwargs,path=path)
        elif dataset_name=='CAMUS':
            dataset=load_pairwise_CAMUS(kwargs,path=path)
    else:
        dataset=load_videoloops(data_dir=data_dir,kwargs=kwargs)





    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]

            try:
                optim.load_state_dict(checkpoint['opt_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
            except ValueError:
                f.write("Optimizer state incompatible with current freeze setup; reinitializing optimizer.\n")

            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'test', 'val']:
                start_time = time.time()
                # Switch from partial fine-tuning to full fine-tuning
                if use_staged_finetuning and epoch == freeze_epochs:
                    print("Unfreezing all layers for full fine-tuning.", flush=True)

                    set_all_trainable(model)

                    finetune_lr = lr * finetune_lr_factor
                    optim = make_optimizer(model, finetune_lr, weight_decay)

                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optim,
                        lr_step_period
                    )

                    print_trainable_layers(model)

                print("Epoch #{}".format(epoch), flush=True)
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                print(phase,sklearn.metrics.r2_score(y, yhat),np.mean(np.abs(y-yhat)),
                      loss,np.min(yhat),np.max(yhat),np.min(y),np.max(y),flush=True)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                print('SAVE')
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["val", "test"]:


                if pairwise:
                    dataloader =load_without_test_time_pairwise(kwargs,path,split,batch_size,device,num_workers)
                else:
                    dataloader =load_without_test_time_loops(kwargs,split,device,data_dir,batch_size,num_workers)


                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                if pairwise:
                    dataloader =load_with_test_time_pairwise(kwargs,split,path,num_workers,device)
                else:
                    dataloader =load_with_test_time_loops(kwargs,split,device,data_dir,num_workers)

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()
                '''
                Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                '''
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)
def fine_tune(
    path_to_pretrained_model='',
    path='',
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=True,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,

    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    pairwise=0
):
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    if pairwise:
        dataset=load_pairwise_ds(kwargs,path=path)
    else:
        dataset=load_videoloops(data_dir=data_dir,kwargs=kwargs)





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
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["val", "test"]:
                if pairwise:
                    dataloader =load_without_test_time_pairwise(kwargs,path,split,batch_size,device,num_workers)
                else:
                    dataloader =load_without_test_time_loops(kwargs,split,device,data_dir,batch_size,num_workers)


                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                if pairwise:
                    dataloader =load_with_test_time_pairwise(kwargs,split,path,num_workers,device)
                else:
                    dataloader =load_with_test_time_loops(kwargs,split,device,data_dir,num_workers)

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()
                '''
                Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                '''
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y
