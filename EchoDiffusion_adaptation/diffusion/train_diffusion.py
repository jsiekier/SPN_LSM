import os, shutil
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import time
import torch
import numpy as np
from PIL import Image
import wandb
import torch
from einops import rearrange
from scipy.ndimage import zoom

from transformers import T5Tokenizer

MODEL_DIR = "/lustre/project/ki-topml/jsiekiera/t5_base"
# ensure the model file exists
assert os.path.isfile(os.path.join(MODEL_DIR, "spiece.model"))
tok = T5Tokenizer.from_pretrained(MODEL_DIR,local_files_only=True)

#tok = T5Tokenizer.from_pretrained("google/t5-v1_1-base", local_files_only=True)


from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer, Imagen
from diffusion.dataset import EchoVideo, EchoPair

#from dataset import EchoVideo, EchoPair
print("Torch CUDA available:", torch.cuda.is_available())
print("CUDA version seen by torch:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
def get_current_unet(trainer, stage):
    return trainer.imagen.get_unet(stage)


def set_all_trainable(module):
    for p in module.parameters():
        p.requires_grad = True


def freeze_all(module):
    for p in module.parameters():
        p.requires_grad = False


def set_trainable_by_name(module, keywords):
    for name, p in module.named_parameters():
        p.requires_grad = any(k in name for k in keywords)


def print_trainable_params(module, max_items=50):
    trainable = [name for name, p in module.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)}")
    for name in trainable[:max_items]:
        print(" ", name)
def delay2str(t):
    t = int(t)
    secs = t%60
    mins = (t//60)%60
    hours = (t//3600)%24
    days = t//86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string

def get_reference_videos(config, val_ds, indices):
    vformat = lambda x: (x*255).permute(1,2,3,0).numpy().astype(np.uint8)
    videos = [ vformat(val_ds[e][0]) for e in indices ]
    # Resize depending on unet
    undex_index = config.stage - 1 # 0 or 1
    size = config.imagen.image_sizes[undex_index]
    videos = [ zoom(v, (1, size/v.shape[1], size/v.shape[2], 1), order=1) for v in videos ]

    return videos

def create_save_folder(save_folder):
    os.makedirs(save_folder, exist_ok = True)
    shutil.copy(args.config, os.path.join(save_folder, "config.yaml"))
    os.makedirs(os.path.join(save_folder, "videos"), exist_ok = True)
    os.makedirs(os.path.join(save_folder, "models"), exist_ok = True)

def delete_save_folder(save_folder):
    try:
        shutil.rmtree(save_folder)
    except:
        pass

def one_line_log(cur_step, loss, batch_per_epoch, start_time, validation=False):
    s_step = f'Step: {cur_step:<6}'
    s_loss = f'Loss: {loss:<6.4f}' if not validation else f'Val loss: {loss:<6.4f}'
    s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f} , {(cur_step):<4.0f} '
    s_mvid = f'Mvid: {(cur_step*config.dataloader.batch_size/1e6):<6.4f}'
    s_delay = f'Elapsed time: {delay2str(time.time() - start_time):<10}'
    print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore
    if cur_step % 1000 == 0:
        print() # Start new line every 1000 steps
    '''
    wandb.log({
        "loss" if not validation else "val_loss" : loss, 
        "step": cur_step, 
        "epoch": cur_step//batch_per_epoch, 
        "mvid": cur_step*config.dataloader.batch_size/1e6
    })
    '''

def start_wandb(config, exp_name, train_days):
    wandb.init(
        name=f"{exp_name}_[{train_days}d]",
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True) # type: ignore
    )

if __name__ == "__main__":

    torch.hub.set_dir(".cache")

    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--stage", type=int, default="1")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--lr", type=float, default="-1")
    parser.add_argument("--ignore_time", type=float, default="0.25", help="probability of setting ignore_time=true")
    parser.add_argument("--steps", type=int, default=-1, help="diffusion steps")
    parser.add_argument("--uname", type=str, default="", help="unique name for experiment")
    parser.add_argument("--new_noise", action="store_true", help="use new noise", default=False)
    parser.add_argument("--temp_loss", type=float, default=0.0, help="Temporal smoothness loss weight")
    parser.add_argument("--ds_name", type=str, default="stanford_seg")
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="Checkpoint used only for initializing weights, e.g. Stanford merged checkpoint.")

    parser.add_argument("--freeze_warmup_steps", type=int, default=0,
                        help="Number of initial steps with partial fine-tuning.")

    parser.add_argument("--finetune_lr_factor", type=float, default=1.0,
                        help="LR multiplier after unfreezing.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))

    # Define experiment name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = args.config.split("/")[-1].split(".")[0] # get config file name
    exp_name = f"{exp_name}_{args.ds_name}_stage{args.stage}"
    if args.uname != "":
        exp_name = f"{exp_name}_{args.uname}"

    # Overwrite config values with args
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if args.bs != -1:
        config.dataloader.batch_size = args.bs
        config.dataloader.num_workers = args.bs
        config.checkpoint.batch_size = min(args.bs, config.checkpoint.batch_size)
    if args.lr != -1:
        config.trainer.lr = args.lr
    if args.steps != -1:
        if config.imagen.get("elucidated", True) == True:
            config.imagen.num_sample_steps = args.steps
        else:
            config.imagen.timesteps = args.steps


    # Create datasets
    #train_ds = EchoVideo(config, ["TRAIN"]) # type: ignore
    #val_ds = EchoVideo(config, ["VAL"]) # type: ignore

    train_ds = EchoPair(config, ["TRAIN"],ds_name=args.ds_name) # type: ignore
    val_ds = EchoPair(config, ["VAL"],ds_name=args.ds_name) # type: ignore


    # Create models and trainer
    unets = []
    for i, (k, v) in enumerate(config.unets.items()):
        unets.append(Unet3D(**v, lowres_cond=(i>0))) # type: ignore

    imagen_klass = ElucidatedImagen if config.imagen.elucidated == True else Imagen
    del config.imagen.elucidated
    imagen = imagen_klass(
        unets = unets,
        **OmegaConf.to_container(config.imagen), # type: ignore
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        **config.trainer,
    ).to(device)

    #print("ImagenTrainer attributes containing opt/lr:")
    #for name in dir(trainer):
    #    if "opt" in name.lower() or "lr" in name.lower():
    #        print(name, type(getattr(trainer, name, None)))

    if args.init_ckpt is not None:
        trainer.accelerator.print(f"Initializing from checkpoint: {args.init_ckpt}")
        trainer.load(args.init_ckpt)
        trainer.accelerator.print("Initialization checkpoint loaded. This checkpoint will not be overwritten.")


    # Create save folder and resume logic
    save_folder = os.path.join(config.checkpoint.path, exp_name)
    if args.resume not in ['auto', 'overwrite']:
        raise ValueError("Got wrong resume value: ", args.resume)

    # Delete previous experiment if requested
    if args.resume == 'overwrite':
        trainer.accelerator.print("Overwriting previous experiment")
        if trainer.is_main:
            delete_save_folder(save_folder)

    # Create save folder if it doesn't exist and copy config file
    if trainer.is_main:
        create_save_folder(save_folder)
    trainer.accelerator.wait_for_everyone()

    base_exp_name = args.config.split("/")[-1].split(".")[0]

    if args.stage > 1:
        trainer.accelerator.print(f"Stage {args.stage}: loading checkpoints of previous stages...")

        for prev_stage in range(1, args.stage):
            prev_exp_name = f"{base_exp_name}_stage{prev_stage}"
            print('TODO Is this the right stage name?',prev_exp_name)
            prev_folder = os.path.join(config.checkpoint.path, prev_exp_name, "models")
            #print(prev_folder)
            #prev_folder=prev_folder[2:-1]
            #print(prev_folder)
            if not os.path.exists(prev_folder):
                raise FileNotFoundError(
                    f"ERROR: Expected previous stage checkpoint folder not found:\n{prev_folder}\n"
                )

            checkpoints = sorted(os.listdir(prev_folder))
            if len(checkpoints) == 0:
                raise FileNotFoundError(
                    f"ERROR: No checkpoints found in {prev_folder}. "
                    f"Did you train stage {prev_stage}?"
                )

            # Load latest checkpoint of previous stage
            ckpt_path = os.path.join(prev_folder, checkpoints[-1])
            trainer.accelerator.print(f" → Loading Stage {prev_stage} checkpoint: {ckpt_path}")

            # Load weights into the multi-UNet Imagen model
            trainer.load(ckpt_path)

        trainer.accelerator.print(f"Successfully loaded previous {args.stage - 1} stages.")
        trainer.accelerator.print("UNet weights for earlier stages are now initialized and will be frozen.")

    # Resume training if requested and possible
    if args.resume == 'auto' and len(os.listdir(os.path.join(save_folder, "models"))) > 0:
        checkpoints = sorted(os.listdir(os.path.join(save_folder, "models")))
        train_days = int(checkpoints[-1].split(".")[1])+1 # format is ckpt.X.pt
        weight_path = os.path.join(save_folder, "models", checkpoints[-1])
        trainer.accelerator.print(f"Resuming training from {weight_path}")
        additional_data = trainer.load(weight_path)
        start_time = time.time() - additional_data["time_elapsed"] # type: ignore
    else:
        train_days = 0
        start_time = time.time()
        trainer.accelerator.print("Training from scratch")

    ###########################################################################
    current_unet = get_current_unet(trainer, args.stage)

    if args.freeze_warmup_steps > 0:
        trainer.accelerator.print(
            f"Warmup fine-tuning: freezing most of UNet {args.stage} for "
            f"{args.freeze_warmup_steps} steps."
        )

        # konservativ: nur output-nahe Layer trainieren
        set_trainable_by_name(
            current_unet,
            keywords=[
                "final",
                "to_out",
                "out",
                "resnet_block2",
                "ups"
            ]
        )

        if trainer.is_main:
            print_trainable_params(current_unet)
    else:
        set_all_trainable(current_unet)

    ############################################################################



    trainer.add_train_dataset(
        train_ds, 
        batch_size = config.dataloader.batch_size, 
        num_workers = config.dataloader.num_workers,
    )
    trainer.add_valid_dataset(
        val_ds, 
        batch_size = config.dataloader.batch_size, 
        num_workers = config.dataloader.num_workers,
    )

    batch_per_epoch = (len(train_ds) // config.dataloader.batch_size)+1



    # Create wandb logger
    #if trainer.is_main:
    #    start_wandb(config, exp_name, train_days)

    # Save reference videos and get test embeddings
    trainer.accelerator.print("Getting reference videos...")
    indices = (torch.rand(config.checkpoint.batch_size)*len(val_ds)).long().tolist()
    videos_ref = get_reference_videos(config, val_ds, indices)
    videos_ref = np.concatenate([np.array(img) for img in videos_ref], axis=-2) # T x H x W x C
    _T, _H, _W, _C = videos_ref.shape
    videos_pil = [Image.fromarray(i) for i in videos_ref]
    if trainer.is_main:
        videos_pil[0].save(os.path.join(save_folder, "videos", f"_reference.gif"), save_all=True, append_images=videos_pil[1:], duration=1000/config.dataset.fps, loop=0)
        videos_ref = rearrange(videos_ref, 't h w c -> t c h w') # type: ignore
        #wandb.log({"videos": wandb.Video(videos_ref, fps=config.dataset.fps)})
        videos_ref = rearrange(videos_ref, "t c h (b w) -> b t c h w", b=len(indices)) # type: ignore
        print("Saved reference videos.")

    valid_embeddings = torch.stack([val_ds[e][1] for e in indices]) # type: ignore
    valid_cond_images = torch.stack([val_ds[e][2] for e in indices]) # type: ignore
    
    # Create kwargs for sampling and logging
    sample_kwargs = {}
    sample_kwargs["start_at_unet_number"] = config.stage
    sample_kwargs["stop_at_unet_number"] = config.stage
    if config.imagen.condition_on_text:
        sample_kwargs["text_embeds"] = valid_embeddings
    if config.unets.get(f"unet{args.stage}").get('cond_images_channels') != None:
        sample_kwargs["cond_images"] = valid_cond_images
    if config.stage > 1: # Condition on low res image
        sample_kwargs["start_image_or_video"] = torch.stack([val_ds[e][0] for e in indices]) # type: ignore

    kwargs = {
        "images": None,
        "ignore_time": False,
    }

    if args.new_noise:
        kwargs['noise'] = ['part_det']
    if args.temp_loss > 0:
        kwargs['temporal_loss_w'] = args.temp_loss
        

    # Start training loop 
    print("Starting training loop...",batch_per_epoch,config.dataloader.batch_size)
    trainer.accelerator.wait_for_everyone()
    cur_step = 0
    test_counter=0
    start_time_loop=time.time()
    unfrozen = False
    while ((time.time()-start_time)<24*60*60):
    #while True: # let slurm handle the stopping
        kwargs['ignore_time'] = True if args.ignore_time > 0 and cur_step%(1/args.ignore_time) == 0 else False
        #print('step', test_counter)
        test_counter+=1
        loss = trainer.train_step(unet_number = args.stage, **kwargs)

        cur_step +=1 #trainer.steps[args.stage-1].item()# type: ignore
        if cur_step % 50 == 0:
            print(
                f"step={cur_step} loss={loss:.4f}",
                flush=True
            )
        if (args.freeze_warmup_steps > 0
                and not unfrozen
                and cur_step >= args.freeze_warmup_steps):
            trainer.accelerator.print(f"Unfreezing full UNet {args.stage}.")

            set_all_trainable(current_unet)

            current_optimizer = getattr(trainer, f"optim{args.stage - 1}")

            for group in current_optimizer.param_groups:
                group["lr"] *= args.finetune_lr_factor

            trainer.accelerator.print(
                f"New LR for UNet {args.stage}: {current_optimizer.param_groups[0]['lr']}"
            )

            #for opt in trainer.optimizers:
            #    for group in opt.param_groups:
            #        group["lr"] *= args.finetune_lr_factor


            if trainer.is_main:
                print_trainable_params(current_unet)

            unfrozen = True

        if trainer.is_main and cur_step%500==0:
            one_line_log(cur_step, loss, batch_per_epoch, start_time)

        if cur_step % config.checkpoint.save_every_x_it == 1:
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print()
            trainer.accelerator.print(f'Saving model and videos to wandb (it. {cur_step})')
            
            if trainer.is_main:
                with torch.no_grad():
                    s_videos = trainer.sample(
                        batch_size=config.checkpoint.batch_size, 
                        cond_scale=config.checkpoint.cond_scale,
                        video_frames=config.dataset.num_frames,
                        **sample_kwargs,
                    ) # B x C x T x H x W
                
                # Upscale videos to match reference videos - if necessary
                s_videos = torch.nn.functional.interpolate(s_videos, size=(videos_ref.shape[1],*videos_ref.shape[3:])) # type: ignore

                videos = rearrange(s_videos.detach().cpu(), 'b c t h w -> b t c h w') # type: ignore
                videos = (videos*255).numpy().astype(np.uint8)
                videos = np.concatenate([videos, videos_ref], axis=-2) # type: ignore
                videos = rearrange(videos, 'b t c h w -> t c h (b w)') # type: ignore
                #wandb.log({"videos": wandb.Video(videos, fps=config.dataset.fps)})

                videos = rearrange(s_videos.detach().cpu(), 'b c t h w -> t h (b w) c') # type: ignore
                videos = (videos*255).numpy().astype(np.uint8)
                videos = np.concatenate([videos, rearrange(videos_ref, 'b t c h w -> t h (b w) c')], axis=1)
                videos = [Image.fromarray(i) for i in videos]
                videos[0].save(os.path.join(save_folder, "videos", f"sample-{str(cur_step).zfill(10)}.gif"), save_all=True, append_images=videos[1:], duration=1000/config.dataset.fps, loop=0)                        

            trainer.accelerator.wait_for_everyone()

            valid_loss = np.mean([trainer.valid_step(unet_number = args.stage) for _ in range(10)])
            if trainer.is_main:
                one_line_log(cur_step, loss, batch_per_epoch, start_time, validation=True)
                print()
            
            trainer.accelerator.wait_for_everyone()

            additional_data = {
                "time_elapsed": time.time() - start_time,
            }
            trainer.save(os.path.join(save_folder, "models", f"ckpt.{train_days}.pt"), **additional_data) # type: ignore

            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print(f'DONE: Saving model (it. {cur_step}): {os.path.join(save_folder, "models", f"ckpt.{train_days}.pt")}')


