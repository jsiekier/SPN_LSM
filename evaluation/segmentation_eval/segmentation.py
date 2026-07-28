import os
import torch
import echonet
import skimage
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scipy


def segmentate(model, x: np.ndarray,
               device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               mean_segmentation=1.0,
               std_segmentation=0.0):
    """
    Segments frames using a PyTorch model and computes left ventricular volume
    based on segmentation area (proxy for volume).

    Args:
        model (torch.nn.Module): Trained segmentation model.
        x (np.ndarray): Input video batch, shape (batch, frames, height, width, channels).
        device (torch.device): Device to run model on (CPU/GPU).

    Returns:
        volumes (np.ndarray): Segmented volume per frame, shape (batch, frames).
        segmentations (np.ndarray): Binary segmentation mask per frame, shape (batch, frames, height, width, 1).
    """
    model.eval()
    #x=x*255
    x -= mean_segmentation
    x /= std_segmentation
    #x=np.repeat(x,repeats=3,axis=-1)


    with torch.no_grad():
        # Unpack and reshape input to (N, C, H, W)
        #b, h, w,f, c = x.shape
        #x_reshaped = x.reshape(-1, h, w, c)
        #x_reshaped=x_reshaped.transpose(0, 3, 1, 2) # (batch * frames, channels, height, width)
        x_tensor = torch.from_numpy(x).float().to(device)

        # Run segmentation
        y_logits = model(x_tensor)["out"]  # (N, 1, H, W)
        y_probs = torch.sigmoid(y_logits[:, 0, :, :])  # (N, H, W), probabilities
        y_mask = (y_probs > 0.2).float()  # Binary mask

        # Compute volume (pixel count)
        volume_per_frame = y_mask.cpu().numpy().sum(axis=(1, 2))  # (N,)

        # Reshape for outputs
        #volumes = volume_per_frame.reshape(b, f)
        #segmentations = np.expand_dims(y_mask.cpu().numpy(),axis=1).transpose(0,2,3,1)
        #segmentations=segmentations.reshape(b,  h, w, f,1)
        volumes = volume_per_frame
        segmentations = y_mask.cpu().numpy()

    return volumes, segmentations




def segmentation(output=None,num_workers=4,model=None,_video_collate_fn=None,save_video=True,dataset=None,device =None,batch_size=None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False,
                                             pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate(
                        [model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in
                         range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        #video *= std.reshape(1, 3, 1, 1)
                        #video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write(
                                "{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0,
                                                             1 if frame == small_index[i] else 0,
                                                             1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf

                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape(
                                    (1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape(
                                    (1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))),
                                                     4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset