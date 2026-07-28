import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist
from scipy import stats
import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
lpips_metric = LearnedPerceptualImagePatchSimilarity(
    net_type='vgg',  # recommended
    normalize=True   # auto converts [0,1] → [-1,1]


).to('cuda')
# Load InceptionV3 model from TF-Hub
#FID_INCEPTION = 'https://tfhub.dev/google/tfgan/eval/inception/1'
##inception_model = hub.load(FID_INCEPTION)


# Function to preprocess images for Inception
def preprocess_images(images, image_size=299):
    """Resizes and normalizes images for InceptionV3."""
    images = tf.image.resize(images, (image_size, image_size))  # Resize
    images = (images*2)-1#(images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images

# Function to compute FID

def compute_fid2D(dataset1, dataset2):
    """Computes FID score between two datasets using TF-GAN."""

    dataset1 = preprocess_images(dataset1)
    dataset2 = preprocess_images(dataset2)
    dataset1=np.repeat(dataset1,repeats=3,axis=-1)
    dataset2 = np.repeat(dataset2, repeats=3, axis=-1)

    fid_score = tfgan.eval.frechet_inception_distance(dataset1, dataset2)
    return fid_score.numpy()  # Convert tensor to NumPy


def compute_fid(dataset1, dataset2,conv_dim=2):
    """Computes FID score between two datasets using TF-GAN."""
    if conv_dim==2:
        return compute_fid2D(dataset1, dataset2)
    else:
        all_frame_fid=0
        for frame in range(dataset1.shape[3]):
            frame_fid=compute_fid2D(dataset1[:,:,:,frame], dataset2[:,:,:,frame])
            all_frame_fid+=frame_fid
        return all_frame_fid/dataset1.shape[3]





def split_by_class(X, y,unique_classes):
    #tmp=np.where(y == 0)
    split_data = {cls: X[np.where(y == cls)[0]] for cls in unique_classes}
    return split_data

def compute_class_fid(dataset1, dataset2,dataset3,y_cf_expected_goal, y_org,y_cf_result):
    y_old=np.mod(y_cf_expected_goal+1,2)
    class_labels=np.unique(y_org)
    ds_org=split_by_class(dataset1,y_org,class_labels)
    ds_cf=split_by_class(dataset2,y_cf_result,class_labels)
    ds_rec=split_by_class(dataset3,y_old,class_labels)

    rel_FID=0

    for class_lab in class_labels:
        partial_org=ds_org[class_lab]
        patrial_cf=ds_cf[class_lab]
        patrial_rec=ds_rec[class_lab]

        fid_rec=compute_fid(partial_org,patrial_rec)
        fid_cf = compute_fid(partial_org, patrial_cf)
        rel_FID+=(fid_rec-fid_cf)


    rel_FID=rel_FID/len(class_labels)


    return rel_FID


def compute_rel_L2(dataset1, dataset2,dataset3,conv_dim):
    #org cf rec
    L2_rec = compute_MSE(dataset1, dataset3,conv_dim) # org rec
    L2_cf = compute_MSE(dataset1, dataset2,conv_dim) # org cf
    rel_L2 = L2_rec[1] - L2_cf[1]
    return rel_L2

def get_importance_region_information(x_org,x_cf_strat,threshold=0.5):
    diff=x_org-x_cf_strat

    # Calculate thresholds
    threshold_pos = 0.5 * np.max(diff)
    threshold_neg = 0.5 * np.min(diff)

    # Generate masks for positive and negative changes
    mask_pos = diff > threshold_pos
    mask_neg = diff < threshold_neg

    # Label connected components in each mask
    #labeled_pos = label(mask_pos)
    #labeled_neg = label(mask_neg)

    ######################
    # Connected Component Analysis
    result_arr=[[],[]]
    for bin_idx,(binary_mask, title) in enumerate(zip([mask_pos,mask_neg],['label pos','label neg'])):
        for idx in range(x_org.shape[0]):
            labeled_mask = label(binary_mask[idx,:,:,0])
            regions = regionprops(labeled_mask)

            # Analyze regions
            blob_sizes = [region.area for region in regions]
            blob_count = len(blob_sizes)
            if len(blob_sizes):
                result_arr[bin_idx].append([sum(blob_sizes)/len(blob_sizes),blob_count])

    #######################

    #result_arr=np.asarray(result_arr)
    result=np.zeros((2,2))
    for e_idx,entry in enumerate(result_arr):
        entry = np.asarray(entry)
        entry=np.mean(entry,axis=0)
        result[e_idx]=entry
    return result


def compute_MSE(dataset1, dataset2,conv_dim):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    sqr=np.square(dataset1 - dataset2)
    if conv_dim==2:
        mse=np.mean(sqr, axis=(1,2))
        L2=np.sqrt(np.sum(sqr,axis=(1,2)))
    elif conv_dim==3:
        mse=np.mean(sqr, axis=(1,2,3))
        L2=np.sqrt(np.sum(sqr,axis=(1,2,3)))
    rmse = np.sqrt(mse)
    L2_mean=np.mean(L2)
    rmse_mean=np.mean(rmse)
    return rmse_mean,L2_mean


def compute_MSE_mean(dataset1, dataset2,conv_dim):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1=np.expand_dims(dataset1,1)
    sqr=np.square(dataset1 - dataset2)
    if conv_dim == 2:
        mae=np.mean(sqr, axis=(2,3))
        L2=np.sqrt(np.sum(sqr,axis=(2,3)))
    elif conv_dim == 3:
        mae=np.mean(sqr, axis=(2,3,4))
        L2=np.sqrt(np.sum(sqr,axis=(2,3,4)))
    L2_mean=np.mean(np.mean(L2,axis=1))
    mae_mean=np.mean(mae)
    return mae_mean,L2_mean


def compute_MAE(dataset1, dataset2,conv_dim):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    abs=np.abs(dataset1 - dataset2)
    if conv_dim == 2:
        mae=np.mean(abs, axis=(1,2))
        L1=np.sum(abs,axis=(1,2))
    elif conv_dim == 3:
        mae=np.mean(abs, axis=(1,2,3))
        L1=np.sum(abs,axis=(1,2,3))
    L1_mean=np.mean(L1)
    mae_mean=np.mean(mae)
    return mae_mean,L1_mean


@torch.no_grad()
def compute_ssim(org_x, gen_x, conv_dim):
    """
    Expected shape (conv3d):
    [B, C, T, H, W]
    Range: [0,1]
    """
    print(org_x.shape,'expected [B, C, T, H, W]')
    if conv_dim == 2:
        raise NotImplementedError("conv_dim=2 not implemented")

    elif conv_dim == 3:
        #B, C, T, H, W = gen_x.shape
        B, H, W, T,C  = gen_x.shape

        # flatten time into batch
        gen_x = gen_x.transpose((0, 3,4 , 1, 2)).reshape((-1, C, H, W))
        org_x = org_x.transpose((0, 3, 4, 1, 2)).reshape((-1, C, H, W))
        gen_x = np.repeat(gen_x, 3, axis=1)
        org_x = np.repeat(org_x, 3, axis=1)

        org_x = torch.as_tensor(org_x, dtype=torch.float32)
        gen_x = torch.as_tensor(gen_x, dtype=torch.float32)

        device = next(lpips_metric.parameters()).device
        org_x = org_x.to(device)
        gen_x = gen_x.to(device)
        # TODO loop over batches else memory issues:

        batch_size=100
        ssim_metric.reset()

        for i in range(0, gen_x.shape[0], batch_size):
            g = gen_x[i:i + batch_size]
            o = org_x[i:i + batch_size]

            ssim_metric.update(g, o)

        score = ssim_metric.compute().item()

    return score


'''
def compute_LPIPS(org_x,gen_x,conv_dim):
    #import lpips_tf

    #b t h w c
    org_x=np.squeeze(org_x)
    gen_x = np.squeeze(gen_x)
    lpips_model = lpips_tf.LPIPS(net='vgg')
    gen_x = gen_x * 2.0 - 1.0
    org_x = org_x * 2.0 - 1.0
    #[B, H, W, C]

    if conv_dim == 2:
        print('not implemented')
    elif conv_dim == 3:
        b, t, h, w, c = gen_x.shape
        gen_flat = tf.reshape(gen_x, [-1, h, w, c])
        ref_flat = tf.reshape(org_x, [-1, h, w, c])
        score = lpips_model(gen_flat, ref_flat)#[b, t]
    ssim_mean = tf.reduce_mean(score)
    return ssim_mean
'''
@torch.no_grad()
def compute_lpips(org_x, gen_x, conv_dim):
    """
    Range must be [0,1] when normalize=True
    """
    print(org_x.shape, 'expected [B, C, T, H, W]')
    if conv_dim == 2:
        raise NotImplementedError("conv_dim=2 not implemented")

    elif conv_dim == 3:
        #B, C, T, H, W = gen_x.shape

        #gen_flat = gen_x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        #org_flat = org_x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        B, H, W, T,C  = gen_x.shape

        # flatten time into batch
        gen_x = gen_x.transpose((0, 3,4 , 1, 2)).reshape((-1, C, H, W))
        org_x = org_x.transpose((0, 3, 4, 1, 2)).reshape((-1, C, H, W))

        gen_x = np.repeat(gen_x, 3, axis=1)
        org_x = np.repeat(org_x, 3, axis=1)

        org_x = torch.as_tensor(org_x, dtype=torch.float32)
        gen_x = torch.as_tensor(gen_x, dtype=torch.float32)

        device = next(lpips_metric.parameters()).device
        org_x = org_x.to(device)
        gen_x = gen_x.to(device)


        batch_size=100
        lpips_metric.reset()

        for i in range(0, gen_x.shape[0], batch_size):
            g = gen_x[i:i + batch_size]
            o = org_x[i:i + batch_size]

            lpips_metric.update(g, o)

        score = lpips_metric.compute().item()

    return score




def compute_MAE_mean(dataset1, dataset2):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1=np.expand_dims(dataset1,1)
    abs=np.abs(dataset1 - dataset2)
    mae=np.mean(abs, axis=(2,3))
    L1=np.sum(abs,axis=(2,3))
    L1_mean=np.mean(np.mean(L1,axis=1))
    mae_mean=np.mean(mae)
    return mae_mean,L1_mean


def compute_validity(expected_y_cf,predicted_y_cf,num_classes=2):
    #expected_y_cf=np.argmax(expected_y_cf)
    #predicted_y_cf=np.argmax(predicted_y_cf)
    if num_classes==2:
        same=(expected_y_cf==predicted_y_cf)
        validity=np.mean(same)
        return validity
    else:
        mae = np.mean(np.abs(expected_y_cf - predicted_y_cf))
        rmse = np.sqrt(np.mean((expected_y_cf - predicted_y_cf) ** 2))
        res = stats.spearmanr(expected_y_cf, predicted_y_cf)[0]
        return mae,rmse,res


def mean_lsp_calculation(latent_representations,metric='euclidean'):

    N, k, d = latent_representations.shape
    diversity_scores = []

    for i in range(N):
        reps = latent_representations[i]  # shape (k, d)
        if k < 2:
            diversity_scores.append(0.0)
        else:
            pairwise_dists = pdist(reps, metric=metric)
            avg_dist = np.mean(pairwise_dists)
            diversity_scores.append(avg_dist)

    return np.mean(diversity_scores)



'''

if __name__ == '__main__':
    # test:
    # Example: Two random datasets (Replace with real images)
    dataset1 = tf.random.uniform([100, 64, 64, 3], 0, 255, dtype=tf.float32)  # Fake dataset 1
    dataset2 = tf.random.uniform([100, 64, 64, 3], 0, 255, dtype=tf.float32)  # Fake dataset 2

    # Compute FID
    fid_value = compute_fid(dataset1, dataset2)
    print(f'FID Score: {fid_value}')
'''
