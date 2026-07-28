import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def plot_heatmap(corr_matrix,plot_path):
    # Summary stats
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    mean_corr = np.mean(corr_values)
    median_corr = np.median(corr_values)

    print(f"Mean correlation: {mean_corr:.3f}")
    print(f"Median correlation: {median_corr:.3f}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title(f"Latent Corr \nMean: {mean_corr:.3f}, Median: {median_corr:.3f}")

    plt.savefig(plot_path)
    plt.clf()
def plot_correlations(gt,
                      predictions,
                      pred_results,
                      method_names,
                     debugging_plot_path):
    if len(gt.shape) > 1:
        gt = gt.flatten()

    x_axis_name = 'gt'
    figsize=4
    fig, axes = plt.subplots(1, len(method_names), figsize=(figsize*len(method_names), figsize))
    for i,(pred,model_name,results) in enumerate(zip(predictions,method_names,pred_results)):
        if len(pred.shape) > 1:
            pred = pred.flatten()
        dataFrame = pd.DataFrame({x_axis_name: gt, model_name: pred})
        sns.scatterplot(dataFrame, x=x_axis_name, y=model_name,ax=axes[i])
        mae, mse,rmse,sp = results
        title=' Sp:'+str(np.round(sp,2))+' MAE:'+str(np.round(mae,3))+' RMSE:'+str(np.round(rmse,3))
        axes[i].set_title(title)

    # plt.show()
    plt.tight_layout()
    plt.savefig(debugging_plot_path)
    plt.clf()
    plt.close()


'''
def plot_examples(sorted_data, input_data_name,plot_path,cnn_spn_model):
    for key,value in sorted_data.items():
        vals=value[:16]
        if len(vals)==16:
            # create reconstructions:
            gt, predictions, pred_exp,images=zip(*vals)
            images=np.asarray(list(images))

            test_rec = cnn_spn_model.vae_rec(images)

            # Create a figure with two sets of 4x4 grids (one 8x8 plot with 16 subplots)
            fig, axs = plt.subplots(4, 8, figsize=(12, 6))

            # First grid (left 4x4) for test_rec
            plot_image_grid(test_rec, axs[:, :4], "Reconstructed Images")

            # Second grid (right 4x4) for images
            plot_image_grid(images, axs[:, 4:], "Original Images")

            # Add an overall title
            plt.suptitle(key+' '+input_data_name, fontsize=18)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the overall title

            # Display the combined plot
            #plt.show()
            plt.savefig(plot_path+key+'_'+input_data_name+'.png')
            plt.clf()

'''
def plot_examples_3D(X_arrs,names,debugging_plot_path,cmap='gray', alpha=1.0):
    for X,name in zip(X_arrs,names):
        print('X shape',X.shape,flush =True)
        for frame in range(2):
            fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            for i in range(9):

                ax[i // 3, i % 3].imshow(X[i,frame], cmap=cmap, alpha=alpha)
                ax[i // 3, i % 3].axis('off')  # Turn off axis for each image


            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the overall title
            # Display the combined plot
            #plt.show()
            print(debugging_plot_path[:-4]+'_'+name+'_frame_'+str(frame)+'.png',flush=True)
            fig.savefig(debugging_plot_path[:-4]+'_'+name+'_frame_'+str(frame)+'.png')
            plt.clf()


def save_example_rec(test_x, test_x_rec,metric_results,metric_names,out_path='',cmap='gray', alpha=1.0):
    title=''
    for metric_name, metric_result in zip(metric_names, metric_results):
        title+=metric_name
        title+=':'+str(np.round(metric_result,decimals=3))+' '

    fig, axis = plt.subplots(test_x.shape[0], 4, figsize=(10, 10))
    for img_idx in range(test_x.shape[0]):
        for frame in range(2):
            ax_org=axis[img_idx,frame*2]
            ax_rec=axis[img_idx,(frame*2)+1]
            #######################################################
            ax_org.imshow(test_x[img_idx, frame], cmap=cmap, alpha=alpha)
            ax_org.axis('off')  # Turn off axis for each image
            #######################################################
            ax_rec.imshow(test_x_rec[img_idx, frame], cmap=cmap, alpha=alpha)
            ax_rec.axis('off')  # Turn off axis for each image
    fig.suptitle(title)
    #plt.show()
    fig.savefig(out_path)
    plt.clf()
