import numpy as np
import pickle as pkl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def make_individual_plot(image_num, betas, gammas, plot_path,save_file_path_plot,model_name,summarizing_fkt ,model_n,subtitle,use_background=True):
    fig_size=4
    print(gammas,betas)
    if  len(gammas)==1:
        fig, axis = plt.subplots(len(gammas),len(betas), figsize=(fig_size * len(betas), fig_size * 1))
    else:
        fig, axis = plt.subplots( len(gammas),len(betas), figsize=(fig_size*len(betas), fig_size*len(gammas)))
    last_img=None
    all_axis=[]

    for b_i, beta in enumerate(betas):
        for g_i, gamma in enumerate(gammas):
            #save_file_path = plot_path + 'datab' + str(b_i) + 'g' + str(g_i) + model_name + '_1000.pkl'
            save_file_path = plot_path + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + model_name + '_1000.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)
            #data = all_data[image_num]

            #reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class,coordinates, x, mean_cls, y=data
            #reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class, coordinates, x = data
            #x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,coordinates, x_org, mean_cls, y_org = data#zip(*all_data)
            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal, coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            #print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org), np.min(x_rec), np.max(x_rec),x_cf.shape)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org=np.asarray(x_org)
            coordinates=np.asarray(coordinates)[image_num]

            differences_mean = summarizing_fkt(x_cf[image_num], x_rec[image_num])


            mini_title ='beta:' + str(beta) + ' gamma:' + str(gamma) #+ "\ny_np:{:.2f},y_cf{:.0f};L:{:.2f};MSE{:.2f};\nlog(p(y|z)):{:.2f},p(z):{:.2f}".format(
                                                                      ##                                 opposite_class,arg_max_mean,
                                                                       #                                loss_mean,
                                                                       #                                distance_mean,
                                                                       #                                log_pred_mean,
                                                                       #                                p_z_mean)
            if len(gammas)>1:
                ax = axis[g_i, b_i]
            else:
                ax = axis[ b_i]
            if use_background:
                ax.imshow(x_org[image_num][0, :, :, 0], cmap='gray')

                last_img=ax.imshow(differences_mean, cmap='jet', alpha=0.5)#,vmax=vmax,vmin=vmin)
                # Add a colorbar specific to this subplot
                fig.colorbar(last_img, ax=ax, orientation='vertical')
            else:
                last_img = ax.imshow(differences_mean, cmap='gray')
            ax.set_title(mini_title)
            if len(coordinates):
                x_co, y_co, w_co, h_co = coordinates
                factor = 128 / 1024
                rect = patches.Rectangle((x_co * factor, y_co * factor), w_co * factor, h_co * factor, linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            all_axis.append(ax)
        # Add a single shared colorbar
    '''
    if use_background:
        cbar = fig.colorbar(last_img, ax=all_axis, location='right', shrink=0.8)
        cbar.set_label('Intensity')
    '''
    fig.suptitle(subtitle + ' model:' + str(model_name))
    #fig.suptitle(model_n+' y'+str(y))
    plt.tight_layout()
    print('save',save_file_path_plot)
    plt.show()
    plt.savefig(save_file_path_plot)

def make_raw_plot(image_num, betas, gammas, plot_path,save_file_path_plot,model_name,summarizing_fkt ,model_n,subtitle,use_background=False):
    fig_size=4
    print(gammas,betas)
    if  len(gammas)==1:
        fig, axis = plt.subplots(len(gammas),len(betas), figsize=(fig_size * len(betas), fig_size * 1))
    else:
        fig, axis = plt.subplots( len(gammas),len(betas), figsize=(fig_size*len(betas), fig_size*len(gammas)))
    last_img=None
    all_axis=[]

    for b_i, beta in enumerate(betas):
        for g_i, gamma in enumerate(gammas):
            save_file_path = plot_path + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + model_name + '_1000.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)
            #data = all_data[image_num]
            #x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,coordinates, x_org, mean_cls, y_org = data#zip(*all_data)
            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,
             coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org=np.asarray(x_org)
            coordinates = np.asarray(coordinates)[image_num]

            differences_mean = np.mean(x_cf[image_num],axis=0)


            mini_title ='beta:' + str(beta) + ' gamma:' + str(gamma) #+ "\ny_np:{:.2f},y_cf{:.0f};L:{:.2f};MSE{:.2f};\nlog(p(y|z)):{:.2f},p(z):{:.2f}".format(
                                                                      ##                                 opposite_class,arg_max_mean,
                                                                       #                                loss_mean,
                                                                       #                                distance_mean,
                                                                       #                                log_pred_mean,
                                                                       #                                p_z_mean)
            if len(gammas)>1:
                ax = axis[g_i, b_i]
            else:
                ax = axis[ b_i]
            if use_background:
                ax.imshow(x_org[image_num][0, :, :, 0], cmap='gray')

                #last_img=ax.imshow(differences_mean, cmap='jet', alpha=0.5)#,vmax=vmax,vmin=vmin)
                # Add a colorbar specific to this subplot
                fig.colorbar(last_img, ax=ax, orientation='vertical')
            else:
                last_img = ax.imshow(differences_mean, cmap='gray')
            ax.set_title(mini_title)
            if len(coordinates):
                x_co, y_co, w_co, h_co = coordinates
                factor = 128 / 1024
                rect = patches.Rectangle((x_co * factor, y_co * factor), w_co * factor, h_co * factor, linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            all_axis.append(ax)
        # Add a single shared colorbar

    fig.suptitle(subtitle + ' model:' + str(model_name))
    #fig.suptitle(model_n+' y'+str(y))
    plt.tight_layout()
    print('save',save_file_path_plot)
    plt.show()
    plt.savefig(save_file_path_plot)


