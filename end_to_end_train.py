import numpy
import tensorflow as tf
import numpy as np
from CNN_functions import train_embedding_cnn, get_layer_embeddings, load_CNN,\
    train_embedding_VAE, load_VAE_and_eval
from SPN_functions import create_SPN, create_SPN_diff_privacy, create_CSPN
import pickle as pkl
from CNN_SPN import CNN_SPN_Parts, test_model_no_mpe, train_model_parts, \
    train_generative_spn, train_MLP
from checkpoint_handler import create_checkpoint_paths, init_checkpoint_manager, create_checkpoint_manager
from CXR_loader import load_CXR_process
from dataset_loader import EchoTF
from tf2_spn import create_multiple_tf_spn_parts
import time
import os
import gc
import shutil
import pandas as pd
from utils import data_to_batch



def load_dataset(dataset_name,binary,load_net,machine,path='',grid_params=None):
    if dataset_name=='mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train=np.expand_dims(x_train,axis=-1)/ 255.0
        x_test = np.expand_dims(x_test, axis=-1)/ 255.0
        num_classes=10
        if binary:
            binary_idx_train=np.where(numpy.logical_or(y_train==1,y_train==0 ))
            binary_idx_test = np.where(numpy.logical_or(y_test == 1 , y_test == 0))

            x_train, y_train =x_train[binary_idx_train[0]], y_train[binary_idx_train[0]]
            x_test, y_test= x_test[binary_idx_test[0]], y_test[binary_idx_test[0]]
            num_classes=2


        return (x_train, y_train), (x_test, y_test),num_classes
    elif dataset_name=='stanford':
        data_path='sf_data.pkl'
        print(path+data_path)
        [train, test, val] = pkl.load(open(path+data_path, 'rb'))
        if grid_params.conv_dim==3:
            train[0] = np.expand_dims(train[0], axis=-1)
            test[0] = np.expand_dims(test[0], axis=-1)
            val[0] = np.expand_dims(val[0], axis=-1) #batchsize,112,112, 2,1

            train[0] = np.transpose(train[0], (0,3,1,2,4))
            test[0] = np.transpose(test[0], (0,3,1,2,4))
            val[0] = np.transpose(val[0], (0,3,1,2,4))

        return ([train,val]),test,1 # batchsize, 2,112,112,1
    elif dataset_name=="CAMUS":
        data_path = 'camus_echonet_style_mobi.npz'
        data = np.load(path + data_path)
        target_type = 'bi'
        train, test, val = [data["train_max_min_frame"], data["train_ef_" + target_type + "targets"]], \
            [data["test_max_min_frame"], data["test_ef_" + target_type + "targets"]], \
            [data["val_max_min_frame"], data["val_ef_" + target_type + "targets"]]
        if grid_params.conv_dim == 3:
            train[0] = np.transpose(train[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            test[0] = np.transpose(test[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            val[0] = np.transpose(val[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0 ## batchsize,1 112,112,2


            train[1] /=100
            test[1] /=100
            val[1] /=100
            print('CAMUS x y data min max', np.max(train[0]),np.min(train[0]), np.max(test[0]), np.max(val[0]),train[0].shape,
                  np.max(train[1]),np.min(train[1]),)
        print(train[0].shape,val[0].shape,test[0].shape)
        return ([train, val]), test, 1 # batchsize, 2,112,112,1


    elif dataset_name=='stanford_seg' or dataset_name=='stanford_seg_estim_ef':
        if dataset_name=='stanford_seg':
            data_path = 'segment_paper_multiple_imgs.npz'
        else:
            data_path = 'segment_paper_multiple_imgs_estimated_ef.npz'
        #'sf_segmentation_frames_mask_09.npz'  #'sf_segmentation_frames_mask_09.pkl'
        print(path + data_path)
        #/lustre/project/ki-topml/jsiekiera/cf_data/sf_segmentation_frames_mask_09.pkl
        #result= pkl.load(open(path + data_path, 'rb'))
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
        train_data,test_data, val_data=result['train'],result['test'],result['val']
        train,test,val=[train_data[2],train_data[0]],[test_data[2],test_data[0]],[val_data[2],val_data[0]]
        print(train[0].shape)
        if grid_params.conv_dim == 3:
            train[0] = np.transpose(train[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            test[0] = np.transpose(test[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0
            val[0] = np.transpose(val[0], (0, 4, 2, 3, 1))[:,:,:,:,0:1]/255.0 ## batchsize,1 112,112,2
            print(np.max(train[0]),np.max(test[0]),np.max(val[0]))

            train[1] /=100
            test[1] /=100
            val[1] /=100

        return ([train, val]), test, 1 # batchsize, 2,112,112,1

    elif dataset_name == 'stanford_video':
        #return load_videos('')
        data_dir=['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/data_in/EchoNet-Dynamic'][1]
        train=EchoTF(root=data_dir, split="train", target_type="EF")
        val=EchoTF(root=data_dir, split="val", target_type="EF")
        test=EchoTF(root=data_dir, split="test", target_type="EF")
        return ([train, val]), test, 1


    elif dataset_name=='CXR_process':
        data_path='CXR_process.pkl'
        #if os.path.exists(data_path):
        if not load_net:
            [train, test] = pkl.load(open(data_path, 'rb'))
        #if load_net:
        #    train[0]=np.repeat(train[0], 3, axis=3)
        #    test[0]=np.repeat(test[0], 3, axis=3)
        else:
            [train, test] = load_CXR_process(data_path,machine)
            #print('Data shape',train[0].shape,train[0][:1],flush=True)
            train[0]=np.repeat(train[0], 3, axis=3)
            test[0]=np.repeat(test[0], 3, axis=3)
        return train, test, 2
    elif dataset_name=='CXR_process_org_distr':
        data_path = path+'data_analysis/x_ray_own_preprocessing_50_percent.pkl'
        [train_data, test_data] = pkl.load(open(data_path, 'rb'))
        if grid_params is not None and grid_params.separate_view:
            view_label_int_train = train_data[2][:, 2].astype(int)
            #############################################################
            train_data[0]=train_data[0][view_label_int_train == 1]
            train_data[1] = train_data[1][view_label_int_train == 1]
            train_data[2] = train_data[2][view_label_int_train == 1]
            view_label_int_test = test_data[2][:, 2].astype(int)
            test_data[0]=test_data[0][view_label_int_test == 1]
            test_data[1] = test_data[1][view_label_int_test == 1]
            test_data[2] = test_data[2][view_label_int_test == 1]


        train_other=np.concatenate([np.expand_dims(train_data[1],axis=1),train_data[2]],axis=1)
        train_data=[train_data[0],train_other,train_data[3]]
        test_other = np.concatenate([np.expand_dims(test_data[1], axis=1), test_data[2]],axis=1)
        test_data = [test_data[0], test_other,test_data[3]]
        if not load_net:
            #    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
            train_data[0]=np.expand_dims(train_data[0]/255.0, axis=-1)
            test_data[0]=np.expand_dims(test_data[0]/255.0, axis=-1)
            #train_data[0]=np.expand_dims( (train_data[0] - 127.5) / 127.5,axis=-1)
            #test_data[0] = np.expand_dims((test_data[0] - 127.5) / 127.5,axis=-1)

        else:
            train_data[0]=np.repeat(np.expand_dims(train_data[0],axis=-1), 3, axis=3)
            test_data[0]=np.repeat(np.expand_dims(test_data[0],axis=-1), 3, axis=3)
        return train_data,test_data,2
    elif dataset_name=='chexpert':
        data_path = path+'chexpert.pkl'
        data = pkl.load(open(data_path, 'rb'))
        #[X, int_data, float_data]=val
        train_data=data['train']
        #'train', 'validation', 'test'
        val_data=data['validation']
        test_data=data['test']

        #normalize float data
        max_floats=np.max(train_data[2],axis=0)
        train_data=[train_data[0],np.concatenate([train_data[1],train_data[2]/max_floats],axis=1)]
        val_data = [val_data[0], np.concatenate([val_data[1], val_data[2] / max_floats], axis=1)]
        test_data = [test_data[0], np.concatenate([test_data[1], test_data[2] / max_floats], axis=1)]
        print(val_data[0].shape,val_data[1].shape)
        if not load_net:
            print('short data example',train_data[0].shape,train_data[0][0,:5])
            train_data[0]=np.expand_dims(train_data[0]/255.0,axis=-1)
            test_data[0]=np.expand_dims(test_data[0]/255.0,axis=-1)
            val_data[0] =np.expand_dims( val_data[0] / 255.0,axis=-1)
            #train_data[0]= np.expand_dims((train_data[0] - 127.5) / 127.5,axis=-1)
            #test_data[0] = np.expand_dims((test_data[0] - 127.5) / 127.5,axis=-1)
            #val_data[0]= np.expand_dims((val_data[0] - 127.5) / 127.5,axis=-1)



        else:
            train_data[0]=np.repeat(np.expand_dims(train_data[0],axis=-1), 3, axis=3)
            test_data[0]=np.repeat(np.expand_dims(test_data[0],axis=-1), 3, axis=3)
            val_data[0] = np.repeat(np.expand_dims(val_data[0], axis=-1), 3, axis=3)
        print('load chex')
        return ([train_data,val_data]),test_data,2


    elif dataset_name=='kmnist':
        data_path = 'kmnist.pkl'
       # (x_train, y_train), (x_test, y_test),num_classes=load_kmnist(data_path)
        ((x_train, y_train), (x_test, y_test),num_classes)=pkl.load(open(data_path, 'rb'))
        if load_net:
            ## rescale:
            # Function to resize a single image
            #def resize_image(image):
            #    resized = cv2.resize(image, (56, 56))
            #    return np.expand_dims(resized, axis=-1)

            # Resize the batch of images
            x_train = tf.image.resize(x_train, [56, 56])#np.array([resize_image(image) for image in x_train])
            x_test = tf.image.resize(x_test, [56, 56]) #np.array([resize_image(image) for image in x_test])

            x_train=np.repeat(x_train, 3, axis=3)
            x_test=np.repeat(x_test, 3, axis=3)
    elif dataset_name == 'TUF':
        # Attention this is only the data of fold 0
        base_path = '/lustre/miifs01/project/m2_datamining/sync_data/TMED-156-52/fold0/'
        pkl_path = base_path + 'diagnosis.pkl'

        data = pkl.load(open(pkl_path, 'rb'))
        x_train,y_train = data['train'][0], data['train'][1]
        x_val,y_val = data['val'][0], data['val'][1]
        x_test,y_test = data['test'][0], data['test'][1]
        num_classes = len(np.unique(y_train))
        #print(x_train.shape())

        #x_train, x_val=np.zeros((5,20)),np.zeros((5,20))


        x_train_=np.concatenate([x_train,x_val],axis=0)
        y_train_=np.concatenate([y_train,y_val],axis=0)
        print(x_train_.shape)


        return (x_train, y_train_), (x_test, y_test),num_classes





    else:
        return None,None,None



def SPN_structure_train(grid_params,model, layer_names, train_dataset, test_dataset,get_max,num_test_instances,
                        num_classes,debugging,add_info):
    embedding_arr, all_labels ,other_info= get_layer_embeddings(grid_params,model, layer_names, [train_dataset, test_dataset],
                                                     get_max=get_max,add_info=add_info)  # valloader

    # normalize embedding:

    [train_embedding, test_embedding] = embedding_arr
    if grid_params.use_add_info:
        print(other_info[0].shape)
        train_embedding=np.concatenate([train_embedding,other_info[0]],axis=-1)
        test_embedding = np.concatenate([test_embedding, other_info[1]], axis=-1)

    # 3. create spn structure +evaluate SPN structure

    train_data_spn = np.concatenate([np.expand_dims(all_labels[0], axis=-1), train_embedding], axis=-1)
    out_labels = np.full((num_test_instances, 1), np.nan)
    test_data_spn = np.concatenate([out_labels, test_embedding], axis=1)

    if num_classes==1:
        #handle regression with multiple CSPNs
        spn_output = create_CSPN(grid_params,train_data_spn, test_data_spn, all_labels[1],
                   input_dim=train_embedding.shape[1], debugging=debugging)
    else:
        if grid_params.diff_privacy:
            #spn_clf, label_ids
            spn_output = create_SPN_diff_privacy(grid_params, train_data_spn, test_data_spn, all_labels[1],
                                            input_dim=train_embedding.shape[1], debugging=debugging)
        else:
            # spn_clf, label_ids
            spn_output = create_SPN(grid_params,train_data_spn, test_data_spn, all_labels[1], input_dim=train_embedding.shape[1],debugging=debugging)

    return spn_output



def train_cnn_spn(grid_params,train_data,test_data,val_data,num_classes,debugging=False,split=0,fold_path=''):
    train_data[0] = train_data[0].astype(np.float32)
    val_data[0] = val_data[0].astype(np.float32)
    test_data[0] = test_data[0].astype(np.float32)

    train_data[1] = train_data[1].astype(np.float32)
    val_data[1] = val_data[1].astype(np.float32)
    test_data[1] = test_data[1].astype(np.float32)
    get_max=False
    load_data=False
    if grid_params.VAE_fine_tune=="SPN_VAE":
        max_iterations = 10
    else:
        max_iterations=1

    ckp_folder_names=['vae_unsup_checkpoints/','vae_unsup_checkpoints_tmp/',
                      'vae_checkpoints/','vae_checkpoints_tmp/','cnn_spn_checkpoints/',
                      'spn_checkpoints/','spn_checkpoints_finetune/',
                      'spn_z_checkpoints/','spn_z_checkpoints_finetune/','mlp_checkpoints/','mlp_checkpoints_finetune/']
    checkpoint_paths = create_checkpoint_paths(fold_path, ckp_folder_names)

    fine_tune_rate=grid_params.fine_tune_rate
    last_num_filters = grid_params.latent_dim
    layer_names = ['flatten']

    # 1. train embedding CNN + evaluate performance of CNN+
    # save CNN weights
    add_info=grid_params.use_add_info
    print('add info: spn training',add_info)

    if grid_params.dataset_name=='stanford_video':

        input_shape = (16,112,112,1)#train_data[0].shape[1:] #(112,112,3)#
    else:
        input_shape = train_data[0].shape[1:]

    print(input_shape)


    if grid_params.use_VAE:
        #1. train VAE only on reconstruction goal
        #model,manager,ckpt,_,all_debugging_stuff=train_embedding_VAE(grid_params,train_data,test_data,val_data,input_shape,
        #                                       num_classes,checkpoint_paths[0],load_data,split,add_info,checkpoint_paths[1],
        #                                                             pretrain=True,model_path=fold_path)
        # copy checkpoint stuff
        for filename in os.listdir(checkpoint_paths[0] + '/'):
            shutil.copy2(checkpoint_paths[0] + '/' + filename, checkpoint_paths[2] + '/' + filename)

        #2. fine-tune VAE on reconstruction goal + y target
        print('#################################################################################################')
        print('train supervised')
        model,manager,ckpt,_,all_debugging_stuff=train_embedding_VAE(grid_params,train_data,test_data,val_data,input_shape,
                                               num_classes,checkpoint_paths[2],load_data,split,add_info,checkpoint_paths[3],
                                                                     pretrain=False,model_path=fold_path)
        # save debugging stuff:
        if grid_params.VAE_debug:
            print('save debugging file')
            pkl.dump(all_debugging_stuff,open(fold_path+'_debugg.pkl','wb'))
    else:
        model,manager,ckpt=train_embedding_cnn(grid_params,val_data,input_shape,
                                               num_classes,checkpoint_paths[2],load_data,split,train_data,add_info)

    # save cnn + mlp in another folder:
    ##################################
    model_path_copy=fold_path+'vae_checkpoints_copy/'#'models/'+model_name
    if not os.path.exists(model_path_copy):
        os.mkdir(model_path_copy)
    model_path_copy+= '/tf_ckpts_last'
    if not os.path.exists(model_path_copy):
        os.mkdir(model_path_copy)
    # Copy each file from dir1 to dir2
    for filename in os.listdir(checkpoint_paths[2]+'/'):
        shutil.copy2(checkpoint_paths[2]+'/'+filename, model_path_copy+'/'+filename)

    gc.collect()
    del model
    tf.keras.backend.clear_session()

    ##########################################################
    # 5. create Cnn_spn
    it_counter=0
    all_accs=[]

    #if num_classes == 1:
    #    curr_acc = float('inf')
    #else:
    curr_acc = 0

    MLP_evals=[]
    all_vae_debug_stuff=[]

    train_dataset,test_dataset,val_dataset=data_to_batch(train_data,test_data,val_data,add_info,grid_params)
    print('before mask',train_data[0].shape)
    presence = np.percentile(train_data[0], 99, axis=(0, 1, 4))  # (H,W)
    cone_mask = (presence > 0.02).astype(np.float32)
    cone_mask_tf = tf.convert_to_tensor(cone_mask, dtype=tf.float32)

    # -> (1,1,H,W,1)
    cone_mask_tf = cone_mask_tf[tf.newaxis, tf.newaxis, :, :, tf.newaxis]
    print('after mask',cone_mask.shape,cone_mask_tf.shape)


    while it_counter<max_iterations:
        # load saved model:
        if grid_params.use_VAE:
            model,ckpt,manager,MLP_eval=load_VAE_and_eval(grid_params, input_shape, num_classes,
                                                 debugging,ckpt,manager,checkpoint_paths[2],
                                                 train_dataset,test_dataset,val_dataset,add_info,cone_mask_tf)
        else:
            model,ckpt,manager,MLP_eval=load_CNN(grid_params, input_shape, num_classes,
                                                 debugging,ckpt,manager,checkpoint_paths[2],
                                                 train_data,test_data,val_data,add_info)

        print('MLP eval', MLP_eval, flush=True)
        MLP_evals.append(MLP_eval)

        block_start_time = time.time()

        decoder = None
        if grid_params.use_VAE:
            vae_prep_start = time.time()
            cnn_embedding = model.clf_model()
            # if grid_params.VAE_fine_tune:
            decoder = model.decoder
            print('[time debug] prepare cnn_embedding via clf_model():',
                  np.round(time.time() - vae_prep_start, 3), 'sec', flush=True)
        else:
            cnn_prep_start = time.time()
            cnn_layer_out = model.get_layer(layer_names[0]).output
            cnn_embedding = tf.keras.Model(inputs=model.input, outputs=cnn_layer_out)
            print('[time debug] prepare cnn_embedding via get_layer/model():',
                  np.round(time.time() - cnn_prep_start, 3), 'sec', flush=True)

        if grid_params.use_add_info and add_info:
            spn_input_shape = (last_num_filters + train_data[1].shape[1],)
        else:
            spn_input_shape = (last_num_filters + 1,)

        print('[time debug] spn_input_shape:', spn_input_shape, flush=True)

        # ------------------------------------------------------------------
        # SPN structure training
        # ------------------------------------------------------------------
        spn_structure_start = time.time()
        print('[time debug] BEFORE SPN_structure_train', flush=True)
        if grid_params.VAE_fine_tune=='SPN_VAE_MLP':
            print('ATTENTION LOAD OLD SPN_finetune',grid_params.VAE_fine_tune)
            # 2. load spn_structure
            spn_data = pkl.load(open(fold_path  + 'spn_old.pkl', 'rb'))
            spn_clf = spn_data['spn_x']
            spn_input_shape = spn_data['data_shape']
            label_ids = spn_data['label_ids']
            bin_borders = spn_data['bin_borders']
            bin_widths_scaled = spn_data['bin_widths_scaled']
            spn_z = spn_data['spn_z']
            spn_output = spn_clf, label_ids, bin_borders, bin_widths_scaled, spn_z


        else:
            print('ATTENTION CREATE NEW SPN structure',grid_params.dataset_name)
            spn_output = SPN_structure_train(grid_params, model, layer_names, train_dataset, test_dataset, get_max,
                                             num_test_instances=test_data[0].shape[0], num_classes=num_classes,
                                             debugging=debugging, add_info=add_info)

        spn_structure_end = time.time()
        print('[time debug] AFTER SPN_structure_train:',
              np.round(spn_structure_end - spn_structure_start, 3), 'sec',
              '(' + str(np.round((spn_structure_end - spn_structure_start) / 60, 3)) + ' min )',
              flush=True)

        # ------------------------------------------------------------------
        # Unpack spn_output
        # ------------------------------------------------------------------
        spn_unpack_start = time.time()

        if num_classes > 1:
            spn_clf, label_ids = spn_output[0], spn_output[1]
            spn_z = None
            bin_borders = None
            bin_widths_scaled = None
        else:
            spn_clf, label_ids, bin_borders, bin_widths_scaled, spn_z = spn_output

        spn_unpack_end = time.time()
        print('[time debug] unpack spn_output:',
              np.round(spn_unpack_end - spn_unpack_start, 3), 'sec', flush=True)

        # ------------------------------------------------------------------
        # create_multiple_tf_spn_parts
        # ------------------------------------------------------------------
        create_tf_spn_start = time.time()
        print('[time debug] BEFORE create_multiple_tf_spn_parts', flush=True)

        all_spn_x_y_model, all_prior, spn_model_z = create_multiple_tf_spn_parts(spn_out_data=spn_output,
                                                                                 num_classes=num_classes,
                                                                                 data_shape=spn_input_shape,
                                                                                 trainable_leaf=grid_params.fine_tune_leafs)

        create_tf_spn_end = time.time()
        print('[time debug] AFTER create_multiple_tf_spn_parts:',
              np.round(create_tf_spn_end - create_tf_spn_start, 3), 'sec',
              '(' + str(np.round((create_tf_spn_end - create_tf_spn_start) / 60, 3)) + ' min )',
              flush=True)

        # ------------------------------------------------------------------
        # Build CNN_SPN_Parts
        # ------------------------------------------------------------------
        cnn_spn_build_start = time.time()
        print('[time debug] BEFORE CNN_SPN_Parts init', flush=True)

        cnn_spn = CNN_SPN_Parts(num_classes=num_classes,
                                learning_rate=fine_tune_rate*0.1 if grid_params.dataset_name=='CAMUS' else fine_tune_rate,
                                all_spn_x_y_model=all_spn_x_y_model,
                                all_prior=all_prior,
                                cnn=cnn_embedding,
                                get_max=get_max,
                                gauss_embeds=grid_params.gauss_embeds,
                                use_add_info=grid_params.use_add_info,
                                VAE_fine_tune=grid_params.VAE_fine_tune,
                                decoder=decoder,
                                loss_weights=grid_params.loss_weights,
                                load_pretrain_model=grid_params.load_pretrain_model,
                                use_GAN=grid_params.GAN,
                                filter_size=grid_params.filter_size,
                                num_layer=grid_params.num_layer,
                                # input_shape=input_shape,
                                batchnorm_integration=grid_params.batchnorm_integration,
                                shortcut=grid_params.shortcut,
                                activation=grid_params.activation,
                                num_filter=grid_params.num_filter_encoder,
                                strides=grid_params.strides_encoder,
                                end_dim=100,
                                dropout=grid_params.dropout, clf_mlp=model.classifier,
                                bin_borders=bin_borders,
                                bin_widths_scaled=bin_widths_scaled,
                                spn_model_z=spn_model_z,
                                conv_dim=grid_params.conv_dim,
                                rec_loss_name=grid_params.rec_loss
                                )

        if grid_params.VAE_fine_tune=='SPN_VAE_MLP':
            start = time.time()
            print('use finetuned spns')
            ckpt_clf_spn = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.clf_spn)
            manager_clf_spn = tf.train.CheckpointManager(ckpt_clf_spn,
                                                         fold_path + 'spn_checkpoints_finetune_old/tf_ckpts_last',
                                                         max_to_keep=1)
            ckpt_clf_spn.restore(manager_clf_spn.latest_checkpoint)
            print("Restored from {}".format(manager_clf_spn.latest_checkpoint))

            ckpt_z_spn = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.spn_model_z)
            manager_z_spn = tf.train.CheckpointManager(ckpt_z_spn,
                                                       fold_path + 'spn_z_checkpoints_finetune_old/tf_ckpts_last',
                                                       max_to_keep=1)
            ckpt_z_spn.restore(manager_z_spn.latest_checkpoint)
            print("Restored from {}".format(manager_z_spn.latest_checkpoint), (time.time() - start) / 60, flush=True)


            # set batchnorm layer trainable = False!
            print('######################trainable batchnorm?!')
            for layer in cnn_spn.submodules:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    print(layer.name, layer.trainable)
            count = 0
            for layer in cnn_spn.submodules:#.embedding
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                    print(layer.name)
                    count += 1
            print('BN counter',count)

            for layer in cnn_spn.submodules:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    print(layer.name, layer.trainable)
            print('######################trainable batchnorm?!',flush=True)






        cnn_spn_build_end = time.time()
        print('[time debug] AFTER CNN_SPN_Parts init:',
              np.round(cnn_spn_build_end - cnn_spn_build_start, 3), 'sec', flush=True)

        # ------------------------------------------------------------------
        # Dummy tensors
        # ------------------------------------------------------------------
        dummy_build_start = time.time()
        x_dummy = tf.random.normal(tuple([3] + list(input_shape)))
        y_dummy = tf.random.normal((3, 1))
        embedding_dummy = tf.random.normal((3, grid_params.latent_dim))
        other_data_dummy = None
        if len(train_data[1].shape) > 1:
            other_data_dummy = tf.random.normal((3, train_data[1].shape[1] - 1))
        dummy_build_end = time.time()
        print('[time debug] create dummy tensors:',
              np.round(dummy_build_end - dummy_build_start, 3), 'sec', flush=True)

        # this triggers variable creation & graph connection
        # _=cnn_spn.clf_spn.spn_regression_exp()
        # _ = cnn_spn.model_execution_vae(x_dummy,y_dummy,other_data_dummy,train_vae=False,train_clf=False)
        warmup_start = time.time()
        print('[time debug] BEFORE warmup model_execution_X', flush=True)
        _ = cnn_spn.model_execution_X(x_dummy, other_data_dummy, train_vae=False, train_clf=False)
        warmup_end = time.time()
        print('[time debug] AFTER warmup model_execution_X:',
              np.round(warmup_end - warmup_start, 3), 'sec',
              '(' + str(np.round((warmup_end - warmup_start) / 60, 3)) + ' min )',
              flush=True)

        #########################################################################################################

        # ------------------------------------------------------------------
        # Save SPN pkl
        # ------------------------------------------------------------------
        spn_save_start = time.time()
        spn_save_data = {'spn_x': spn_clf,
                         'data_shape': spn_input_shape,
                         'label_ids': label_ids,
                         'spn_z': spn_z,
                         'bin_borders': bin_borders,
                         'bin_widths_scaled': bin_widths_scaled}
        pkl.dump(spn_save_data, open(fold_path + 'spn.pkl', 'wb'))
        spn_save_end = time.time()
        print('[time debug] save spn.pkl:',
              np.round(spn_save_end - spn_save_start, 3), 'sec', flush=True)

        # ------------------------------------------------------------------
        # Checkpoint manager
        # ------------------------------------------------------------------
        ckpt_build_start = time.time()
        sub_models = [cnn_spn, cnn_spn.clf_spn, cnn_spn.clf_spn, cnn_spn.spn_model_z, cnn_spn.spn_model_z,
                      cnn_spn.clf_mlp, cnn_spn.clf_mlp]
        ckps, managers = create_checkpoint_manager(sub_models, checkpoint_paths[4:])
        ckpt_build_end = time.time()
        print('[time debug] create_checkpoint_manager:',
              np.round(ckpt_build_end - ckpt_build_start, 3), 'sec', flush=True)

        if not it_counter:
            init_ckpt_start = time.time()
            init_checkpoint_manager(ckps, managers)
            print('initial save of all checkpoints', flush=True)
            print('[time debug] init_checkpoint_manager:',
                  np.round(time.time() - init_ckpt_start, 3), 'sec', flush=True)
        # '''
        #########################################################################################################
        #'''
        # print('FIRST SPN TEST',flush=True)
        print('BEFORE CNN+SPN TRAIN', flush=True)

        # ------------------------------------------------------------------
        # Eval before train
        # ------------------------------------------------------------------
        eval_before_start = time.time()
        eval_before_train = test_model_no_mpe(cnn_spn, [train_data, val_data, test_data], num_classes,
                                              grid_params.batch_size, add_info=add_info)
        print('FIRST EVAL RESULTS',eval_before_train)
        eval_before_end = time.time()
        print('[time debug] eval_before_train:',
              np.round(eval_before_end - eval_before_start, 3), 'sec',
              '(' + str(np.round((eval_before_end - eval_before_start) / 60, 3)) + ' min )',
              flush=True)

        if num_classes == 1:
            print('With spearman', curr_acc,eval_before_train[1][2])
            if eval_before_train[1][2] > curr_acc:
                curr_acc = eval_before_train[1][2]  # spearman
                print('init son with spearman',curr_acc)

        else:
            if eval_before_train[1][2] > curr_acc:
                curr_acc = eval_before_train[1][2]  # balanced accuracy

        # print('train MLP')
        # train_MLP(grid_params, cnn_spn, ckps[6], managers[6], train_data, val_data, curr_log_likelihood=1000000)

        print('TRAIN CNN+SPN', flush=True)

        # ------------------------------------------------------------------
        # Fine-tune CNN+SPN
        # ------------------------------------------------------------------
        finetune_start = time.time()
        eval_after_train, vae_debugg_stuff = train_model_parts(grid_params, cnn_spn, train_data,
                                                               val_data, test_data,
                                                               ckpt=[ckps[0], ckpt, ckps[2]],
                                                               manager=[managers[0], manager, managers[2]],
                                                               val_entropy=eval_before_train[1][1],
                                                               val_acc=curr_acc, add_info=add_info,
                                                               num_classes=num_classes)
        finetune_end = time.time()
        print('[time debug] train_model_parts:',
              np.round(finetune_end - finetune_start, 3), 'sec',
              '(' + str(np.round((finetune_end - finetune_start) / 60, 3)) + ' min )',
              flush=True)
        #'''
        if grid_params.VAE_fine_tune=='SPN_VAE_MLP':
            print('start fine_tune MLP')
            train_MLP(grid_params, cnn_spn, ckps[6], managers[6], train_data, val_data, curr_log_likelihood=1000000)
            print('finish fine_tune MLP')

        #'''
        # 6 train the method and fine tune both CNN + SPN
        print('finetune z spn', flush=True)

        # ------------------------------------------------------------------
        # Generative SPN
        # ------------------------------------------------------------------

        gen_spn_start = time.time()
        train_generative_spn(grid_params, cnn_spn, ckps[4], managers[4], train_data, val_data,
                             curr_log_likelihood=1000000)
        gen_spn_end = time.time()
        print('[time debug] train_generative_spn:',
              np.round(gen_spn_end - gen_spn_start, 3), 'sec',
              '(' + str(np.round((gen_spn_end - gen_spn_start) / 60, 3)) + ' min )',
              flush=True)

        all_vae_debug_stuff.append(vae_debugg_stuff)

        # save this:
        # save debugging stuff:
        # if grid_params.de:
        print('save debugging file', flush=True)

        debug_save_start = time.time()
        pkl.dump(all_vae_debug_stuff, open(fold_path + '_SPN_debug.pkl', 'wb'))
        debug_save_end = time.time()
        print('[time debug] save _SPN_debug.pkl:',
              np.round(debug_save_end - debug_save_start, 3), 'sec', flush=True)

        # print('AFTER TRAINING weights', model.trainable_variables[0].numpy()[0, 0, 0, :5])
        # 7 test the performance
        # print('SECOND SPN TEST',flush=True)
        if not len(eval_after_train):
            eval_after_train = eval_before_train

        if num_classes == 1:
            if eval_after_train[1][5] > curr_acc:
                curr_acc = eval_after_train[1][5]  # spearman
        else:
            if eval_after_train[1][2] > curr_acc:
                curr_acc = eval_after_train[1][2]  # balanced accuracy

        eval_diff = eval_after_train[1][0] - eval_before_train[1][0]

        all_accs.extend([eval_before_train, eval_after_train])

        print('ACC increase:', eval_after_train[1][0], eval_before_train[1][0], eval_diff,
              flush=True)  # eval_after_train,eval_before_train

        print('[time debug] total block time:',
              np.round(time.time() - block_start_time, 3), 'sec',
              '(' + str(np.round((time.time() - block_start_time) / 60, 3)) + ' min )',
              flush=True)


        #'''
        del cnn_spn
        del spn_clf
        del model
        gc.collect()
        tf.keras.backend.clear_session()

        it_counter+=1



    return all_accs,MLP_evals
    #'''

def load_CXR():
    train_data_path = '/lustre/miifs01/project/m2_datamining/curAIscid_data/train/train'

    filepaths_train = []
    labels_train = []
    train_labels_ = []

    folds = os.listdir(train_data_path)

    for fold in folds:
        if 'true' in fold:
            label = 1
        else:
            label = 0
        f_path = os.path.join(train_data_path, fold)
        filelists = os.listdir(f_path)

        for file in filelists:
            filepaths_train.append(os.path.join(f_path, file))
            labels_train.append(fold)
            train_labels_.append(label)

    # Concat data paths with labels
    Fseries = pd.Series(filepaths_train, name='filepaths')
    Lseries = pd.Series(labels_train, name='label')
    train_df = pd.concat([Fseries, Lseries], axis=1)

    # test_data_path = 'C:/Users/siekiera/Documents/curAIscid_data/test/test'
    test_data_path = '/lustre/miifs01/project/m2_datamining/curAIscid_data/test/test'
    filepaths_test = []
    labels_test = []
    test_labels_ = []

    folds = os.listdir(test_data_path)

    for fold in folds:

        if 'true' in fold:
            label = 1
        else:
            label = 0

        f_path = os.path.join(test_data_path, fold)
        filelists = os.listdir(f_path)

        for file in filelists:
            filepaths_test.append(os.path.join(f_path, file))
            labels_test.append(fold)
            test_labels_.append(label)

    # Concat data paths with labels
    Fseries = pd.Series(filepaths_test, name='filepaths')
    Lseries = pd.Series(labels_test, name='label')
    test_df = pd.concat([Fseries, Lseries], axis=1)
    return train_df,test_df


