import pickle as pkl
import numpy as np



def load_dataset(dataset_name,binary,load_net,machine,path='',grid_params=None):

    if dataset_name=='stanford':
        data_path='sf_data.pkl'
        print(path+data_path)
        [train, test, val] = pkl.load(open(path+data_path, 'rb'))
        if grid_params.conv_dim==3:
            train[0] = np.expand_dims(train[0], axis=-1)
            test[0] = np.expand_dims(test[0], axis=-1)
            val[0] = np.expand_dims(val[0], axis=-1)
        print('DEBUG stanford org shape:',train[0].shape,val[0].shape,test[0].shape)
        return ([train,val]),test,1


    elif dataset_name == 'stanford_seg':
        data_path = 'sf_segmentation_frames_mask_09.npz'  #'sf_segmentation_frames_mask_09.pkl'
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
            train[0] = np.transpose(train[0], (0,  2, 3,4, 1))[:,:,:,:,0:1]/255.0
            test[0] = np.transpose(test[0], (0,  2, 3,4, 1))[:,:,:,:,0:1]/255.0
            val[0] = np.transpose(val[0], (0,  2, 3,4, 1))[:,:,:,:,0:1]/255.0 # batchsize,1 112,112,2
            print(np.max(train[0]),np.max(test[0]),np.max(val[0]))

            train[1] /=100
            test[1] /=100
            val[1] /=100

        return ([train, val]), test, 1 ##batchsize,112,112, 2,1



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
        print('DEBUG CAMUS org shape:',test[0].shape,val[0].shape,test[0].shape)
        test[0] = np.transpose(test[0], (0, 2, 3, 1, 4))
        print('DEBUG CAMUS org shape:', test[0].shape, val[0].shape, test[0].shape)
        return ([train, val]), test, 1 # batchsize, 2,112,112,1






    else:
        return None,None,None
