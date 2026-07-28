import pickle as pkl
import glob
import imageio.v3 as iio
import numpy as np



def load_CXR_process(save_path,machine):
    if machine=='laptop':
        data_path='C:/Users/siekiera/Documents/curAIscid_data/'
    elif machine=='mogon':
        data_path = '/lustre/miifs01/project/m2_datamining/curAIscid_data/'
    else:
        data_path = '/lustre/project/ki-topml/curAIscid_data/'
    x_data=[[],[]]
    y_data=[[],[]]
    for data_idx,path_appendix in enumerate(['train/train/','test/test/']):
        for label,path_label in enumerate(['false/','true/']):
            png_path=data_path+path_appendix+path_label+'*'
            num_files=0
            for im_path in glob.glob(png_path):
                im = iio.imread(im_path)
                x_data[data_idx].append(im)
                y_data[data_idx].append(label)
                num_files+=1
            print('num files',num_files)
        x_data[data_idx]=np.expand_dims(np.asarray(x_data[data_idx]),axis=-1)#/255
        y_data[data_idx] = np.asarray(y_data[data_idx])
    #pkl.dump([[x_data[0],y_data[0]],[x_data[1],y_data[1]]],open(save_path,'wb'))
    return [x_data[0],y_data[0]],[x_data[1],y_data[1]]


def load_kmnist(data_path):
    (ds_train, ds_test), ds_info = tfds.load(
        'kmnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    x_train = np.asarray(list(ds_train.map(lambda x, y: x)))
    y_train = np.asarray(list(ds_train.map(lambda x, y: y)))
    x_test = np.asarray(list(ds_test.map(lambda x, y: x)))
    y_test = np.asarray(list(ds_test.map(lambda x, y: y)))

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    num_classes = 10
    pkl.dump(((x_train, y_train), (x_test, y_test), num_classes), open(data_path, 'wb'))
    print('finish loading')
    return (x_train, y_train), (x_test, y_test), num_classes



