import tensorflow as tf
import random



'''
def data_to_batch(train_data,test_data,val_data,add_info,params,randomize=False):

    if params.dataset_name=='stanford_video':
        train_dataset = train_data.get_tf_dataset(batch_size=params.batch_size).shuffle(buffer_size=10)
        val_dataset = val_data.get_tf_dataset(batch_size=params.batch_size).shuffle(buffer_size=10)
        test_dataset = test_data.get_tf_dataset(batch_size=params.batch_size, shuffle=False)
        return train_dataset, test_dataset, val_dataset
    idx_train = list(range(train_data[0].shape[0]))

    if randomize:
        random.shuffle(idx_train)
        if add_info:
            # print(val_data[0].shape,val_data[1].shape,'val data shape')
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train],
                                                                train_data[1][idx_train, 0].astype(int),
                                                                train_data[1][idx_train, 1:])).shuffle(10).batch(
                params.batch_size).cache().prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (val_data[0], val_data[1][:, 0].astype(int), val_data[1][:, 1:])).shuffle(10).batch(
                params.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[0][idx_train], train_data[1][idx_train])).shuffle(10).batch(params.batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).shuffle(10).batch(params.batch_size)


        return train_dataset,0,val_dataset

    if add_info:
        #print(val_data[0].shape,val_data[1].shape,'val data shape')
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train], train_data[1][idx_train,0].astype(int),train_data[1][idx_train,1:])).shuffle(10).batch(params.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1][:,0].astype(int),test_data[1][:,1:])).shuffle(10).batch(params.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1][:,0].astype(int),val_data[1][:,1:])).shuffle(10).batch(params.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train], train_data[1][idx_train])).shuffle(10).batch(params.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).shuffle(10).batch(params.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).shuffle(10).batch(params.batch_size)



    return train_dataset, test_dataset, val_dataset
'''
#'''
def data_to_batch(train_data, test_data, val_data, add_info, params):
    batch_size = params.batch_size

    def make_dataset(x, y, add_info_data=None, shuffle=False):
        if add_info_data is not None:
            ds = tf.data.Dataset.from_tensor_slices(
                (x, y[:, 0].astype(int), y[:, 1:])
            )
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(x),
                reshuffle_each_iteration=True
            )

        return (
            ds.batch(batch_size)
              .prefetch(tf.data.AUTOTUNE)
        )

    # Training dataset (SHUFFLED)
    train_dataset = make_dataset(
        train_data[0],
        train_data[1],
        add_info_data=train_data[1] if add_info else None,
        shuffle=True
    )

    # Validation dataset (NO shuffle)
    val_dataset = make_dataset(
        val_data[0],
        val_data[1],
        add_info_data=val_data[1] if add_info else None,
        shuffle=False
    )

    # Test dataset (NO shuffle)
    test_dataset = make_dataset(
        test_data[0],
        test_data[1],
        add_info_data=test_data[1] if add_info else None,
        shuffle=False
    )

    return train_dataset, test_dataset, val_dataset
'''
def data_to_batch(train_data, test_data, val_data, add_info, params):
    batch_size = params.batch_size

    def random_frame_flip_x(X):
        # X shape: (batch, 2, ...)
        batch_n = tf.shape(X)[0]

        flip_mask = tf.random.uniform((batch_n,), 0, 1) < 0.5

        indices_normal = tf.tile(tf.constant([[0, 1]]), [batch_n, 1])
        indices_flipped = tf.tile(tf.constant([[1, 0]]), [batch_n, 1])

        flip_mask = tf.cast(flip_mask, tf.int32)[:, None]
        indices = flip_mask * indices_flipped + (1 - flip_mask) * indices_normal

        X = tf.gather(X, indices, batch_dims=1)
        return X

    def train_augment_xy(X, y):
        X = random_frame_flip_x(X)
        return X, y

    def train_augment_xy_info(X, y_main, y_info):
        X = random_frame_flip_x(X)
        return X, y_main, y_info

    def make_dataset(x, y, add_info_data=None, shuffle=False, augment=False):
        if add_info_data is not None:
            ds = tf.data.Dataset.from_tensor_slices(
                (x, y[:, 0].astype(int), y[:, 1:])
            )
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(x),
                reshuffle_each_iteration=True
            )

        ds = ds.batch(batch_size)

        if augment:
            if add_info_data is not None:
                ds = ds.map(train_augment_xy_info, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(train_augment_xy, num_parallel_calls=tf.data.AUTOTUNE)

        return ds.prefetch(tf.data.AUTOTUNE)

    train_dataset = make_dataset(
        train_data[0],
        train_data[1],
        add_info_data=train_data[1] if add_info else None,
        shuffle=True,
        augment=True
    )

    val_dataset = make_dataset(
        val_data[0],
        val_data[1],
        add_info_data=val_data[1] if add_info else None,
        shuffle=False,
        augment=False
    )

    test_dataset = make_dataset(
        test_data[0],
        test_data[1],
        add_info_data=test_data[1] if add_info else None,
        shuffle=False,
        augment=False
    )

    return train_dataset, test_dataset, val_dataset
'''
