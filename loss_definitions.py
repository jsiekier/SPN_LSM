import tensorflow as tf

def gaussian_kernel(size: int, sigma: float):
    """Returns a 2D Gaussian kernel."""
    coords = tf.range(size, dtype=tf.float32) - size // 2
    g = tf.exp(-(coords ** 2) / (2. * sigma ** 2))
    g_kernel = tf.tensordot(g, g, axes=0)
    return g_kernel / tf.reduce_sum(g_kernel)

def apply_gaussian_blur(img, size=11, sigma=1.5):
    """Applies Gaussian blur to 4D tensor [B, H, W, C]."""
    kernel = gaussian_kernel(size, sigma)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, img.shape[-1], 1])
    return tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding='SAME')

def ssim_per_scale(x, y, C1=0.01**2, C2=0.03**2):
    """Compute SSIM between x and y for one scale."""
    mu_x = apply_gaussian_blur(x)
    mu_y = apply_gaussian_blur(y)

    sigma_x = apply_gaussian_blur(x * x) - mu_x ** 2
    sigma_y = apply_gaussian_blur(y * y) - mu_y ** 2
    sigma_xy = apply_gaussian_blur(x * y) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return tf.reduce_mean(ssim_map)
def ssim_per_scale_masked(x, y, mask, C1=0.01**2, C2=0.03**2):
    mu_x = apply_gaussian_blur(x)
    mu_y = apply_gaussian_blur(y)

    sigma_x = apply_gaussian_blur(x * x) - mu_x ** 2
    sigma_y = apply_gaussian_blur(y * y) - mu_y ** 2
    sigma_xy = apply_gaussian_blur(x * y) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    ssim_map = tf.clip_by_value(ssim_map, 0.0, 1.0)
    weights = 0.2 + 0.8 * mask
    #weights = tf.cast(weights, ssim_map.dtype)
    #return tf.reduce_sum(ssim_map * mask) / (tf.reduce_sum(mask) + 1e-8)
    return tf.reduce_sum(ssim_map * weights) / (tf.reduce_sum(weights) + 1e-8)

def ms_ssim_custom(x, y, max_val=1.0, num_scales=3):
    """Multi-Scale SSIM using native TensorFlow ops."""
    weights = tf.constant([0.3, 0.3, 0.4], dtype=tf.float32)[:num_scales]
    msssim = []

    for _ in range(num_scales):
        ssim = ssim_per_scale(x, y)
        msssim.append(ssim)
        # Downsample
        x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
        y = tf.nn.avg_pool2d(y, ksize=2, strides=2, padding='SAME')

    msssim = tf.stack(msssim)
    return tf.reduce_prod(msssim ** weights)
def ms_ssim_custom_masked(x, y,mask, max_val=1.0, num_scales=3):
    """Multi-Scale SSIM using native TensorFlow ops."""
    weights = tf.constant([0.3, 0.3, 0.4], dtype=tf.float32)[:num_scales]
    msssim = []

    for _ in range(num_scales):
        ssim = ssim_per_scale_masked(x, y,mask)
        #ssim = tf.clip_by_value(ssim, 1e-6, 1.0)
        msssim.append(ssim)
        # Downsample
        x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
        y = tf.nn.avg_pool2d(y, ksize=2, strides=2, padding='SAME')
        mask = tf.nn.avg_pool2d(mask, ksize=2, strides=2, padding='SAME')

    msssim = tf.stack(msssim)
    return tf.reduce_prod(msssim ** weights)

def ms_ssim_loss3D(y_true, y_pred, mask=None,num_scales=3):
    """Applies MS-SSIM slice-wise to 3D volumes (shape: B, H, W, D, C)."""
    depth = tf.shape(y_true)[3]
    ssim_vals = []

    #for d in range(y_true.shape[3]):
    for d in range(y_true.shape[1]):
        #x = y_true[:, :, :, d, :]
        #y = y_pred[:, :, :, d, :]
        x = y_true[:, d]
        y = y_pred[:, d]
        val = ms_ssim_custom(x, y, num_scales=num_scales)
        ssim_vals.append(val)

    return 1.0 - tf.reduce_mean(tf.stack(ssim_vals))

def ms_ssim_loss3D_mask(y_true, y_pred,cone_mask, num_scales=3):
    """Applies MS-SSIM slice-wise to 3D volumes (shape: B, H, W, D, C)."""
    weights = tf.constant([0.3, 0.3, 0.4], dtype=tf.float32)[:num_scales]
    depth = tf.shape(y_true)[3]
    ssim_vals = []

    #for d in range(y_true.shape[3]):
    for d in range(y_true.shape[1]):
        #x = y_true[:, :, :, d, :]
        #y = y_pred[:, :, :, d, :]
        x = y_true[:, d]
        y = y_pred[:, d]
        mask = cone_mask[:, 0]  # (1,H,W,1), broadcastbar auf (B,H,W,1)
        val = ms_ssim_custom_masked(x, y, mask,num_scales=num_scales)
        ssim_vals.append(val)

    return 1.0 - tf.reduce_mean(tf.stack(ssim_vals))

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))



def ssim_loss3D(y_true, y_pred):#(B,H,W,T,C) - (B,T,H,W,C)
    # Combine batch and time: (batch*time, height, width, channels)
    #y_true=tf.transpose(y_true,(0,3,1,2,4))
    #y_pred = tf.transpose(y_pred, (0, 3, 1, 2, 4))
    y_true_flat = tf.reshape(y_true, [-1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])
    y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0))

#def ms_ssim_loss3D(y_true, y_pred):
#    print("DEBUG decoding shape:", y_pred.shape,y_true.shape)
#    y_true_flat = tf.reshape(y_true, [-1, y_true.shape[1], y_true.shape[2], y_true.shape[4]])
#    y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[1], y_pred.shape[2], y_pred.shape[4]])
#    print("DEBUG decoding shape:", y_pred_flat.shape, y_true_flat.shape)
#    return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true_flat, y_pred_flat, max_val=1.0))

def ms_ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))


def ms_ssim_loss3D_huber(y_true, y_pred,alpha=0.5):
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    ms_ssim=ms_ssim_loss3D(y_true, y_pred)
    return alpha * huber + (1 - alpha) * ms_ssim

def ms_ssim_loss3D_huber_mask(y_true, y_pred,mask,alpha=0.5):
    #huber = tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)
    delta = 0.1
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = (0.5 * tf.square(quadratic) + delta * linear) / delta

    # mask weighting
    weights = 0.2 + 0.8 * mask

    huber = tf.reduce_sum(huber * weights, axis=(1,2,3,4)) / (
        tf.reduce_sum(weights, axis=(1,2,3,4)) + 1e-8
    )  # -> (B,)
    huber = tf.reduce_mean(huber)



    ms_ssim=ms_ssim_loss3D_mask(y_true, y_pred,mask)
    #ms_ssim = tf.clip_by_value(ms_ssim, 1e-6, 1.0)


    return alpha * huber + (1 - alpha) * ms_ssim
def ms_ssim_loss_huber(y_true, y_pred,alpha=0.5):
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    ms_ssim=1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))
    return alpha * huber + (1 - alpha) * ms_ssim



def mae_loss2D_per_sample(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))


def mse_loss2D_per_sample(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2, 3))


def huber_loss2D_per_sample(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + delta * linear
    return tf.reduce_mean(loss, axis=(1, 2, 3))


def ssim_loss2D_per_sample(y_true, y_pred):
    return 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0)


def ms_ssim_loss2D_per_sample(y_true, y_pred):
    h = y_true.shape[1]
    w = y_true.shape[2]

    # tf.image.ssim_multiscale braucht effektiv >= 11x11
    if h is not None and w is not None and (h < 11 or w < 11):
        vals = []
        for b in range(tf.shape(y_true)[0]):
            vals.append(ms_ssim_custom(y_true[b:b+1], y_pred[b:b+1], num_scales=3))
        vals = tf.stack(vals, axis=0)
        return 1.0 - vals

    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)


def ms_ssim_huber_2d_per_sample(y_true, y_pred, alpha=0.5):
    huber = huber_loss2D_per_sample(y_true, y_pred)
    ms_ssim = ms_ssim_loss2D_per_sample(y_true, y_pred)
    return alpha * huber + (1.0 - alpha) * ms_ssim

def mae_loss3D_per_sample(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3, 4))


def mse_loss3D_per_sample(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2, 3, 4))


def huber_loss3D_per_sample(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + delta * linear
    return tf.reduce_mean(loss, axis=(1, 2, 3, 4))


def ssim_loss3D_per_sample(y_true, y_pred):
    # y_true, y_pred: (B, T, H, W, C)
    b = tf.shape(y_true)[0]
    t = tf.shape(y_true)[1]
    h = tf.shape(y_true)[2]
    w = tf.shape(y_true)[3]
    c = tf.shape(y_true)[4]

    y_true_flat = tf.reshape(y_true, (b * t, h, w, c))
    y_pred_flat = tf.reshape(y_pred, (b * t, h, w, c))

    ssim_vals = tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0)   # (B*T,)
    ssim_vals = tf.reshape(ssim_vals, (b, t))                          # (B,T)

    return 1.0 - tf.reduce_mean(ssim_vals, axis=1)                     # (B,)


def _ms_ssim_custom_single(inputs):
    img_true, img_pred = inputs
    # img_true/img_pred: (H, W, C)
    img_true = tf.expand_dims(img_true, axis=0)   # (1,H,W,C)
    img_pred = tf.expand_dims(img_pred, axis=0)
    return ms_ssim_custom(img_true, img_pred, num_scales=3)            # scalar


def ms_ssim_loss3D_per_sample(y_true, y_pred,mask=None):
    # y_true, y_pred: (B, T, H, W, C)
    b = tf.shape(y_true)[0]
    t = tf.shape(y_true)[1]
    h = tf.shape(y_true)[2]
    w = tf.shape(y_true)[3]
    c = tf.shape(y_true)[4]

    y_true_flat = tf.reshape(y_true, (b * t, h, w, c))   # (B*T,H,W,C)
    y_pred_flat = tf.reshape(y_pred, (b * t, h, w, c))

    ms_vals = tf.map_fn(
        _ms_ssim_custom_single,
        (y_true_flat, y_pred_flat),
        fn_output_signature=tf.float32
    )                                                     # (B*T,)

    ms_vals = tf.reshape(ms_vals, (b, t))                 # (B,T)
    return 1.0 - tf.reduce_mean(ms_vals, axis=1)         # (B,)


def ms_ssim_huber_3d_per_sample(y_true, y_pred,mask, alpha=0.5):
    huber = huber_loss3D_per_sample(y_true, y_pred)
    ms_ssim = ms_ssim_loss3D_per_sample(y_true, y_pred)
    return alpha * huber + (1.0 - alpha) * ms_ssim

def rec_loss_creation_per_sample(rec_loss_name, conv_dim):
    if conv_dim == 3:
        if rec_loss_name == 'MSE':
            return mse_loss3D_per_sample
        elif rec_loss_name == 'MAE':
            return mae_loss3D_per_sample
        elif rec_loss_name == 'HUBER':
            return huber_loss3D_per_sample
        elif rec_loss_name == 'SSIM':
            return ssim_loss3D_per_sample
        elif rec_loss_name == 'MS_SSIM':
            return ms_ssim_loss3D_per_sample
        elif rec_loss_name == 'MS_SSIM_HUBER':
            print( ' loss   MS_SSIM_HUBER')
            return ms_ssim_huber_3d_per_sample
    else:
        if rec_loss_name == 'MSE':
            return lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2, 3))
        elif rec_loss_name == 'MAE':
            return lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))
        elif rec_loss_name == 'HUBER':
            def huber_2d(y_true, y_pred, delta=1.0):
                error = y_true - y_pred
                abs_error = tf.abs(error)
                quadratic = tf.minimum(abs_error, delta)
                linear = abs_error - quadratic
                loss = 0.5 * tf.square(quadratic) + delta * linear
                return tf.reduce_mean(loss, axis=(1, 2, 3))
            return huber_2d
        elif rec_loss_name == 'SSIM':
            return lambda y_true, y_pred: 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0)
        elif rec_loss_name == 'MS_SSIM':
            return lambda y_true, y_pred: 1.0 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
        elif rec_loss_name == 'MS_SSIM_HUBER':
            def ms_ssim_huber_2d(y_true, y_pred, alpha=0.5):
                error = y_true - y_pred
                abs_error = tf.abs(error)
                quadratic = tf.minimum(abs_error, 1.0)
                linear = abs_error - quadratic
                huber = tf.reduce_mean(0.5 * tf.square(quadratic) + linear, axis=(1, 2, 3))
                ms_ssim = 1.0 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
                return alpha * huber + (1.0 - alpha) * ms_ssim
            return ms_ssim_huber_2d

    raise ValueError(f"Unknown rec loss: {rec_loss_name}")
def mse_per_sample(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
def rec_loss_creation(rec_loss_name,conv_dim):
    if rec_loss_name=='MSE':
        return tf.keras.losses.MeanSquaredError()
    elif rec_loss_name=='MAE':
        return tf.keras.losses.MeanAbsoluteError()
    elif rec_loss_name=='SSIM':
        if conv_dim==3:
            return ssim_loss3D
        else:
            return ssim_loss
    elif rec_loss_name=='MS_SSIM':
        if conv_dim==3:
            return ms_ssim_loss3D
        else:
            return  ms_ssim_loss
    elif rec_loss_name== 'MS_SSIM_HUBER':
        if conv_dim==3:
            return ms_ssim_loss3D_huber
        else:
            return ms_ssim_loss_huber
    elif rec_loss_name == 'HUBER':
        return tf.keras.losses.Huber(delta=1.0)