import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import time
import pickle as pkl
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tensorflow as tf

from counterfactuals.analyse_model import load_model, load_VAE_model
from end_to_end_train import load_dataset

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------
# Globals for worker process
# -------------------------
_WORKER_MODEL = None
_WORKER_CONFIG = None


def configure_tf_for_cpu_worker():
    # Critical to avoid oversubscription
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def init_worker(model_name, add_info, num_classes, input_shape, data_path_fold, params, model_path, fine_tune,gamma):
    """
    Loads the model once per worker process.
    """
    global _WORKER_MODEL, _WORKER_CONFIG

    configure_tf_for_cpu_worker()

    if model_name == 'MLP':
        _WORKER_MODEL = load_VAE_model(
            add_info,
            num_classes,
            input_shape,
            data_path_fold,
            params,
            path=model_path,
            fine_tune=fine_tune,
            gamma=gamma
        )
    else:
        _WORKER_MODEL = load_model(
            add_info,
            num_classes,
            input_shape,
            data_path_fold,
            params,
            path=model_path,
            fine_tune=fine_tune,
        )

    _WORKER_CONFIG = {
        "model_name": model_name,
        "num_classes": num_classes,
    }

    print(f'worker initialized pid={os.getpid()}', flush=True)


def get_num_workers(requested=None):
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    available = int(slurm_cpus) if slurm_cpus is not None else os.cpu_count()

    if requested is None or requested <= 0:
        return max(1, available)

    return max(1, min(requested, available))


def sample_outside_eps(y, min_bound=0.0, max_bound=1.0, eps=0.1,seed=0):
    if seed:
        np.random.seed(seed)
        print(min_bound,max_bound,eps,seed)
    y = np.asarray(y)

    a = np.maximum(min_bound, y - eps)
    b = np.minimum(max_bound, y + eps)

    left_len = a - min_bound
    right_len = max_bound - b
    total_len = left_len + right_len

    if np.any(total_len <= 0):
        raise ValueError("No valid interval to sample from. Epsilon too large.")

    p_left = left_len / total_len
    rand = np.random.rand(*y.shape)
    choose_left = rand < p_left

    y_cf = np.empty_like(y, dtype=float)

    y_cf[choose_left] = np.random.uniform(min_bound, a[choose_left])
    y_cf[~choose_left] = np.random.uniform(b[~choose_left], max_bound)

    return y_cf


def get_counterfactual_infos(
    cnn_spn_model,
    X,
    additional_info,
    opposite_class,
    model_type,
    opt_weights=[10, 0.005],
    learning_rate=0.05,
    num_steps=150,
    num_classes=2,
        spn_z=None
):
    [z_embed, _, _] = cnn_spn_model.embedding(X, training=False)

    y_opposite = tf.Variable(
        opposite_class,#np.asarray([opposite_class] * z_embed.shape[0]),
        trainable=True,
        dtype=tf.dtypes.float32
    )
    '''
    y_opposite_log = tf.Variable(
        np.log(opposite_class),#np.asarray([np.log(opposite_class)] * z_embed.shape[0]),
        trainable=True,
        dtype=tf.dtypes.float32
    )
    '''

    z = tf.Variable(z_embed, trainable=False, dtype=tf.dtypes.float32)
    z_cf = tf.Variable(z_embed, trainable=True, dtype=tf.dtypes.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0)

    def get_lr(step):
        if step < 10:
            return 0.2
        elif step < 30:
            return 0.1
        else:
            return 0.05

    if num_classes > 1:
        myloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        myloss = tf.keras.losses.MeanAbsoluteError()

    if cnn_spn_model.use_add_info:
        add_info = tf.Variable(additional_info, trainable=False, dtype=tf.dtypes.float32)
        embedding_org = tf.concat([z, add_info], axis=-1)
    else:
        add_info = None
        embedding_org = z

    pred_org, p_z_org = cnn_spn_model.spn_clf(embedding_org, training=False)
    if model_type == 'SPN':
        p_z_org = cnn_spn_model.spn_model_z(embedding_org, training=False)
        #print(p_z_org.shape)
    elif model_type=='MLP' and opt_weights[1]:
        p_z_org = spn_z(embedding_org, training=False)
    #print("y_opposite.shape,y_opposite_log.shape", y_opposite.shape, y_opposite_log.shape, z.shape)
    @tf.function
    def counterfactual_loss():
        if cnn_spn_model.use_add_info:
            embedding = tf.concat([z_cf, add_info], axis=-1)
        else:
            embedding = z_cf

        pred, p_z = cnn_spn_model.spn_clf(embedding, training=False)

        distance = tf.reduce_mean(tf.losses.MAE(z, z_cf))
        distance_ind = tf.reduce_mean(tf.abs(z - z_cf), axis=-1)
        plausability_diff = distance
        plausability_diff_ind=distance_ind

        if model_type == 'MLP':
            #pred_flat = tf.reshape(pred, [tf.shape(z_cf)[0], -1])
            #y_flat = tf.reshape(y_opposite, [tf.shape(z_cf)[0], -1])
            if num_classes > 1:
                test = myloss(y_opposite, pred)
            else:
                #test = tf.reduce_mean(tf.abs(pred_flat - y_flat), axis=-1)
                test_ind = tf.abs(pred - y_opposite)

                test = myloss(y_opposite, pred)
                y_diff_sum = tf.reduce_sum(tf.abs(pred - y_opposite), axis=-1)
            if opt_weights[1]:
                #spn_z
                p_z = spn_z(embedding, training=False)
                plausability_diff = tf.reduce_sum(tf.abs(p_z - p_z_org), axis=-1)
                delta = 0
                #plausability_diff = tf.reduce_sum(tf.nn.relu(p_z_org - p_z - delta), axis=-1)

                plausability_diff_ind = plausability_diff#tf.abs(p_z - p_z_org)
                plausability_diff=tf.reduce_mean(plausability_diff)
                loss = test + (distance * opt_weights[0])+ (plausability_diff * opt_weights[1])
                loss_ind = test_ind + (distance_ind * opt_weights[0]) + (plausability_diff_ind * opt_weights[1])
            else:
                loss = test + (distance * opt_weights[0])
                loss_ind = test_ind + (distance_ind * opt_weights[0])
        elif model_type == 'SPN':
            p_z = cnn_spn_model.spn_model_z(embedding, training=False)
            #print(p_z.shape,p_z[:2])
            plausability_diff = tf.reduce_sum(tf.abs(p_z - p_z_org), axis=-1)# OLD
            plausability_diff_ind = plausability_diff
            plausability_diff = tf.reduce_mean(plausability_diff)
            delta=0
            #plausability_diff =tf.reduce_sum(tf.nn.relu(p_z_org - p_z - delta), axis=-1)
            #lambda_down = 1.0
            #lambda_abs = 0.1
            #plaus_loss_down = tf.reduce_sum(tf.nn.relu(p_z_org - p_z - delta), axis=-1)
            #plaus_loss_abs =  tf.reduce_sum(tf.abs(p_z - p_z_org), axis=-1)

            #plausability_diff = lambda_down * plaus_loss_down + lambda_abs * plaus_loss_abs




            #plausability_diff_ind = plausability_diff# tf.abs(p_z - p_z_org)
            #plausability_diff = tf.reduce_mean(plausability_diff)
            if num_classes > 1:
                test = -pred[:, opposite_class]
            else:
                #pred_flat = tf.reshape(pred, [tf.shape(z_cf)[0], -1])
                #ylog_flat = tf.reshape(y_opposite_log, [tf.shape(z_cf)[0], -1])

                # per replicate
                #test_ind = tf.abs(pred - y_opposite_log)
                #test = tf.reduce_sum(tf.abs(pred - y_opposite_log), axis=-1)

                test_ind = tf.abs(tf.exp(pred) - y_opposite)
                #test = tf.reduce_mean(tf.abs(tf.exp(pred) - y_opposite), axis=-1)
                test = myloss(y_opposite, tf.exp(pred))
                y_diff_sum=tf.reduce_sum(tf.abs(tf.exp(pred) - y_opposite), axis=-1)


            loss = test + (distance * opt_weights[0]) + (plausability_diff * opt_weights[1])
            loss_ind = test_ind + (distance_ind * opt_weights[0]) + (plausability_diff_ind * opt_weights[1])
        plaus_loss_abs, plaus_loss_down=0,0
        return loss, test, distance, pred, plausability_diff,loss_ind,plaus_loss_abs,plaus_loss_down,y_diff_sum,test_ind,plausability_diff_ind,distance_ind

    for s in range(num_steps):
        #lr_var.assign(get_lr(s))
        optimizer.learning_rate.assign(get_lr(s))
        with tf.GradientTape() as tape:
            loss, log_pred, distance, pred, p_z_diff,loss_ind,plaus_abs,plaus_down ,y_diff_sum,test_ind_,plausability_diff_ind_,distance_ind_= counterfactual_loss()
            #print(s,np.mean(loss.numpy()),np.mean(y_diff_sum.numpy()),np.mean(log_pred.numpy()),np.mean(distance.numpy()),np.mean(p_z_diff.numpy()),np.mean(plaus_abs.numpy()),np.mean(plaus_down.numpy()))
            #print(loss_ind.numpy()[2],test_ind_.numpy()[2],distance_ind_.numpy()[2],plausability_diff_ind_.numpy()[2])
            #print(np.mean(loss_ind.numpy()),np.mean(test_ind_.numpy()),np.mean(distance_ind_.numpy()),np.mean(plausability_diff_ind_.numpy()))
            loss_opt = tf.reduce_mean(loss_ind)

        grads = tape.gradient(loss_opt, [z_cf])
        #grad_norm = tf.linalg.global_norm(grads)
        #current_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))

        #update_norm = tf.linalg.global_norm([
        #    tf.cast(current_lr, grads[0].dtype) * grads[0]
        #])

        #print("lr", current_lr, "grad_norm", grad_norm.numpy(), "update_norm", update_norm.numpy())

        #print(grad_norm.numpy(),update_norm.numpy())
        optimizer.apply_gradients(zip(grads, [z_cf]))
        #real_pred=np.squeeze(pred.numpy()[:5]) if model_type == 'MLP' else np.squeeze(np.exp(pred.numpy()[:5]))
        #gt=np.squeeze(y_opposite.numpy())[:5]
        #print('DEBUG CF LOSS',np.mean(loss.numpy()),real_pred,gt,np.abs(real_pred-gt),distance.numpy()[:5],p_z_diff.numpy()[:5],log_pred.numpy()[:5],loss.numpy().shape)

    rec_cf = cnn_spn_model.reconstruct(z_cf)
    rec_gt = cnn_spn_model.reconstruct(z)
    p_z_diff = np.mean(p_z_diff.numpy())
    log_pred = log_pred.numpy()
    loss_ind= loss_ind.numpy()

    if num_classes > 1:
        arg_max = np.argmax(pred.numpy(), axis=1)
    else:
        arg_max = pred.numpy()

    return (
        rec_cf.numpy(),
        rec_gt.numpy(),
        z_cf.numpy(),
        z.numpy(),
        arg_max,
        loss_ind,
        log_pred,
        p_z_diff,
        pred.numpy(),
    )


def build_batch_payload(input_data, params, i, end):
    X = np.asarray(input_data[0][i:end])

    if params.add_info:
        # Keep your original slicing logic unchanged
        y = np.asarray(input_data[1][i:end][0])
        additional_info = np.asarray([input_data[1][i:end][1:]])
    else:
        y = np.asarray(input_data[1][i:end])
        additional_info = None

    return X, y, additional_info


def compute_batch_metadata_serial(
    input_data,
    model_name,
    add_info,
    num_classes,
    input_shape,
    data_path_fold,
    params,
    model_path,
    y_min_max,
    fine_tune,
    batch_size,
    args=None
):
    """
    Compute pred/mean_cls/opposite_class once per batch in the main process,
    so all replicate chunks of the same batch share the exact same opposite_class
    as in the original code.
    """
    start_time = time.time()

    if model_name == 'MLP':
        metadata_model = load_VAE_model(
            add_info,
            num_classes,
            input_shape,
            data_path_fold,
            params,
            path=model_path,
            fine_tune=fine_tune,
        )
    else:
        metadata_model = load_model(
            add_info,
            num_classes,
            input_shape,
            data_path_fold,
            params,
            path=model_path,
            fine_tune=fine_tune,
        )

    print('metadata model loaded', (time.time() - start_time) / 60, flush=True)

    total_samples = input_data[0].shape[0]
    batch_infos = []

    for batch_idx, i in enumerate(range(0, total_samples, batch_size)):
        end = min(total_samples, i + batch_size)
        X, y, additional_info = build_batch_payload(input_data, params, i, end)

        print('start first pred batch', batch_idx, flush=True)
        time_pred = time.time()

        pred = metadata_model.model_execution_X(X, additional_info, train_vae=False, train_clf=False)

        if num_classes > 1:
            print('not implemented', flush=True)
            raise NotImplementedError("num_classes > 1 is not implemented in this version.")
        else:
            if model_name == 'MLP':
                pred = tf.keras.activations.sigmoid(pred).numpy()
                pred=np.squeeze(pred)
            else:
                pred = np.exp(pred.numpy())

            mean_cls = pred
            if args.show_case:
                opposite_class = sample_outside_eps(
                    y,
                    min_bound=y_min_max[0],
                    max_bound=y_min_max[1],
                    eps=0.05,
                    seed=1
                )
                print('org y:',y)
                print('DEBUGGING: CF targets',opposite_class)
                print('show case')
            else:
                opposite_class = sample_outside_eps(
                    mean_cls,
                    min_bound=y_min_max[0],
                    max_bound=y_min_max[1],
                    eps=0.05,
                )

        print(
            'end first pred',
            (time.time() - time_pred) / 60,
            'batch',
            batch_idx,
            'y,pred,opposite_class',
            y[:10],
            mean_cls[:10],
            opposite_class[:10],
            flush=True
        )

        batch_infos.append({
            "batch_idx": batch_idx,
            "i": i,
            "end": end,
            "X": X,
            "y": y,
            "additional_info": additional_info,
            "pred": pred,
            "mean_cls": mean_cls,
            "opposite_class": opposite_class,
        })

    return batch_infos


def process_replicate_chunk_worker(task):
    """
    One task = one batch + one chunk of replicates.
    Model is cached per worker via init_worker().
    """
    global _WORKER_MODEL, _WORKER_CONFIG

    (
        batch_idx,
        i,
        end,
        X,
        additional_info,
        pred,
        mean_cls,
        opposite_class,
        rep_start,
        rep_end,
        opt_weights,
        learning_rate,
        num_steps,
    ) = task

    task_start = time.time()

    if isinstance(_WORKER_MODEL, tuple):
        if len(_WORKER_MODEL) == 2:
            cnn_spn_model, spn_z = _WORKER_MODEL
        else:
            [cnn_spn_model] = _WORKER_MODEL
            spn_z=None
    else:
        cnn_spn_model = _WORKER_MODEL
        spn_z=None


    model_name = _WORKER_CONFIG["model_name"]
    num_classes = _WORKER_CONFIG["num_classes"]

    print(
        f'worker pid={os.getpid()} start batch={batch_idx} reps=[{rep_start},{rep_end})',
        flush=True
    )

    rep_results = []
    for rep_idx in range(rep_start, rep_end):
        time_cf = time.time()

        rec_cf, rec_gt, z_cf, z, arg_max, loss, log_pred, p_z_diff, pred_cf = get_counterfactual_infos(
            cnn_spn_model,
            X,
            additional_info,
            opposite_class,
            model_name,
            opt_weights,
            learning_rate=learning_rate,
            num_steps=num_steps,
            num_classes=num_classes,
            spn_z=spn_z
        )

        rep_results.append((
            rep_idx,
            (rec_cf, rec_gt, z_cf, z, arg_max, loss, log_pred, p_z_diff),
        ))

        print(
            'end cf',
            (time.time() - time_cf) / 60,
            'batch',
            batch_idx,
            'rep',
            rep_idx,
            flush=True
        )

    print(
        f'worker pid={os.getpid()} done batch={batch_idx} reps=[{rep_start},{rep_end}) '
        f'minutes={(time.time() - task_start)/60:.3f}',
        flush=True
    )

    return batch_idx, {
        "pred": pred,
        "opposite_class": opposite_class,
        "indices": [i, end],
        "mean_cls": mean_cls,
        "rep_results": rep_results,
    }


def process_image_parallel(
    input_data,
    model_name,
    opt_weights,
    learning_rate,
    num_steps,
    replicates,
    add_info,
    num_classes,
    input_shape,
    data_path_fold,
    params,
    model_path,
    bin_widths_scaled,
    y_min_max=[0, 1],
    fine_tune=0,
    batch_size=300,
    num_workers=None,
    replicate_chunk_size=5,
    args=None
):
    """
    Parallelization strategy:
    - keep batches large
    - split replicates into chunks
    - cache model once per worker
    - preserve original output format exactly
    """
    start_time = time.time()

    num_workers = get_num_workers(num_workers)

    # 1) Compute per-batch metadata once, serially, preserving old semantics
    batch_infos = compute_batch_metadata_serial(
        input_data=input_data,
        model_name=model_name,
        add_info=add_info,
        num_classes=num_classes,
        input_shape=input_shape,
        data_path_fold=data_path_fold,
        params=params,
        model_path=model_path,
        y_min_max=y_min_max,
        fine_tune=fine_tune,
        batch_size=batch_size,
        args=args
    )

    total_chunks = 0
    tasks = []

    for batch_info in batch_infos:
        batch_idx = batch_info["batch_idx"]

        for rep_start in range(0, replicates, replicate_chunk_size):
            rep_end = min(replicates, rep_start + replicate_chunk_size)
            total_chunks += 1

            tasks.append((
                batch_idx,
                batch_info["i"],
                batch_info["end"],
                batch_info["X"],
                batch_info["additional_info"],
                batch_info["pred"],
                batch_info["mean_cls"],
                batch_info["opposite_class"],
                rep_start,
                rep_end,
                opt_weights,
                learning_rate,
                num_steps,
            ))

    print(
        f'parallel start: batches={len(batch_infos)}, replicates={replicates}, '
        f'chunk_size={replicate_chunk_size}, total_tasks={total_chunks}, '
        f'workers={num_workers}, batch_size={batch_size}',
        flush=True
    )

    # preserve original result structure:
    # (pred, opposite_class, [i, end], mean_cls, rep_results)
    assembled = {
        batch_info["batch_idx"]: {
            "pred": batch_info["pred"],
            "opposite_class": batch_info["opposite_class"],
            "indices": [batch_info["i"], batch_info["end"]],
            "mean_cls": batch_info["mean_cls"],
            "rep_results": [None] * replicates,
        }
        for batch_info in batch_infos
    }

    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_ctx,
        initializer=init_worker,
        initargs=(
            model_name,
            add_info,
            num_classes,
            input_shape,
            data_path_fold,
            params,
            model_path,
            fine_tune,
            opt_weights[1]#gamma
        ),
    ) as executor:
        futures = [executor.submit(process_replicate_chunk_worker, task) for task in tasks]

        for future in as_completed(futures):
            batch_idx, payload = future.result()

            for rep_idx, rep_result in payload["rep_results"]:
                assembled[batch_idx]["rep_results"][rep_idx] = rep_result

    result = []
    for batch_idx in range(len(batch_infos)):
        entry = assembled[batch_idx]
        result.append((
            entry["pred"],
            entry["opposite_class"],
            entry["indices"],
            entry["mean_cls"],
            entry["rep_results"],
        ))

    print('Time of all cf in min ', (time.time() - start_time) / 60, flush=True)
    return result


def multiple_vanilla_counterfactuals(
    dataset_name,
    params,
    num_train_eval_runs,
    data_path_grid,
    add_info,
    path='../',
    model_path='../',
    learning_rate=0.01,
    num_steps=150,
    fold_idx=0,
    replicates=50,
    args=None,
    batch_size=300,
    num_workers=None,
    replicate_chunk_size=5,
):
    beta = args.beta
    gamma = args.gamma
    model_name = args.model_name
    fine_tune = args.fine_tune
    if model_name=='MLP' and gamma==0 and fine_tune==True:
        print('Solo MLP finetune not implemented')
        return

    train, test, num_classes = load_dataset(
        dataset_name,
        binary=False,
        load_net=params.load_pretrain_model,
        machine=params.machine,
        path=path,
        grid_params=params
    )

    data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'
    ([train_data, val_data]) = train

    print(test[0].shape)

    # Full test set
    #
    if args.show_case:
        input_data = (test[0][:30], test[1][:30])
    else:
        input_data = (test[0], test[1])
    input_shape = input_data[0].shape[1:]

    plot_path = data_path_fold + 'counterfactual_imgs/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    if num_classes > 1:
        bin_widths_scaled = None
    else:
        spn_data = pkl.load(open(data_path_fold + 'spn.pkl', 'rb'))
        bin_widths_scaled = spn_data['bin_widths_scaled']

    if fine_tune:
        if args.show_case:
            save_file_path = (
                plot_path
                + 'datab' + str(beta).replace('.', '-')
                + 'g' + str(gamma).replace('.', '-')
                + model_name + '_1000_new_minmax_tmp_fine_show.pkl'
            )
        else:
            save_file_path = (
                plot_path
                + 'datab' + str(beta).replace('.', '-')
                + 'g' + str(gamma).replace('.', '-')
                + model_name + '_1000_new_minmax_tmp_fine.pkl'
            )
    else:
        save_file_path = (
            plot_path
            + 'datab' + str(beta).replace('.', '-')
            + 'g' + str(gamma).replace('.', '-')
            + model_name + '_1000_new_minmax_tmp.pkl'
        )

    print('beta', beta, 'gamma', gamma, 'model', model_name, flush=True)

    #if os.path.exists(save_file_path):
    #    print(save_file_path, 'already exists')
    #    return

    start_time = time.time()
    print(
        'start_processing images',
        'replicates=', replicates,
        'batch_size=', batch_size,
        'workers=', get_num_workers(num_workers),
        'replicate_chunk_size=', replicate_chunk_size,
        flush=True
    )

    y_min_max = [np.min(train_data[1])+0.05 , np.max(train_data[1])-0.05]

    result = process_image_parallel(
        input_data=input_data,
        model_name=model_name,
        opt_weights=[beta, gamma],
        learning_rate=learning_rate,
        num_steps=num_steps,
        replicates=replicates,
        add_info=add_info,
        num_classes=num_classes,
        input_shape=input_shape,
        data_path_fold=data_path_fold,
        params=params,
        model_path=model_path,
        bin_widths_scaled=bin_widths_scaled,
        y_min_max=y_min_max,
        fine_tune=fine_tune,
        batch_size=batch_size,
        num_workers=num_workers,
        replicate_chunk_size=replicate_chunk_size,
        args=args
    )

    print('end_processing images', replicates, flush=True)

    with open(save_file_path, 'wb') as f:
        pkl.dump(result, f)
        print("wrote", f)
    #print('TODO I DID NOT save anything')

    print('total runtime in min', (time.time() - start_time) / 60, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a file path.")

    parser.add_argument("--filepath", type=str, default='stan_KLD4_MS_SSIM_HUBER_64_relu_DS_debug')
    parser.add_argument("--model_name", type=str, default='MLP')
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--fine_tune", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=300)#300
    parser.add_argument("--num_workers", type=int, default=14)#24
    parser.add_argument("--replicate_chunk_size", type=int, default=1)#5
    parser.add_argument("--dataset_name", type=str, default="stanford", help="Path to the dataset")
    parser.add_argument("--show_case", type=int, default=0)

    work_from = -1
    fold_idx = 0
    num_train_eval_runs = 3
    grid_idx = 0

    input_data_path = ['../','','/lustre/project/ki-topml/jsiekiera/cf_data/'][work_from]

    args = parser.parse_args()
    print(args)

    save_folder = [
        '../results/',
        '/lustre/miifs01/project/m2_datamining/sync_data/cnn_spn_models/',
        '/lustre/project/ki-topml/jsiekiera/cnn_spn_models/'
    ][work_from]

    model_name = args.filepath

    data_path_grid = save_folder + model_name + '/grid' + str(grid_idx) + '/'
    params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))
    add_info = params.add_info
    dataset_name = args.dataset_name

    multiple_vanilla_counterfactuals(
        dataset_name,
        params,
        num_train_eval_runs,
        data_path_grid,
        add_info,
        path=input_data_path,
        learning_rate=0.2,#0.05
        num_steps=61 ,
        fold_idx=fold_idx,
        args=args,
        replicates=10,#20
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replicate_chunk_size=args.replicate_chunk_size,
    )