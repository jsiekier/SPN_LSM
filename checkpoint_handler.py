import os
import tensorflow as tf

def create_checkpoint_path(fold_path,name):
    model_path=fold_path+name#'models/'+model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+ 'tf_ckpts_last/')
    return  model_path + 'tf_ckpts_last'


def create_checkpoint_paths(fold_path,names):
    checkpoint_paths=[]
    for name in names:
        checkpoint_path = create_checkpoint_path(fold_path,name)
        checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths

def create_checkpoint_manager(models ,ckp_paths):
    ckps=[]
    managers=[]
    for model,ckp_path in zip(models,ckp_paths):
        ckps.append(tf.train.Checkpoint(step=tf.Variable(1), net=model) )#optimizer=cnn_spn.optimizer,
        managers.append(tf.train.CheckpointManager(ckps[-1], ckp_path, max_to_keep=1))
    return ckps,managers

def init_checkpoint_manager(ckps, managers):
    for ckp,manager in zip(ckps,managers):
        ckp.step.assign_add(1)
        manager.save()





'''

    checkpoint_path = create_checkpoint_path(fold_path,'vae_checkpoints/')
    checkpoint_path_tmp=  create_checkpoint_path(fold_path,'vae_checkpoints_tmp/')
    checkpoint_path_cnn_spn=  create_checkpoint_path(fold_path,'cnn_spn_checkpoints/')
    checkpoint_path_spn=  create_checkpoint_path(fold_path,'spn_checkpoints/')
    checkpoint_path_spn_finetune=  create_checkpoint_path(fold_path,'spn_checkpoints_finetune/')

    ckpt_cnn_spn = tf.train.Checkpoint(step=tf.Variable(1), optimizer=cnn_spn.optimizer, net=cnn_spn)
    manager_cnn_spn = tf.train.CheckpointManager(ckpt_cnn_spn, checkpoint_path_cnn_spn, max_to_keep=1)

    ckpt_spn = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.clf_spn)
    manager_spn = tf.train.CheckpointManager(ckpt_spn, checkpoint_path_spn, max_to_keep=1)

    ckpt_spn_finetune = tf.train.Checkpoint(step=tf.Variable(1), net=cnn_spn.clf_spn)
    manager_spn_finetune = tf.train.CheckpointManager(ckpt_spn_finetune, checkpoint_path_spn_finetune, max_to_keep=1)
'''