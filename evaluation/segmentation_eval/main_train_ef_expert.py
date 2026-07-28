import video
import torch


if __name__ == '__main__':
    machine=1
    pairwise=1
    device_str='cuda'#'cpu'#
    data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][machine]
    output = ['./tmp/','/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_seg'][machine]#'/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw'
    path=['C:/Users/siekiera/Documents/git_repos/GradientSPN/',
          '/lustre/project/ki-topml/jsiekiera/cf_data/'][machine]
    weights_ef=None



    # train model on pairwise EF data
    dataset='stanford_seg'
    #video.run(path=path,data_dir=data_dir,output=output,device = torch.device(device_str),pairwise=pairwise,dataset_name=dataset)
    # fine-tune on camus
    #dataset = 'CAMUS'
    '''
    data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][machine]
    weights_ef=['C:/Users/siekiera/Documents/git_repos/CF_eval/ef_pred_weights_echos/best.pt',
               '/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw/best.pt'][machine]
    path=['C:/Users/siekiera/Documents/curAIscid_data/camus_echonet_style/',
          '/lustre/project/ki-topml/jsiekiera/cf_data/'][machine]
    output = ['./tmp/camu','/lustre/project/ki-topml/jsiekiera/ef_pred_weights_echos_pw_camu_mo'][machine]
    '''
    #
    video.run(path=path,
              data_dir=data_dir,
              output=output,
              device=torch.device(device_str), pairwise=pairwise,
              dataset_name=dataset,
              weights=weights_ef,
              lr=1e-4,
              staged_finetuning=False,#True,
              freeze_epochs=0,#5,
              warmup_trainable_layers=("fc", "layer4"),
              finetune_lr_factor=1)#0.1)   # danach also 1e-5)