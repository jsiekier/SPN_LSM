from cf_analysis import load_segmentation_model
import torch

if __name__ == '__main__':
    machine=1
    data_dir = ['C:/Users/siekiera/Documents/SPNs/data/echos/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic',
                '/lustre/project/ki-topml/jsiekiera/EchoNet-Dynamic'][machine]
    output = ['./tmp/','/lustre/project/ki-topml/jsiekiera/segmentation_weights_echos'][machine]
    '''
    load_segmentation_model(device=torch.device("cpu"),
                            model_name="deeplabv3_resnet50",
                            pretrained=False,
                            weights=None,
                            train=True,
                            run_test=True,
                            num_epochs=50,
                            lr=1e-5,
                            weight_decay=1e-5,
                            lr_step_period=None,
                            num_train_patients=None,
                            num_workers=4,
                            batch_size=20,
                            seed=0, data_dir=data_dir, output=output)
    '''

    output = ['./tmp/',
              '/lustre/project/ki-topml/jsiekiera/segmentation_weights_camus'][machine]
    path=['C:/Users/siekiera/Documents/curAIscid_data/camus_echonet_style/',
          '/lustre/project/ki-topml/jsiekiera/cf_data/'][machine]
    weights_ef=['C:/Users/siekiera/Documents/git_repos/CF_eval/weights/deeplabv3_resnet50_random.pt',
                'deeplabv3_resnet50_random.pt'][machine]
    device=["cpu",'cuda'][machine]
    load_segmentation_model(device=torch.device(device),
                            model_name="deeplabv3_resnet50",
                            pretrained=True,
                            weights=weights_ef,
                            train=True,
                            run_test=True,
                            num_epochs=50,
                            lr=1e-5,
                            weight_decay=1e-5,
                            lr_step_period=None,
                            num_train_patients=None,
                            num_workers=4,
                            batch_size=20,
                            seed=0,
                            data_dir=data_dir,
                            output=output,
                            staged_finetuning=True,
                            warmup_trainable_layers=("classifier", "backbone.layer4"),
                            finetune_lr_factor=0.1,
                            path='',#path,
                            )