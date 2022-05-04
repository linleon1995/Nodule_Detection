import torch
import os
from model import unet3d
from config import models_genesis_config
from data import build_dataset
from utils.train_utils import create_training_path
from config import models_genesis_config

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = models_genesis_config()
config.display()


#Declare the Dice Loss
def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    # y_true_f = torch.flatten(y_true)
    # y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true * y_pred)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth))

def main():
    # prepare your own data

    # # TODO: annotation json and cv
    # train_cases = data_utils.get_pids_from_coco(
    #     [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_train.json') for dataset_name in cfg.DATA.NAMES])
    # # valid_cases = data_utils.get_pids_from_coco(
    #     [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_test.json') for dataset_name in cfg.DATA.NAMES])


    # train_cases = [f'1m{idx:04d}' for idx in range(1, 37)] + [f'1B{idx:04d}' for idx in range(1, 21)]
    # valid_cases = [f'1m{idx:04d}' for idx in range(37, 39)] + [f'1B{idx:04d}' for idx in range(21, 23)]
    # key = '32x64x64-10-shift-8'
    # key = '32x64x64-10'
    # input_roots = [
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
    # ]
    # target_roots = [
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
    #     os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
    # ]


    train_cases = [f'subset{idx}' for idx in range(1)]
    valid_cases = [f'subset{idx}' for idx in range(1, 2)]

    input_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data']
    target_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_mask']
    # target_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_hu_mask']

    # train_cases = [f'luna16-{idx:04d}' for idx in range(852)]
    # valid_cases = [f'luna16-{idx:04d}' for idx in range(852, 963)]
    # train_cases = [f'luna16-{idx:04d}' for idx in range(112)]
    # valid_cases = [f'luna16-{idx:04d}' for idx in range(112, 240)]
    # input_roots = [
    #     rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\positive\Image',
    #     rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\negative\Image',
    #     # rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1000\positive\Image\subset1'
    # ]
    # target_roots = [
    #    rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\positive\Mask',
    #    rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\negative\Mask',
    # #    rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1000\positive\Mask\subset1'
    # ]


    train_loader, valid_loader = build_dataset.build_dataloader_mhd(
        input_roots, target_roots, train_cases, config.annot_path, valid_cases, train_batch_size=config.batch_size, 
        class_balance=config.class_balance, remove_empty_sample=config.remove_empty_sample)


    # Model Genesis provided LUNA16
    # train_loader, valid_loader, train_samples, valid_samples = build_generate_pair(config, seg_model=True)

    # prepare the 3D model
    model = unet3d.UNet3D()
    #Load pre-trained weights
    # from torchsummary import summary
    # summary(model, (1, 64, 64, 32), device='cpu')

    if config.weights is not None:
        checkpoint = torch.load(config.weights)
        state_dict = checkpoint['state_dict']
        # state_dict = checkpoint['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        model.load_state_dict(unParalled_state_dict)

    model.to(device)
    # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    from torchsummary import summary
    summary(model, (1, 64, 64, 32))

    criterion = torch_dice_coef_loss
    # criterion = DiceLoss(normalization='none')
    # optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.8), gamma=0.5)
    intial_epoch = 0
    checkpoint_saving_steps = 10
    checkpoint_path = os.path.join(config.model_path, config.exp_name)
    run_path = create_training_path(checkpoint_path)

    # trainer = Trainer(model,
    #                   criterion=criterion,
    #                   optimizer=optimizer,
    #                   train_dataloader=train_loader,
    #                   valid_dataloader=None,
    #                   logger=logger,
    #                   device=configuration.get_device(),
    #                   n_class=config.nb_class,
    #                   exp_path=exp_path,
    #                   train_epoch=config.nb_epoch,
    #                   batch_size=config.batch_size,
    #                   valid_activation=valid_activation,
    #                   history=checkpoint_path,
    #                   checkpoint_saving_steps=checkpoint_saving_steps)

    # train the model
    print('Start training')
    print(f'Train Number {len(train_loader.dataset)}')
    print(f'Valid Number {len(valid_loader.dataset)}')
    print(60*'-')
    min_loss = 10000
    for epoch in range(intial_epoch, config.nb_epoch):
        scheduler.step(epoch)
        print(f'learning rate {scheduler.get_last_lr()}')
        model.train()
        train_loss = []
        for batch_ndx, (x, y) in enumerate(train_loader):
            x, y = x.float().to(device), y.float().to(device)
            x = torch.split(x, 1, dim=1)
            x = torch.cat(x, dim=0)
            y = torch.split(y, 1, dim=1)
            y = torch.cat(y, dim=0)

            pred = model(x)
            loss = criterion(y, pred)

            # x_np, y_np, pred_np = x.cpu().detach().numpy(), y.cpu().detach().numpy(), pred.cpu().detach().numpy()
            # pred_np = np.where(pred_np>0.5, 1, 0)
            # if batch_ndx%1 == 0:
            #     for n in range(6):
            #         for s in range(0, 32):
            #             if np.sum(y_np[n,0,...,s]) > 0:
            #                 # print(np.sum(y_np[n,0,...,s]))
            #                 plt.imshow(x_np[n,0,...,s], 'gray')
            #                 # plt.imshow(y_np[n,0,...,s]+2*pred_np[n,0,...,s], alpha=0.2, vmin=0, vmax=3)
            #                 plt.imshow(y_np[n,0,...,s], alpha=0.2, vmin=0, vmax=3)
            #                 plt.title(f'n: {n} s: {s}')
            #                 # plt.savefig(f'figures/plot/train-{epoch}-{batch_ndx}-{n}-{s}.png')
            #                 plt.show()

            if batch_ndx%20 == 0:
                print(f'Step {batch_ndx} Loss {loss}')
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = sum(train_loss)/len(train_loss)

        model.eval()
        valid_loss = []
        for batch_ndx, (valid_x, valid_y) in enumerate(valid_loader):
            valid_x, valid_y = valid_x.float().to(device), valid_y.float().to(device)
            valid_x = torch.split(valid_x, 1, dim=1)
            valid_x = torch.cat(valid_x, dim=0)
            valid_y = torch.split(valid_y, 1, dim=1)
            valid_y = torch.cat(valid_y, dim=0)

            valid_pred = model(valid_x)
            loss = criterion(valid_y, valid_pred)
            valid_loss.append(loss.item())
        valid_loss = sum(valid_loss)/len(valid_loss)
        print(20*'-')
        print('\n')
        print(f'Epoch {epoch} Train Loss {train_loss} Valid Loss {valid_loss}')

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        if valid_loss < min_loss:
            print(f'Valid loss improves from {min_loss} to {valid_loss}')
            min_loss = valid_loss
            torch.save(checkpoint, os.path.join(run_path, f'ckpt-best.pt'))

        if epoch%checkpoint_saving_steps == 0:
            torch.save(checkpoint, os.path.join(run_path, f'ckpt-{epoch:03d}.pt'))


if __name__ == '__main__':
    # import numpy as np
    # import matplotlib.pyplot as plt
    # f = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\keras\downstream_tasks\data\ncs\x_train_64x64x32.npy'
    # a = np.load(f)
    # print(np.max(a))
    # for i in range(a.shape[0]):
    #     # for j in range(a.shape[3]):
    #     plt.imshow(a[i, :, :, 0, 0])
    #     plt.savefig(f'plot/stuff/MG_{i}.png')
    # b= a
    main()

        