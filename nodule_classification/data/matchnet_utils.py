import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

from utils import trainer
from utils import metrics
from nodule_classification.model.ResNet_3d import build_3d_resnet


def build_support_set(n, n_class):
    n_benign = n//2
    n_malignant = n - n_benign
    data_root = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\crop\32x64x64-10'
    support_set_df = pd.read_csv(os.path.join(data_root, 'data_samples.csv'))
    # malignancy_df = support_set_df[support_set_df['malignancy']=='benign']
    benign_df = support_set_df.iloc[np.where((support_set_df['malignancy']=='benign'))]
    malignant_df = support_set_df.iloc[np.where((support_set_df['malignancy']=='malignant'))]

    sample_df = pd.concat([
        benign_df.sample(n_benign, random_state=0),
        malignant_df.sample(n_malignant, random_state=0)
    ])
    support_set_x = []
    support_set_y = []
    for idx, row in sample_df.iterrows():
        data = np.load(os.path.join(data_root, row['path']))
        support_set_x.append(data)
        if row['malignancy'] == 'benign':
            support_set_y.append(0)
        elif row['malignancy'] == 'malignant':
            support_set_y.append(1)
    support_set_x = np.stack(support_set_x, axis=0)
    support_set_y = np.array(support_set_y)
    support_set_y_onehot = np.eye(n_class)[support_set_y]
    support_set_y_onehot = np.asarray(support_set_y_onehot, np.float32)
    return support_set_x, support_set_y_onehot






class matchingnet_trainer(trainer.Trainer):
    def __init__(self,
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 n_class,
                 exp_path,
                 support_set_x,
                 support_set_y,
                 train_epoch=10,
                 batch_size=1,
                 lr_scheduler=None,
                 valid_activation=None,
                 USE_TENSORBOARD=True,
                 USE_CUDA=True,
                 history=None,
                 checkpoint_saving_steps=20,
                 patience=10,
                 ):
        super().__init__(model, 
                        criterion, 
                        optimizer, 
                        train_dataloader, 
                        valid_dataloader,
                        logger,
                        device,
                        n_class,
                        exp_path,
                        train_epoch,
                        batch_size,
                        lr_scheduler,
                        valid_activation,
                        USE_TENSORBOARD,
                        USE_CUDA,
                        history,
                        checkpoint_saving_steps,
                        patience)
        self.support_set_x = torch.from_numpy(support_set_x).to(self.device)
        self.support_set_y = torch.from_numpy(support_set_y).to(self.device)
        self.suppoer_set_size = self.support_set_x.shape[0]
    
    def train(self):
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.train_epoch}')
        train_samples = len(self.train_dataloader.dataset)
        total_train_loss = 0.0
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            input_var, target_var = data['input'], data['target']
            input_var, target_var = input_var.float(), target_var.float()
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)

            self.optimizer.zero_grad()
            accuracy, loss, _ = self.predict(input_var, target_var)
            
            # loss = self.criterion(batch_output, target_var)
            loss.backward()
            self.optimizer.step()
        
            loss = loss.item()
            total_train_loss += loss*input_var.shape[0]
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, i)

            if i%self.display_step == 0:
                self.logger.info(f'Step {i}  Step loss {loss} Step Acc {accuracy}')
        self.iterations = i
        train_loss = total_train_loss/train_samples
        self.logger.info(f'Training loss {train_loss}')
        if self.USE_TENSORBOARD:
            # TODO: correct total_train_loss
            self.train_writer.add_scalar('Loss/epoch', train_loss, self.epoch)

    def validate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.n_class, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        valid_samples = len(self.valid_dataloader.dataset)
        total_labels, total_preds = [], []
        for idx, data in enumerate(self.valid_dataloader):
            test_n_iter += 1

            input_var, labels = data['input'], data['target']
            input_var, labels = input_var.float(), labels.float()
            input_var, labels = input_var.to(self.device), labels.to(self.device)
            
            _, loss, pred = self.predict(input_var, labels)
            total_labels.append(labels.squeeze().item())
            total_preds.append(pred.item())

            # loss = loss_func(outputs, torch.argmax(labels, dim=1)).item()
            total_test_loss += loss.item()

        total_labels = np.array(total_labels)
        total_preds = np.array(total_preds)
        self.avg_test_acc = accuracy_score(total_labels, total_preds)
        self.test_loss = total_test_loss/valid_samples
        self.logger.info(f'Testing loss {self.test_loss}')
        self.valid_writer.add_scalar('Accuracy/epoch', self.avg_test_acc, self.epoch)
        self.valid_writer.add_scalar('Loss/epoch', self.test_loss, self.epoch)

    def predict(self, target_image, target_y):
        # rand_indices = torch.randint(self.suppoer_set_size, (self.batch_size,))
        # support_set_images = self.support_set_x[rand_indices]
        # support_set_y_one_hot = self.support_set_y[rand_indices]

        # TODO: ugly
        # b = target_image.shape[0]
        # support_set_images = torch.unsqueeze(self.support_set_x, dim=0)
        # support_set_y_one_hot = torch.unsqueeze(self.support_set_y, dim=0)
        support_set_images = torch.tile(
            torch.unsqueeze(self.support_set_x, dim=1), (1, 3, 1, 1, 1))
        support_set_y_one_hot = torch.unsqueeze(self.support_set_y, dim=1)

        support_set_images = support_set_images.to(torch.float)
        accuracy, crossentropy_loss, prob, pred = self.model(
            support_set_images, support_set_y_one_hot, target_image, target_y)
        return accuracy, crossentropy_loss, pred