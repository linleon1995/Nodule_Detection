import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
import sys


class Trainer(object):
    def __init__(self,
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 n_class,
                 train_epoch=10,
                 batch_size=1,
                 activation_func=None,
                 USE_TENSORBOARD=True,
                 USE_CUDA=True,
                 history=None,
                 ):
        self.n_class = n_class
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        # TODO: remove all config
        # TODO: better way to warp or combine parameters
        # self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.history = history
        self.model = model
        if history is not None:
            self.load_model_from_checkpoint(self.history)
        self.activation_func = activation_func
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.checkpoint_path = self.config.CHECKPOINT_PATH
        if self.USE_TENSORBOARD:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'train'))
            self.valid_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'valid'))
        if USE_CUDA:
            self.model.cuda()
        self.max_acc = 0

    def fit(self):
        for self.epoch in range(1, self.config.TRAIN.EPOCH + 1):
            self.train()
            with torch.no_grad():
                self.validate()
                self.save_model()

        if self.USE_TENSORBOARD:
            self.train_writer.close()
            self.valid_writer.close()
            
    def train(self):
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.train_epoch}')
        train_samples = len(self.train_dataloader.dataset)
        total_train_loss = 0.0
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            input_var, target_var = data['input'], data['target']
            target_var = target_var.long()
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)

            batch_output = self.model(input_var)
            
            self.optimizer.zero_grad()
            loss = self.criterion(batch_output, target_var)
            loss.backward()
            self.optimizer.step()
        
            loss = loss.item()
            total_train_loss += loss
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, i)

            display_step = train_utils.calculate_display_step(num_sample=train_samples, batch_size=self.batch_size)
            # TODO: display_step = 10
            display_step = 20
            if i%display_step == 0:
                self.logger.info('Step {}  Step loss {}'.format(i, loss))
        self.iterations = i
        if self.USE_TENSORBOARD:
            # TODO: correct total_train_loss
            self.train_writer.add_scalar('Loss/epoch', self.batch_size*total_train_loss/train_samples, self.epoch)

    def validate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.n_class, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        valid_samples = len(self.valid_dataloader.dataset)
        for idx, data in enumerate(self.valid_dataloader):
            test_n_iter += 1

            input_var, labels = data['input'], data['target']
            labels = labels.long()

            input_var, labels = input_var.to(self.device), labels.to(self.device)
            outputs = self.model(input_var)

            loss = self.criterion(outputs, labels)

            # loss = loss_func(outputs, torch.argmax(labels, dim=1)).item()
            total_test_loss += loss.item()

            # TODO: torch.nn.functional.sigmoid(outputs)
            # prob = torch.nn.functional.softmax(outputs, dim=1)
            # prob = torch.sigmoid(outputs)
            if self.activation_func:
                if self.config.MODEL.ACTIVATION == 'softmax':
                    prob = self.activation_func(outputs, dim=1)
                else:
                    prob = self.activation_func(outputs)
            else:
                prob = outputs
            prediction = torch.argmax(prob, dim=1)
            labels = labels[:,0]

            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            evals = self.eval_tool(labels, prediction)

        self.avg_test_acc = metrics.accuracy(
                np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        self.valid_writer.add_scalar('Accuracy/epoch', self.avg_test_acc, self.epoch)
        self.valid_writer.add_scalar('Loss/epoch', total_test_loss/valid_samples, self.epoch)

    def load_model_from_checkpoint(self, ckpt):
        state_key = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state_key['net'])
        self.model = self.model.to(self.device)

    def save_model(self):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch
            }
        if self.avg_test_acc > self.max_acc:
            self.max_acc = self.avg_test_acc
            self.logger.info(f"-- Saving best model with testing accuracy {self.max_acc:.3f} --")
            checkpoint_name = 'ckpt_best.pth'
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))

        if self.epoch%self.config.TRAIN.CHECKPOINT_SAVING_STEPS == 0:
            self.logger.info(f"Saving model with testing accuracy {self.avg_test_acc:.3f} in epoch {self.epoch} ")
            checkpoint_name = 'ckpt_best_{:04d}.pth'.format(self.epoch)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))



if __name__ == '__main__':
    pass