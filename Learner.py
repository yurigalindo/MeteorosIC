from __future__ import print_function, division 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time

class Learner:

    def __init__(self,model,criterion):
        self.model=model
        self.criterion=criterion
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_datasets(self,training,valid):
        self.training_dataset=train
        self.validation_dataset=valid
    def change_optimizer(self,optimizer,scheduler):
        self.optimizer=optimizer
        self.scheduler=scheduler

    def train_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs=self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        _,preds = torch.max(outputs, 1)
        corrects = torch.sum(preds==labels.data).item()
        return loss.item(),corrects

    def eval_batch(self, inputs, labels):
        with torch.no_grad():
            outputs=self.model(inputs)
            loss = self.criterion(outputs, labels)
            _,preds = torch.max(outputs, 1)
        corrects = torch.sum(preds==labels.data).item()
        return loss.item(),corrects

    def print_statistics(self,loss,accuracy):
        print("Total Loss={}\nAccuracy={}".format(loss,accuracy))

    def train(self,num_epochs,training_batchsize,validation_batchsize):
        """Train the learner's model for num_epochs on training batchsize, printing the train and validation loss and accuracy
        """
        train_dataloader=torch.utils.data.DataLoader(self.training_dataset,batch_size=training_batchsize,shuffle=True)
        valid_dataloader=torch.utils.data.DataLoader(self.validation_dataset,batch_size=validation_batchsize,shuffle=True)
        training_batches=len(self.training_dataset)/training_batchsize
        validation_batches=len(self.validation_dataset)/validation_batchsize
        since = time.time()
        for epoch in range(num_epochs):
            print("Epoch {} of {}".format(epoch,num_epochs))
            self.scheduler.step()
            

            self.model.train()
            #Train model
            loss = 0
            corrects = 0
            iterator = iter(train_dataloader)
            for batch in range(training_batches):
                inputs,labels=next(iterator)
                new_loss , new_corrects = self.train_batch(inputs.to(self.device),labels.to(self.device))
                del inputs
                del labels
                loss+=new_loss
                corrects+=new_corrects

            print("Training:")
            self.print_statistics(loss,corrects/len(self.training_dataset))

            self.model.eval()
            #Evaluate model
            loss = 0
            corrects = 0
            iterator = iter(valid_dataloader)
            for batch in range(validation_batches):
                inputs,labels=next(iterator)
                new_loss , new_corrects=self.eval_batch(inputs.to(self.device),labels.to(self.device))
                del inputs
                del labels
                loss+=new_loss
                corrects+=new_corrects

            print("Validation:")
            self.print_statistics(loss,corrects/len(self.validation_dataset))
            print()
        time_elapsed = time.time() - since
        print("Total time:{}".format(time_elapsed))

