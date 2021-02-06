import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse

from data.dataset_train import DAVIS_MO_Test
from model.models import resnet18
from model.selector_net import selector_net

import os
import itertools

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Solver:
  def __init__(self, args, train_loader, test_loader=None):

    self.args= args
    self.lr = args.lr
    self.name = args.name
    self.criterion = nn.MSELoss()
    self.epochs = args.epochs
    self.log_step = args.log_step
    self.ckpt_step = args.ckpt_step

    self.log_dir = args.log_dir
    self.check_dir = args.checkpoint_dir
    self.writer = SummaryWriter(os.path.join(self.log_dir, args.name))
    self.train_loader = train_loader
    self.test_loader = test_loader

    self.model = selector_net()
    self.optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.fc1.parameters(), self.fc2.parameters()), self.lr)
    self.lr_sch = get_scheduler(self.optimizer, args)



  def get_norm(self, params):
    total_norm = 0
    for p in params:
      param_norm = p.grad.data.norm(2)
      total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1. / 2)
    return total_norm

  def test(self):
    dataloader = torch.utils.data.DataLoader(self.test_loader, batch_size=1, shuffle=True, num_workers=2)
    avg_loss = 0.0
    acc = 0.0
    self.model.eval()
    for itr, data in tqdm(enumerate(dataloader)):

      img1, img2, label1, label2 = data["input1"], data["input2"], data["output1"], data["output2"] 
      img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)

      with torch.no_grad():
        output = self.model(img1, img2)

      
      l1 = label1.cpu().numpy()[0]
      l2 = label2.cpu().numpy()[0]
      o = output.cpu().detach().numpy()[0]
      
      if o[0] > o[1]:
        res = np.array([1.,0.])
      else:
        res = np.array([0.,1.])

      if l1 > 0.5:
        l = np.array([1.,0.])
      else:
        l = np.array([0.,1.])
     

      if np.array_equal(l, res):
        acc += 1
    
      loss = np.fabs(res-l1)
      avg_loss = avg_loss*itr + loss
      avg_loss = avg_loss/(itr+1)
    print(acc/len(dataloader))

  def train(self):


    data_loader = self.train_loader

    print("Starting Training")
       

    for i in range(self.epochs):
      self.lr_sch.step()
      self.model.eval()
      self.test()
      self.model.train()
      for itr, data in tqdm(enumerate(data_loader)):

        img1, img2, label1, label2 = data["input1"], data["input2"], data["output1"], data["output2"] 
        img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)

        label1, label2 = torch.unsqueeze(label1, 1), torch.unsqueeze(label2,1)
        label = torch.cat((label1, label2), 1)
        
        output = self.model(img1, img2)
        
        loss = self.criterion(output, label.float())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        norm = self.get_norm(itertools.chain(self.model.parameters(), self.fc1.parameters(), self.fc2.parameters()))
        self.optimizer.step()

        if(itr%self.log_step == 0):
          
          self.writer.add_scalar('loss', loss, itr + len(data_loader)*i)
          self.writer.add_scalar('Norm', norm, itr + len(data_loader)*i)
          lr1 = get_lr(self.optimizer)
          self.writer.add_scalar('LR',lr1, itr + len(data_loader)*i)
          self.writer.flush()

        if(i%self.ckpt_step == 0 and itr==0):
          p = os.path.join(os.path.join(self.check_dir,self.name) , "epoch" + str(i))
          if(not os.path.isdir(p)):
            os.makedirs(p)
          torch.save({"model":self.model.state_dict()}, os.path.join(p, "model.ckpt"))





def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']



def get_scheduler(optimizer, opt):
  """Return a learning rate scheduler
  Parameters:
    optimizer          -- the optimizer of the network
    opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
  For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
  and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
  For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
  See https://pytorch.org/docs/stable/optim.html for more details.
  """
  if opt.lr_policy == 'linear':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + 20- opt.epochs) / float(20 + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
  elif opt.lr_policy == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
  else:
    return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler



# parser = argparse.ArgumentParser()

# parser.add_argument('--train_dataset_path',
#           default='./data')
# parser.add_argument('--val_dataset_path',
#           default='./data')
# parser.add_argument('--load',
#           default=0)
# parser.add_argument('--batch_size',
#           default=64)
# parser.add_argument('--epochs', default=21, type=int,
#           help='epochs to train for')
# parser.add_argument('--lr', default=0.0001, type=float, help='LR')#used 0.01
# parser.add_argument('--lr_policy', default="linear", type=str, help='LR policy')
# parser.add_argument('--log_step', type=int, default=10,
#           help='frequency to display')
# parser.add_argument('--ckpt_step', type=int, default=2,
#           help='frequency to display')
# parser.add_argument('--checkpoint_dir', type=str, default="checkpoint",
#           help='frequency to display')
# parser.add_argument('--log_dir', type=str, default="logs",
#           help='frequency to display')
# parser.add_argument('--name', type=str, default="initial_tes",
#           help='name')

# args = parser.parse_args()
# print("Input arguments:")
# for key, val in vars(args).items():
#   print("{:16} {}".format(key, val))


# test_loader = DAVIS_MO_Test(imset="val.txt")
# train_loader = DAVIS_MO_Test()
# print("Loaded")
# dataloader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=2)
# print("Ready")
# solve = Solver(args, dataloader, test_loader)

# solve.train()

