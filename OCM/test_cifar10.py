import sys, argparse
import numpy as np
import torch
from torch.nn.functional import relu, avg_pool2d
from buffer import Buffer
# import utils
import datetime
from torch.nn.functional import relu
import torch
import torch.nn as nn
import torch.nn.functional as F
from InfoNCE import tao as TL
#from CSL import classifier as C
from InfoNCE.utils import normalize
from InfoNCE.contrastive_learning import get_similarity_matrix,Supervised_NT_xent_pre,Supervised_NT_xent_n,Supervised_NT_xent_uni
import torch.optim.lr_scheduler as lr_scheduler
#from CSL.shedular import GradualWarmupScheduler
import torch
import torchvision.transforms as transforms
import  torchvision
from torch.cuda.amp import GradScaler,autocast
scaler = GradScaler()

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='cifar-10', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='OCMM', type=str, required=False, help='(default=%(default)s)')
#parser.add_argument('--nepochs', default=25, type=int, required=False, help='(default=%(default)d)')
#parser.add_argument('--lr', default=0.02, type=float, required=False, help='(default=%(default)f)')
#parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--dataset', type=str, default='cifar', help='(default=%(default)s)')
parser.add_argument('--input_size', type=str, default=[3, 32, 32], help='(default=%(default)s)')
parser.add_argument('--buffer_size', type=int, default=200, help='(default=%(default)s)')
parser.add_argument('--gen', type=str, default=True, help='(default=%(default)s)')
parser.add_argument('--n_classes', type=int, default=512, help='(default=%(default)s)')
parser.add_argument('--buffer_batch_size', type=int, default=64, help='(default=%(default)s)')
args = parser.parse_args()
import os
gpus = [0]#, 1, 2, 3,5,6,7]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
from apex import amp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use gpu0,1
def rot_inner_all(x):
    num=x.shape[0]

    #print(num)
    R=x.repeat(4,1,1,1)
    a=x.permute(0,1,3,2)
    a = a.view(num,3, 2, 16, 32)
    import pdb
   # pdb.set_trace()
  #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2,0, 1, 3, 4)
    s1=a[0]#.permute(1,0, 2, 3)#, 4)
    s2=a[1]#.permute(1,0, 2, 3)
    #print("a",a.shape,a[:63][0].shape)
    s1_1 = torch.rot90(s1, 2, (2, 3))
    s2_2 = torch.rot90(s2, 2, (2, 3))#R[3*num:]

    R[num:2*num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)
    R[3*num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)
    R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)

    return R
def Rotation(x):
   # print(x.shape)
  # if r<=-1:
        X = rot_inner_all(x)#, 1, 0)
       # x = torch.rot90(x,r,(2,3))
    #import pdb
   # pdb.set_trace()
       # X=[]
   # print(x.shape)


        #X.append(x)


       # X=rot_inner_all(x, 1, 0)


       # X.append(rot_inner(x, 0, 1))
    #else:
     #   x1=rot_inner(x,0,1)
       # X.append(rot_inner(x,1,1))
        return torch.cat((X,torch.rot90(X,2,(2,3)),torch.rot90(X,1,(2,3)),torch.rot90(X,3,(2,3))),dim=0)
   #else:
       #if r <= 0:
     #   x = torch.rot90(x, r, (2, 3))
    #    return x

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' )
print('=' * 100)
########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()
import cifar as dataloader
# import owm as approach
# import cnn_owm as network
#from minimodel import net as s_model
from Resnet18 import resnet18 as b_model
from buffer import Buffer as buffer
# imagenet200 import SequentialTinyImagenet as STI
from torch.optim import Adam, SGD  # ,SparseAdam
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
def imshow(img):
    img=img/2+0.5
    npimg=img.cpu().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
def test_model(loder,i,model):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):

        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        model.eval()
        pred=model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]

    #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
            test_loss, correct, num,
            100. * correct / num, ))
    return test_accuracy
print('Load data...')
oop=16

data, taskcla, inputsize, Loder, test_loder = dataloader.get_fast(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)
buffero = buffer(args).cuda()
Basic_model = b_model(10).cuda()
llabel = {}
Optimizer = Adam(Basic_model.parameters(), lr=0.001, betas=(0.9, 0.99),weight_decay=1e-4)#SGD(Basic_model.parameters(), lr=0.02, momentum=0.9)
Basic_model, Optimizer = amp.initialize(Basic_model, Optimizer,opt_level="O1")
#Basic_model = nn.DataParallel(Basic_model.cuda(), device_ids=gpus, output_device=gpus[0])
hflip = TL.HorizontalFlipLayer().cuda()
cutperm = TL.CutPerm().cuda()
with torch.no_grad():
    resize_scale = (0.3, 1.0)  # resize scaling factor,default [0.08,1]

  #  color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).cuda()
    color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
    simclr_aug = transform = torch.nn.Sequential(
       # color_jitter,  # 这个不会变换大小，但是会变化通道值，新旧混杂
        hflip,
        color_gray, # 这个也不会，混搭)
        resize_crop, )
Max_acc=[]
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ')
print('=' * 100)
class_holder=[]
buffer_per_class =7
for run in range(1):
    rank=torch.arange(0, 5)

    for i in range(len(Loder)):
        print(i)
        task_id=i
        if i==0:
            train_loader = Loder[rank[i].item()]['train']
            #Optimizer = Adam(Basic_model.parameters(), lr=0.02, momentum=0.9)
            for epoch in range(1):
                Basic_model.train()
                num_d=0
                for batch_idx, (x, y) in enumerate(train_loader):
                    num_d+=x.shape[0]
                    if num_d%5000==0:
                        print(num_d,num_d/10000)
                  #  if batch_idx>3:
                   #     continue
                    llabel[i] = []

                    Y = deepcopy(y)
                    for j in range(len(Y)):
                        if Y[j] not in class_holder:
                            class_holder.append(Y[j].detach())

                    Optimizer.zero_grad()
                    # if args.cuda:
                    x, y = x.cuda(), y.cuda()

                    x = x.requires_grad_()
                   # imshow(torchvision.utils.make_grid(x[2]))
                    #positive_data2 = x.data
                    #images1, images2 = hflip(positive_data2.repeat(2, 1, 1, 1)).chunk(2)
                    images1 = Rotation(x)#torch.cat([Rotation(x, r) for r in range(4)])
                   # images2 = Rotation(x)#torch.cat([Rotation(x, r) for r in range(4)])#.detach()  # 4B

                    #images1 = torch.cat([Rotation(x, r) for r in range(4)])
                   # images2 = torch.cat([Rotation(x, r) for r in range(4)])

                    images_pair = torch.cat([images1, simclr_aug(images1)], dim=0)  # 8B

                 #   labels1 = y.cuda()
                    # print("LLLL",labels1.shape)
                    rot_sim_labels = torch.cat([y.cuda() + 100 * i for i in range(oop)], dim=0)
                  #  Rot_sim_labels = torch.cat([labels1 + 0 * i for i in range(oop)], dim=0)
              #      rot_sim_labels1 = torch.cat([torch.arange(0,labels1.shape[0]).cuda()+ 10*i for i in range(4)], dim=0)
                    #print(labels1)
                    #print(Rot_sim_labels)
                    # print("RRRR1",rot_sim_labels)da()  # 这个label其实是用来mask的，4个rotate
                 #   rot_sim_labels = rot_sim_labels.cuda()
                  #  images_pair = simclr_aug(images_pair)  # simclr augment
                    # print("X",input_x.shape,images_pair.shape)
                  #  for n,w in Basic_model.named_parameters():
                   #     print(n,w.shape)
                    feature_map,outputs_aux = Basic_model(images_pair, is_simclr=True)
                   # outputs_aux1 = Basic_model(images_pair, is_simclr1=True)# , penultimate=True)

                    simclr = normalize(outputs_aux)  # normalize
                    feature_map_out = normalize(feature_map[:images_pair.shape[0]])
                   # num1 = feature_map_out.shape[1] // simclr.shape[1]
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
             #       id1_2 = torch.randperm(num1)[1]
                    size = simclr.shape[1]
                 #   sim_matrix = torch.zeros((simclr.shape[0], simclr.shape[0])).cuda()
                    #sim_matrix_r = torch.zeros((simclr_r.shape[0], simclr_r.shape[0])).cuda()
                    #   sim_matrix_r_pre = torch.zeros((pre_r.shape[0], pre_r.shape[0])).cuda()

                    #for index in range(num1):
                        # pdb.set_trace()
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).cuda())#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim1 = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                   temperature=0.07)
                    lo1 = 1 * loss_sim1 #+0*loss_sim2

                    if task_id==0:
                       # hidden_pred = Basic_model.f_train()
                        y_pred = Basic_model.forward(simclr_aug(x))
                        loss = 1*F.cross_entropy(y_pred, y)+1*lo1
                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()
           #         loss.backward()

#                    print(i, epoch, loss)
                    #optimizer[i].step()
                    Optimizer.step()


                    buffero.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=i)
            Previous_model = deepcopy(Basic_model)
            for j in range(i + 1):
                print("ori", rank[j].item())
                a = test_model(Loder[rank[j].item()]['test'], j,Basic_model)
                if j == i:
                    Max_acc.append(a)
                if a > Max_acc[j]:
                    Max_acc[j] = a


        else:
            train_loader = Loder[rank[i].item()]['train']
            #optimizer.append(Adam(S_model[i].parameters(), lr=0.001, betas=(0.9, 0.99)))  # ,momentum=0.9))
            for epoch in range(1):
            #    S_model[i].train()
                num_d=0
                Basic_model.train()
                for batch_idx, (x, y) in enumerate(train_loader):
                    num_d+=x.shape[0]
                    if num_d%5000==0:
                        print(num_d,num_d/10000)



                    Y = deepcopy(y)
                    for j in range(len(Y)):
                        if Y[j] not in class_holder:
                            class_holder.append(Y[j].detach())
                    task_id=i

                    # print("idx",batch_idx)
                    # buffero.add_reservoir(x=x, y=y, logits=None, t=i)
             #       optimizer[i].zero_grad()
                    Optimizer.zero_grad()
                    # if args.cuda:
                    x, y = x.cuda(), y.cuda()
                    # x, y = Variable(x), Variable(y)
                    x = x.requires_grad_()
                    buffer_batch_size = min(64,buffer_per_class*len(class_holder))
                    mem_x, mem_y,bt = buffero.sample(buffer_batch_size, exclude_task=None)

                    mem_x = mem_x.requires_grad_()

                    images1 = Rotation(x)

                    images1_r = Rotation(mem_x)#torch.cat([Rotation(mem_x, r) for r in range(4)])
                  #  images2_r = Rotation(mem_x)#torch.cat([Rotation(mem_x, r) for r in range(4)])#.detach()
                    #images2_r.requires_grad=False
                    images_pair = torch.cat([images1, simclr_aug(images1)], dim=0)
                    images_pair_r = torch.cat([images1_r, simclr_aug(images1_r)], dim=0)

                    t =torch.cat((images_pair,images_pair_r),dim=0)
                    feature_map, u = Basic_model.forward(t, is_simclr=True)
                    _, pre_u = Previous_model.forward(images1_r, is_simclr=True)
                    feature_map_out = normalize(feature_map[:images_pair.shape[0]])
                    feature_map_out_r = normalize(feature_map[images_pair.shape[0]:])
              #      pre_feature_map_out_r = normalize(pre_u_feature)




                    images_out = u[:images_pair.shape[0]]
                    images_out_r = u[images_pair.shape[0]:]
                    pre_u = normalize(pre_u)#torch.cat((images_out_r,pre_u),dim=0)

                    simclr = normalize(images_out)
                    simclr_r = normalize(images_out_r)
               #     simclr_pre = normalize(pre_feature_map_out_r)
                   # simclr_now=normalize(u[images_pair.shape[0]:images_pair.shape[0]+images1_r.shape[0]])

                    rot_sim_labels = torch.cat([y.cuda()+ 100 * i for i in range(oop)],dim=0)
            #        rot_sim_labels1 = torch.cat([torch.arange(0,y.shape[0]).cuda() +10*i for i in range(4)], dim=0)
             #       rot_sim_labels_r1 = torch.cat([torch.arange(0,mem_y.shape[0]).cuda()+10*i for i in range(4)], dim=0)
                    rot_sim_labels_r = torch.cat([mem_y.cuda()+ 100 * i for i in range(oop)],dim=0)

                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
             #       id1_2 = torch.randperm(num1)[1]
                    id2=torch.randperm(num1)[0]
              #      id2_2 = torch.randperm(num1)[0]
                  #  id3 = torch.randperm(num1)[0]
                    size = simclr.shape[1]

                    sim_matrix = 0.5*torch.matmul(simclr, feature_map_out[:, id1:id1 + size].t())
                    sim_matrix_r = 0.5*torch.matmul(simclr_r,
                                                     feature_map_out_r[:, id2:id2 + size].t())


                    sim_matrix += 0.5 * get_similarity_matrix(
                        simclr)  # *(1-torch.eye(simclr.shape[0]).cuda())#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r += 0.5 * get_similarity_matrix(simclr_r)
                    sim_matrix_r_pre = torch.matmul(simclr_r[:images1_r.shape[0]],pre_u.t())
                  #  Label=torch.cat((rot_sim_labels.repeat(2),rot_sim_labels_r.repeat(2)),dim=0)
                  #  sim_matrix /= (num1 + 1)
                 #   sim_matrix_r /= (num1 + 1)
               #
             #       loss_sim_mix1=Supervised_NT_xent(sim_matrix_mix,labels=rot_sim_labels_mix1,temperature=0.07)
                    loss_sim_r =Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
             #       loss_sim_mix2= Supervised_NT_xent(sim_matrix_mix,labels=rot_sim_labels_mix2,temperature=0.07)
                    loss_sim_pre = Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                   # y_feature = Basic_model.f_train(simclr_aug(mem_x))
                    y_label = Basic_model.forward(simclr_aug(mem_x))
                    y_label_pre = Previous_model(simclr_aug(mem_x))
                    #  y_label = Basic_model.linear(y_feature)
                    # Pre_y_feature = Previous_model.f_train(simclr_aug(mem_x))
                    # hidden_pred = Basic_model.f_train(simclr_aug(x))
                    # y_label_1 = Basic_model.linear(hidden_pred)

                    loss = 1 * F.cross_entropy(y_label, mem_y) + lo1  +1 * F.mse_loss(y_label_pre[:, :2 * task_id],
                                                                                       y_label[:,
                                                                                       :2 * task_id])  #+1*F.mse_loss(y_feature, logits)+ 1*F.mse_loss(y_feature, Pre_y_feature)
                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()
               #     loss.backward()

                    Optimizer.step()




                    buffero.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=i)
           # import pdb
          #  pdb.set_trace()
            Previous_model = deepcopy(Basic_model)

    print('=' * 100)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ')
    print('=' * 100)
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(test_loder):

        data, target = data.cuda(), target.cuda()
        Basic_model.eval()
        pred=Basic_model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(
            test_loss, correct, num,
            100. * correct / num, ))



