import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record,load_pre_model
from model import Model
# from preload_dataset import Dataset
from dataset import Dataset
from train import train
from test import test
import option
from tqdm import tqdm
# from utils import Visualizer
from config import *
import time

from thop import profile
import copy
# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
from  torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR


def main( depth,heads,topk,LR , indx):
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(args.feature_size, args.batch_size, depth,heads,topk,device)
    # pre_model = Model(args.feature_size, args.batch_size, device)
    # pre_model = pre_model.to(device)
    # model = load_pre_model('../ckpt/0120_ucf3d_0.0001_fc1fc21024_fd6h4_3_0.8011_0.1917.pkl',model)
    model_name = '0417_shi3d_{}_fc1fc21024_td{}h{}k{}_{}'.format(LR,depth,heads,topk,indx)

    best_dict = model.state_dict()
    model = model.to(device)
    # macs, params=profile(model,(torch.randn(1,10,10,2048).to(device),))
    # print(f"macs = {macs / 1e9}G")
    # print(f"params = {params / 1e6}M")
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total / 1e6))

    if not os.path.exists('../ckpt'):
        os.makedirs('../ckpt')
    

    optimizer = optim.Adam(model.parameters(),
                           lr=LR, weight_decay=0.005)
    # optimizer2 = optim.Adam(loss_model.parameters(),
    #                         lr=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=2, T_mult=2)
    # scheduler = CosineAnnealingLR(optimizer,T_max=2)
    # scheduler2 = CosineAnnealingLR(optimizer2,T_max=2)
    test_info = {"epoch": [], "test_AUC": [],'ap':[]}
    best_ROC_AUC = -1
    best_AP_AUC = -1
    output_path = '../record'  # put your own path here
    torch.cuda.synchronize()
    test_start=time.time()
    roc_auc,ap = test(test_loader, model, args, device)
    torch.cuda.synchronize()
    # print(time.time()-test_start)
    file_path = os.path.join(output_path, model_name + '.txt')
    if os.path.exists(file_path):
        os.remove(file_path)
    # for data in iter(train_nloader):
    #         print(data[0].shape)
    # for data in iter(train_aloader):
    #         print(data[0].shape)
    # print('??????:',next(iter(train_nloader)))
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        # if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print(optimizer2.state_dict()['param_groups'][0]['lr'])
        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device,None)
        scheduler.step(step/len(train_nloader))
        if step % 10 == 0 and step > 10:

            roc_auc,ap_auc = test(test_loader, model, args, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(roc_auc)
            test_info["ap"].append(ap_auc)
            if test_info["test_AUC"][-1] > best_ROC_AUC:
                best_ROC_AUC = test_info["test_AUC"][-1]
                best_AP_AUC = test_info["ap"][-1]
                best_dict = copy.deepcopy(model.state_dict())

                save_best_record(test_info, file_path)
            # if step == 500 and best_ROC_AUC < 0.90:
            #     print("over!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     break

    os.rename(file_path, file_path.replace('.txt', '_' + str(round(best_ROC_AUC, 4))+'_'+str(round(best_AP_AUC, 4)) + '.txt'))
    torch.save(best_dict, '../ckpt/' + model_name + '_' + str(round(best_ROC_AUC, 4)) + '_'+str(round(best_AP_AUC, 4)) +'.pkl')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    count = 0
    
    for depth in [6]:
        for heads in [4]:
            for topk in [3]:
                for five in range(1):
                    for LR in [1e-4]:
                        main(depth,heads,topk,LR, count)
                        count += 1
