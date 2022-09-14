"
Model training and testing.

Written by Jing Luo from Xi'an University of Technology, China.

luojing@xaut.edu.cn
"

import logging
import os
import os.path
import time
from collections import OrderedDict
import sys
import pandas as pd
import numpy as np
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from bcic_iv_2b import BCICompetition4Set2B
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from tqdm import tqdm
from SMT import SMT

log = logging.getLogger(__name__)
gpus=[0]
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))

def mergeSet(sat1,sat2):
    tempX = np.concatenate((sat1.X,sat2.X))
    tempY = np.concatenate((sat1.y,sat2.y))
    return SignalAndTarget(tempX,tempY)

def pick2Class(set):
    tempX = set.X
    tempY = set.y
    trialNum = tempY.size
    for t in range(trialNum-1,-1,-1):
        if tempY[t] in [2,3]:
            tempX = np.delete(tempX,t,axis=0)
            tempY = np.delete(tempY,t,axis=0)
    return SignalAndTarget(tempX,tempY)

def process_2a(data_folder, subject_id, low_cut_hz, high_cut_hz):
    ival = [-500, 4000]
    factor_new = 1e-3
    init_block_size = 1000
    # Data loading
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')
    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # train_cnt
    train_cnt = train_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    assert len(train_cnt.ch_names) == 3
    # convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    # bandpass
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    # test_cnt
    test_cnt = test_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    assert len(test_cnt.ch_names) == 3
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    # left and right 2-category EEG
    train_set = pick2Class(train_set)
    test_set = pick2Class(test_set)
    return train_set,test_set

def read_data_2a(data_folder,low_cut_hz,high_cut_hz):
    train_set1, test_set1 = process_2a(data_folder, 1, low_cut_hz, high_cut_hz)
    for subject_id in range(2, 10, 1):
        train_set, test_set = process_2a(data_folder, subject_id, low_cut_hz, high_cut_hz)
        train_set1 = mergeSet(train_set1, train_set)
        test_set1 = mergeSet(test_set1, test_set)

    train_set = train_set1
    test_set = test_set1
    train_temp = train_set.X.copy()
    train_temp_label = train_set.y.copy()
    # mirror EEG
    train_temp[:, (0, 2), :] = train_temp[:, (2, 0), :]
    where_0 = np.where(train_temp_label == 0)
    where_1 = np.where(train_temp_label == 1)
    train_temp_label[where_0] = 1
    train_temp_label[where_1] = 0

    train_data.append(train_set.X)
    data_label.append(train_set.y)
    train_data.append(train_temp)
    data_label.append(train_temp_label)
    test_data.append(test_set.X)
    test_data_label.append(test_set.y)

def process_2b(data_folder, subject_id, low_cut_hz, high_cut_hz):
    ival = [-500, 4000]
    factor_new = 1e-3
    init_block_size = 1000

    test_filename4 = 'B{:02d}04E.gdf'.format(subject_id)
    test_filename5 = 'B{:02d}05E.gdf'.format(subject_id)
    test_filepath4 = os.path.join(data_folder, test_filename4)
    test_filepath5 = os.path.join(data_folder, test_filename5)
    test_label_filepath4 = test_filepath4.replace('.gdf', '.mat')
    test_label_filepath5 = test_filepath5.replace('.gdf', '.mat')
    test_loader4 = BCICompetition4Set2B(
        test_filepath4, labels_filename=test_label_filepath4)
    test_loader5 = BCICompetition4Set2B(
        test_filepath5, labels_filename=test_label_filepath5)
    test_cnt4 = test_loader4.load()
    test_cnt5 = test_loader5.load()    
    # Preprocessing
    test_cnt4 = test_cnt4.pick_channels(['EEG:C3', 'EEG:C4', 'EEG:Cz'])
    assert len(test_cnt4.ch_names) == 3
    test_cnt4 = mne_apply(lambda a: a * 1e6, test_cnt4)
    test_cnt4 = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt4.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt4)
    test_cnt4 = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt4)

    test_cnt5 = test_cnt5.pick_channels(['EEG:C3', 'EEG:C4', 'EEG:Cz'])
    assert len(test_cnt5.ch_names) == 3
    test_cnt5 = mne_apply(lambda a: a * 1e6, test_cnt5)
    test_cnt5 = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt5.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt5)
    test_cnt5 = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt5)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    tempSet4 = create_signal_target_from_raw_mne(test_cnt4, marker_def, ival)
    tempSet5 = create_signal_target_from_raw_mne(test_cnt5, marker_def, ival)
    test_set = mergeSet(tempSet4, tempSet5)
    return test_set

def read_data_2b(data_folder,low_cut_hz,high_cut_hz):
    test_set1 = process_2b(data_folder, 1, low_cut_hz, high_cut_hz)
    for subject_id in range(2, 10, 1):
        test_set = process_2b(data_folder, subject_id, low_cut_hz, high_cut_hz)
        test_set1 = mergeSet(test_set1, test_set)
    test_set = test_set1
    test_data_2b.append(test_set.X)
    test_data_label_2b.append(test_set.y)

def train(opt,train_data,data_label,test_data,test_data_label,test_data_2b,test_data_label_2b):
    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(data_label)
    test_data, test_data_label = torch.from_numpy(test_data), torch.from_numpy(test_data_label) 
    test_data_2b, test_data_label_2b = torch.from_numpy(test_data_2b), torch.from_numpy(test_data_label_2b)
    data_set_test_2b = TensorDataset(test_data_2b, test_data_label_2b)
    dataloader_test_2b = DataLoader(data_set_test_2b, batch_size=opt.batch_size, shuffle=True)

    dataset = TensorDataset(train_data, train_label)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    data_set_test = TensorDataset(test_data,test_data_label)
    dataloader_test = DataLoader(data_set_test,batch_size=opt.batch_size,shuffle=True)

    eeg_size = train_data.shape[-1]    
    model = SMT(eeg_size=eeg_size)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.functional.nll_loss
    
    for e in range(opt.epochs):
        list_excel=[str(e+1)]
        train_loss = 0
        train_acc = 0
        model.train()
        # set progress bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (t_data, t_label) in pbar:
            t_data = Variable(t_data.cuda().type(torch.cuda.FloatTensor))
            t_label = Variable(t_label.cuda().type(torch.cuda.LongTensor))
            pred = model(t_data)          
            pred_train = torch.max(pred, 1)[1]
            train_acc += (pred_train.cpu().numpy() == t_label.cpu().numpy()).sum()
            if i == len(pbar) - 1:
                l = len(dataset)
            else:
                l = (i + 1) * len(t_label)          
            loss = criterion(torch.log(torch.clamp(pred,min=1e-6)), t_label)
            train_loss += loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                'epoch : {}, train loss: {:.3f}, train acc: {:.3f}'.format(e, train_loss / (i + 1), train_acc / l))
            if l == len(dataset):

                list_excel.append(train_acc / l)
                list_excel.append(1 - train_acc / l)

        model.eval()

        # 2a shallow transformer
        test_acc = 0.0
        # 2a shallow mirror transformer
        test_acc_twin = 0.0
        # 2b shallow transformer
        test_acc_2b = 0.0
        # 2b shallow mirror transformer
        test_acc_2b_twin = 0.0

        with torch.no_grad():
            # 2a
            for test_temp in dataloader_test:
                test_digit, test_output = test_temp
                temp = model(test_digit)
                # mirror EEG
                test_digit_lr = test_digit
                test_digit_lr[:, :, (0, 2), :] = test_digit_lr[:, :, (2, 0), :]
                temp_lr = model(test_digit_lr)
                temp_lr[:, (0, 1)] = temp_lr[:, (1, 0)]
                # porb emsemble
                temp_merge = temp_lr + temp
                pred_test = torch.max(temp, 1)[1]
                test_acc += (pred_test.cpu().numpy() == test_output.cpu().numpy()).sum()
                pred_test = torch.max(temp_merge, 1)[1]
                test_acc_twin += (pred_test.cpu().numpy() == test_output.cpu().numpy()).sum()
            # 2b
            for test_temp in dataloader_test_2b:
                test_digit, test_output = test_temp
                temp = model(test_digit)
                # mirror EEG
                test_digit_lr = test_digit
                test_digit_lr[:, :, (0, 2), :] = test_digit_lr[:, :, (2, 0), :]
                temp_lr = model(test_digit_lr)
                temp_lr[:, (0, 1)] = temp_lr[:, (1, 0)]
                # porb emsemble
                temp_merge = temp_lr + temp
                pred_test = torch.max(temp, 1)[1]
                test_acc_2b += (pred_test.cpu().numpy() == test_output.cpu().numpy()).sum()
                pred_test = torch.max(temp_merge, 1)[1]
                test_acc_2b_twin += (pred_test.cpu().numpy() == test_output.cpu().numpy()).sum()

            test_acc = test_acc / (test_data.shape[0])
            test_acc_twin = test_acc_twin / (test_data.shape[0])
            test_acc_2b = test_acc_2b / (test_data_2b.shape[0])
            test_acc_2b_twin = test_acc_2b_twin / (test_data_2b.shape[0])

            list_excel.append(test_acc)
            list_excel.append(1 - test_acc)
            list_excel.append(test_acc_twin)
            list_excel.append(1 - test_acc_twin)
            
            list_excel.append(test_acc_2b)
            list_excel.append(1 - test_acc_2b)
            list_excel.append(test_acc_2b_twin)
            list_excel.append(1 - test_acc_2b_twin)
            print('test acc: ', test_acc)
        if e==0:
            df = pd.DataFrame([['epoch','train_acc','train_misclass', 'test_acc','test_misclass','test_acc_twin','test_misclass_twin','test_acc_newsub','test_misclass_newsub','test_acc_newsub_twin','test_misclass_newsub_twin']])  # 列名
            data = pd.DataFrame([list_excel])
            data = df.append(data)
        else:
            data1 = pd.DataFrame([list_excel])
            data = data.append(data1)
        torch.save(model.module.state_dict(), './Vit'+str(low_cut_hz)+'hz.pth')       
    data.to_excel('./acc_' + str(low_cut_hz) + 'hz.xlsx',header=None ,index=False)

if __name__ == '__main__':

    time_start=time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
    level=logging.DEBUG, stream=sys.stdout)

    data_folder_2a = r'D:\BCI\BCICIV_2a_gdf'
    data_folder_2b = r'D:\BCI\BCICIV_2b_gdf'
    cuda =True
    low_cut_hz =0
    high_cut_hz =38
    
    train_data = []
    data_label = []
    test_data = []
    test_data_label = []
    test_data_2b = []
    test_data_label_2b = []
    
    read_data_2a(data_folder_2a, low_cut_hz, high_cut_hz)
    train_data = np.concatenate(train_data, 0)
    data_label = np.concatenate(data_label, 0)
    train_data = np.expand_dims(train_data, 1)
    test_data = np.concatenate(test_data, 0)
    test_data_label = np.concatenate(test_data_label, 0)
    test_data = np.expand_dims(test_data, 1)

    read_data_2b(data_folder_2b, low_cut_hz, high_cut_hz)
    test_data_2b = np.concatenate(test_data_2b, 0)
    test_data_label_2b = np.concatenate(test_data_label_2b, 0)
    test_data_2b = np.expand_dims(test_data_2b, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500)
    opt = parser.parse_args()
    train(opt,train_data,data_label,test_data,test_data_label,test_data_2b,test_data_label_2b)
