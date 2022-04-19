import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
def cal_false_alarm(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    # false_num=0.
    # _len=len(labels)
    # for score,label in zip(scores,labels):
    #     if label!=score:
    #         false_num+=1
    fp=np.sum(scores*(1-labels))
    return fp/np.sum(1-labels)
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def plot_roc_curve(fpr,tpr, thresholds):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])

    plt.savefig('roc_and_threshold.png')
    plt.close()
def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits,_,_ = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            pred = torch.cat((pred, sig))
        # print('pred:',pred.shape)
        if args.dataset == 'sh':
            gt = np.load('../list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('../list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        roc_auc = auc(fpr, tpr)
        print('roc_auc : ' + str(roc_auc))

        # plot_roc_curve(fpr, tpr, threshold)
        # best_threshold=threshold[np.where((tpr-fpr)==np.max(tpr-fpr))]

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)
        # np.save('{}_res.npy'.format(args.dataset),pred)
        # print(cal_false_alarm(pred,gt,best_threshold))
        torch.cuda.empty_cache()

        return roc_auc,pr_auc

