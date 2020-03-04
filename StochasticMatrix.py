import os
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score,roc_curve)
from scipy import sparse
from scipy.sparse import csr_matrix
import jieba.analyse
import re

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')
indptr=[0]
indices=[]
data=[]
vocabulary={}
with codecs.open('CJO_word_cut_1103_K300_tf_commonwords.txt',encoding='utf-8',errors='ignore') as onetxt:
    abcdlines=onetxt.readlines()
    alist=abcdlines[0:320]
    blist=abcdlines[404:964]
    clist=abcdlines[320:404]
    dlist=abcdlines[964:1103]
    ablist=alist+blist
    cdlist=clist+dlist
    lines=ablist+cdlist
    for line in lines:
        items=line.strip('\r\n').split('\t')
        for item in items:
            wordc=re.findall('(\w+):',item)
            if len(wordc)==0:
                continue
            else:
                word=wordc[0]
            freqc=re.findall(':(\d+)',item)
            if len(freqc)==0:
                continue
            else:
                freq=int(freqc[0])
            for i in range(freq):
                index=vocabulary.setdefault(word,len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))
        
vsmtf_common_words=csr_matrix((data,indices,indptr),dtype=int).toarray()
vsmtf_common_words.shape

def information_gain_ratio(X, y):
    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)
        tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)
        
    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain_ratio = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain_ratio.append(0)
            ig = _calIg()
            information_gain_ratio.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain_ratio.append(ig)
    return np.asarray(information_gain_ratio)

y_train=np.zeros((880,1))
for i in range(320):
    y_train[i]=1

y_test=np.zeros((223,1))
for i in range(84):
    y_test[i]=1

y0=np.vstack((y_train,y_test))
y=np.zeros(len(y0))
for i in range(len(y)):
    y[i]=y0[i][0]

igr=information_gain_ratio(vsmtf_common_words,y)
igrsorted=sorted(igr)
x_igrsorted=range(1,41624)

plt.plot(x_igrsorted,igrsorted,label='Information Gain Ratio(IGR)')
plt.xlabel('index of the feature')
plt.ylabel('information gain ratio of the individual feature')
plt.legend()
plt.show()

threshold=igrsorted[20000]
vsmtf_common_words_T=vsmtf_common_words.T
firstcol=0
for i in range(len(igr)):
    if igr[i]>threshold:
        firstcol=i
        break

vsmtf_common_words_T_igr_selected=vsmtf_common_words_T[firstcol]
for i in range(firstcol+1,len(igr)):
    if igr[i]>threshold:
        vsmtf_common_words_T_igr_selected=np.vstack((vsmtf_common_words_T_igr_selected,vsmtf_common_words_T[i]))

vsmtf_common_words_igr_selected=vsmtf_common_words_T_igr_selected.T
vsmtf_common_words_igr_selected.shape

def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    """ Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] """
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class
    for i in range(num_class):
        """ cal precision for each class: right / predict """
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]
        """ cal recall for each class: right / gold """
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]
        """ cal recall for each class: 2 pr / (p+r) """
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")
    """ PRF for each label or PRF for all labels """
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def test_prf(pred, labels):
    """
    4. log and return prf scores
    :return:
    """
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --")
    print("  ****** Neg|Neu|Pos ******")
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy))
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1))
    return accuracy
    
    
