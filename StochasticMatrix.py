
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
import math
############################################################################################################################################### evaluation function
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
############################################################################################################################################### build vector space model of legal terms
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')

indptr=[0]
indices=[]
data=[]
legal_terms_vocabulary={}
with codecs.open('CJO_word_cut_1103_K300_tf_legalterms.txt',encoding='utf-8',errors='ignore') as onetxt:
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
                index=legal_terms_vocabulary.setdefault(word,len(legal_terms_vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_legal_terms=csr_matrix((data,indices,indptr),dtype=int).toarray()
vsmtf_legal_terms.shape
############################################################################################################################################### build vector space model of common words
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')
indptr=[0]
indices=[]
data=[]
common_words_vocabulary={}
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
                index=common_words_vocabulary.setdefault(word,len(common_words_vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_common_words=csr_matrix((data,indices,indptr),dtype=int).toarray()
vsmtf_common_words.shape
###############################################################################################################################################the sum of features is 1154
x_sm_legalterms=vsmtf_legal_terms
x_train=x_sm_legalterms[0:880]
x_test=x_sm_legalterms[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1
y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_sm_legalterms=vsmtf_legal_terms
x_train=x_sm_legalterms[0:880]
x_validation_positive=x_sm_legalterms[880:922]
x_validation_negative=x_sm_legalterms[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation
y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1
y_test=y_validation
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_train=x_sm[0:880]
x_test_positive=x_sm[922:964]
x_test_negative=x_sm[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))
y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
############################################################################################################################################### the sum of features is 41623 
x_sm_commonwords=vsmtf_common_words
x_train=x_sm_commonwords[0:880]
x_test=x_sm_commonwords[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1
y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_commonwords_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_commonwords_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_validation_positive=x_sm_commonwords[880:922]
x_validation_negative=x_sm_commonwords[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation
y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1
y_test=y_validation
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_commonwords_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_commonwords_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_test_positive=x_sm_commonwords[922:964]
x_test_negative=x_sm_commonwords[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))
y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_commonwords_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_commonwords_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
############################################################################################################################################### the sum of features is 42777
x_sm=np.hstack((vsmtf_legal_terms,vsmtf_common_words))
x_train=x_sm[0:880]
x_test=x_sm[880:1103]
y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test
x_validation_positive=x_sm[880:922]
x_validation_negative=x_sm[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation
y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1
y_test=y_validation
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_test_positive=x_sm[922:964]
x_test_negative=x_sm[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))
y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
sm_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
sm_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
############################################################################################################################################### calculate the imbalance of each feature
vsmtf_common_words_positive=vsmtf_common_words[0:320]
vsmtf_common_words_negative=vsmtf_common_words[320:880]
vsmtf_common_words_positive_T=vsmtf_common_words_positive.T
vsmtf_common_words_negative_T=vsmtf_common_words_negative.T

vsmtf_common_words_positive_mean=[]
vsmtf_common_words_positive_var=[]
for i in range(len(vsmtf_common_words_positive_T)):
    vsmtf_common_words_positive_mean.append(np.mean(vsmtf_common_words_positive_T[i]))
    vsmtf_common_words_positive_var.append(np.mean(vsmtf_common_words_positive_T[i]))

vsmtf_common_words_negative_mean=[]
vsmtf_common_words_negative_var=[]
for i in range(len(vsmtf_common_words_negative_T)):
    vsmtf_common_words_negative_mean.append(np.mean(vsmtf_common_words_negative_T[i]))
    vsmtf_common_words_negative_var.append(np.mean(vsmtf_common_words_negative_T[i]))

imbalance=[]
for i in range(len(vsmtf_common_words_positive_T)):
    mean2=math.pow(abs(vsmtf_common_words_positive_mean[i]-vsmtf_common_words_negative_mean[i]),2)
    variance=abs(vsmtf_common_words_positive_var[i]-vsmtf_common_words_negative_var[i])
    imb=mean2+variance
    imbalance.append(imb)

for k,v in common_words_vocabulary.items():
    if v==0:
        print(k)

remove_word_list=[]
vsmtfcolsort=sorted(imbalance)
threshold=vsmtfcolsort[35000]
for i in range(len(imbalance)):
    if imbalance[i]<threshold:
        for k,v in common_words_vocabulary.items():
            if v==i:
                remove_word_list.append(k)

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')
legal_terms_list=[]
file_name_legalterms='./CJO_legal_terms.txt'
legal_terms_vocabulary={}
with codecs.open(file_name_legalterms,encoding='utf-8',errors='ignore') as onetxt:
    legalwords_lines=onetxt.readlines()
    for legalword in legalwords_lines:
        item=legalword.strip('\r\n').split('t')
        index=legal_terms_vocabulary.setdefault(item[0],len(legal_terms_vocabulary))

for item in legal_terms_vocabulary.items():
    legal_terms_list.append(item[0])

remove_word_vocabulary = {}
for word in remove_word_list:
    index = remove_word_vocabulary.setdefault(word, len(remove_word_vocabulary))
   
indptr = [0]
indices = []
data = []
common_words_tf_selected_vocabulary = {}
with codecs.open('CJO_word_cut_1103_K300_tf_commonwords.txt','r',encoding='utf-8',errors='ignore') as onetxt:
    lines_abcd_list=onetxt.readlines()
    alist=lines_abcd_list[0:320]
    blist=lines_abcd_list[404:964]
    ablist=alist+blist
    clist=lines_abcd_list[320:404]
    dlist=lines_abcd_list[964:1103]
    cdlist=clist+dlist
    lines=ablist+cdlist
    for line in lines:
        item=line.strip('\r\n').split('\t')
        for wordfreq in item:
            wordc=re.findall("(\w+):",wordfreq)
            if len(wordc)==0:
                continue
            word=wordc[0]
            if word not in remove_word_vocabulary:
                freqc=re.findall(":(\d+)",wordfreq)
            else:
                continue
            if len(freqc)==0:
                continue
            else:
                freq=int(freqc[0])
            for i in range(freq):
                index = common_words_tf_selected_vocabulary.setdefault(word, len(common_words_tf_selected_vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_commonwords_imbalance_selected=csr_matrix((data, indices, indptr), dtype=int).toarray()
vsmtf_commonwords_imbalance_selected.shape

############################################################################################################################################### the sum of features is 7779(1154+6625)

x_imb=np.hstack((vsmtf_legal_terms,vsmtf_commonwords_imbalance_selected))
x_train=x_imb[0:880]
x_test=x_imb[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1
y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
imb_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
imb_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_validation_positive=x_imb[880:922]
x_validation_negative=x_imb[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation
y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1
y_test=y_validation
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
imb_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
imb_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
x_test_positive=x_imb[922:964]
x_test_negative=x_imb[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))
y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
imb_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
imb_auc
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)


x_ir=vsmtf_legal_terms
x_train=x_ir[0:880]
x_test=x_ir[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## validation 112
x_pca=vsmtf_legal_terms

x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

y_test=y_validation
doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## test 111
x_pca=vsmtf_legal_terms

x_train=x_pca[0:880]
x_test_positive=x_pca[922:964]
x_test_negative=x_pca[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## test 223
x_pca=vsmtf_common_words
x_train=x_pca[0:880]
x_test=x_pca[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

x_pca=vsmtf_common_words

x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

y_test=y_validation


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## test 111
x_pca=vsmtf_common_words

x_train=x_pca[0:880]
x_test_positive=x_pca[922:964]
x_test_negative=x_pca[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

######################################################################################################## feature 42777,1154 legalterms+41623 commonwords
################################################################## test 223
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_common_words))
x_train=x_pca[0:880]
x_test=x_pca[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## validation 112
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_common_words))

x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

y_test=y_validation


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## test 111
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_common_words))

x_train=x_pca[0:880]
x_test_positive=x_pca[922:964]
x_test_negative=x_pca[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')
indptr=[0]
indices=[]
data=[]
common_words_vocabulary={}
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
                index=common_words_vocabulary.setdefault(word,len(common_words_vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_common_words=csr_matrix((data,indices,indptr),dtype=int).toarray()

vsmtf_common_words_positive=vsmtf_common_words[0:320]
vsmtf_common_words_negative=vsmtf_common_words[320:880]
vsmtf_common_words_positive_average=sum(vsmtf_common_words_positive)/320
vsmtf_common_words_negative_average=sum(vsmtf_common_words_negative)/560
vsmtf_common_words_cols_mse=abs(vsmtf_common_words_positive_average-vsmtf_common_words_negative_average)

for k,v in common_words_vocabulary.items():
    if v==0:
        print(k)

remove_word_list=[]
vsmtfcolsort=sorted(vsmtf_common_words_cols_mse)
threshold=vsmtfcolsort[35000]
for i in range(len(vsmtf_common_words_cols_mse)):
    if vsmtf_common_words_cols_mse[i]<threshold:
        for k,v in common_words_vocabulary.items():
            if v==i:
                remove_word_list.append(k)

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')
legal_terms_list=[]
file_name_legalterms='./CJO_legal_terms.txt'
legal_terms_vocabulary={}
with codecs.open(file_name_legalterms,encoding='utf-8',errors='ignore') as onetxt:
    legalwords_lines=onetxt.readlines()
    for legalword in legalwords_lines:
        item=legalword.strip('\r\n').split('t')
        index=legal_terms_vocabulary.setdefault(item[0],len(legal_terms_vocabulary))

for item in legal_terms_vocabulary.items():
    legal_terms_list.append(item[0])

remove_word_vocabulary = {}
for word in remove_word_list:
    index = remove_word_vocabulary.setdefault(word, len(remove_word_vocabulary))
   
indptr = [0]
indices = []
data = []
common_words_tf_selected_vocabulary = {}
with codecs.open('CJO_word_cut_1103_K300_tf_commonwords.txt','r',encoding='utf-8',errors='ignore') as onetxt:
    lines_abcd_list=onetxt.readlines()
    alist=lines_abcd_list[0:320]
    blist=lines_abcd_list[404:964]
    ablist=alist+blist
    clist=lines_abcd_list[320:404]
    dlist=lines_abcd_list[964:1103]
    cdlist=clist+dlist
    lines=ablist+cdlist
    for line in lines:
        item=line.strip('\r\n').split('\t')
        for wordfreq in item:
            wordc=re.findall("(\w+):",wordfreq)
            if len(wordc)==0:
                continue
            word=wordc[0]
            if word not in remove_word_vocabulary:
                freqc=re.findall(":(\d+)",wordfreq)
            else:
                continue
            if len(freqc)==0:
                continue
            else:
                freq=int(freqc[0])
            for i in range(freq):
                index = common_words_tf_selected_vocabulary.setdefault(word, len(common_words_tf_selected_vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_commonwords_tf_selected=csr_matrix((data, indices, indptr), dtype=int).toarray()
vsmtf_commonwords_tf_selected.shape

######################################################################################################## feature 7779,1154 legalterms+6625 tf_selected on 41623 commonwords
################################################################## test 223
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_commonwords_tf_selected))

x_train=x_pca[0:880]
x_test=x_pca[880:1103]
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## validation 112
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_commonwords_tf_selected))

x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
x_test=x_validation

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

y_test=y_validation


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

################################################################## test 111
x_pca=np.hstack((vsmtf_legal_terms,vsmtf_commonwords_tf_selected))

x_train=x_pca[0:880]
x_test_positive=x_pca[922:964]
x_test_negative=x_pca[1034:1103]
x_test=np.vstack((x_test_positive,x_test_negative))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(111)
for i in range(42):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test
linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

