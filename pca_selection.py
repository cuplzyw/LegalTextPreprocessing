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

##################################################### evaluation function
def cal_prf(pred, right, gold, formation=True, metric_type=""):
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

##################################################### build vector space model of legal terms
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO')

indptr=[0]
indices=[]
data=[]
vocabulary={}
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
                index=vocabulary.setdefault(word,len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

vsmtf_legal_terms=csr_matrix((data,indices,indptr),dtype=int).toarray()

##################################################### build vector space model of common words

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

##################################################### calculate accurancy,auc and matrix of confuse when number of pricipal components is from 2 to 1102
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO/fig/PCA')
fp_pcsoncommonwords_confusematrix_list_noproba=open('pcsoncommonwords_2to1102_confusematrix_list_noproba.txt',mode='w',encoding='utf-8')
pcsoncommonwords_acc_list_noproba=[]
fp_pcsoncommonwords_confusematrix_list_proba=open('pcsoncommonwords_2to1102_confusematrix_list_proba.txt',mode='w',encoding='utf-8')
pcsoncommonwords_acc_list_proba=[]
pcsoncommonwords_auc_list=[]
for pcs in range(2,1103):
    estimator = PCA(n_components=pcs,svd_solver='arpack')
    x_pca=estimator.fit_transform(vsmtf_common_words)    
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
    total = len(y_test)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[y_predict[i]] += 1
        if y_predict[i] == y_test[i]:
            pred_right[y_predict[i]] += 1
        gold[y_test[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --",file=fp_pcsoncommonwords_confusematrix_list_noproba)
    print("  ****** Neg|Neu|Pos ******",file=fp_pcsoncommonwords_confusematrix_list_noproba)
    accuracy = 1.0 * sum(pred_right) / total
    pcsoncommonwords_acc_list_noproba.append(accuracy)
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy),file=fp_pcsoncommonwords_confusematrix_list_noproba)
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1),file=fp_pcsoncommonwords_confusematrix_list_noproba)
    linear_svc = SVC(kernel="linear",probability=True)
    linear_svc.fit(doc_term_matrix_train_vec, y_train)
    #### pause    
    y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
    y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
    for i in range(len(y_predict_proba_1_vec)):
        y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]
    ##### pause
    pcs_auc=roc_auc_score(y_test, y_predict_proba_1_vec)
    pcsoncommonwords_auc_list.append(pcs_auc)   
    y_predict=np.zeros(len(y_predict_proba_1_vec))
    for i in range(len(y_predict)):
        if y_predict_proba_1_vec[i][0]>0.5:
            y_predict[i]=1
    y_predict=y_predict.astype(np.int64)
    y_test=y_test.astype(np.int64)
    total = len(y_test)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[y_predict[i]] += 1
        if y_predict[i] == y_test[i]:
            pred_right[y_predict[i]] += 1
        gold[y_test[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --",file=fp_pcsoncommonwords_confusematrix_list_proba)
    print("  ****** Neg|Neu|Pos ******",file=fp_pcsoncommonwords_confusematrix_list_proba)
    accuracy = 1.0 * sum(pred_right) / total
    pcsoncommonwords_acc_list_proba.append(accuracy)
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy),file=fp_pcsoncommonwords_confusematrix_list_proba)
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1),file=fp_pcsoncommonwords_confusematrix_list_proba)

fp_pcsoncommonwords_confusematrix_list_noproba.flush()
fp_pcsoncommonwords_confusematrix_list_noproba.close()
fp_pcsoncommonwords_confusematrix_list_proba.flush()
fp_pcsoncommonwords_confusematrix_list_proba.close()

fp_pcsoncommonwords_acc_list_noproba=open('pcsoncommonwords_acc_list_noproba.txt',mode='w',encoding='utf-8')
for i in range(len(pcsoncommonwords_acc_list_noproba)):
    st=str(pcsoncommonwords_acc_list_noproba[i])
    fp_pcsoncommonwords_acc_list_noproba.write(st+'\r\n')

fp_pcsoncommonwords_acc_list_noproba.flush()
fp_pcsoncommonwords_acc_list_noproba.close()
##################################################### when pcs(n_components) is 796
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca=estimator.fit_transform(vsmtf_common_words)    
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################

x=x_train.T[0:121]
y=x_train.T[122:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:121]
yt=x_test.T[122:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
##########################################################################################################
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca=estimator.fit_transform(vsmtf_common_words)    
x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

x_test=x_validation
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################

x=x_train.T[0:121]
y=x_train.T[122:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:121]
yt=x_test.T[122:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
##########################################################################################################
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca=estimator.fit_transform(vsmtf_common_words)    
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################

x=x_train.T[0:121]
y=x_train.T[122:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:121]
yt=x_test.T[122:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

########################################################################################################## including legal terms(1154) and top pcs(796)
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca_oncommonwords=estimator.fit_transform(vsmtf_common_words)   
x_pca=np.hstack((vsmtf_legal_terms,x_pca_oncommonwords))
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################

x=x_train.T[0:1275]
y=x_train.T[1276:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:1275]
yt=x_test.T[1276:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

##########################################################################################################
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca_oncommonwords=estimator.fit_transform(vsmtf_common_words)   
x_pca=np.hstack((vsmtf_legal_terms,x_pca_oncommonwords))
x_train=x_pca[0:880]
x_validation_positive=x_pca[880:922]
x_validation_negative=x_pca[964:1034]
x_validation=np.vstack((x_validation_positive,x_validation_negative))
y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_validation=np.zeros(112)
for i in range(42):
    y_validation[i]=1

x_test=x_validation
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

pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################

x=x_train.T[0:1275]
y=x_train.T[1276:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:1275]
yt=x_test.T[1276:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

##########################################################################################################
estimator = PCA(n_components=796,svd_solver='arpack')
x_pca_oncommonwords=estimator.fit_transform(vsmtf_common_words)   
x_pca=np.hstack((vsmtf_legal_terms,x_pca_oncommonwords))
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

#####################################################
x=x_train.T[0:1275]
y=x_train.T[1276:]
xy=np.vstack((x,y))
x_train_delete_pcs=xy.T

xt=x_test.T[0:1275]
yt=x_test.T[1276:]
xyt=np.vstack((xt,yt))
x_test_delete_pcs=xyt.T

doc_term_matrix_train_vec=x_train_delete_pcs
doc_term_matrix_test_vec=x_test_delete_pcs
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

pca_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pca_auc

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i][0]>0.5:
        y_predict[i]=1

y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

