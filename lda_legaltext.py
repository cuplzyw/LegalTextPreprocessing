import os
import sys
import codecs
import re
import scipy.sparse
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve,roc_auc_score)
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from slda.topic_models import LDA
from sklearn.linear_model import LinearRegression

###############################################################################################################################################################
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
###############################################################################################################################################################
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

x_train=vsmtf_legal_terms[0:880]
x_test=vsmtf_legal_terms[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

###############################################################################################################################################################
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO/fig/forpaper/LDA')

V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

K_LDA_lr_acc_list=[]
K_LDA_lr_auc_list=[]
fp_K_LDA_lr_confusematrix_list=open('K_LDA_lr_confusematrix_list.txt',mode='w',encoding='utf-8')
for K in range(2,1154):
    alpha=np.repeat(1.,K)
    lda=LDA(K,alpha,beta,n_iter,seed=42)    
    lda.fit(doc_term_matrix_train_vec)
    theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
    lr=LinearRegression(fit_intercept=False)
    lr.fit(lda.theta,y_train)
    y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
    pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
    K_LDA_lr_auc_list.append(pcs_auc)
    y_predict=np.zeros(len(y_predict_proba_1_vec))
    for i in range(len(y_predict)):   
        if y_predict_proba_1_vec[i]>0.5:
            y_predict[i]=1
    y_test=y_test.astype(np.int64)
    y_predict=y_predict.astype(np.int64)    
    total = len(y_test)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[y_predict[i]] += 1
        if y_predict[i] == y_test[i]:
            pred_right[y_predict[i]] += 1
        gold[y_test[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --",file=fp_K_LDA_lr_confusematrix_list)
    print("  ****** Neg|Neu|Pos ******",file=fp_K_LDA_lr_confusematrix_list)
    accuracy = 1.0 * sum(pred_right) / total
    K_LDA_lr_acc_list.append(accuracy)
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy),file=fp_K_LDA_lr_confusematrix_list)
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1),file=fp_K_LDA_lr_confusematrix_list)

fp_K_LDA_lr_confusematrix_list.flush()
fp_K_LDA_lr_confusematrix_list.close()

fp_K_LDA_lr_acc_list=open('K_LDA_lr_acc_list.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_acc_list)):
    st=str(K_LDA_lr_acc_list[i])
    fp_K_LDA_lr_acc_list.write(st+'\r\n')

fp_K_LDA_lr_acc_list.flush()
fp_K_LDA_lr_acc_list.close()


fp_K_LDA_lr_auc_list=open('K_LDA_lr_auc_list.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_auc_list)):
    st=str(K_LDA_lr_auc_list[i])
    fp_K_LDA_lr_auc_list.write(st+'\r\n')

fp_K_LDA_lr_auc_list.flush()
fp_K_LDA_lr_auc_list.close()
max(K_LDA_lr_acc_list)
maxi=0
for i in range(len(K_LDA_lr_acc_list)):
    if K_LDA_lr_acc_list[i]==max(K_LDA_lr_acc_list):
        maxi=i
        break

maxi
max(K_LDA_lr_auc_list)
maxi=0
for i in range(len(K_LDA_lr_auc_list)):
    if K_LDA_lr_auc_list[i]==max(K_LDA_lr_auc_list):
        maxi=i
        break
x_range=range(2,1154)
plt.plot(x_range,K_LDA_lr_acc_list,label='Accurancy Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Accurancy')
plt.legend()
plt.show()
x_range=range(2,1154)
plt.plot(x_range,K_LDA_lr_auc_list,label='AUC Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Area Under Curve')
plt.legend()
plt.show()


###############################################################################################################################################################
K=356
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)

pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

###############################################################################################################################################################
vsmtf_phi_word_samples=lda.phi.T
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(800):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)
linear_svc.coef_.shape
linear_svc.intercept_[0]

x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))


x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

###############################################################################################################################################################
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

x_train=vsmtf_common_words[0:880]
x_test=vsmtf_common_words[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

###############################################################################################################################################################
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO/fig/forpaper/LDA')

V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

K_LDA_lr_acc_list=[]
K_LDA_lr_auc_list=[]
fp_K_LDA_lr_confusematrix_list=open('K_LDA_lr_confusematrix_list_common_words.txt',mode='w',encoding='utf-8')
for K in range(2,V):
    alpha=np.repeat(1.,K)
    lda=LDA(K,alpha,beta,n_iter,seed=42)    
    lda.fit(doc_term_matrix_train_vec)
    theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
    lr=LinearRegression(fit_intercept=False)
    lr.fit(lda.theta,y_train)
    y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
    pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
    K_LDA_lr_auc_list.append(pcs_auc)
    y_predict=np.zeros(len(y_predict_proba_1_vec))
    for i in range(len(y_predict)):   
        if y_predict_proba_1_vec[i]>0.5:
            y_predict[i]=1
    y_test=y_test.astype(np.int64)
    y_predict=y_predict.astype(np.int64)    
    total = len(y_test)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[y_predict[i]] += 1
        if y_predict[i] == y_test[i]:
            pred_right[y_predict[i]] += 1
        gold[y_test[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --",file=fp_K_LDA_lr_confusematrix_list)
    print("  ****** Neg|Neu|Pos ******",file=fp_K_LDA_lr_confusematrix_list)
    accuracy = 1.0 * sum(pred_right) / total
    K_LDA_lr_acc_list.append(accuracy)
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy),file=fp_K_LDA_lr_confusematrix_list)
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1),file=fp_K_LDA_lr_confusematrix_list)

fp_K_LDA_lr_confusematrix_list.flush()
fp_K_LDA_lr_confusematrix_list.close()

fp_K_LDA_lr_acc_list=open('K_LDA_lr_acc_list_common_words.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_acc_list)):
    st=str(K_LDA_lr_acc_list[i])
    fp_K_LDA_lr_acc_list.write(st+'\r\n')

fp_K_LDA_lr_acc_list.flush()
fp_K_LDA_lr_acc_list.close()


fp_K_LDA_lr_auc_list=open('K_LDA_lr_auc_list_common_words.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_auc_list)):
    st=str(K_LDA_lr_auc_list[i])
    fp_K_LDA_lr_auc_list.write(st+'\r\n')

fp_K_LDA_lr_auc_list.flush()
fp_K_LDA_lr_auc_list.close()
max(K_LDA_lr_acc_list)
maxi=0
for i in range(len(K_LDA_lr_acc_list)):
    if K_LDA_lr_acc_list[i]==max(K_LDA_lr_acc_list):
        maxi=i
        break

maxi
max(K_LDA_lr_auc_list)
maxi=0
for i in range(len(K_LDA_lr_auc_list)):
    if K_LDA_lr_auc_list[i]==max(K_LDA_lr_auc_list):
        maxi=i
        break
x_range=range(2,V)
plt.plot(x_range,K_LDA_lr_acc_list,label='Accurancy Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Accurancy')
plt.legend()
plt.show()
x_range=range(2,V)
plt.plot(x_range,K_LDA_lr_auc_list,label='AUC Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Area Under Curve')
plt.legend()
plt.show()

###############################################################################################################################################################
x_train=vsmtf_common_words[0:880]
x_test=vsmtf_common_words[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=250
alpha=np.repeat(1.,K)
V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)

pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
lda.theta.shape
theta_test_lda_vec.shape
lda.phi.shape
###############################################################################################################################################################
vsmtf_phi_word_samples=lda.phi.T
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(20000):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)
linear_svc.coef_.shape
linear_svc.intercept_[0]

x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))


x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)


import os
import sys
import codecs
import re
import scipy.sparse
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve,roc_auc_score)
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from slda.topic_models import LDA
from sklearn.linear_model import LinearRegression

###############################################################################################################################################################
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
###############################################################################################################################################################

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

x_train=vsmtf_legal_terms[0:880]
x_test=vsmtf_legal_terms[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=15
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)

linear_svc=SVC(kernel='linear')
linear_svc.fit(lda.theta,y_train)
y_predict=linear_svc.predict(theta_test_lda_vec)
y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
####################################### K=15
  Prediction: [56, 167]  Right: [52, 80]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 132/223 = 0.591928
    Precision: [0.9285714285714286, 0.47904191616766467]
    Recall   : [0.37410071942446044, 0.9523809523809523]
    F1 score : [0.5333333333333333, 0.6374501992031872]
    Macro F1 score on test (Neg|Neu|Pos) is 0.682922
####################################### K=50
 Prediction: [76, 147]  Right: [73, 81]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 154/223 = 0.690583
    Precision: [0.9605263157894737, 0.5510204081632653]
    Recall   : [0.5251798561151079, 0.9642857142857143]
    F1 score : [0.6790697674418604, 0.7012987012987012]
    Macro F1 score on test (Neg|Neu|Pos) is 0.750212


linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(lda.theta,y_train)
y_predict_proba_vec=linear_svc.predict_proba(theta_test_lda_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    if y_predict_proba_vec[i][1]>0.5:
        y_predict_proba_1_vec[i]=1

pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
####################################### K=15
0.7185680027406647
####################################### K=50
0.7365536142514559
###############################################################################################################################################################

lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)

pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
####################################### K=15
0.789397053785543
####################################### K=50
0.8196300102774924
####################################### K=354+2=356
0.8524323398424118

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
####################################### K=15
  Prediction: [76, 147]  Right: [69, 77]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 146/223 = 0.654709
    Precision: [0.9078947368421053, 0.5238095238095238]
    Recall   : [0.49640287769784175, 0.9166666666666666]
    F1 score : [0.641860465116279, 0.6666666666666667]
    Macro F1 score on test (Neg|Neu|Pos) is 0.711163
####################################### K=50
  Prediction: [80, 143]  Right: [76, 80]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 156/223 = 0.699552
    Precision: [0.95, 0.5594405594405595]
    Recall   : [0.5467625899280576, 0.9523809523809523]
    F1 score : [0.6940639269406393, 0.7048458149779737]
    Macro F1 score on test (Neg|Neu|Pos) is 0.752137
####################################### K=354+2=356
  Prediction: [101, 122]  Right: [92, 75]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 167/223 = 0.748879
    Precision: [0.9108910891089109, 0.6147540983606558]
    Recall   : [0.6618705035971223, 0.8928571428571429]
    F1 score : [0.7666666666666666, 0.7281553398058253]
    Macro F1 score on test (Neg|Neu|Pos) is 0.770025

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO/fig/forpaper/LDA')

V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

K_LDA_lr_acc_list=[]
K_LDA_lr_auc_list=[]
fp_K_LDA_lr_confusematrix_list=open('K_LDA_lr_confusematrix_list.txt',mode='w',encoding='utf-8')
for K in range(2,1154):
    alpha=np.repeat(1.,K)
    lda=LDA(K,alpha,beta,n_iter,seed=42)    
    lda.fit(doc_term_matrix_train_vec)
    theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
    lr=LinearRegression(fit_intercept=False)
    lr.fit(lda.theta,y_train)
    y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
    pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
    K_LDA_lr_auc_list.append(pcs_auc)
    y_predict=np.zeros(len(y_predict_proba_1_vec))
    for i in range(len(y_predict)):   
        if y_predict_proba_1_vec[i]>0.5:
            y_predict[i]=1
    y_test=y_test.astype(np.int64)
    y_predict=y_predict.astype(np.int64)    
    total = len(y_test)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[y_predict[i]] += 1
        if y_predict[i] == y_test[i]:
            pred_right[y_predict[i]] += 1
        gold[y_test[i]] += 1
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold," -- for all labels --",file=fp_K_LDA_lr_confusematrix_list)
    print("  ****** Neg|Neu|Pos ******",file=fp_K_LDA_lr_confusematrix_list)
    accuracy = 1.0 * sum(pred_right) / total
    K_LDA_lr_acc_list.append(accuracy)
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print( "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy),file=fp_K_LDA_lr_confusematrix_list)
    print( "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" %(p, r, f1, macro_f1),file=fp_K_LDA_lr_confusematrix_list)

fp_K_LDA_lr_confusematrix_list.flush()
fp_K_LDA_lr_confusematrix_list.close()

######################################

fp_K_LDA_lr_acc_list=open('K_LDA_lr_acc_list.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_acc_list)):
    st=str(K_LDA_lr_acc_list[i])
    fp_K_LDA_lr_acc_list.write(st+'\r\n')

fp_K_LDA_lr_acc_list.flush()
fp_K_LDA_lr_acc_list.close()

######################################

fp_K_LDA_lr_auc_list=open('K_LDA_lr_auc_list.txt',mode='w',encoding='utf-8')
for i in range(len(K_LDA_lr_auc_list)):
    st=str(K_LDA_lr_auc_list[i])
    fp_K_LDA_lr_auc_list.write(st+'\r\n')

fp_K_LDA_lr_auc_list.flush()
fp_K_LDA_lr_auc_list.close()

######################################
######################################
max(K_LDA_lr_acc_list)
0.7488789237668162
maxi=0
for i in range(len(K_LDA_lr_acc_list)):
    if K_LDA_lr_acc_list[i]==max(K_LDA_lr_acc_list):
        maxi=i
        break

maxi
354
K_LDA_lr_acc_list[373]
0.6816143497757847
######################################
max(K_LDA_lr_auc_list)
0.8636519355943816
maxi=0
for i in range(len(K_LDA_lr_auc_list)):
    if K_LDA_lr_auc_list[i]==max(K_LDA_lr_auc_list):
        maxi=i
        break

maxi
373
K_LDA_lr_auc_list[354]
0.8524323398424118
######################################
x_range=range(2,1154)
plt.plot(x_range,K_LDA_lr_acc_list,label='Accurancy Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Accurancy')
plt.legend()
plt.show()
######################################
x_range=range(2,1154)
plt.plot(x_range,K_LDA_lr_auc_list,label='AUC Under Top K Topics')
plt.xlabel('Top K Topics')
plt.ylabel('Area Under Curve')
plt.legend()
plt.show()

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

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

x_train=vsmtf_common_words[0:880]
x_test=vsmtf_common_words[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

######################################
K=15
alpha=np.repeat(1.,K)
V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
######################################
lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)

############################################################################
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
####################################### K=15,commonwords
0.5803357314148682
####################################### K=50,commonwords
0.6647824597464885
####################################### K=100,commonwords
0.7945357999314834
####################################### K=150,commonwords
0.7724391915039396
####################################### K=200,commonwords
0.7778348749571771
####################################### K=250,commonwords
0.7878554299417608
####################################### K=300,commonwords
0.8013874614594039
####################################### K=350,commonwords
0.7922233641658102
####################################### K=400,commonwords
0.8006166495375129
####################################### K=450,commonwords
0.803100376841384
####################################### K=500,commonwords
0.7525693730729702
####################################### K=550,commonwords
0.7658444672833162
####################################### K=600,commonwords
0.657416923603974
####################################### K=1000,commonwords
0.6789140116478246

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
####################################### K=15,commonwords
  Prediction: [148, 75]  Right: [100, 36]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 136/223 = 0.609865
    Precision: [0.6756756756756757, 0.48]
    Recall   : [0.7194244604316546, 0.42857142857142855]
    F1 score : [0.6968641114982578, 0.4528301886792452]
    Macro F1 score on test (Neg|Neu|Pos) is 0.575911
####################################### K=50,commonwords
  Prediction: [101, 122]  Right: [77, 60]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 137/223 = 0.614350
    Precision: [0.7623762376237624, 0.4918032786885246]
    Recall   : [0.5539568345323741, 0.7142857142857143]
    F1 score : [0.6416666666666667, 0.5825242718446602]
    Macro F1 score on test (Neg|Neu|Pos) is 0.630586
####################################### K=100,commonwords
  Prediction: [119, 104]  Right: [94, 59]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 153/223 = 0.686099
    Precision: [0.7899159663865546, 0.5673076923076923]
    Recall   : [0.6762589928057554, 0.7023809523809523]
    F1 score : [0.7286821705426356, 0.6276595744680851]
    Macro F1 score on test (Neg|Neu|Pos) is 0.683924
####################################### K=150,commonwords
  Prediction: [109, 114]  Right: [88, 63]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 151/223 = 0.677130
    Precision: [0.8073394495412844, 0.5526315789473685]
    Recall   : [0.6330935251798561, 0.75]
    F1 score : [0.7096774193548386, 0.6363636363636364]
    Macro F1 score on test (Neg|Neu|Pos) is 0.685717
####################################### K=200,commonwords
  Prediction: [101, 122]  Right: [87, 70]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 157/223 = 0.704036
    Precision: [0.8613861386138614, 0.5737704918032787]
    Recall   : [0.6258992805755396, 0.8333333333333334]
    F1 score : [0.7249999999999999, 0.6796116504854368]
    Macro F1 score on test (Neg|Neu|Pos) is 0.723547
####################################### K=250,commonwords
  Prediction: [129, 94]  Right: [106, 61]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 167/223 = 0.748879
    Precision: [0.8217054263565892, 0.648936170212766]
    Recall   : [0.762589928057554, 0.7261904761904762]
    F1 score : [0.791044776119403, 0.6853932584269663]
    Macro F1 score on test (Neg|Neu|Pos) is 0.739828
####################################### K=300,commonword
  Prediction: [103, 120]  Right: [92, 73]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 165/223 = 0.739910
    Precision: [0.8932038834951457, 0.6083333333333333]
    Recall   : [0.6618705035971223, 0.8690476190476191]
    F1 score : [0.7603305785123967, 0.7156862745098039]
    Macro F1 score on test (Neg|Neu|Pos) is 0.758043
####################################### K=350,commonword
  Prediction: [100, 123]  Right: [87, 71]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 158/223 = 0.708520
    Precision: [0.87, 0.5772357723577236]
    Recall   : [0.6258992805755396, 0.8452380952380952]
    F1 score : [0.7280334728033473, 0.6859903381642513]
    Macro F1 score on test (Neg|Neu|Pos) is 0.729544
####################################### K=400,commonword
  Prediction: [114, 109]  Right: [96, 66]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 162/223 = 0.726457
    Precision: [0.8421052631578947, 0.6055045871559633]
    Recall   : [0.6906474820143885, 0.7857142857142857]
    F1 score : [0.758893280632411, 0.6839378238341969]
    Macro F1 score on test (Neg|Neu|Pos) is 0.730922
####################################### K=450,commonword
  Prediction: [109, 114]  Right: [93, 68]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 161/223 = 0.721973
    Precision: [0.8532110091743119, 0.5964912280701754]
    Recall   : [0.6690647482014388, 0.8095238095238095]
    F1 score : [0.7500000000000001, 0.6868686868686869]
    Macro F1 score on test (Neg|Neu|Pos) is 0.732001
####################################### K=500,commonword
  Prediction: [103, 120]  Right: [82, 63]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 145/223 = 0.650224
    Precision: [0.7961165048543689, 0.525]
    Recall   : [0.5899280575539568, 0.75]
    F1 score : [0.6776859504132231, 0.6176470588235295]
    Macro F1 score on test (Neg|Neu|Pos) is 0.665228
####################################### K=550,commonword
  Prediction: [102, 121]  Right: [83, 65]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 148/223 = 0.663677
    Precision: [0.8137254901960784, 0.5371900826446281]
    Recall   : [0.5971223021582733, 0.7738095238095238]
    F1 score : [0.6887966804979252, 0.6341463414634146]
    Macro F1 score on test (Neg|Neu|Pos) is 0.680425
####################################### K=600,commonword
  Prediction: [77, 146]  Right: [59, 66]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 125/223 = 0.560538
    Precision: [0.7662337662337663, 0.4520547945205479]
    Recall   : [0.4244604316546763, 0.7857142857142857]
    F1 score : [0.5462962962962963, 0.5739130434782609]
    Macro F1 score on test (Neg|Neu|Pos) is 0.607109
####################################### K=1000,commonword
  Prediction: [71, 152]  Right: [61, 74]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 135/223 = 0.605381
    Precision: [0.8591549295774648, 0.4868421052631579]
    Recall   : [0.43884892086330934, 0.8809523809523809]
    F1 score : [0.5809523809523809, 0.6271186440677966]
    Macro F1 score on test (Neg|Neu|Pos) is 0.666385


###############################################################################################################################################################
###############################################################################################################################################################
############################################################################################################################################# vsmtf_legal_terms
x_train=vsmtf_legal_terms[0:880]
x_test=vsmtf_legal_terms[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=356
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
######################################
lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8524323398424118

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#0.748879

lda.theta.shape
#(880, 356)
theta_test_lda_vec.shape
#(223, 356)
lda.phi.shape
#(356, 1154)

vsmtf_phi_word_samples=lda.phi.T
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(500):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)
linear_svc.coef_.shape
#(1, 356)
linear_svc.intercept_[0]
#-1.0024206198097196

###################################### same as the above 
x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))


x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8432682425488182
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

  Prediction: [100, 123]  Right: [91, 75]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 166/223 = 0.744395
    Precision: [0.91, 0.6097560975609756]
    Recall   : [0.6546762589928058, 0.8928571428571429]
    F1 score : [0.7615062761506276, 0.7246376811594203]
    Macro F1 score on test (Neg|Neu|Pos) is 0.766759
###################################### renormalize
vsmtf_phi_word_samples=lda.phi.T
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(800):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)

x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))

x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
############# for i in range(400):
#############     y_train[i]=1
0.843268242548818
############# for i in range(800):
#############     y_train[i]=1
0.8325625214114423

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
############# for i in range(400):
#############     y_train[i]=1
  Prediction: [100, 123]  Right: [90, 74]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 164/223 = 0.735426
    Precision: [0.9, 0.6016260162601627]
    Recall   : [0.6474820143884892, 0.8809523809523809]
    F1 score : [0.7531380753138076, 0.7149758454106281]
    Macro F1 score on test (Neg|Neu|Pos) is 0.757456
############# for i in range(800):
#############     y_train[i]=1
  Prediction: [119, 104]  Right: [103, 68]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 171/223 = 0.766816
    Precision: [0.865546218487395, 0.6538461538461539]
    Recall   : [0.7410071942446043, 0.8095238095238095]
    F1 score : [0.7984496124031009, 0.7234042553191489]
    Macro F1 score on test (Neg|Neu|Pos) is 0.767402

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
########################################################################################################################################### vsmtf_common_words
x_train=vsmtf_common_words[0:880]
x_test=vsmtf_common_words[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=250
alpha=np.repeat(1.,K)
V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda=LDA(K,alpha,beta,n_iter=100,seed=42)
lda.fit(doc_term_matrix_train_vec)
theta_test_lda_vec=lda.transform(doc_term_matrix_test_vec)
######################################
lr=LinearRegression(fit_intercept=False)
lr.fit(lda.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.7878554299417608

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#0.748879

lda.theta.shape
#(880, 250)
theta_test_lda_vec.shape
#(223, 250)
lda.phi.shape
#(250, 41623)

vsmtf_phi_word_samples=lda.phi.T
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(15000):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)
linear_svc.coef_.shape
#(1, 250)
linear_svc.intercept_[0]
#-1.0008647563633617

###################################### Result same as the above
x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))


x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.7880267214799588
y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

  Prediction: [121, 102]  Right: [102, 65]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 167/223 = 0.748879
    Precision: [0.8429752066115702, 0.6372549019607843]
    Recall   : [0.7338129496402878, 0.7738095238095238]
    F1 score : [0.7846153846153846, 0.6989247311827957]
    Macro F1 score on test (Neg|Neu|Pos) is 0.746900

###################################### renormalize
x_train=vsmtf_phi_word_samples
y_train=np.zeros(len(x_train))
for i in range(20000):
    y_train[i]=1

linear_svc=SVC(kernel='linear')
linear_svc.fit(x_train,y_train)

x_train_weight=np.zeros((len(lda.theta),len(lda.theta[1])))
x_train_weight_normalize=np.zeros((len(lda.theta),len(lda.theta[1])))
for i in range(len(lda.theta)):
    for j in range(len(lda.theta[i])):
        x_train_weight[i][j]=lda.theta[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_train_weight)):
    for j in range(len(x_train_weight[i])):
        x_train_weight_normalize[i][j]=x_train_weight[i][j]/(sum(x_train_weight[i]))

x_test_weight=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
x_test_weight_normalize=np.zeros((len(theta_test_lda_vec),len(theta_test_lda_vec[1])))
for i in range(len(theta_test_lda_vec)):
    for j in range(len(theta_test_lda_vec[i])):
        x_test_weight[i][j]=theta_test_lda_vec[i][j]*linear_svc.coef_[0][j]

for i in range(len(x_test_weight)):
    for j in range(len(x_test_weight[i])):
        x_test_weight_normalize[i][j]=x_test_weight[i][j]/(sum(x_test_weight[i]))

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1

lr=LinearRegression(fit_intercept=False)
lr.fit(x_train_weight_normalize,y_train)
y_predict_proba_1_vec=lr.predict(x_test_weight_normalize)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
############# for i in range(15000):
#############     y_train[i]=1
0.7714114422747516
############# for i in range(20000):
#############     y_train[i]=1
0.7817745803357313
############# for i in range(25000):
0.6928742720109626

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.5:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
############# for i in range(15000):
#############     y_train[i]=1
  Prediction: [130, 93]  Right: [103, 57]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 160/223 = 0.717489
    Precision: [0.7923076923076923, 0.6129032258064516]
    Recall   : [0.7410071942446043, 0.6785714285714286]
    F1 score : [0.7657992565055761, 0.6440677966101694]
    Macro F1 score on test (Neg|Neu|Pos) is 0.706179
############# for i in range(20000):
#############     y_train[i]=1
  Prediction: [120, 103]  Right: [101, 65]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 166/223 = 0.744395
    Precision: [0.8416666666666667, 0.6310679611650486]
    Recall   : [0.7266187050359713, 0.7738095238095238]
    F1 score : [0.77992277992278, 0.6951871657754012]
    Macro F1 score on test (Neg|Neu|Pos) is 0.743226
############# for i in range(25000):
  Prediction: [107, 116]  Right: [81, 58]  Gold: [139, 84]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 139/223 = 0.623318
    Precision: [0.7570093457943925, 0.5]
    Recall   : [0.5827338129496403, 0.6904761904761905]
    F1 score : [0.6585365853658536, 0.58]
    Macro F1 score on test (Neg|Neu|Pos) is 0.632529
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
########################################################################################################################################### vsmtf_legal_terms
########################################################################################################################################### vsmtf_common_words
x_train=vsmtf_legal_terms[0:880]
x_test=vsmtf_legal_terms[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=356
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_legal_terms=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_legal_terms.fit(doc_term_matrix_train_vec)
theta_test_lda_legal_terms_vec=lda_legal_terms.transform(doc_term_matrix_test_vec)
######################################
x_train=vsmtf_common_words[0:880]
x_test=vsmtf_common_words[880:1103]

y_train=np.zeros(880)
for i in range(320):
    y_train[i]=1

y_test=np.zeros(223)
for i in range(84):
    y_test[i]=1


doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=250
alpha=np.repeat(1.,K)
V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_common_words=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_common_words.fit(doc_term_matrix_train_vec)
theta_test_lda_common_words_vec=lda_common_words.transform(doc_term_matrix_test_vec)
######################################
lda_theta=np.hstack((lda_legal_terms.theta,lda_common_words.theta))
theta_test_lda_vec=np.hstack((theta_test_lda_legal_terms_vec,theta_test_lda_common_words_vec))
lr=LinearRegression(fit_intercept=False)
lr.fit(lda_theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
0.5295477903391572
######################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
for i in range(10):
    for k,v in legal_terms_vocabulary.items():
        if v==i:
            print(k)

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_CJO/fig/forpaper/LDA')
fp_legal_terms_topic_top20words=open('legal_terms_topic_top20words.txt',mode='w',encoding='utf-8')
for i in range(len(lda_legal_terms.phi)):
    lis=lda_legal_terms.phi[i]
    topic_N_top_words_index=sorted(range(len(lis)),key=lambda k:lis[k],reverse=True)
    topic_N_top20_words=[]
    for j in range(20):
        index=topic_N_top_words_index[j]
        for k,v in legal_terms_vocabulary.items():
            if v==index:
                topic_N_top20_words.append(k)
    for word in topic_N_top20_words:
        fp_legal_terms_topic_top20words.write(word+'\t')
    fp_legal_terms_topic_top20words.write('\r\n')

fp_legal_terms_topic_top20words.flush()
fp_legal_terms_topic_top20words.close()

linear_svc.coef_.shape
#(1,356)
linear_svc.intercept_[0]
#0.9956329058704136
fp_legal_terms_words_topics_weight=open('legal_terms_words_topics_weight.txt',mode='w',encoding='utf-8')
for i in range(len(linear_svc.coef_[0])):
    st=str(linear_svc.coef_[0][i])
    fp_legal_terms_words_topics_weight.write(st+'\r\n')

fp_legal_terms_words_topics_weight.flush()
fp_legal_terms_words_topics_weight.close()

######################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
from sklearn.metrics.pairwise import cosine_similarity
def cos_sim(vector_a,vector_b):
    vector_a=np.mat(vector_a)
    vector_b=np.mat(vector_b)
    num=float(vector_a*vector_b.T)
    denom=np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    sim=num/denom
    return sim

lda.theta.shape
theta_test_lda_vec.shape
x_train_weight_normalize.shape
x_test_weight_normalize.shape

doc_sim_original=[]
for i in range(len(lda.theta)):
    st=cos_sim(lda.theta[i],theta_test_lda_vec[222])
    doc_sim_original.append(st)

x_doc_sim=range(1,len(lda.theta)+1)
plt.plot(x_doc_sim,doc_sim_original,label='cosine similarity between two documents')
plt.xlabel('number of the document from the traindata')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()


doc_sim_weighted=[]
for i in range(len(x_train_weight_normalize)):
    st=cos_sim(x_train_weight_normalize[i],x_test_weight_normalize[222])
    doc_sim_weighted.append(st)

x_doc_sim=range(1,len(lda.theta)+1)
plt.plot(x_doc_sim,doc_sim_weighted,label='cosine similarity after weighting for topics distribution')
plt.xlabel('number of the document from the traindata')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()

##############################################################################################################################################################################
####################################################################################### draw the supvised information of  FCA labels
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
legal_terms_list=[]
file_name_legalterms='./FCA_legal_terms.txt'
legal_terms_vocabulary={}
with codecs.open(file_name_legalterms,encoding='utf-8',errors='ignore') as onetxt:
    legalwords_lines=onetxt.readlines()
    for legalword in legalwords_lines:
        item=legalword.strip('\r\n').split('\t')
        index=legal_terms_vocabulary.setdefault(item[0],len(legal_terms_vocabulary))

#######################################################################################
fp_legal_terms_vocabulary_1006_paper=open('legal_terms_vocabulary_1259_paper.txt',mode='w',encoding='utf-8')
fp_legal_terms_vocabulary_1006_ID_paper=open('legal_terms_vocabulary_1259_ID_paper.txt',mode='w',encoding='utf-8')
for k,v in legal_terms_vocabulary.items():
    fp_legal_terms_vocabulary_1006_paper.write(k+'\r\n')
    fp_legal_terms_vocabulary_1006_ID_paper.write(k+':'+str(v)+'\r\n')

fp_legal_terms_vocabulary_1259_paper.flush()
fp_legal_terms_vocabulary_1259_paper.close()
fp_legal_terms_vocabulary_1259_ID_paper.flush()
fp_legal_terms_vocabulary_1259_ID_paper.close()
#######################################################################################
##############################################################################################################################################################################
####################################################################################### draw the supvised information of  FCA labels
len(legal_terms_vocabulary)
#1259
def doc_word_cut(filename):
    with codecs.open(filename,encoding='utf-8',errors='ignore') as onetxt:
        lines=onetxt.readlines()
        onedoc=''
        for line in lines:
            onedoc+=line
        content=onedoc.strip('\r\n')
        keywords=jieba.analyse.extract_tags(content,topK=300, withWeight=True, allowPOS=())
        for item in keywords:
            if item[0] in legal_terms_vocabulary:
                st=item[0]+':'+str(item[1])
                word_list.append(st) 

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
fp=open('FCA_word_cut_3886_K300_tf_legalterms.txt', mode="w", encoding="utf-8")

os.chdir('/home/zzqzyq//Downloads/dataset/corpus/fulltext')
path='/home/zzqzyq//Downloads/dataset/corpus/fulltext'
filelist=os.listdir(path)
for filename in filelist:
    word_list=[]
    doc_word_cut(filename)
    if len(word_list)==0:
        print(file_name)
    for word in word_list:
        fp.write(word+'\t')
    fp.write('\r\n')

fp.flush()
fp.close()
#######################################################################################
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
indptr=[0]
indices=[]
data=[]
legal_terms_vocabulary_indataset={}
with codecs.open('FCA_word_cut_3886_K300_tf_legalterms.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
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
                index=legal_terms_vocabulary_indataset.setdefault(word,len(legal_terms_vocabulary_indataset))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

len(legal_terms_vocabulary_indataset)
#1086
vsmtf_legal_terms=csr_matrix((data,indices,indptr),dtype=int).toarray()
vsmtf_legal_terms.shape
#(3886, 1086)
#######################################################################################
##############################################################################################################################################################################
####################################################################################### draw the supvised information of  FCA labels
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
jieba.analyse.set_stop_words('./FCA_legal_terms.txt')
def doc_word_cut(filename):
    with codecs.open(filename,encoding='utf-8',errors='ignore') as onetxt:
        lines=onetxt.readlines()
        onedoc=''
        for line in lines:
            onedoc+=line
        content=onedoc.strip('\r\n')
        keywords=jieba.analyse.extract_tags(content,topK=300, withWeight=True, allowPOS=())
        for item in keywords:
            st=item[0]+':'+str(item[1])
            word_list.append(st) 

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
fp1=open('FCA_word_cut_3886_K300_tf_commonwords.txt', mode="w", encoding="utf-8")

os.chdir('/home/zzqzyq//Downloads/dataset/corpus/fulltext')
path='/home/zzqzyq//Downloads/dataset/corpus/fulltext'
filelist=os.listdir(path)
for filename in filelist:
    word_list=[]
    doc_word_cut(filename)
    if len(word_list)==0:
        print(file_name)
    for word in word_list:
        fp1.write(word+'\t')
    fp1.write('\r\n')

fp1.flush()
fp1.close()
#######################################################################################
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
indptr=[0]
indices=[]
data=[]
common_words_vocabulary={}
with codecs.open('FCA_word_cut_3886_K300_tf_commonwords.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
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

len(common_words_vocabulary)
#43891
vsmtf_common_words=csr_matrix((data,indices,indptr),dtype=int).toarray()
vsmtf_common_words.shape
#(3886, 43891)
#######################################################################################
##############################################################################################################################################################################
####################################################################################### draw the supvised information of  FCA labels
x_train=vsmtf_legal_terms[0:3000]
x_test=vsmtf_legal_terms[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=65
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_legal_terms=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_legal_terms.fit(doc_term_matrix_train_vec)
theta_test_lda_legal_terms_vec=lda_legal_terms.transform(doc_term_matrix_test_vec)
######################################
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
lda_legal_terms=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_legal_terms.fit(vsmtf_legal_terms)
#disMat = sch.distance.pdist(lda_legal_terms.theta,'euclidean')
#Z=sch.linkage(disMat,method='average')
#P=sch.dendrogram(Z)
#os.getcwd()
#plt.savefig('plot_dendrogram.png')
#cluster= sch.fcluster(Z, t=0.35, criterion='distance')
#cluster= sch.fcluster(Z, t=0.35, criterion='inconsistent')
#print("Original cluster by hierarchy clustering:\n",cluster)

data=whiten(lda_legal_terms.theta)
centroid=kmeans(data,2)[0] 
centroid.shape
#(2,65)
labels=vq(data,centroid)[0] 

os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
fp=open('labels.txt',mode='w',encoding='utf-8')
for i in range(len(labels)):
    st=str(labels[i])
    fp.write(st+'\r\n')

fp.flush()
fp.close()

labels=vq(data,centroid)[0] 
labels[3800:]=1
labels[3200:3300]=0
os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
fp1=open('MR.task.labels.txt',mode='w',encoding='utf-8')
for i in range(len(labels)):
    st=str(labels[i])
    fp1.write(st+'\r\n')

fp1.flush()
fp1.close()
