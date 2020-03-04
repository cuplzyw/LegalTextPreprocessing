import os
import sys
import codecs
import numpy as np
import re
import jieba.analyse
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve,roc_auc_score)
from slda.topic_models import LDA
from slda.topic_models import BLSLDA
from sklearn.linear_model import LinearRegression

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


vsmtf_legal_terms=csr_matrix((data,indices,indptr),dtype=int).toarray()
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

vsmtf_common_words=csr_matrix((data,indices,indptr),dtype=int).toarray()
############################################################################################################################################################################## vsmtf_legal_terms
####################################################################################### vsmtf_legal_terms
################################################################################ svm
x_train=vsmtf_legal_terms[0:3000]
x_test=vsmtf_legal_terms[3000:3886]


os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
labels=[]
with codecs.open('MR.task.labels.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
    for line in lines:
        item=line.strip('\r\n').split('\t')
        st=int(item[0])
        labels.append(st)


y_train=np.zeros(3000)
for i in range(len(y_train)):
    y_train[i]=labels[i]

y_test=np.zeros(886)
for i in range(len(y_test)):
    j=i+3000
    y_test[i]=labels[j]

y_train=y_train.astype(np.int64)
y_test=y_test.astype(np.int64)

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

  Prediction: [252, 634]  Right: [227, 568]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 795/886 = 0.897291
    Precision: [0.9007936507936508, 0.8958990536277602]
    Recall   : [0.7747440273037542, 0.9578414839797639]
    F1 score : [0.8330275229357798, 0.9258353708231459]
    Macro F1 score on test (Neg|Neu|Pos) is 0.882028

linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc
#0.8777115263972742

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
####################################################################################### LDA+lr
x_train=vsmtf_legal_terms[0:3000]
x_test=vsmtf_legal_terms[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=100
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_legal_terms=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_legal_terms.fit(doc_term_matrix_train_vec)
theta_test_lda_legal_terms_vec=lda_legal_terms.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda_legal_terms.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_legal_terms_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8788079355852408(K=100)

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
#K=100,p=0.5
  Prediction: [223, 663]  Right: [205, 575]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 780/886 = 0.880361
    Precision: [0.9192825112107623, 0.8672699849170438]
    Recall   : [0.6996587030716723, 0.96964586846543]
    F1 score : [0.7945736434108527, 0.9156050955414012]
    Macro F1 score on test (Neg|Neu|Pos) is 0.862970

####################################################################################### SLDA
x_train=vsmtf_legal_terms[0:3000]
x_test=vsmtf_legal_terms[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100
blslda_legal_terms = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)

blslda_legal_terms.fit(doc_term_matrix_train_vec, y_train)

burn_in = max(n_iter - 100, int(n_iter / 2))
eta_pred = blslda_legal_terms.eta[burn_in:].mean(axis=0)
thetas_test_blslda = blslda_legal_terms.transform(doc_term_matrix_test_vec)

def bern_param(eta, theta):
    return np.exp(np.dot(eta, theta)) / (1 + np.exp(np.dot(eta, theta)))

D=len(thetas_test_blslda)
y_blslda = [bern_param(eta_pred, thetas_test_blslda[i]) for i in range(D)]

pcs_auc=roc_auc_score(y_test,y_blslda)
pcs_auc

#0.869610760349699(_K=100)
#0.8697834232139466(_K=95)
#0.8691388151874255(_K=90)
#0.8682179465781097(_K=85)
#0.8679474414241232(_K=80)
#0.8714409867107149(_K=75)
#0.8660423944886013(_K=70)
#0.8686726254539594(_K=65)
#0.873455386793593(_K=60)
#0.8705028518149744(_K=55)
#0.8727417136213734(_K=50)
#0.8707676015401528(_K=45)
#0.870002129508659(_K=40)
#0.8661690139223822(_K=35)
#0.8703244335219195(_K=30)
#0.8732597022141134(_K=25)
#0.8708366666858514(_K=20)
#0.867193480250246(_K=15)
#0.8625143166291604(_K=10)
#0.8675848494092052(_K=5)
#0.8633430983775446(_K=2)
#0.5(_K=1)

x_range=[2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
y_auc=[0.8633430983775446,0.8675848494092052,0.8625143166291604,0.867193480250246,0.8708366666858514,0.8732597022141134,0.8703244335219195,0.8661690139223822,0.870002129508659,
       0.8707676015401528,0.8727417136213734,0.8705028518149744,0.873455386793593,0.8686726254539594,0.8660423944886013,0.8714409867107149,0.8679474414241232,0.8682179465781097,
       0.8691388151874255,0.8697834232139466,0.869610760349699]
plt.plot(x_range,y_auc,label='AUC under the different K ')
plt.xlabel('the sum of topics')
plt.ylabel('Area Under Curve(AUC)')
plt.legend()
plt.show()

fpr,tpr,_=roc_curve(y_test,y_blslda)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_blslda))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_blslda))
for i in range(len(y_predict)):
    if y_blslda[i]>0.7:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.7,K=60
  Prediction: [249, 637]  Right: [222, 566]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 788/886 = 0.889391
    Precision: [0.891566265060241, 0.8885400313971743]
    Recall   : [0.757679180887372, 0.954468802698145]
    F1 score : [0.8191881918819188, 0.9203252032520326]
    Macro F1 score on test (Neg|Neu|Pos) is 0.872733
####################################################################################### SLDA+lr
x_train=vsmtf_legal_terms[0:3000]
x_test=vsmtf_legal_terms[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_legal_terms = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)
blslda_legal_terms.fit(doc_term_matrix_train_vec, y_train)

thetas_test_blslda_legal_terms_vec = blslda_legal_terms.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(blslda_legal_terms.theta,y_train)
y_predict_proba_1_vec=lr.predict(thetas_test_blslda_legal_terms_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8740136633879909

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.6:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.6,K=60
  Prediction: [246, 640]  Right: [222, 569]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 791/886 = 0.892777
    Precision: [0.9024390243902439, 0.8890625]
    Recall   : [0.757679180887372, 0.9595278246205734]
    F1 score : [0.823747680890538, 0.9229521492295216]
    Macro F1 score on test (Neg|Neu|Pos) is 0.876784

#######################################################################################
##############################################################################################################################################################################
####################################################################################### vsmtf_common_words
x_train=vsmtf_common_words[0:3000]
x_test=vsmtf_common_words[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
labels=[]
with codecs.open('MR.task.labels.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
    for line in lines:
        item=line.strip('\r\n').split('\t')
        st=int(item[0])
        labels.append(st)


y_train=np.zeros(3000)
for i in range(len(y_train)):
    y_train[i]=labels[i]

y_test=np.zeros(886)
for i in range(len(y_test)):
    j=i+3000
    y_test[i]=labels[j]

y_train=y_train.astype(np.int64)
y_test=y_test.astype(np.int64)

linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

  Prediction: [243, 643]  Right: [202, 552]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 754/886 = 0.851016
    Precision: [0.831275720164609, 0.8584758942457231]
    Recall   : [0.689419795221843, 0.9308600337268128]
    F1 score : [0.753731343283582, 0.8932038834951456]
    Macro F1 score on test (Neg|Neu|Pos) is 0.827143


linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc
#0.84771135373441

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

####################################################################################### LDA
x_train=vsmtf_common_words[0:3000]
x_test=vsmtf_common_words[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=100
alpha=np.repeat(1.,K)
V=vsmtf_common_words.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_common_words=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_common_words.fit(doc_term_matrix_train_vec)
theta_test_lda_common_words_vec=lda_common_words.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda_common_words.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_common_words_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.867912908851274(K=100)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.7:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#K=100,p=0.7
  Prediction: [256, 630]  Right: [214, 551]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 765/886 = 0.863431
    Precision: [0.8359375, 0.8746031746031746]
    Recall   : [0.7303754266211604, 0.9291736930860034]
    F1 score : [0.7795992714025501, 0.9010629599345871]
    Macro F1 score on test (Neg|Neu|Pos) is 0.842330


####################################################################################### SLDA
x_train=vsmtf_common_words[0:3000]
x_test=vsmtf_common_words[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100
blslda_common_words = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)

blslda_common_words.fit(doc_term_matrix_train_vec, y_train)

burn_in = max(n_iter - 100, int(n_iter / 2))
eta_pred = blslda_common_words.eta[burn_in:].mean(axis=0)
thetas_test_blslda = blslda_common_words.transform(doc_term_matrix_test_vec)

def bern_param(eta, theta):
    return np.exp(np.dot(eta, theta)) / (1 + np.exp(np.dot(eta, theta)))

D=len(thetas_test_blslda)
y_blslda = [bern_param(eta_pred, thetas_test_blslda[i]) for i in range(D)]

pcs_auc=roc_auc_score(y_test,y_blslda)
pcs_auc
#0.8637862663957778(_K=60)

y_predict_proba_1_vec=y_blslda
fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_blslda))
for i in range(len(y_predict)):
    if y_blslda[i]>0.8:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.8,_K=60
  Prediction: [247, 639]  Right: [210, 556]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 766/886 = 0.864560
    Precision: [0.8502024291497976, 0.8701095461658842]
    Recall   : [0.7167235494880546, 0.9376053962900506]
    F1 score : [0.7777777777777778, 0.9025974025974025]
    Macro F1 score on test (Neg|Neu|Pos) is 0.843338

####################################################################################### SLDA+lr
x_train=vsmtf_common_words[0:3000]
x_test=vsmtf_common_words[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_common_words = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)
blslda_common_words.fit(doc_term_matrix_train_vec, y_train)

thetas_test_blslda_common_words_vec = blslda_common_words.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(blslda_common_words.theta,y_train)
y_predict_proba_1_vec=lr.predict(thetas_test_blslda_common_words_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8638207989686272(_K=60)


fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.67:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.67,_K=60
  Prediction: [228, 658]  Right: [200, 565]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 765/886 = 0.863431
    Precision: [0.8771929824561403, 0.8586626139817629]
    Recall   : [0.6825938566552902, 0.9527824620573356]
    F1 score : [0.7677543186180422, 0.9032773780975221]
    Macro F1 score on test (Neg|Neu|Pos) is 0.842059

#######################################################################################
##############################################################################################################################################################################
####################################################################################### vsmtf_legal_terms+vsmtf_common_words
vsmtf=np.hstack((vsmtf_legal_terms,vsmtf_common_words))
x_train=vsmtf[0:3000]
x_test=vsmtf[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test


linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)

  Prediction: [240, 646]  Right: [215, 568]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 783/886 = 0.883747
    Precision: [0.8958333333333334, 0.8792569659442725]
    Recall   : [0.7337883959044369, 0.9578414839797639]
    F1 score : [0.8067542213883677, 0.9168684422921711]
    Macro F1 score on test (Neg|Neu|Pos) is 0.866178


linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc
#0.8729489090584693

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

####################################################################################### LDA+lr
x_train=vsmtf[0:3000]
x_test=vsmtf[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=100
alpha=np.repeat(1.,K)
V=vsmtf.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_all_features=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_all_features.fit(doc_term_matrix_train_vec)
theta_test_lda_all_features_vec=lda_all_features.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda_all_features.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_all_features_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8729604199160859(K=100)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.64:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#K=100,p=0.64
  Prediction: [238, 648]  Right: [214, 569]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 783/886 = 0.883747
    Precision: [0.8991596638655462, 0.8780864197530864]
    Recall   : [0.7303754266211604, 0.9595278246205734]
    F1 score : [0.8060263653483992, 0.9170024174053183]
    Macro F1 score on test (Neg|Neu|Pos) is 0.866237

####################################################################################### SLDA
x_train=vsmtf[0:3000]
x_test=vsmtf[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_all_features = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)

blslda_all_features.fit(doc_term_matrix_train_vec, y_train)

burn_in = max(n_iter - 100, int(n_iter / 2))
eta_pred = blslda_all_features.eta[burn_in:].mean(axis=0)
thetas_test_blslda = blslda_all_features.transform(doc_term_matrix_test_vec)

def bern_param(eta, theta):
    return np.exp(np.dot(eta, theta)) / (1 + np.exp(np.dot(eta, theta)))

D=len(thetas_test_blslda)
y_blslda = [bern_param(eta_pred, thetas_test_blslda[i]) for i in range(D)]

pcs_auc=roc_auc_score(y_test,y_blslda)
pcs_auc
#0.8670380836724241

y_predict_proba_1_vec=y_blslda
fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_blslda))
for i in range(len(y_predict)):
    if y_blslda[i]>0.81:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.81,_K=60
  Prediction: [248, 638]  Right: [219, 564]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 783/886 = 0.883747
    Precision: [0.8830645161290323, 0.8840125391849529]
    Recall   : [0.7474402730375427, 0.9510961214165261]
    F1 score : [0.8096118299445472, 0.9163281884646628]
    Macro F1 score on test (Neg|Neu|Pos) is 0.866064

####################################################################################### SLDA+lr
x_train=vsmtf[0:3000]
x_test=vsmtf[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = x_train.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_all_features = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)
blslda_all_features.fit(doc_term_matrix_train_vec, y_train)

thetas_test_blslda = blslda_all_features.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(blslda_all_features.theta,y_train)
y_predict_proba_1_vec=lr.predict(thetas_test_blslda)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.8683157888678495(_K=60)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.67:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.67,_K=60
  Prediction: [232, 654]  Right: [211, 572]  Gold: [293, 593]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 783/886 = 0.883747
    Precision: [0.9094827586206896, 0.8746177370030581]
    Recall   : [0.7201365187713311, 0.9645868465430016]
    F1 score : [0.8038095238095238, 0.917401764234162]
    Macro F1 score on test (Neg|Neu|Pos) is 0.866494

####################################################################################### weighted by dc
os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
from preprocessing import preprocessing


corpus,labelset,vocafreq=preprocessing(vsmtf_legal_terms,legal_terms_vocabulary_indataset)
os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
labels=[]
with codecs.open('MR.task.labels.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
    for line in lines:
        item=line.strip('\r\n').split('\t')
        st=int(item[0])
        labels.append(st)

def init(package):
	package["voca"] = []
	package["labelset"] = []
	package["vocafreq"] = {}
	package["weights"] = {}
	package["doclist"] = []	
	package["docname"] = set()

package = {}
global package
init(package)

package["labelset"] = labelset
package["vocafreq"] = vocafreq

voca = vocafreq.keys()
package["voca"] = voca

os.chdir('/home/zzqzyq/Downloads/dataset/preprocessing_FCA')
from tf_idf import tf_idf
test=0
weights = tf_idf(corpus,test,package,vsmtf_legal_terms,legal_terms_vocabulary_indataset)

vsmtf_legal_terms_weighted=np.zeros(vsmtf_legal_terms.shape)
doccount = {}
for i in range(len(vsmtf_legal_terms)):
    label = labels[i]
    if label not in doccount:
        doccount[label] = 0
    doccount[label] += 1    
    docname = str(label)+str(doccount[label])
    allvalues_i=weights[docname].values()
    inval=(max(allvalues_i)-min(allvalues_i))/3
    p1=min(allvalues_i)+inval
    p2=max(allvalues_i)-inval
    word_id_list=[]
    for j in range(len(vsmtf_legal_terms[i])):
        vsmtf_legal_terms_weighted[i][j]=vsmtf_legal_terms[i][j]
        if vsmtf_legal_terms[i][j]>0:
            word_id_list.append(j)
    for idx in word_id_list:
        for k,v in legal_terms_vocabulary_indataset.items():
            if v==idx:
                if weights[docname][k]>p2:
                    vsmtf_legal_terms_weighted[i][idx]=vsmtf_legal_terms[i][idx]*3
                if weights[docname][k]>p1 and weights[docname][k]<p2:
                    vsmtf_legal_terms_weighted[i][idx]=vsmtf_legal_terms[i][idx]*2 

vsmtf_legal_terms_weighted=vsmtf_legal_terms_weighted.astype(np.int64)                
 


#######################################################################################
##############################################################################################################################################################################
####################################################################################### vsmtf_legal_terms+weighted
x_train=vsmtf_legal_terms_weighted[0:3000]
x_test=vsmtf_legal_terms_weighted[3000:3886]


os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
labels=[]
with codecs.open('MR.task.labe1s.txt',encoding='utf-8',errors='ignore') as onetxt:
    lines=onetxt.readlines()
    for line in lines:
        item=line.strip('\r\n').split('\t')
        st=int(item[0])
        labels.append(st)


y_train=np.zeros(3000)
for i in range(len(y_train)):
    y_train[i]=labels[i]

y_test=np.zeros(886)
for i in range(len(y_test)):
    j=i+3000
    y_test[i]=labels[j]

y_train=y_train.astype(np.int64)
y_test=y_test.astype(np.int64)

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

linear_svc=SVC(kernel='linear')
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict=linear_svc.predict(doc_term_matrix_test_vec)
y_predict=y_predict.astype(np.int64)
y_test=y_test.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#weighted
  Prediction: [254, 632]  Right: [228, 578]  Gold: [282, 604]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 806/886 = 0.909707
    Precision: [0.8976377952755905, 0.9145569620253164]
    Recall   : [0.8085106382978723, 0.956953642384106]
    F1 score : [0.8507462686567165, 0.9352750809061489]
    Macro F1 score on test (Neg|Neu|Pos) is 0.894262


linear_svc=SVC(kernel='linear',probability=True)
linear_svc.fit(doc_term_matrix_train_vec,y_train)
y_predict_proba_vec=linear_svc.predict_proba(doc_term_matrix_test_vec)
y_predict_proba_1_vec=np.zeros((len(y_predict_proba_vec),1))
for i in range(len(y_predict_proba_1_vec)):
    y_predict_proba_1_vec[i]=y_predict_proba_vec[i][1]

igr_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
igr_auc
#0.9066330844018599(weighted)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

####################################################################################### LDA+weighted+lr
x_train=vsmtf_legal_terms_weighted[0:3000]
x_test=vsmtf_legal_terms_weighted[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

K=100
alpha=np.repeat(1.,K)
V=vsmtf_legal_terms_weighted.shape[1]
beta=np.repeat(0.01,V)
n_iter=100

lda_all_features=LDA(K,alpha,beta,n_iter=100,seed=42)
lda_all_features.fit(doc_term_matrix_train_vec)
theta_test_lda_all_features_vec=lda_all_features.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(lda_all_features.theta,y_train)
y_predict_proba_1_vec=lr.predict(theta_test_lda_all_features_vec)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.9139425109201069(K=100)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.6:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#K=100,p=0.6
  Prediction: [252, 634]  Right: [229, 581]  Gold: [282, 604]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 810/886 = 0.914221
    Precision: [0.9087301587301587, 0.916403785488959]
    Recall   : [0.8120567375886525, 0.9619205298013245]
    F1 score : [0.8576779026217228, 0.938610662358643]
    Macro F1 score on test (Neg|Neu|Pos) is 0.899596


####################################################################################### SLDA+weighted
x_train=vsmtf_legal_terms_weighted[0:3000]
x_test=vsmtf_legal_terms_weighted[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = vsmtf_legal_terms_weighted.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_all_features = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)

blslda_all_features.fit(doc_term_matrix_train_vec, y_train)

burn_in = max(n_iter - 100, int(n_iter / 2))
eta_pred = blslda_all_features.eta[burn_in:].mean(axis=0)
thetas_test_blslda = blslda_all_features.transform(doc_term_matrix_test_vec)

def bern_param(eta, theta):
    return np.exp(np.dot(eta, theta)) / (1 + np.exp(np.dot(eta, theta)))

D=len(thetas_test_blslda)
y_blslda = [bern_param(eta_pred, thetas_test_blslda[i]) for i in range(D)]

pcs_auc=roc_auc_score(y_test,y_blslda)
pcs_auc
#0.9155276877553894

y_predict_proba_1_vec=y_blslda
fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_blslda))
for i in range(len(y_predict)):
    if y_blslda[i]>0.6:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.6,_K=60
   Prediction: [238, 648]  Right: [223, 589]  Gold: [282, 604]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 812/886 = 0.916479
    Precision: [0.9369747899159664, 0.9089506172839507]
    Recall   : [0.7907801418439716, 0.9751655629139073]
    F1 score : [0.8576923076923076, 0.9408945686900959]
    Macro F1 score on test (Neg|Neu|Pos) is 0.902525


####################################################################################### SLDA+weighted+lr
x_train=vsmtf_legal_terms_weighted[0:3000]
x_test=vsmtf_legal_terms_weighted[3000:3886]

doc_term_matrix_train_vec=x_train
doc_term_matrix_test_vec=x_test

_K = 60
_alpha = np.repeat(1., _K)
V = vsmtf_legal_terms_weighted.shape[1]
_mu = 0.
_nu2 = 1.
_beta = np.repeat(0.01, V)
_b = 7.25
n_iter = 100

blslda_all_features = BLSLDA(_K, _alpha, _beta, _mu, _nu2, _b, n_iter, seed=42)
blslda_all_features.fit(doc_term_matrix_train_vec, y_train)

thetas_test_blslda = blslda_all_features.transform(doc_term_matrix_test_vec)

lr=LinearRegression(fit_intercept=False)
lr.fit(blslda_all_features.theta,y_train)
y_predict_proba_1_vec=lr.predict(thetas_test_blslda)
pcs_auc=roc_auc_score(y_test,y_predict_proba_1_vec)
pcs_auc
#0.918339908881687(_K=60)

fpr,tpr,_=roc_curve(y_test,y_predict_proba_1_vec)
plt.plot(fpr,tpr,label=('AUC = {:.3f}'.format(roc_auc_score(y_test,y_predict_proba_1_vec))))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_predict=np.zeros(len(y_predict_proba_1_vec))
for i in range(len(y_predict)):
    if y_predict_proba_1_vec[i]>0.6:
        y_predict[i]=1

y_test=y_test.astype(np.int64)
y_predict=y_predict.astype(np.int64)
accurancy=test_prf(y_predict,y_test)
#p=0.6,_K=60
  Prediction: [248, 638]  Right: [229, 585]  Gold: [282, 604]  -- for all labels --
  ****** Neg|Neu|Pos ******
    Accuracy on test is 814/886 = 0.918736
    Precision: [0.9233870967741935, 0.9169278996865203]
    Recall   : [0.8120567375886525, 0.9685430463576159]
    F1 score : [0.8641509433962263, 0.9420289855072465]
    Macro F1 score on test (Neg|Neu|Pos) is 0.904982


##############################################################################################################################################################################
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

##############################################################################################################################################################################
########### 1086-svm-878-hist
_m = plt.hist(y_predict_proba_1_vec, bins=30)
plt.show()
plt.show()
