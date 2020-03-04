
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

