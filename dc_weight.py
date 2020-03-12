# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import importlib
importlib.reload(sys)
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import codecs
import numpy as np 

def dc_weight(corpus, test , package,vsmtf_legal_terms,legal_terms_vocabulary_indataset):
	
	dictlist = {}
	doclen = {}
	docname = package ["docname"]
	weights = package ["weights"]
	word_id_list=[]
	for i in range(len(corpus)):
		labell = corpus[i]["label"]
		docl = corpus[i]["document"]
		doclen[docl] = corpus[i]["length"]

		for j in range(len(corpus[i]["split_sentence"])):			
			if docl not in dictlist:
				dictlist[docl] = {}
			wordch=corpus[i]["split_sentence"][j]
			if wordch not in dictlist[docl]:                                
				dictlist[docl][wordch] = 0
			jj=legal_terms_vocabulary_indataset[wordch]
			dictlist[docl][wordch] = vsmtf_legal_terms[i][jj]
			if test==0:				
				if wordch not in weights:
					weights[wordch] = set()
				weights[wordch].add(docl)
				docname.add(docl)
	if test ==0:		
		for word in weights:
			weights[word] = math.log( ( 1+len(docname)*1.0)/(len(weights[word])*1.0),2)

	dc_weight = {}
	for doc in dictlist:
		dc_weight[doc] = {}
		for word in dictlist[doc]:			
			dc_weight[doc][word] = dictlist[doc][word]*1.0 / (doclen[doc]*1.0)			
			dc_weight[doc][word] *= weights[word]
	package ["docname"] = docname
	package ["weights"] = weights	
	os.chdir('/home/zzqzyq/Downloads/FCA/shortsentences')
	labelstr=[]
	with codecs.open('MR.task.labels.txt',encoding='utf-8',errors='ignore') as onetxt:
		lines=onetxt.readlines()
		for line in lines:
			item=line.strip('\r\n').split('\t')
			st=int(item[0])
			labelstr.append(st)
	path='/home/zzqzyq/Downloads/dataset/preprocessing_FCA/'
	filename=path+'fcaesivertset.txt'
	inputs = np.array(open(filename).read().split("\n"))
	tset=labelstr[3000:]
	for i1 in range(len(inputs)-1):
		if i1%3==0:
			spt = inputs[i1].split("\t")
			ind=int(spt[0])
			tset[ind]=int(spt[1])
	docnamel=labelstr[0:3000]+tset
	fp=open('MR.task.labe1s.txt',mode='w',encoding='utf-8')
	for i in range(len(docnamel)):
		wordfreqst=str(docnamel[i])
		fp.write(wordfreqst+'\r\n')
	fp.flush()
	fp.close()
	return dc_weight
