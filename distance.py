####
import numpy as np
from numpy import linalg as LA
import math
import itertools
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
#from processData import Process

def docToMesh(name):
    docpath = './'+name
    f = open(docpath+'/regular_meshes.txt','r')
    meshes = f.readlines()
    meshes = map(lambda x: x.strip(), meshes)
    N = len(meshes)  # the number of mesh terms
    f.close()
    f = open(docpath+'/pmids.txt','r')
    M = len(f.readlines())# the number of docs
    Doc_Mesh = np.zeros((M,N))
    f.close()
    i = 0
    for line in open(docpath+'/meshes_f.txt','r'):
        items = line.strip().split('|')
        for item in items:
            if len(item)>1:
                if '*' in item:
                    item = item.replace('*','')
                    if '/' in item:
                        w = item.split('/')[0]
                    else:
                        w = item
                    if w in meshes:
                        j = meshes.index(w)
                        Doc_Mesh[i,j] +=1
                    
                else:
                    if '/' in item:
                        w = item.split('/')[0]
                    else:
                        w = item
                    if w in meshes:
                        j = meshes.index(w)
                        Doc_Mesh[i,j] +=1
                
        i +=1
    return Doc_Mesh
def selectMeSH(name):
    Doc_MeSH = docToMesh(name)
    docpath = './'+name
    f = open(docpath+'/pmids_and_refer.txt','r')
    Doc_ref = f.readlines()
    Doc_ref = map(lambda x:int(x.strip().split()[-1]), Doc_ref)
    f.close()
    f = open(docpath+'/regular_meshes.txt','r')
    meshes = f.readlines()
    meshes = map(lambda x: x.strip(), meshes)
    f.close()
    max_f =0
    max_index = 0
    for i in range(20):
        TP=0
        TN=0
        FP=0
        FN=0
        for j in range(len(Doc_ref)):
            if Doc_MeSH[j,i]>0 and Doc_ref[j] == 1:
                TP +=1
            if Doc_MeSH[j,i]>0 and Doc_ref[j] ==-1:
                FP +=1
            if Doc_MeSH[j,i]==0 and Doc_ref[j] ==-1:
                TN +=1
            if Doc_MeSH[j,i]==0 and Doc_ref[j] ==1:
                FN +=1
        if TP >0:
            p = 1.0*TP/(TP+FP)
            r = 1.0*TP/(TP+FN)
            f = 2*p*r/(p+r)
        else:
            f = 0
        if f>max_f:
            max_f = f
            max_index = i
        print meshes[i], f
    return [max_f, meshes[max_index]]
def test():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines', 'UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,12,13,4,6,4]
    nameMap1 = sorted(nameMap)
    max_fs =[]
    max_meshes=[]
    for i in range(14,len(nameindex)):
        name = nameMap1[i]
        [max_f, max_mesh]=selectMeSH(name)
        max_fs.append(max_f)
        max_meshes.append(max_mesh)
        exit()
    print max_meshes
    print max_fs

def docToWords(name):
    docpath = './'+name
    f = open(docpath+'/unqiue_words.txt','r')
    words = f.readlines()
    words = map(lambda x: x.strip(), words)
    f.close()
    f = open(docpath+'/docs.txt','r')
    docs = f.readlines()
    docs = map(lambda x: x.strip(), docs)
    M = len(docs)
    N = len(words)
    Doc_Word = np.zeros((M,N))
    print 'processing Docs to Words Matrix'
    for i in range(M):
        if i%1000 ==0:
            print i, ' of', M
        process = Process()
        [s, s_words] = process.preprocessformesh(docs[i])
        for item in s:
            if item[0] in words:
                Doc_Word[i,words.index(item[0])]=item[1]
    np.savetxt(docpath+'/doc_words.txt', Doc_Word)
    
    ## tf-idf
    print 'processing Docs to Words matrix tf-idf'
    for j1 in range(M):
        if j1%1000 ==0:
            print j1, ' of', M
        for j2 in range(N):
            if Doc_Word[j1,j2]>0:
                Doc_Word[j1,j2]=(Doc_Word[j1,j2]*1.0/sum(Doc_Word[j1,:]))*math.log(M*1.0/(np.count_nonzero(Doc_Word[:,j2])))  ## tf*idf
    np.savetxt(docpath+'/doc_words_tfidf.txt', Doc_Word)
    print 'Doc to word matrix done!!!'
    print '-'*100
    return Doc_Word
    
    
def docToMeshTopics(name, T):
    T = int(T)
    docpath = './'+name
    ldapath = './'+name+'/resultforlda'
    Mesh_Topic = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(T)+'.txt')
    Doc_Topic = np.loadtxt(ldapath+str(T)+'/final.gamma')
    Doc_Mesh_topic = np.dot(Doc_Topic, Mesh_Topic.transpose())
    ## local normalize this matrix  ///? will consider about the gloable normalize
    MeshMax = Doc_Mesh_topic.max(0)
    MeshMin = Doc_Mesh_topic.min(0)
    MeshDis = MeshMax-MeshMin
    Doc_Mesh_topic = (Doc_Mesh_topic-MeshMin)*(1.0/MeshDis)
    '''MeshSum = sum(Doc_Mesh_topic)*1.0
    Doc_Mesh_topic = Doc_Mesh_topic/MeshSum'''
    return Doc_Mesh_topic

def kl_divergence(M1, M2): ## one way kl divergence
    kl_score = 0
    for i in range(len(M1)):
        kl_score += M2[i]*np.log(M2[i]/M1[i])
    return kl_score

def entropypart(p):
    if not p:
        return 0
    return -p * np.log2(p)
def mutual_information(M1,M2):
    pairs = zip(M1,M2)
    probs = []
    for pair in itertools.product(M1,M2):
        probs.append(1.0 * sum([p == pair for p in pairs])/len(pairs))
    return sum([entropypart(p) for p in probs])

def cosine(M1,M2):
    nor = sum([a*b for a, b in zip(M1, M2)])
    denor = math.sqrt(sum([math.pow(i,2) for i in M1]))*math.sqrt(sum([math.pow(j,2) for j in M2]))
    return (nor*1.0)/denor
def dot_product(va,vb):
    dmax = sum([a*b for a,b in zip(sorted(va, reverse=True),sorted(vb, reverse=True))])
    dmin = sum([a*b for a,b in zip(sorted(va, reverse=True),sorted(vb))])
    c = sum([a*b for a,b in zip(va,vb)])
    score = (c-dmin)/(dmax-dmin)
    return score

def setupROCCurvePlot(plt,name):

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve: "+name, fontsize=14)
    
def saveROCCurvePlot(plt, fname, randomline=True):

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    #plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    
def hist_threshold():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    nameMap1 = sorted(nameMap)
    for i in range(len(nameindex)):
        mh_index = nameindex[i]
        name = nameMap1[i]
        docpath = './'+name
        fname = docpath+'/auc_mesh_topic'
        Doc_Mesh = docToMesh(name)
        ones = np.where(Doc_Mesh[:,mh_index]==1)[0].tolist()
        zeros = np.where(Doc_Mesh[:,mh_index]==0)[0].tolist()
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(2)
            else:
                refer_list.append(1)
        TT = range(5,300,5)
        aucs = []
        #aucs0 = [tem_auc0]*len(TT)
        pre_auc=0
        T_f =0
        for T in TT:
            Doc_Mesh_topic = docToMeshTopics(name, T)
            fpr, tpr, thresholds = metrics.roc_curve(np.asarray(refer_list), Doc_Mesh_topic[:, mh_index], pos_label=2)
            tem_auc = metrics.auc(fpr, tpr)
            aucs.append(tem_auc)
            print name, T
            if tem_auc>pre_auc:
                #print tem_auc, T, M, len(refer_list)
                pre_auc = tem_auc
                FPR = fpr
                TPR = tpr
                label = 'With '+str(T)+ ' topics AUC: '+str("{0:.4f}".format(pre_auc))
                T_f = T
        Doc_Mesh_topic = docToMeshTopics(name, T_f)
        th_ones = Doc_Mesh_topic[ones,mh_index].tolist()
        th_zeros= Doc_Mesh_topic[zeros,mh_index].tolist()
        bins = np.linspace(0,1,50)
        plt.figure()
        plt.hist(th_zeros, bins, alpha=0.5, label='0,non_relevent')
        plt.hist(th_ones, bins, alpha=0.5, label='1,relevent')
        plt.legend(fontsize=10, loc='best')
        plt.savefig(docpath+'/hist_re_nre.jpg')
        plt.close()
        
def hist_threshold_refer():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    nameMap1 = sorted(nameMap)
    for i in range(len(nameindex)):
        mh_index = nameindex[i]
        name = nameMap1[i]
        docpath = './'+name
        fname = docpath+'/auc_mesh_topic'
        Doc_Mesh = docToMesh(name)
        ones = np.where(Doc_Mesh[:,mh_index]==1)[0].tolist()
        zeros = np.where(Doc_Mesh[:,mh_index]==0)[0].tolist()
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(2)
            else:
                refer_list.append(1)
        refers=np.asarray(refer_list)
        ones = np.where(refers==2)[0].tolist()
        zeros = np.where(refers==1)[0].tolist()
        TT = range(5,300,5)
        aucs = []
        #aucs0 = [tem_auc0]*len(TT)
        pre_auc=0
        T_f =0
        for T in TT:
            Doc_Mesh_topic = docToMeshTopics(name, T)
            fpr, tpr, thresholds = metrics.roc_curve(np.asarray(refer_list), Doc_Mesh_topic[:, mh_index], pos_label=2)
            tem_auc = metrics.auc(fpr, tpr)
            aucs.append(tem_auc)
            print name, T
            if tem_auc>pre_auc:
                #print tem_auc, T, M, len(refer_list)
                pre_auc = tem_auc
                FPR = fpr
                TPR = tpr
                label = 'With '+str(T)+ ' topics AUC: '+str("{0:.4f}".format(pre_auc))
                T_f = T
        Doc_Mesh_topic = docToMeshTopics(name, T_f)
        th_ones = Doc_Mesh_topic[ones,mh_index].tolist()
        th_zeros= Doc_Mesh_topic[zeros,mh_index].tolist()
        bins = np.linspace(0,1,100)
        plt.figure()
        plt.hist(th_zeros, bins, alpha=0.5, label='0,non_relevent')
        plt.hist(th_ones, bins, alpha=0.5, label='1,relevent')
        plt.legend(fontsize=10, loc='best')
        plt.title('TopicMeSH histgram')
        plt.savefig(docpath+'/topic_hist_refer.jpg')
        plt.close()
        

def prf(y0,y1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y0)):
        if y0[i] == 0 and y1[i]==0:
            TN +=1
        if y0[i] == 0 and y1[i]==1:
            FP +=1
        if y0[i] == 1 and y1[i]==0:
            FN +=1
        if y0[i] == 1 and y1[i]==1:
            TP +=1
    if TP > 0:
        p = TP*1.0/(TP+FP)
        r = TP*1.0/(TP+FN)
        f = 2*p*r/(p+r)
    else:
        p=0
        r=0
        f=0
    return p,r,f
        
def evaluate_TM():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    nameMap1 = sorted(nameMap)
    refer_list = []

    for i in range(len(nameindex)):
        name = nameMap1[i]
        docpath = './'+name
        ldapath = './'+name+'/resultforlda'
        #Mesh_Topic = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(T)+'.txt')
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(2)
            else:
                refer_list.append(1)
        L1 = np.where(np.asarray(refer_list)==2)[0].tolist()
        TT = range(5,300,5)
        ## test on single topic
        F_fs=[]
        F_f95s=[]
        for T in TT:
            print name, T
            Doc_Topic = np.loadtxt(ldapath+str(T)+'/final.gamma')
            [M,N] = Doc_Topic.shape
            F_f=0
            F_f95=0
            for i in range(N):
                L2 = np.where(Doc_Topic[:,i]>0.1)[0].tolist()
                L3 = set(L1).intersection(set(L2))
                if len(L3)>0:
                    P = len(L3)*1.0/len(L2)
                    R = len(L3)*1.0/len(L1)
                    F = 2*P*R/(P+R)
                    print P, R, F
                    if F>F_f:
                        F_f=F
                    if F>F_f95 and R >0.8:
                        F_f95=F
            
            F_fs.append(F_f)
            F_f95s.append(F_f95)
       
        plt.figure()
        plt.plot(TT, F_fs,linestyle='dashed', color='red', label='F-score')
        plt.plot(TT, F_f95s,linestyle='dashed', color='green', label='F-score recall@95')
        plt.xlabel('T : Number of Topics')
        plt.ylabel('F_score')
        plt.title('Fscores on topics :'+ str(name) )
        plt.legend(fontsize=10, loc='best')
        plt.savefig(docpath+'/tm_fscore_on_topics.jpg')
        plt.close()
  
        
        
        

def ml_valuation_dt():
    #nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    #nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    nameMap={'UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[4]
    nameMap1 = sorted(nameMap)
    P0s=[]
    P1s=[]
    R0s=[]
    R1s=[]
    F0s=[]
    F1s=[]
    for i in range(len(nameindex)):
        name = nameMap1[i]
        print 'processing ', name
        docpath = './'+name
        Doc_Word = docToWords(name)  ## Doc to Words matrix
        refer_list = []  ## Gold standard
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(1)
            else:
                refer_list.append(0)
        refer_list = np.asarray(refer_list)
        Doc_Mesh = docToMesh(name)  ## original Doc to mesh matrix
        [M,N] = Doc_Mesh.shape
        if N > 295:
            T = 295
        else:
            T = (math.floor(N/10.0))*10+5    
        Doc_Mesh_topic = docToMeshTopics(name, T) ## new doc to mesh-Topic matrix
        skf = StratifiedKFold(refer_list, 5)
        c = 1
        P0 = 0
        P1 = 0
        R0 = 0
        R1 = 0
        F0 = 0
        F1 = 0
        
        for train,test in skf:
            ## testing on Doc_Mesh
            print 'testing on original Documents, MeSH datasets + Words'
            print '-'*100
            '''X = np.append(Doc_Word[train,:],Doc_Mesh[train,:], axis=1)
            #X = Doc_Mesh[train,:]
            y = refer_list[train]
            print 'training DT: ', c, name
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X, y)
            X_test = np.append(Doc_Word[test,:],Doc_Mesh[test,:], axis=1)
            #X_test = Doc_Mesh[test,:]
            y_test = refer_list[test]
            y_pred = clf.predict(X_test)
            [p0, r0, f0] = prf(y_test, y_pred)
            P0 +=p0
            R0 +=r0
            F0 +=f0
            print p0, r0, f0
            
            print 'testing on Documents, MeSH with Topic Models results+Words'
            print '-'*100
            X = np.append(Doc_Word[train,:], Doc_Mesh_topic[train,:],axis=1)
            #X = Doc_Mesh_topic[train,:]
            y = refer_list[train]
            print 'training DT: ', c, name
            c +=1
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X, y)
            X_test = np.append(Doc_Word[test,:],Doc_Mesh_topic[test,:], axis=1)
            #X_test = Doc_Mesh_topic[test,:]
            y_test = refer_list[test]
            y_pred = clf.predict(X_test)
            [p1, r1, f1] = prf(y_test, y_pred)
            P1 +=p1
            R1 +=r1
            F1 +=f1
            print p1,r1, f1'''
            
            ### test on Doc to words matrix
            print 'testing on Documents, words datasets'
            print '-'*100
            X = Doc_Word[train,:]
            y = refer_list[train]
            print 'training DT: ', c, name
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X, y)
            X_test = Doc_Word[test,:]
            y_test = refer_list[test]
            y_pred = clf.predict(X_test)
            [p0, r0, f0] = prf(y_test, y_pred)
            P0 +=p0
            R0 +=r0
            F0 +=f0
            print p0, r0, f0

        P0 = P0/5.0
        P1 = P1/5.0
        R0 = R0/5.0
        R1 = R1/5.0
        F0 = F0/5.0
        F1 = F1/5.0
        
        P0s.append(P0)
        P1s.append(P1)
        R0s.append(R0)
        R1s.append(R1)
        F0s.append(F0)
        F1s.append(F1)
        print F0
        

    
    print F0s
    print F1s
    print P0s
    print P1s
    print R0s
    print R1s
    
def ml_valuation_lr():
    #nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    #nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    nameMap={'UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[4]
    nameMap1 = sorted(nameMap)
    P0s=[]
    P1s=[]
    R0s=[]
    R1s=[]
    F0s=[]
    F1s=[]
    for i in range(len(nameindex)):
        name = nameMap1[i]
        print 'processing ', name
        docpath = './'+name
        Doc_Word = docToWords(name)  ## Doc to Words matrix
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(1)
            else:
                refer_list.append(0)
        refer_list = np.asarray(refer_list)
        Doc_Mesh = docToMesh(name)
        [M,N] = Doc_Mesh.shape
        if N > 295:
            T = 295
        else:
            T = (math.floor(N/10.0))*10+5    
        Doc_Mesh_topic = docToMeshTopics(name, T)
        skf = StratifiedKFold(refer_list, 5)
        c = 1
        P0 = 0
        P1 = 0
        R0 = 0
        R1 = 0
        F0 = 0
        F1 = 0
        
        for train,test in skf:
            ## testing on Doc_Mesh
            f0_p=0
            r0_p=0
            p0_p=0
            print 'testing on original Documents, MeSH datasets'
            print '-'*100
            #X = np.append(Doc_Word[train,:],Doc_Mesh[train,:], axis=1)
            '''X= Doc_Mesh[train,:]
            y = refer_list[train]
            #X_test = np.append(Doc_Word[test,:],Doc_Mesh[test,:], axis=1)
            X_test= Doc_Mesh[test,:]
            y_test = refer_list[test]
            print 'training LR: ', c, name
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                model = LogisticRegression(C = C, penalty='l2', tol=0.01)
                model = model.fit(X, y)
                y_pred = model.predict(X_test)
                [p0, r0, f0] = prf(y_test, y_pred)
                if f0>f0_p:
                    f0_p=f0
                    p0_p=p0
                    r0_p=r0        
            P0 +=p0_p
            R0 +=r0_p
            F0 +=f0_p
            print p0_p, r0_p, f0_p
            
            print 'testing on Documents, MeSH with Topic Models results'
            print '-'*100
            #clf = svm.SVC()
            #X = np.append(Doc_Word[train,:], Doc_Mesh_topic[train,:],axis=1)
            X = Doc_Mesh_topic[train,:]
            y = refer_list[train]
            #X_test = np.append(Doc_Word[test,:],Doc_Mesh_topic[test,:], axis=1)
            X_test= Doc_Mesh_topic[test,:]
            y_test = refer_list[test]
            print 'training DT: ', c, name
            c +=1
            f1_p=0
            r1_p=0
            p1_p=0
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                model = LogisticRegression(C = C, penalty='l2', tol=0.01)
                model = model.fit(X, y)
                y_pred = model.predict(X_test)
                [p1, r1, f1] = prf(y_test, y_pred)
                if f1>f1_p:
                    f1_p=f1
                    p1_p=p1
                    r1_p=r1            
            P1 +=p1_p
            R1 +=r1_p
            F1 +=f1_p
            print p1_p, r1_p, f1_p'''
            
            ### test on Doc to words matrix
            print 'testing on Documents, words datasets'
            print '-'*100
            X = Doc_Word[train,:]
            y = refer_list[train]
            X_test = Doc_Word[test,:]
            y_test = refer_list[test]
            print 'training DT: ', c, name
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                model = LogisticRegression(C = C, penalty='l2', tol=0.01)
                model = model.fit(X, y)
                y_pred = model.predict(X_test)
                [p0, r0, f0] = prf(y_test, y_pred)
                if f0>f0_p:
                    f0_p=f0
                    p0_p=p0
                    r0_p=r0
            P0 +=p0_p
            R0 +=r0_p
            F0 +=f0_p
            print p0_p, r0_p, f0_p
        P0 = P0/5.0
        P1 = P1/5.0
        R0 = R0/5.0
        R1 = R1/5.0
        F0 = F0/5.0
        F1 = F1/5.0
        
        P0s.append(P0)
        P1s.append(P1)
        R0s.append(R0)
        R1s.append(R1)
        F0s.append(F0)
        F1s.append(F1)
    print F0s
    print F1s
    print P0s
    print P1s
    print R0s
    print R1s

def tfidfDocMeSH(Doc_Mesh):
    [M,N] = Doc_Mesh.shape
    Doc_Mesh=np.asarray(Doc_Mesh)
    #print sum(Doc_Mesh)
    Docs_idf = (M+1.0)/(np.sum(Doc_Mesh,0)+1.0/M)
    Docs_idf = map(lambda x:math.log(x), Docs_idf)
    for i in range(M):
        Doc_Mesh[i] = Doc_Mesh[i]*Docs_idf  
    return Doc_Mesh
    
    
def ml_valuation_svm():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines','UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8,4]
    #nameMap={'UrinaryIncontinence':'Urinary Incontinence'}
    #nameindex=[4]
    nameMap1 = sorted(nameMap)
    P0s=[]
    P1s=[]
    R0s=[]
    R1s=[]
    F0s=[]
    F1s=[]
    for i in range(len(nameindex)):
        name = nameMap1[i]
        docpath = './'+name
        print 'processing ', name
        #Doc_Word = docToWords(name)  ## Doc to Words matrix
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(1)
            else:
                refer_list.append(0)
        C_0 = refer_list.count(0)
        C_1 = refer_list.count(1)
        cc = C_0*1.0/C_1
        print cc
        refer_list = np.asarray(refer_list)
        Doc_Mesh_0 = docToMesh(name)
        ##tfidf mesh
        Doc_Mesh = tfidfDocMeSH(Doc_Mesh_0)
        [M,N] = Doc_Mesh.shape
        if N > 295:
            T = 295
        else:
            T = (math.floor(N/10.0))*10+5    
        #Doc_Mesh_topic = docToMeshTopics(name, T)
        ## get doc topics and doc mesh to one matrix
        ldapath = './'+name+'/resultforlda'
        Doc_Topic = np.loadtxt(ldapath+str(int(T))+'/final.gamma')
        
        skf = StratifiedKFold(refer_list, 5)
        c = 1
        P0 = 0
        P1 = 0
        R0 = 0
        R1 = 0
        F0 = 0
        F1 = 0
        
        for train,test in skf:
            ## testing on Doc_Mesh
            '''f0_p=0
            r0_p=0
            p0_p=0
            print 'testing on original Documents, MeSH tfidf'
            print '-'*100
            #clf = svm.SVC()
            #rf = RandomForestClassifier(n_estimators=10)
            #X = np.append(Doc_Word[train,:],Doc_Mesh[train,:], axis=1)
            X= Doc_Mesh[train,:]
            y = refer_list[train]
            y_1 = np.count_nonzero(y)
            y_0 = len(y)-y_1
            cc = y_0*1.0/y_1
            #X_test = np.append(Doc_Word[test,:],Doc_Mesh[test,:], axis=1)
            X_test= Doc_Mesh[test,:]
            y_test = refer_list[test]
            print 'training SVM: ', c, name
            f0_p=0
            r0_p=0
            p0_p=0
            #for pair in itertools.product([0.01, 0.1, 1, 10, 100, 1000],[0, 2, 4, 8, 16, 32, 64]):
            #    C = pair[0]
            #    g = pair[1]
            #    print 'pair:', C, g
            model = svm.SVC(kernel='linear',class_weight={1:cc})
            model = model.fit(X, y)
            y_pred = model.predict(X_test)
            [p0, r0, f0] = prf(y_test, y_pred)
            if f0>f0_p:
                f0_p=f0
                p0_p=p0
                r0_p=r0
            
            P0 +=p0_p
            R0 +=r0_p
            F0 +=f0_p
            print p0_p, r0_p, f0_p'''
            
            print 'testing on Documents, MeSH and Topic Models simple combination results'
            print '-'*100
            #X = np.append(Doc_Mesh_0[train,:], Doc_Topic[train,:],axis=1)
            X= Doc_Topic[train,:]
            y = refer_list[train]
            #X_test = np.append(Doc_Mesh_0[test,:],Doc_Topic[test,:], axis=1)
            X_test= Doc_Topic[test,:]
            y_test = refer_list[test]
            print 'training SVM: ', c, name
            c +=1
            f1_p=0
            r1_p=0
            p1_p=0
            #for C in [0.01, 0.1, 1, 10, 100, 100]:            
            model = svm.SVC(kernel='linear',class_weight={1:cc})
            model = model.fit(X, y)
            y_pred = model.predict(X_test)
            [p1, r1, f1] = prf(y_test, y_pred)
            if f1>f1_p:
                f1_p=f1
                p1_p=p1
                r1_p=r1            
            P1 +=p1_p
            R1 +=r1_p
            F1 +=f1_p
            print p1_p, r1_p, f1_p
            
            '''### test on Doc to words matrix
            print 'testing on Documents, words datasets'
            print '-'*100
            X = Doc_Word[train,:]
            y = refer_list[train]
            X_test = Doc_Word[test,:]
            y_test = refer_list[test]
            y_1 = np.count_nonzero(y)
            y_0 = len(y)-y_1
            cc = y_0*1.0/y_1
            print 'training SVM: ', c, name
            #for C in [0.01, 0.1, 1, 10, 100, 1000]:
            model = svm.SVC(kernel='linear',class_weight={1:cc})
            model = model.fit(X, y)
            y_pred = model.predict(X_test)
            [p0, r0, f0] = prf(y_test, y_pred)
            if f0>f0_p:
                f0_p=f0
                p0_p=p0
                r0_p=r0
            P0 +=p0_p
            R0 +=r0_p
            F0 +=f0_p
            print p0_p, r0_p, f0_p'''
            
        P0 = P0/5.0
        P1 = P1/5.0
        R0 = R0/5.0
        R1 = R1/5.0
        F0 = F0/5.0
        F1 = F1/5.0
        
        P0s.append(P0)
        P1s.append(P1)
        R0s.append(R0)
        R1s.append(R1)
        F0s.append(F0)
        F1s.append(F1)
        print F0 , F1

    print F0s
    print F1s
    print P0s
    print P1s
    print R0s
    print R1s
    
   # nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines','UrinaryIncontinence':'Urinary Incontinence'}
   # nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8,4]    
   
def mean_average_precision():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines', 'UrinaryIncontinence':'Urinary Incontinence'}
    nameindexs=[3,13,4,1,5,6,3,1,1,2,12,13,4,6,4]   
    nameMap1 = sorted(nameMap)
    C_0s=[]
    C_1s=[]
    p_meshes=[]
    r_meshes=[]
    TP_meshes=[]
    FP_meshes=[]
    #topicalmesh
    p_topicalmeshes=[]
    TP_topicalmeshes=[]
    FP_topicalmeshes=[]
    for i in range(len(nameindexs)):
        nameindex = nameindexs[i]
        name = nameMap1[i]
        #name='UrinaryIncontinence'
        #nameindex=4
        docpath = './'+name
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(1)
            else:
                refer_list.append(0)
        C_0 = refer_list.count(0)
        C_1 = refer_list.count(1)
        C_0s.append(C_0)
        C_1s.append(C_1)

        print name, len(refer_list), C_0, C_1, nameindex
        #refer_list = np.asarray(refer_list)
        Doc_Mesh = docToMesh(name)
        [M,N] = Doc_Mesh.shape
        if N > 295:
            T = 295
        else:
            T = (math.floor(N/10.0))*10+5   
        print T
        Doc_Mesh_topic = docToMeshTopics(name, T)
        doc_mesh = Doc_Mesh[:,nameindex].tolist()
        doc_mesh = map(int,doc_mesh)
        #print doc_mesh
        #print refer_list
        doc_topicalmesh = Doc_Mesh_topic[:,nameindex].tolist()

        p_mesh = 0
        r_mesh = 0
        TP_mesh=0
        FP_mesh=0
        p_topicalmesh = []
        TP_topicalmesh =[]
        FP_topicalmesh =[]

        #indices = range(len(refer_list))
        #indices.sort(lambda x,y: -cmp(doc_mesh[x], doc_mesh[y]))

        indices1 = range(len(refer_list))
        indices1.sort(lambda x,y: -cmp(doc_topicalmesh[x], doc_topicalmesh[y]))
        
        for j1 in range(len(refer_list)):
            #print refer_list[j1], doc_mesh[j1]
            if refer_list[j1]==1 and doc_mesh[j1]==1:
                TP_mesh += 1
            if refer_list[j1]==0 and doc_mesh[j1]==1:
                FP_mesh += 1
            #if refer_list[indices[j1]] ==1:
            #    c1 +=1
            #    p_mesh.append(c1*1.0/(j1+1))
        p_mesh = TP_mesh*1.0/(TP_mesh+FP_mesh)
        r_mesh = TP_mesh*1.0/C_1
        p_meshes.append(p_mesh)
        r_meshes.append(r_mesh)
        TP_meshes.append(TP_mesh)
        FP_meshes.append(FP_mesh)
        ##TopicalMesh
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1
                if c2 >= C_1*0.1:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.2:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.3:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.4:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.5:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.6:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.7:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.8:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1*0.9:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        c2 = 0
        for j2 in range(len(refer_list)):
            if refer_list[indices1[j2]] ==1:
                c2 +=1            
                if c2 >= C_1:
                    p_topicalmesh.append(c2*1.0/(j2+1)) 
                    TP_topicalmesh.append(c2)
                    FP_topicalmesh.append(j2+1-c2)
                    break
        print len(p_topicalmesh)
        p_topicalmeshes.append(p_topicalmesh)
        TP_topicalmeshes.append(TP_topicalmesh)
        FP_topicalmeshes.append(FP_topicalmesh)
        
    return [p_meshes,r_meshes, TP_meshes, FP_meshes, p_topicalmeshes, TP_topicalmeshes, FP_topicalmeshes, C_1s, C_0s]
    
def evaluate_IR():
    print '123'
    #nameMap1 = sorted(nameMap)
    #for i in range(len(nameindex)):
    name = 'Triptans'
    nameindex = 8
    docpath = './'+name
    print 'processing ', name
    #Doc_Word = docToWords(name)  ## Doc to Words matrix
    refer_list = []
    for line in open(docpath+'/pmids_and_refer.txt','r'):
        if line.strip().split()[-1] == '1':
            refer_list.append(1)
        else:
            refer_list.append(0)
    C_0 = refer_list.count(0)
    C_1 = refer_list.count(1)
    
    print len(refer_list), C_0, C_1
    refer_list = np.asarray(refer_list)
    Doc_Mesh = docToMesh(name)
    [M,N] = Doc_Mesh.shape
    if N > 295:
        T = 295
    else:
        T = (math.floor(N/10.0))*10+5    
    Doc_Mesh_topic = docToMeshTopics(name, T)
    doc_mesh = Doc_Mesh[:,nameindex].tolist()
    doc_topicalmesh = Doc_Mesh_topic[:,nameindex].tolist()
    
    print "processing doc to mesh ranking results"
    indices = range(len(refer_list))
    indices.sort(lambda x,y: -cmp(doc_mesh[x], doc_mesh[y]))
    
    indices1 = range(len(refer_list))
    indices1.sort(lambda x,y: -cmp(doc_topicalmesh[x], doc_topicalmesh[y]))
    
    cc = 0
    for j1 in range(len(refer_list)):
        if refer_list[indices[j1]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.5:
                print 'doc mesh 50% :  ', j1+1
                break
    
            
    cc = 0
    for i1 in range(len(refer_list)):
        if refer_list[indices1[i1]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.5:
                print 'doc topicalmesh 50% :  ', i1+1
                break
                
    cc = 0
    for j2 in range(len(refer_list)):
        if refer_list[indices[j2]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.7:
                print 'doc mesh 70% :  ', j2+1
                break
    
            
    cc = 0
    for i2 in range(len(refer_list)):
        if refer_list[indices1[i2]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.7:
                print 'doc topicalmesh 70% :  ', i2+1
                break
    
    cc = 0
    for j3 in range(len(refer_list)):
        if refer_list[indices[j3]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.9:
                print 'doc mesh 90% :  ', j3+1
                break
    
            
    cc = 0
    for i3 in range(len(refer_list)):
        if refer_list[indices1[i3]] ==1:
            cc +=1
            if cc/(C_1*1.0) > 0.9:
                print 'doc topicalmesh 90% :  ', i3+1
                break
    
    
    cc = 0
    for j4 in range(len(refer_list)):
        if refer_list[indices[j4]] ==1:
            cc +=1
            if cc/(C_1*1.0) == 1.0:
                print 'doc mesh 100% :  ', j4+1
                break
    
            
    cc = 0
    for i4 in range(len(refer_list)):
        if refer_list[indices1[i4]] ==1:
            cc +=1
            if cc/(C_1*1.0) == 1.0:
                print 'doc topicalmesh 100% :  ', i4+1
                break
def pre_recall():
    [p_meshes,r_meshes, TP_meshes, FP_meshes, p_topicalmeshes, TP_topicalmeshes, FP_topicalmeshes, C_1s, C_0s] = mean_average_precision()
    ## Gloable Precision and Recall
    ## MeSH
    ma_p_mesh = sum(p_meshes)/15.0
    ma_r_mesh = sum(r_meshes)/15.0
    ## TopicalMeSH 
    ## recall from 10%-100%
    r_topicalmesh = range(1,11,1)
    ma_r_topicalmesh = [i/10.0 for i in r_topicalmesh]
    ma_p_topicalmesh = []
    for j in range(10):
        p = 0
        for item in p_topicalmeshes:
            p = p+item[j]
        ma_p_topicalmesh.append(p/15.0)
    plt.figure()
    plt.plot(ma_r_topicalmesh, ma_p_topicalmesh,'-o', color='c', label='TopicalMeSH')
    plt.plot([ma_r_mesh], [ma_p_mesh],'D',color='m',label='MeSH')
    plt.ylim([0,0.4])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision and Recall')
    plt.legend(fontsize=10, loc='best')
    plt.savefig('./overall.jpg')
    plt.close()
    
    ## indivisual figures
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines', 'UrinaryIncontinence':'Urinary Incontinence'}
    nameindexs=[3,13,4,1,5,6,3,1,1,2,12,13,4,6,4]
    nameMap1 = sorted(nameMap)
    
    plt.figure()
    for i in range(15):
        name = nameMap1[i]
        print name
        ##MeSH
        p_mesh = p_meshes[i]
        r_mesh = r_meshes[i]
        ## TopicalMeSH
        p_topicalmesh = p_topicalmeshes[i]
        plt.subplot(5,3,i+1)
        plt.plot(ma_r_topicalmesh, p_topicalmesh,'-o', color='c',label='TopicalMeSH')
        plt.plot([r_mesh], [p_mesh],'D',color = 'm',label='MeSH')
        plt.ylim([0,0.8])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(name,fontweight='bold')
        plt.legend(fontsize=10, loc='best')
    plt.savefig('./indivisual.jpg')
    plt.show()
    
def ave_pre_recall():
    [p_meshes,r_meshes, TP_meshes, FP_meshes, p_topicalmeshes, TP_topicalmeshes, FP_topicalmeshes, C_1s, C_0s] = mean_average_precision()
    ## macro_average Precision and Recall
    ## MeSH
    ma_p_mesh = sum(p_meshes)/15.0
    ma_r_mesh = sum(r_meshes)/15.0
    ## TopicalMeSH 
    ## recall from 10%-100%
    r_topicalmesh = range(1,11,1)
    ma_r_topicalmesh = [i/10.0 for i in r_topicalmesh]
    ma_p_topicalmesh = []
    for j in range(10):
        p = 0
        for item in p_topicalmeshes:
            p = p+item[j]
        ma_p_topicalmesh.append(p/15.0)
    
    ## micro_average precision and recall
    ##mesh
    mi_p_mesh = sum(TP_meshes)*1.0/(sum(TP_meshes)+sum(FP_meshes))
    mi_r_mesh = sum(TP_meshes)*1.0/sum(C_1s)
    
    ## TopicalMeSH
    #recall from 10%-100%
    r_topicalmesh = range(1,11,1)
    mi_r_topicalmesh = [i/10.0 for i in r_topicalmesh]
    
    mi_p_topicalmesh = []
    for k in range(10):
        TP=0
        for item1 in TP_topicalmeshes:
            TP = TP + item1[k]
        FP = 0
        for item2 in FP_topicalmeshes:
            FP = FP + item2[k]
        mi_p_topicalmesh.append(TP*1.0/(TP+FP))
    
    print 'Macro P R: MeSH VS TopicalMeSH'
    print ma_p_mesh, ma_r_mesh
    print ma_p_topicalmesh
    print ma_r_topicalmesh
    
    print 'Micro P R: MeSH vs TopicalMeSH'
    print mi_p_mesh, mi_r_mesh
    print mi_p_topicalmesh
    print mi_r_topicalmesh
    
    # Two subplots, the axes array is 1-d
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(ma_r_topicalmesh, ma_p_topicalmesh,'-o', color='blue', label='TopicalMeSH')
    plt.plot([ma_r_mesh], [ma_p_mesh],'or',label='MeSH')
    plt.ylim([0,0.4])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Macro Average Precision and Recall')
    plt.legend(fontsize=10, loc='best')
    
    plt.subplot(1,2,2)
    plt.plot(mi_r_topicalmesh, mi_p_topicalmesh, '-o', color='blue', label='TopicalMeSH')
    plt.plot([mi_r_mesh], [mi_p_mesh],'or' ,label='MeSH')
    plt.ylim([0,0.4])
    plt.xlabel('Recall')
    plt.title('Micro Average Precision and Recall')
    plt.legend(fontsize=10, loc='best')
    plt.savefig('./Micro and Macro_P_R.jpg')
    plt.show()
def meshPerArticle():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines','UrinaryIncontinence':'Urinary Incontinence'}
    for name in nameMap:
        docMeSH = docToMesh(name)
        (M,N) = docMeSH.shape
        print name+ ': ' + str(sum(sum(docMeSH))*1.0/M)
    
def main():
    name = 'ACEInhibitors'
    Doc_Word=docToWords(name)
    print Doc_Word.shape
    exit()
    
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines', 'UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[3,13,4,1,5,6,3,4,1,2,1,13,4,8,4]
    nameMap1 = sorted(nameMap)
    for i in range(len(nameindex)):
        mh_index = nameindex[i]
        name = nameMap1[i]
        docpath = './'+name
        fname = docpath+'/auc_mesh_topic'
        Doc_Mesh = docToMesh(name)
        ones = np.where(Doc_Mesh[:,mh_index]==1)[0].tolist()
        zeros = np.where(Doc_Mesh[:,mh_index]==0)[0].tolist()
        refer_list = []
        for line in open(docpath+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(2)
            else:
                refer_list.append(1) 
        fpr0, tpr0, thresholds0 = metrics.roc_curve(np.asarray(refer_list), Doc_Mesh[:, mh_index], pos_label=2)
        tem_auc0 = metrics.auc(fpr0, tpr0)
        label0 = 'MeSH term '+ name+' AUC: '+str("{0:.4f}".format(tem_auc0))
        ### build auc of roc
        plt.figure()
        setupROCCurvePlot(plt, name)
        TT = range(5,300,5)
        aucs = []
        aucs0 = [tem_auc0]*len(TT)
        pre_auc=0
        T_f =0
        for T in TT:
            Doc_Mesh_topic = docToMeshTopics(name, T)
            fpr, tpr, thresholds = metrics.roc_curve(np.asarray(refer_list), Doc_Mesh_topic[:, mh_index], pos_label=2)
            tem_auc = metrics.auc(fpr, tpr)
            aucs.append(tem_auc)
            if tem_auc>pre_auc:
                #print tem_auc, T, M, len(refer_list)
                pre_auc = tem_auc
                FPR = fpr
                TPR = tpr
                label = 'With '+str(T)+ ' topics AUC: '+str("{0:.4f}".format(pre_auc))
                T_f = T
            print name, T
        plt.plot(FPR, TPR, color='b', linewidth=2, label=label) 
        plt.plot(fpr0, tpr0, color='g', linewidth=2, label=label0) 
        saveROCCurvePlot(plt, fname, 'True')
        
        plt.figure()
        plt.plot(TT, aucs,linestyle='dashed', color='red', label='MeSH+Topics')
        plt.plot(TT, aucs0,linestyle='dashed', color='green', label='MeSH')
        plt.xlabel('T : Number of Topics')
        plt.ylabel('AUC')
        plt.title('AUC differnece on Number of topics'+ str(Doc_Mesh.shape) )
        plt.legend(fontsize=10, loc='best')
        plt.savefig(docpath+'/AUC_and_topics.jpg')
        plt.close()
            
            
            
            
    '''## normalize refer list
    ss = sum(refer_list)
    sd = 1.0/ss
    refer_list = [(j+sd)/ss for j in refer_list]
    
    Doc_Mesh = docToMesh(name)
    [M,N] = Doc_Mesh.shape
    TT = range(5,300,5)
    diss = []
    KLs =[]
    MIs =[]
    COs =[]
    DPs =[]
    KLs0 =[]
    MIs0 =[]
    COs0 =[]
    DPs0 =[]
    KLs1 =[]
    MIs1 =[]
    COs1 =[]
    DPs1 =[]
    for T in TT:
        Doc_Mesh_topic = docToMeshTopics(name, T)
        KLf =0
        MIf =0
        COf =0
        DPf =0
        KLf0 =0
        MIf0 =0
        COf0 =0
        DPf0 =0
        KLf1 =0
        MIf1 =0
        COf1 =0
        DPf1 =0
        for i in range(N):
            L0 = Doc_Mesh[:][i].tolist()
            ss = sum(L0)
            if ss ==0:
                continue
            
            sd = 1.0/ss
            L0 = [(j+sd)/ss for j in L0]
            L1 = Doc_Mesh_topic[:][i].tolist()
            ### one way KL divergence
            #print 'computing KL divergence'
            KL0 = kl_divergence(L0, refer_list) # for mesh
            KL1 = kl_divergence(L1, refer_list) # for topic model and mesh
            KLf0 +=KL0
            KLf1 +=KL1
            KLf +=(KL1-KL0)
            #print KL0, KL1
            ### mutual_information
            #print 'computing Mutual Information'
            MI0 = mutual_information(L0, refer_list) # for mesh
            MI1 = mutual_information(L1, refer_list) # for topic model and mesh
            MIf +=(MI1-MI0)
            #print MI0, MI1
            ### cosine
            #print 'Computing Cosine'
            CO0 = cosine(L0, refer_list)
            CO1 = cosine(L1, refer_list)
            COf0 +=CO0
            COf1 +=CO1
            COf +=(CO1-CO0)
            #print CO0, CO1
            ### dot product
            #print 'computing recaled Dot product'
            DP0 = dot_product(L0, refer_list)
            DP1 = dot_product(L1, refer_list)
            DPf0 +=DP0
            DPf1 +=DP1
            DPf +=(DP1-DP0)
            #print DP0, DP1
        KLs.append(KLf)
        #MIs.append(MIf/N)
        COs.append(COf)
        DPs.append(DPf)
        KLs0.append(KLf0)
        #MIs.append(MIf/N)
        COs0.append(COf0)
        DPs0.append(DPf0)
        KLs1.append(KLf1)
        #MIs.append(MIf/N)
        COs1.append(COf1)
        DPs1.append(DPf1)
    
        ## compute distance
        #diss.append(LA.norm(Doc_Mesh_topic - Doc_Mesh, 'fro'))  ## Frobenius distance
        print T
    #print Doc_Mesh.shape 
    ## plot it
    plt.figure()
    plt.plot(TT, KLs0,linestyle='dashed', color='red', label='MeSH')
    plt.plot(TT, KLs1,linestyle='dashed', color='green', label='MeSH+Topics')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('KL divergence')
    plt.title('KL differnece on Number of topics'+ str(Doc_Mesh.shape) )
    plt.legend(fontsize=10, loc='best')
    plt.savefig(docpath+'/KL_and_topics.jpg')
    #plt.show()  
    plt.close()
    plt.figure()
    plt.plot(TT, MIs)
    plt.xlabel('T : Number of Topics')
    plt.ylabel('difference of Mutual information')
    plt.title('MI differnece on Number of topics'+ str(Doc_Mesh.shape) )
    plt.savefig(docpath+'/MI_and_topics.jpg')
    #plt.show()  
    plt.close()
    
    plt.figure()
    plt.plot(TT, COs0,linestyle='dashed', color='red', label='MeSH')
    plt.plot(TT, COs1,linestyle='dashed', color='green', label='MeSH+Topics')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('Cosine')
    plt.title('CO differnece on Number of topics'+ str(Doc_Mesh.shape) )
    plt.legend(fontsize=10, loc='best')
    plt.savefig(docpath+'/CO_and_topics.jpg')
    #plt.show()  
    plt.close()
    
    plt.figure()
    plt.plot(TT, DPs0,linestyle='dashed', color='red', label='MeSH')
    plt.plot(TT, DPs1,linestyle='dashed', color='green', label='MeSH+Topics')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('Rescled dot product')
    plt.title('RDP differnece on Number of topics'+ str(Doc_Mesh.shape))
    plt.legend(fontsize=10, loc='best')
    plt.savefig(docpath+'/RDP_and_topics.jpg')
    #plt.show()  
    plt.close()'''

pre_recall()
#meshPerArticle()
#ml_valuation_svm()
#test()
