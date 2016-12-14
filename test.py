####### utility test
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from sklearn import metrics

def p_r_mesh(name, mesh_name):
    docpath = './'+name
    mesh_doc_list=[]
    mmesh_doc_list=[]
    a = -1
    for line in open(docpath+'/meshes_f.txt','r'):
        items = line.strip().split('|')
        a +=1
        for item in items:
            if len(item)>1:
                if '*' in item:
                    item = item.replace('*','')
                    if '/' in item:
                        w = item.split('/')[0]
                    else:
                        w = item
                    if w == mesh_name:
                        mmesh_doc_list.append(a)
                        mesh_doc_list.append(a)
                        break
                else:
                    if '/' in item:
                        w = item.split('/')[0]
                    else:
                        w = item
                    if w == mesh_name:
                        mesh_doc_list.append(a)
                        break
    return [mesh_doc_list, mmesh_doc_list]
def p_r_topic(name, T, mh_index):
    docpath = './'+name
    ldapath = './'+name+'/resultforlda'
    M1 = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(T)+'.txt')
    M2 = np.loadtxt(ldapath+str(T)+'/final.gamma')
    
    #topic_list = [j for j,v in enumerate(M2[3,:].tolist()) if v > 0.5]
    
    topic_list=[]
    ## ranking documents based on the similairy of topics and also topics weigts in docs
    docs_weight = np.dot(M2,M1[mh_index,:].transpose()).tolist()
    ## normalized to 0-1
    dw_min = min(docs_weight)
    dw_max= max(docs_weight)
    docs_weight = [(i-dw_min)/float((dw_max-dw_min)) for i in docs_weight]

    '''LL = M1[8,:].tolist()
    #LL_mean = float(sum(LL))/len(LL)
    #II = sorted(LL)
    for item in LL:
        if item>0.55:
            topic_list.append(LL.index(item))'''
    '''if II[-1] > 0:
        topic_list.append(LL.index(II[-1]))
    if II[-2] >0:
        topic_list.append(LL.index(II[-2]))
    if II[-3] >0:
        topic_list.append(LL.index(II[-3]))'''
    '''if II[-4] >0:
        topic_list.append(LL.index(II[-4]))
    if II[-5] >0:
        topic_list.append(LL.index(II[-5]))
    if II[-6] >0:
        topic_list.append(LL.index(II[-6]))'''
    '''topic_doc_list=[]
    for topic in topic_list:
        topic_doc_list += [j1 for j1,v1 in enumerate(M2[:,topic].tolist()) if v1 > 1]'''
    
    return docs_weight

def document_on_topics(refer_list, T, name):
    docpath = './'+name
    ldapath = './'+name+'/resultforlda'
    M1 = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(T)+'.txt')
    M2 = np.loadtxt(ldapath+str(T)+'/final.gamma')
    (M, N) = M2.shape
    print M,N
    topic_index = 0
    P=0
    R=0
    F=0
    for i in range(N):
        topic_doc_list = [j1 for j1,v1 in enumerate(M2[:,i].tolist()) if v1 > 1]
        N = len(set(topic_doc_list).intersection(set(refer_list)))
        P_D = len(set(topic_doc_list))
        R_D = len(refer_list)
        if N > 0:
            P1 = float(N)/P_D
            R1 = float(N)/R_D
            F1 = (2*P1*R1)/(P1+R1)
            if F1 > F:
                F =F1
                P =P1
                R =R1
                topic_index = i
    sim_score = M1[3,topic_index]
    return [P, R, F, sim_score]
        
    
    
def graph(Ps,Rs,Fs, TT,docpath):

    plt.plot(TT,Ps,'r-o',label='Precision')
    plt.plot(TT,Rs,'g-',label='Recall')
    plt.plot(TT,Fs,'b-*',label = 'F_score')
   # plt.plot(TT,Ss,'*',label = 'Sim_score')
    #plt.plot(TT,Fx,'bo', label = 'max F_score similarity threshold')
    #plt.plot(TT,Repeated,'b',label='Repeated')
    plt.grid(True)
    legend(loc='upper right')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('Percent')
    plt.title('Precision, Recall, and F_score vs Number of Topics')
    plt.savefig(docpath+'/prf_mean5.jpg')
    #plt.show()  

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
    #plt.show()
def f_score(refer_list, docs_weight):
    un_th = set(docs_weight)
    f1 = 0
    thres=0
    f1_re95 =0
    thres_re95 =0
    for th in un_th:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(refer_list)):
            if docs_weight[i] >= th and refer_list[i]==2:
                TP +=1
            if docs_weight[i] >= th and refer_list[i]==1:
                FP +=1
            if docs_weight[i] < th and refer_list[i]==2:
                FN +=1
            if docs_weight[i] < th and refer_list[i]==1:
                TN +=1
        if TP > 0:
            P = float(TP)/(TP+FP)
            R = float(TP)/(TP+FN)
            F = 2*P*R/(P+R)
            if F > f1:
                f1 = F
                thres = th
            if F > f1_re95 and R >= 0.95:
                f1_re95 = F
                thres_re95 = th
    return [f1,thres, f1_re95, thres_re95]

def best_f1(name, mh_index, baseF1, baseF1_re95):
    docpath = './'+name
    fname = docpath+'/best_f1s'
    M1 = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(5)+'.txt')
    (M, N) = M1.shape
    refer_list = []
    for line in open(docpath+'/pmids_and_refer.txt','r'):
        if line.strip().split()[-1] == '1':
            refer_list.append(2)
        else:
            refer_list.append(1)
    TT = range(5,300,5)
    F1s = []
    F1_re95s = []
    Thress =[]
    Thres_re95s=[]
    for T in TT:
        #print name, ' processing... ',T
        docs_weight=p_r_topic(name, T, mh_index)
        [f1, thres, f1_re95, thres_re95] = f_score(refer_list, docs_weight)
        F1s.append(f1)
        F1_re95s.append(f1_re95)
        Thress.append(thres)
        Thres_re95s.append(thres_re95)
    
    # best F1 on topics
    BaseF1s = [baseF1]*len(TT)
    plt.figure()
    plt.plot(TT, F1s,linestyle='dashed', color='red', label='F1_score')
    #plt.plot(TT, F1_re95s,linestyle='dashed', color='blue')
    plt.plot(TT, Thress,linestyle='dashed', color='blue', label='Threshold')
    plt.plot(TT, BaseF1s, linestyle='dashed', color='green', label='Baseline F1_score')
    #plt.plot(TT, Thres_re95s,linestyle='*', color='blue')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('percent')
    plt.legend(fontsize=10, loc='best')
    plt.title('%s: F1 score vs Number of Topics' %(name))
    plt.savefig(docpath+'/F1_and_topics.jpg')
    plt.close()
    print name, ': baseline, Best F1_score, Threshold, number of MeSH, number of documents, number of topics'
    print baseF1, ', ', max(F1s), ', ', Thress[F1s.index(max(F1s))],', ', M, ', ', len(refer_list), TT[F1s.index(max(F1s))]

    # best F1 @95% on topics
    BaseF1_re95s = [baseF1_re95]*len(TT)
    plt.figure()
    plt.plot(TT, F1_re95s,linestyle='dashed', color='red', label='F1_score @95%')
    #plt.plot(TT, F1_re95s,linestyle='dashed', color='blue')
    plt.plot(TT, Thres_re95s,linestyle='dashed', color='blue', label='Threshold')
    plt.plot(TT, BaseF1_re95s, linestyle='dashed', color='green', label='Baseline F1_score @95%')
    #plt.plot(TT, Thres_re95s,linestyle='*', color='blue')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('percent')
    plt.legend(fontsize=10, loc='best')
    plt.title('%s: F1 score @95 vs Number of Topics' %(name))
    plt.savefig(docpath+'/F1@95_and_topics.jpg')
    plt.close()
    print name, ': baseline, Best F1_score @95, Threshold, number of MeSH, number of documents, number of topics'
    print baseF1_re95, ', ', max(F1_re95s), ', ', Thres_re95s[F1_re95s.index(max(F1_re95s))],', ', M, ', ', len(refer_list), TT[F1_re95s.index(max(F1_re95s))]
    
        
def roc_curve(name, mh_index):

    docpath = './'+name
    fname = docpath+'/best_roc'
    M1 = np.loadtxt(docpath+'/mesh_topic_cor_matrix_'+str(5)+'.txt')
    (M, N) = M1.shape
    refer_list = []
    for line in open(docpath+'/pmids_and_refer.txt','r'):
        if line.strip().split()[-1] == '1':
            refer_list.append(2)
        else:
            refer_list.append(1)
    y = np.asarray(refer_list)
    #num_pos=refer_list.count(2)
    #num_neg=refer_list.count(1)    
    #plt.figure(figsize=(4, 4), dpi=80)
    plt.figure()
    setupROCCurvePlot(plt, name)
    TT = range(5,305,5)
    #TT = [60]
    #colors = ['b','g','r','c','m','y','k','w']
    pre_auc=0
    T_f = 0
    aucs = []
    for T in TT:
        docs_weight=p_r_topic(name, T, mh_index)
        pred = np.asarray(docs_weight)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
        tem_auc = metrics.auc(fpr, tpr)
        aucs.append(tem_auc)
        if tem_auc>pre_auc:
            #print tem_auc, T, M, len(refer_list)
            pre_auc = tem_auc
            FPR = fpr
            TPR = tpr
            label = 'With '+str(T)+ ' topics AUC: '+str("{0:.4f}".format(pre_auc))
            T_f = T
        
    plt.plot(FPR, TPR, color='b', linewidth=2, label=label)        
    saveROCCurvePlot(plt, fname, 'True')
    docs_weight=p_r_topic(name, T_f, mh_index)


    F_f = 0
    P_f =0
    R_f = 0
    TH = 0
    F_b = 0
    P_b =0
    R_b = 0
    TH_b = 0
    WSS_f = 0
    AN = len(refer_list)
    for item in docs_weight:
        th = item
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(docs_weight)):
            if docs_weight[i] >= th and refer_list[i]==2:
                TP +=1
            if docs_weight[i] >= th and refer_list[i]==1:
                FP +=1
            if docs_weight[i] < th and refer_list[i]==2:
                FN +=1
            if docs_weight[i] < th and refer_list[i]==1:
                TN +=1
        if TP > 0:
            P = float(TP)/(TP+FP)
            R = float(TP)/(TP+FN)
            F = 2*P*R/(P+R)
            #WSS =(TN +FN)/float(AN) -1 + float(TP)/(TP + FN)
            #WSS = float(TP+FP)/AN
            if F>F_f and R>=0.95:
            #if WSS>WSS_f and R>=0.95:
                F_f = F
                P_f = P
                R_f = R
                TH = th
                #WSS_f = WSS
                #print TN, FN, AN, R
            if F > F_b:
                F_b = F
                P_b = P
                R_b = R
                TH_b = th
    print name, 'results @ 95% recall: Precision, Recall, F_score, Threshold, number of MeSH, number of documents, number of topics'
    print P_f, R_f, F_f, TH, M, len(refer_list), T_f, pre_auc
    
    print name, 'The best result:Precision, Recall, F_score, Threshold, number of MeSH, number of documents, number of topics'
    print P_b, R_b, F_b, TH_b, M, len(refer_list), T_f, pre_auc
    ### plot the auc and number of topics
    plt.figure()
    plt.plot(TT, aucs)
    plt.xlabel('T : Number of Topics')
    plt.ylabel('Area under ROC')
    plt.title('Sensitivity of auc and Number of topics')
    plt.savefig(docpath+'/auc_and_topics.jpg')
    plt.close()
    #plt.show()  
    
def main():
    name = 'ADHD'
    docpath = './'+name
    refer_list = []
    a = 0
    for line in open(docpath+'/pmids_and_refer.txt','r'):
        if line.strip().split()[-1] == '1':
            refer_list.append(a)
        a +=1
    [mesh_doc_list, mmesh_doc_list] = p_r_mesh(name, 'Attention Deficit Disorder with Hyperactivity')
    print len(mesh_doc_list), len(mmesh_doc_list), len(refer_list)
    
    T_N = len(set(mesh_doc_list).intersection(set(refer_list)))
    T_N_1 =  len(set(mmesh_doc_list).intersection(set(refer_list)))
    print 'MeSH Term\'s performance'
    v1 = float(T_N)/len(mesh_doc_list)
    v2 = float(T_N)/len(refer_list)
    print float(T_N)/len(mesh_doc_list), float(T_N)/len(refer_list),2*v1*v2/(v1+v2)
    print 'Major MeSH\'s performance'
    if len(mmesh_doc_list)>0:
        v1 = float(T_N_1)/len(mmesh_doc_list)
        v2 = float(T_N_1)/len(refer_list)
        print float(T_N_1)/len(mmesh_doc_list), float(T_N_1)/len(refer_list),2*v1*v2/(v1+v2)
    exit()
    Ps = []
    Rs = []
    Fs = []
    Ss = []
    #TT = range(5,300,5)
    TT = [55,60,65,70,75,80,85,90,95,100]
    for T in TT:
        topic_doc_list = p_r_topic(name, T)
        N = len(set(topic_doc_list).intersection(set(refer_list)))
        P_D = len(set(topic_doc_list))
        R_D = len(refer_list)
        if N >0:
            P = float(N)/P_D
            R = float(N)/R_D
            F = (2*P*R)/(P+R)
            print P,R,F
        else:
            P=0
            R=0
            F=0
                
        #[P, R, F, sim_score] = document_on_topics(refer_list, T, name)
        #print P, R, F
        Ps.append(P)
        Rs.append(R)
        Fs.append(F)
        #Ss.append(sim_score)
    print max(Fs), Ps[Fs.index(max(Fs))], Rs[Fs.index(max(Fs))], TT[Fs.index(max(Fs))]
    #graph(Ps,Rs,Fs,TT,docpath)
    
    

def run():
    nameMap = {'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Statins':'Hydroxymethylglutaryl-CoA Reductase Inhibitors','Triptans':'Tryptamines'}
    nameindex=[3,13,4,1,5,6,3,1,1,2,1,13,4,8]
    BaseF1s = [0.203, 0.226, 0.108, 0.394, 0.2, 0.322, 0.525, 0.463, 0.061, 0.499, 0.312, 0.01, 0.256, 0.179]
    BaseF1s_re95s = [0.075, 0.137, 0, 0.24, 0.05, 0.156, 0.404, 0.279, 0.016, 0.436, 0.09, 0.011, 0.06, 0.07]
    #print len(nameMap)
    nameMap1 = sorted(nameMap)
    #print nameMap1
    for i in range(len(nameindex)):
        mh_index = nameindex[i]
        name = nameMap1[i]
        baseF1 = BaseF1s[i]
        baseF1_re95 = BaseF1s_re95s[i]
        #print name,mh_index,nameMap[name]
        #exit()
        best_f1(name, mh_index, baseF1, baseF1_re95) 
        print '-'*200

main()
    
                    
                        

                            