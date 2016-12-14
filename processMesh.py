## process mesh
import matplotlib.pyplot as plt
from collections import Counter
from processData import Process
import numpy as np
import math
import time
from collections import OrderedDict
import csv
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
def promesh(docpath):
    meshes={}
    meshess={}
    c = 0
    for line in open(docpath+'/meshes_f.txt','r'):
        items = line.strip().split('|')
        for item in items:
            if len(item)>1:
                if '*' in item:
                    item = item.replace('*','')
                    if '/' in item:
                        w = item.split('/')[0]
                        if meshes.has_key(w):
                            meshes[w].append(c)
                        else:
                            meshes[w]=[]
                            meshes[w].append(c)
                        if meshess.has_key(w):
                            meshess[w].append(c)
                        else:
                            meshess[w]=[]
                            meshess[w].append(c)
                        
                    else:
                        w = item
                        if meshes.has_key(w):
                            meshes[w].append(c)
                        else:
                            meshes[w]=[]
                            meshes[w].append(c)
                        if meshess.has_key(w):
                            meshess[w].append(c)
                        else:
                            meshess[w]=[]
                            meshess[w].append(c)
                else:
                    if '/' in item:
                        w = item.split('/')[0]
                        if meshes.has_key(w):
                            meshes[w].append(c)
                        else:
                            meshes[w]=[]
                            meshes[w].append(c)
                        
                    else:
                        w = item
                        if meshes.has_key(w):
                            meshes[w].append(c)
                        else:
                            meshes[w]=[]
                            meshes[w].append(c)
        c +=1
    return [meshes, meshess]
def precision_recall(docpath, ldapath,T):
    rmesh = file(docpath+'regular_mesh.txt','r').readlines()
    rmesh = (lambda x: x.strip(), rmesh)
    docs_mesh = []
    cc=[]
    for line in file(docpath+'meshes.txt','r'):
        mesh=[]
        for item in line.strip().split('|'):
            if len(item)>1:
                item = item.replace('*','')
                if '/' in item:
                    if item.split('/')[0] in rmesh:
                        mesh.append(rmesh.index(item.split('/')[0]))
                else:
                    if item in rmesh:
                        mesh.append(rmesh.index(item))
        docs_mesh.append(mesh)
        cc.append(len(mesh))
    M1 = np.loadtxt(docpath+'mesh_topic_cor_matrix_'+str(T)+'.txt')
    #M1= M1[:500]
    M1_min = M1.min(1).tolist()
    M1_max = M1.max(1).tolist()
    M1_dis = [a - b for a, b in zip(M1_max, M1_min)]
    M2 = np.loadtxt(ldapath+str(T)+'/final.gamma')
    
    Ps = []
    Rs = []
    Fs =[]
    x=[]
    for th in range(0,100,2):
        print '%d of 100' %th
        print T
        
        P = 0
        R = 0
        F = 0
        P_D = 0
        R_D = 0
        T_N = 0
        for i in range(len(docs_mesh)):
            #topic_list=[]
            topic_list = [j for j,v in enumerate(M2[i,:].tolist()) if v > 0.2]
            
            '''LL = M2[i,:].tolist()
            II = sorted(LL)
            if II[-1] > 0.2:
                topic_list.append(LL.index(II[-1]))
            if II[-2] >0.2:
                topic_list.append(LL.index(II[-2]))
            if II[-3] >0.2:
                topic_list.append(LL.index(II[-3]))'''
            '''if II[-4] >0.2:
                topic_list.append(LL.index(II[-4]))'''
            if len(topic_list)==0:
                mesh_list = docs_mesh[i]
                R_D += len(mesh_list)
            else:
                mesh_list = docs_mesh[i]
                b = len(mesh_list)
                topic_mesh_list=[]
                for topic in topic_list:
                    topic_mesh_list += [j1 for j1,v1 in enumerate(M1[:,topic].tolist()) if v1 > (th/100.0)]
                    #if len([j1 for j1,v1 in enumerate(M1[:,topic].tolist()) if v1 > (th/100.0)])>len(topic_mesh_list):
                    #    topic_mesh_list = [j1 for j1,v1 in enumerate(M1[:,topic].tolist()) if v1 > (th/100.0)]
                    
                #print a_sum, a, b, c
                T_N += len(set(mesh_list).intersection(set(topic_mesh_list)))
                P_D += len(set(topic_mesh_list))
                R_D += len(mesh_list)
        #print T_N, P_D, R_D, th
        if T_N>0:
            P = T_N/float(P_D)
            R = T_N/float(R_D)
            F = (2*P*R)/(P+R)
            Ps.append(P)
            Rs.append(R)
            Fs.append(F)
            x.append(th/100.0)
    
    
    aa = [Ps[Fs.index(max(Fs))], Rs[Fs.index(max(Fs))], max(Fs), x[Fs.index(max(Fs))]]
    print T_N, P_D, R_D
    return aa

def pre_recall(docpath, ldapath,T):
    mmesh = file(docpath+'major_mesh.txt','r').readlines()
    mmesh = map(lambda x: x.strip(), mmesh)
    docs_mesh = []
    for line in file(docpath+'meshes.txt','r'):
        mesh=[]
        for item in line.strip().split('|'):
            if len(item)>1:
                if '*' in item:
                    item = item.replace('*','')
                    if '/' in item:
                        if item.split('/')[0] in mmesh:
                            mesh.append(mmesh.index(item.split('/')[0]))
                    else:
                        if item in mmesh:
                            mesh.append(mmesh.index(item))
        docs_mesh.append(mesh)
    M1 = np.loadtxt(docpath+'cor_matrix_'+str(T)+'.txt')
    M2 = np.loadtxt(ldapath+str(T)+'/final.gamma')
    
    Ps = []
    Rs = []
    Fs =[]
    x=[]
    for th in [50]:
        print '%d of 100' %th
        print T
        
        P = 0
        R = 0
        F = 0
        P_D = 0
        R_D = 0
        T_N = 0
        for i in range(len(docs_mesh)):
            topic_list = [j for j,v in enumerate(M2[i,:].tolist()) if v > 10]
            if len(topic_list)==0:
                continue
            else:
                mesh_list = docs_mesh[i]
                a= len(topic_list)
                a_sum = a
                b = len(mesh_list)
                M_sub = M1[mesh_list][:,topic_list]
                #print M_sub.shape
                c = sum([i>(th/100.0) for i in M_sub.max(1).tolist()])
                for k in range(a):
                    a1 = sum([i>(th/100.0) for i in M_sub[:,k].tolist()])
                    if a1 >1:
                        a_sum += a1-1
                        
                '''for k1 in range(b):
                    a2 = sum([i>(th/100.0) for i in M_sub[k1,:].tolist()])
                    if a2>1:
                        a_sum = a_sum-a2+1'''
                #print a_sum, a, b, c
                T_N += c
                P_D += a_sum
                R_D += b
        #print T_N, P_D, R_D, th
        if T_N>0:
            P = T_N/float(P_D)
            R = T_N/float(R_D)
            F = (2*P*R)/(P+R)
            Ps.append(P)
            Rs.append(R)
            Fs.append(F)
            x.append(th/100.0)
    
    
    aa = [Ps[Fs.index(max(Fs))], Rs[Fs.index(max(Fs))], max(Fs), x[Fs.index(max(Fs))]]
    print T_N, P_D, R_D
    return aa

def doc_mesh(docpath, ldapath,T):
    mmesh = file(docpath+'major_mesh.txt','r').readlines()
    mmesh = map(lambda x: x.strip(), mmesh)
    docs_mesh = []
    for line in file(docpath+'meshes.txt','r'):
        mesh=[]
        for item in line.strip().split('|'):
            if len(item)>1:
                if '*' in item:
                    item = item.replace('*','')
                    if '/' in item:
                        if item.split('/')[0] in mmesh:
                            mesh.append(mmesh.index(item.split('/')[0]))
                    else:
                        if item in mmesh:
                            mesh.append(mmesh.index(item))
        docs_mesh.append(mesh)
    M1 = np.loadtxt(docpath+'cor_matrix_'+str(T)+'.txt')
    M2 = np.loadtxt(ldapath+str(T)+'/final.gamma')
    
    Fls = []
    Frs = []
    x=[]
    for th in range(0,100,2):
        print '%d of 100' %th
        print T
        x.append(th/100.0)
        Fl = 0
        for i in range(len(docs_mesh)):
            topic_list = [j for j,v in enumerate(M2[i,:].tolist()) if v > 1]
            if len(topic_list)==0:
                continue
            else:
                mesh_list = docs_mesh[i]
                a= len(topic_list)
                a_sum =a
                b = len(mesh_list)
                M_sub = M1[mesh_list][:,topic_list]
                #print M_sub.shape
                c = sum([i>(th/100.0) for i in M_sub.max(1).tolist()])
                for k in range(a):
                    a1 = sum([i>(th/100.0) for i in M_sub[:,k].tolist()])
                    if a1 >1:
                        a_sum += a1-1
                Fl +=2*c/float(a_sum+b)
        Fl = Fl/len(docs_mesh)
        Fls.append(Fl)
        
        Fr = 0
        [meshes, meshess] = promesh(docpath)    
        for i1 in range(len(mmesh)):
            topics_list = [i2 for i2,v in enumerate(M1[i1,:].tolist()) if v>(th/100.0)]
            c = 0.0
            f=[]
            #print i1, len(mmesh)
            for item in meshess[mmesh[i1]]:
                doc_topics = [i3 for i3,v in enumerate(M2[item,:].tolist()) if v > 1]
                if len(set(topics_list).intersection(set(doc_topics)))>0:
                    c +=1
            for topic in topics_list:
                f += [i4 for i4,v in enumerate(M2[:,topic].tolist()) if v > 1]
            Fr +=2*c/(len(mmesh[i1])+len(set(f)))
        Fr = Fr/len(mmesh)
        Frs.append(Fr)
    aa = [max(Fls),x[Fls.index(max(Fls))], max(Frs), x[Frs.index(max(Frs))]]
    return aa
    
    '''plt.plot(x,Fls,'r',label='Labeling F_measure')
    plt.plot(x,Frs,'g',label='Retrieval F_measure')
    #plt.plot(Frs, Fls)
    plt.grid(True)
    #legend(loc='upper right')
    plt.xlabel('Retrieval F_measure score')
    plt.ylabel('Labeling F_measure score')
    plt.title('number of topic:200, with abs')
    plt.savefig(docpath+'braf_mesh_topics_200_doc1.jpg')
    plt.show()'''
    #print Fl, len(docs_mesh)
    
    '''c = 0.0
        for mesh_id in mesh_list:
            d=0
            for topic_id in topic_list:
                if M1[mesh_id, topic_id]>0.1485:
                    d +=1
            if d >0:
                c +=1'''
        
    
    '''Fr = 0
    [meshes, meshess] = promesh(docpath)    
    for i1 in range(len(mmesh)):
        topics_list = [i2 for i2,v in enumerate(M1[i1,:].tolist()) if v>0.1485]
        c = 0.0
        f=[]
        print i1, len(mmesh)
        for item in meshess[mmesh[i1]]:
            doc_topics = [i3 for i3,v in enumerate(M2[item,:].tolist()) if v > 1]
            if len(set(topics_list).intersection(set(doc_topics)))>0:
                c +=1
        for topic in topics_list:
            f += [i4 for i4,v in enumerate(M2[:,topic].tolist()) if v > 1]
        Fr +=2*c/len(mmesh[i1])+len(set(f))
    Fr = Fr/len(mmesh)'''
    '''    d =[]
            e = 0.0
            
            for k in topic_list:
                if M1[i1,k]>0.1485:
                    e +=len([i3 for i3,v in enumerate(M2[:,k].tolist()) if v > 1])
                d.append(M1[i1,k])
            if len([x for x in d if x>0.1485])>0:
                c +=1
            if e>0:
                f.append(e/len([x for x in d if x>0.1485]))
        Fr +=2*c/(len(mmesh[i1])+sum(f)/(len(f)-1))'''   
    
    
        #for mesh in docs_mesh:
            
            

def processmesh(docpath, filename):
    [meshes, meshess]=promesh(docpath)
    print len(meshes), len(meshess)
    count1 = []
    count2 = []
    meshes_c={}
    meshess_c={}
    c1=0
    c2=0
    for item in meshes:
        if len(meshes[item])>10:
            if len(meshes[item])<2500:  ## J of B I 1200;  _1  1000; _2; 800; _3 600 ## JMAIA   2000; _1 1500 _2 1200
                c1 +=1
                meshes_c[item]=len(meshes[item])
        count1.append(len(meshes[item]))
        
    for item in meshess:
        if len(meshess[item])>10:
            if len(meshess[item])<2500:
                c2 +=1
                meshess_c[item] = len(meshess[item])
        count2.append(len(meshess[item]))

    print c1, c2

    
    ###########  process major mesh terms only, if want to process the whole mesh, change mmesh to mesh
    
    s1 = sorted(meshes_c.items(),key=lambda item:item[0]) #secondary key: sort alphabetically
    s2 = sorted(s1,key=lambda item:item[1], reverse=True) #primary key: sort by count
    
    ## get WO, the unique word list in the decreasing order
    print 'processing whole data\'s unqiue words and frequency'
    docfile = docpath+filename+'.txt'
    f=open(docfile,'r')
    data = f.read()
    f.close()
    process = Process()
    [s, s_words] = process.preprocessformesh(data)
    words=[]  # get all unqiue words and frequency
    words_f = []
    for item in s:
        if item[1]>1:
            words.append(item[0])
            words_f.append(item[1])

    print 'start processing each mesh term'
    mesh_word_dis={}
    for i in range(len(s2)):
        mesh = s2[i][0]
        mesh_word_dis[mesh]=[0]*len(words)
        doc_list = meshes[mesh]
        f=open(docfile,'r')
        docs = f.readlines()
        f.close()
        process1 = Process()
        docs_data=''
        words_df = {}
        for item in doc_list:
            docs_data = docs_data+' '+docs[item]
            [tokens,bi_tokens, tri_tokens] = process1.compute_documents(docs[item]) ## process for the word df
            for token in list(set(tokens)):
                if words_df.has_key(token):
                    words_df[token] +=1
                else:
                    words_df[token]=1            
        [docs_s, docs_words] = process1.preprocessformesh(docs_data)
            
        for item in docs_s:
            word = item[0]
            if word in words:
                w_tf = docs_words[word]   ## tf
                w_df = words_df[word]                 ## df
                w_tfidf = w_tf*math.log(float((len(doc_list)+1))/w_df)
                mesh_word_dis[mesh][words.index(word)]=w_tfidf
                print mesh, word, w_tf, w_df, len(doc_list), w_tfidf
    
    f = open(docpath+'unqiue_words.txt','w')
    for word in words:
        f.write(word+'\n')
    f.close()
    f = open(docpath+'regular_meshes.txt','w')
    f1 = open(docpath+'regular_mesh_words_dist.txt','w')
    for i in range(len(s2)):
        item = s2[i][0]
        f.write(item+'\n')
        for num in mesh_word_dis[item]:
            f1.write(str(num)+' ')
        f1.write('\n')
    f.close()
    f1.close()
    
    
    
    '''c1 = Counter(count1)
    c2 = Counter(count2)
    print c1
    print c2
    plt.hist(count1,100,facecolor='green')
    plt.hist(count2,100,facecolor='red')
    plt.show()'''

def rdp(va,vb):# Rescaled dot product.
    dmax = sum([a*b for a,b in zip(sorted(va, reverse=True),sorted(vb, reverse=True))])
    dmin = sum([a*b for a,b in zip(sorted(va, reverse=True),sorted(vb))])
    c = sum([a*b for a,b in zip(va,vb)])
    score = (c-dmin)/(dmax-dmin)
    return score
def logit(p):
    return np.log(p) - np.log(1 - p)
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def compare_dists(docpath,T,ldapath):
    betafile = ldapath+str(T)+'/final.beta'
    f1 = open(betafile,'r')
    M = len(f1.readlines())
    meshfile = docpath+'regular_mesh_words_dist.txt'
    f2 = open(meshfile,'r')
    N = len(f2.readlines())
    print N, M
    matches = np.zeros((N,M))  # similarity of mesh and topic based on word distribution
    matches_refer=np.zeros((N,N))  # similarity of mesh and mesh based on word distribution
    n=0
    for line in file(meshfile,'r'):
        refertopic = map(float, line.split())
        print min(refertopic), max(refertopic), sum(refertopic), len(refertopic)
        refertopic = [i+1/M for i in refertopic]
        c_sum = sum(refertopic)
        refertopic = [j/c_sum for j in refertopic]
        m = 0
        print [n,T]
        
        for topic in file(betafile,'r'):   # compare to topics
            topic = map(float, topic.split())
            topic = [math.exp(x) for x in topic]
            #print topic[0], refertopic[0]
            score = rdp(refertopic, topic)
            matches[n,m] = score
            m +=1
        '''m=0
        for line_mesh in file(meshfile,'r'): # compare to mesh
            refer = map(float, line_mesh.split())
            refer = [i1+1/M for i1 in refer]
            c_sum = sum(refer)
            refer = [j1/c_sum for j1 in refer]
            score = rdp(refertopic, refer)
            matches_refer[n,m]=score
            m +=1'''
        n +=1
    np.savetxt(docpath+'mesh_topic_cor_matrix_'+str(T)+'.txt',matches)
    #np.savetxt(docpath+'cor_mesh_matrix_doc1.txt',matches_refer)
def compare_sim_dist(docpath):
    M_topic = np.loadtxt(docpath+'cor_matrix_200.txt')
    M_mesh = np.loadtxt(docpath+'cor_mesh_matrix.txt')
    [N,M] = M_topic.shape
    M_topic_mesh = np.zeros([N,M])
    for i in range(N):
        mesh = M_mesh[:,i].tolist()
        m_sum = sum(mesh)
        mesh = [i1/m_sum for i1 in mesh]
        print i
        for j in range(M):
            topic = M_topic[:,j].tolist()
            t_sum = sum(topic)
            topic = [j1/t_sum for j1 in topic]
            score = rdp(mesh, topic)
            M_topic_mesh[i,j] = score
    np.savetxt(docpath+'cor_matrix_topic_mesh.txt', M_topic_mesh)

def topic_diagnostics(docpath, T):
    M1 = np.loadtxt(docpath+'mesh_topic_cor_matrix_'+str(T)+'.txt')
    [N, M] = M1.shape
    c_junk = 0.0
    c_fused = 0.0
    c_match_topic = 0.0
    c_missing = 0.0
    c_repeated=0.0
    c_match_reference = 0.0
    for i in range(M):
        docs_list = [x for x in M1[:,i].tolist() if x>0.55]
        if len(docs_list) == 0:
            c_junk +=1
        if len(docs_list) == 1:
            c_match_topic +=1
        if len(docs_list) >1:
            c_fused +=1
    for j in range(N):
        docs_list = [x for x in M1[j,:].tolist() if x>0.55]
        if len(docs_list) == 0:
            c_missing +=1
        if len(docs_list) == 1:
            c_match_reference +=1
        if len(docs_list) >1:
            c_repeated +=1 
    a = [c_junk, c_match_topic, c_fused, c_junk/M, c_match_topic/M, c_fused/M,c_missing, c_match_reference, c_repeated, c_missing/N, c_match_reference/N, c_repeated/N]
         #[3.0, 2.0, 70.0, 0.04, 0.02666666666666667, 0.9333333333333333, 97.0, 53.0, 139.0, 0.3356401384083045, 0.18339100346020762, 0.4809688581314879]

    print a
    return a
    '''print c_junk, c_match_topic, c_fused, c_junk/M, c_match_topic/M, c_fused/M
    print c_missing, c_match_reference, c_repeated, c_missing/N, c_match_reference/N, c_repeated/N''' 

def evaluation_measures(docpath):
    M1 = np.loadtxt(docpath+'cor_matrix_200.txt')
    M2 = np.loadtxt(docpath+'cor_doc_matrix.txt')
    [N,M] = M1.shape
    Fl = 0
    labels = []
    for i in range(N):
        a = []
        b = []
        c = 0
        for j in range(M):
            if M2[i,j]>0.5:
                c +=1
                a.append(1)
                if M1[i,j]>0.3:
                    b.append(1)
                else:
                    b.append(0)
                #a = a + 2*M1[i,j]*M2[i,j]
                #b = b + M1[i,j]+M2[i,j]
        if c>0:
            Fl = Fl + b.count(1)/float(len(a))
            labels.append(c)
    Fl = Fl/len(labels)
    
    Fr = 0
    labels1=[]
    for i1 in range(M):
        a = []
        b = []
        c = 0
        for j1 in range(N):
            if M2[j1,i1]>0.5:
                c +=1
                a.append(1)
                if M1[j1,i1]>0.3:
                    b.append(1)
                else:
                    b.append(0)
                #a = a + 2*M1[j1,i1]*M2[j1,i1]
                #b = b + M1[j1,i1]+M2[j1,i1]
        if c>0:
            Fr = Fr + b.count(1)/float(len(b))
            labels1.append(c)
    Fr = Fr/len(labels1)
    print Fl, Fr, len(labels), len(labels1)
        
    
def compare_mesh_topic(docpath):
    M1 = np.loadtxt(docpath+'cor_matrix_200.txt')
    M2 = np.loadtxt(docpath+'cor_doc_matrix.txt')
    [N,M] = M1.shape
    C_doc=[]
    
    C_topic_mesh=[]
    C_doc1=[]
    C_topic_mesh1=[]
    for i in range(N):
        for j in range(M):
            if M2[i,j]>0.2:
                C_doc.append(i)
                if M1[i,j]>0.5:
                    C_topic_mesh.append(i)
            if M1[i,j]>0.5:
                C_topic_mesh1.append(i)
                if M2[i,j]>0.2:
                    C_doc1.append(i)
    a = OrderedDict.fromkeys(C_doc).keys()
    b = OrderedDict.fromkeys(C_topic_mesh).keys()
    c = OrderedDict.fromkeys(C_doc1).keys()
    d = OrderedDict.fromkeys(C_topic_mesh1).keys()
    print len(a), len(b), len(b)/float(len(a))
    print len(c), len(d), len(c)/float(len(d))
    print len(C_doc1), len(C_topic_mesh1), float(len(C_doc1))/len(C_topic_mesh1)
                
    
    
def compare_docs_dist(docpath):
    [meshes, meshess]=promesh(docpath)
    meshess_c={}
    for item in meshess:
        if len(meshess[item])>10:
            if len(meshess[item])<3500:
                meshess_c[item] = len(meshess[item])
    s1 = sorted(meshess_c.items(),key=lambda item:item[0]) #secondary key: sort alphabetically
    s2 = sorted(s1,key=lambda item:item[1], reverse=True) #primary key: sort by count
    print s2[0]
    M = np.loadtxt(docpath+'final.gamma')
    [m,n] = M.shape
    topics_docs = [0]*n
    for i in range(n-1):
        doc_list= M[:,i].tolist()
        k=0
        topics_docs[i] = [k for k,x in enumerate(doc_list) if x > 0.0178]
    d=[]
    M_inter_topic = np.zeros((len(s2),n))
    M_inter_mesh = np.zeros((len(s2),n))
    M_inter= np.zeros((len(s2),n))
    for j in range(len(s2)):
        mesh = s2[j][0]
        mesh_docs=meshess[mesh]
        i=0
        c = 0
        for j1 in range(n-1):
            topic_docs = topics_docs[j1]
            if len(topic_docs)>0:
                share_docs = set(mesh_docs).intersection(set(topic_docs))
                p1 = float(len(share_docs))/len(mesh_docs)
                p2 = float(len(share_docs))/len(topic_docs)
                M_inter_topic[j,j1] = p2
                M_inter_mesh[j,j1] = p1
                M_inter[j,j1] = (p1+p2)/2
                if p1 > 0.3:
                    c +=1
                    print p1,p2,(p1+p2)/2, len(mesh_docs),len(topic_docs), mesh
        d.append(c)
    print len(s2), d.count(0)
    np.savetxt(docpath+'cor_doc_matrix_topic.txt',M_inter_topic)
    np.savetxt(docpath+'cor_doc_matrix_mesh.txt',M_inter_mesh)
    np.savetxt(docpath+'cor_doc_matrix.txt',M_inter)
    return()
    
    
def analysismatrx(docpath):
    ### write popular mesh terms
    D = [] # reference topics
    T = [0]*200 # generated topics
    for line in file(docpath+'cor_matrix_100.txt','r'):
        line = map(float, line.split())
        c = 0
        for item in line:
            if item>0.5:
                c +=1
        D.append(c)
        for i in range(len(line)):
            if T[i] < line[i]:
                T[i] = line[i]
                
        print c, max(line)
    print D.count(0), max(D)
    ct=0
    for i1 in T:
        if i1>0.5:
            ct +=1
    print ct
    exit()
    MD = []
    MDC = []
    
    for j in range(len(D)):
        if D[j]>15:
            MD.append(j)
            MDC.append(D[j])
            
    mmesh = file(docpath+'major_mesh.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    mmesh = map(lambda x: x.strip(), mmesh)    
    
    vocab = file(docpath+'unqiue_words.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)
    
    f1 = open(docpath+'major_mesh_words_dist.txt','r')
    mesh_dists = f1.readlines()
    indices1 = [0]*len(MD)
    i=0
    for i in range(len(MD)):
        mesh = mesh_dists[MD[i]]
        mesh = map(float, mesh.split())
        indices1[i] = range(len(vocab))
        indices1[i].sort(lambda x,y: -cmp(mesh[x], mesh[y]))
    
    c = csv.writer(open(docpath+"popular_mesh.csv", "wb"))
    MDC = [str(z) for z in MDC]
    c.writerow(MDC)
    SMD = []
    for item in MD:
        SMD.append(mmesh[item])
    c.writerow(SMD)
    i=0
    for i in range(50):
        Mdata=[]
        for item in indices1:
            Mdata.append(vocab[item[i]])
        c.writerow(Mdata)
    
    ###### write high similarity of topics and meshes
    '''D = []
    T = []
    d = 0
    for line in file(docpath+'cor_matrix.txt','r'):
        line = map(float, line.split())
        t=0
        for item in line:
            if item>0.9:
                D.append(d)
                T.append(t)
            t +=1
        d +=1
    print D, T
    mmesh = file(docpath+'major_mesh.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    mmesh = map(lambda x: x.strip(), mmesh)
    
    vocab = file(docpath+'unqiue_words.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)
    
    f1 = open(docpath+'major_mesh_words_dist.txt','r')
    f2 = open(docpath +'final.beta','r')
    mesh_dists = f1.readlines()
    topic_dists = f2.readlines()
    indices1 = [0]*len(D)
    indices2 = [0]*len(D)
    for i in range(len(D)):
        mesh = mesh_dists[D[i]]
        mesh = map(float, mesh.split())
        indices1[i] = range(len(vocab))
        indices2[i] = range(len(vocab))
        indices1[i].sort(lambda x,y: -cmp(mesh[x], mesh[y]))
        
        topic = topic_dists[T[i]]
        topic = map(float, topic.split())
        indices2[i].sort(lambda x,y: -cmp(topic[x], topic[y]))
    
    c = csv.writer(open(docpath+"mesh_sim_topic.csv", "wb"))
    c.writerow([mmesh[D[0]],"Topic "+str(T[0]) ,mmesh[D[1]],"Topic "+str(T[1]) ,mmesh[D[2]],"Topic "+str(T[2]) ,mmesh[D[3]],"Topic "+str(T[3]) ,mmesh[D[4]],"Topic "+str(T[4]) ,mmesh[D[5]],"Topic "+str(T[5])])
    for i in range(50):
        c.writerow([vocab[indices1[0][i]],vocab[indices2[0][i]],vocab[indices1[1][i]],vocab[indices2[1][i]], vocab[indices1[2][i]],vocab[indices2[2][i]],vocab[indices1[3][i]],vocab[indices2[3][i]],vocab[indices1[4][i]],vocab[indices2[4][i]],vocab[indices1[5][i]],vocab[indices2[5][i]]])
        
    '''
    ### write popular topics----------------------
    '''D = [] # reference topics
    T = np.empty((200,0)).tolist() # generated topics
    for line in file(docpath+'cor_matrix.txt','r'):
        line = map(float, line.split())
        for i in range(len(line)):
            if line[i]>0.5:
                T[i].append(line[i])
        
    for item in T:
        D.append(len(item))
    MD = []
    MDC = []
    for j in range(len(D)):
        if D[j]>20:
            MD.append(j)
            MDC.append(D[j])
            
    mmesh = file(docpath+'major_mesh.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    mmesh = map(lambda x: x.strip(), mmesh)    
    
    vocab = file(docpath+'unqiue_words.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)
    
    f1 = open(docpath+'final.beta','r')
    mesh_dists = f1.readlines()
    indices1 = [0]*len(MD)
    i=0
    for i in range(len(MD)):
        mesh = mesh_dists[MD[i]]
        mesh = map(float, mesh.split())
        indices1[i] = range(len(vocab))
        indices1[i].sort(lambda x,y: -cmp(mesh[x], mesh[y]))
    
    c = csv.writer(open(docpath+"popular_topics.csv", "wb"))
    MDC = [str(z) for z in MDC]
    c.writerow(MDC)
    SMD = []
    for item in MD:
        SMD.append('Topic '+str(item))
    c.writerow(SMD)
    i=0
    for i in range(50):
        Mdata=[]
        for item in indices1:
            Mdata.append(vocab[item[i]])
        c.writerow(Mdata)'''
        
def hdp_num(docpath):
    N1=[]
    M = np.loadtxt(docpath+'final.topics')
    N1.append(((M.max(axis=1))>5).sum())
    print N1
    
def txt2csv(docpath):
    M = np.loadtxt(docpath+'transformed_cor_matrix_200.txt')
    mmesh = file(docpath+'major_mesh.txt', 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    mmesh = map(lambda x: x.strip(), mmesh)  
    c = csv.writer(open(docpath+"transformed_cor_matrix_200.csv", "wb"))
    firstrow = ['matrix']
    for i in range(1,201):
        firstrow.append('Topic' +str(i))
    c.writerow(firstrow)
    for i in range(len(mmesh)):
        print M[i].tolist()
        row = M[i].tolist()
        row.reverse()
        row.append(mmesh[i])
        row .reverse()
        c.writerow(row)
    return()
    
def filterdocs(docpath, docfile, meshfile):
    f1 = open(docfile,'r')
    f2 = open(meshfile,'r')
    docs = f1.readlines()
    docs = map(lambda x: x.strip(), docs)
    meshes = f2.readlines()
    meshes = map(lambda x: x.strip(), meshes)
    f1.close()
    f2.close()
    c=0
    for i in range(len(docs)):
        if '.' not in docs[i]:
            c +=1
            print docs[i]
        #if len(docs[i].split('.')[1])>10:
        #    c +=1
    print c
    return()
#[c_junk, c_match_topic, c_fused, c_junk/M, c_match_topic/M, c_fused/M,c_missing, c_match_reference, c_repeated, c_missing/N, c_match_reference/N, c_repeated/N]
#[max(Fls),x[Fls.index(max(Fls))], max(Frs), x[Frs.index(max(Frs))]] 
def graph(A, TT,docpath):
    '''Junk = []
    Fused=[]
    ReAFused=[]
    Missing=[]
    Resolved=[]
    Repeated=[]'''
    P = []
    R = []
    F = []
    Fx = []
    for item in A:
        '''Junk.append(item[3])
        Fused.append(item[4])
        ReAFused.append(item[5])
        Missing.append(item[9])
        Resolved.append(item[10])
        Repeated.append(item[11])'''
        P.append(item[3])
        R.append(item[4])
        F.append(item[5])
        #Fx.append(item[3])
    print P
    print R
    print F
    #print Fx
    plt.plot(TT,P,'r-o',label='Unmatched')
    plt.plot(TT,R,'g-',label='Matched')
    plt.plot(TT,F,'b-*',label = 'Fused')
    #plt.plot(TT,Fx,'bo', label = 'max F_score similarity threshold')
    #plt.plot(TT,Repeated,'b',label='Repeated')
    plt.grid(True)
    legend(loc='upper right')
    plt.xlabel('T : Number of Topics')
    plt.ylabel('Percent')
    plt.title('UnMatched, Matched, and Fused topic vs Number of Topics')
    plt.savefig(docpath+'umf_0.55.jpg')
    plt.show()
    
    
    
def main():
    docpath = './journal of the american medical informatics association/'
    ldapath='/Users/zyu4/Documents/Research/topicmodels/Topic-Browser/lda-c-dist/journal-of-the-american-medical-informatics-association/resultforlda'
    TT1 = range(200,750,50)
    TT2 = range(5,155,5)
    TT = TT2+TT1
    A = []
    for T in TT:
        #compare_dists(docpath,T,ldapath)
        aa = topic_diagnostics(docpath, T)
        #aa= precision_recall(docpath, ldapath,T)
        #a = topic_diagnostics(docpath, T)
        A.append(aa)
    graph(A, TT, docpath)
    
    
   
#docpath = './journal of the american medical informatics association/'
#filename = 'docs'
#processmesh(docpath,filename)
#analysismatrx(docpath)
#hdp_num(docpath)
#compare_dists(docpath)
#txt2csv(docpath)
#compare_docs_dist(docpath)
#compare_sim_dist(docpath)
#compare_mesh_topic(docpath)
#evaluation_measures(docpath)
#doc_mesh(docpath)
#topic_diagnostics(docpath)
#docfile = docpath+'4387.txt'
#meshfile = docpath + 'meshes.txt'
#filterdocs(docpath, docfile, meshfile)
#processmesh()
#