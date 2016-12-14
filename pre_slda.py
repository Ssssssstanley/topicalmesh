## prepare 5-folder slda data
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import os
from processData import Process

class Solution():
    def __init__(self):
        return
        
    def splitData(self,name):
        if not os.path.isdir('./slda/'+name):
            os.mkdir('./slda/'+name)
        ## get refer list
        refer_list = []
        for line in open('./'+name+'/pmids_and_refer.txt','r'):
            if line.strip().split()[-1] == '1':
                refer_list.append(1)
            else:
                refer_list.append(0)
        C_0 = refer_list.count(0)
        C_1 = refer_list.count(1)
        cc = C_0*1.0/C_1
        print cc
        refer_list = np.asarray(refer_list)
        
        ## get LDA data to list
        f = open('./'+name+'/docs.txt','r')
        data = f.readlines()
        f.close()
        skf = StratifiedKFold(refer_list, 5)
        cnt =1
        for train,test in skf:
            f1 = open('./slda/'+name+'/train_data_'+str(cnt)+'.txt','w')
            f2 = open('./slda/'+name+'/train_label_'+str(cnt)+'.txt','w')
            for i in train:
                f1.writelines(data[i])
                f2.writelines(str(refer_list[i])+'\n')
            f1.close()
            f2.close()
            f1 = open('./slda/'+name+'/test_data_'+str(cnt)+'.txt','w')
            f2 = open('./slda/'+name+'/test_label_'+str(cnt)+'.txt','w')
            for i in test:
                f1.writelines(data[i])
                f2.writelines(str(refer_list[i])+'\n')
            f1.close()
            f2.close()
            cnt +=1
        return 0
    def pro_slda_data(self,name):
        docpath = './slda/'+name
        for i in range(1,6):
            f_train = 'train_data_'+str(i)
            f_test = 'test_data_'+str(i)
            p = Process()
            p.preprocessforlda(docpath, f_train)
            p.preprocessforlda(docpath, f_test)
        ## check 0 element
        for j in range(1,6):
            ### check trainning data
            f1 = open('./slda/'+name+'/lda_train_data_'+str(j)+'.txt','r')
            data = f1.readlines()
            data = map(lambda x:x.strip(), data)
            f1.close()
            f2 = open('./slda/'+name+'/train_label_'+str(j)+'.txt','r')
            refer = f2.readlines()
            refer = map(lambda x:x.strip(), refer)
            f2.close()
            f1 = open('./slda/'+name+'/lda_train_data_'+str(j)+'.txt','w')
            f2 = open('./slda/'+name+'/train_label_'+str(j)+'.txt','w')
            for i in range(len(data)):
                if len(data[i]) <2:
                    print data[i]
                else:
                    f1.writelines(data[i]+'\n')
                    f2.writelines(refer[i]+'\n')
            f1.close()
            f2.close()
            ## check testing data's 0
            f1 = open('./slda/'+name+'/lda_test_data_'+str(j)+'.txt','r')
            data = f1.readlines()
            data = map(lambda x:x.strip(), data)
            f1.close()
            f2 = open('./slda/'+name+'/test_label_'+str(j)+'.txt','r')
            refer = f2.readlines()
            refer = map(lambda x:x.strip(), refer)
            f2.close()
            f1 = open('./slda/'+name+'/lda_test_data_'+str(j)+'.txt','w')
            f2 = open('./slda/'+name+'/test_label_'+str(j)+'.txt','w')
            for i in range(len(data)):
                if len(data[i]) <2:
                    print data[i]
                else:
                    f1.writelines(data[i]+'\n')
                    f2.writelines(refer[i]+'\n')
            f1.close()
            f2.close()
    def prf(self, name):
        P,R, F =[],[],[]
        c = 0
        for i in range(1,6):
            f1 = open('./slda/'+name+'/cv_'+str(i)+'_test/inf-labels.dat','r')
            predicts = f1.readlines()
            predicts = map(lambda x:x.strip(), predicts)
            f1.close()
            f2 = open('./slda/'+name+'/test_label_'+str(i)+'.txt','r')
            test = f2.readlines()
            test = map(lambda x:x.strip(),test)
            f2.close()
            tp,fp,fn = 0, 0, 0
            for i in range(len(test)):
                if test[i] =='1' and predicts[i]=='1':
                    tp +=1
                if test[i] =='1' and predicts[i]=='0':
                    fn +=1
                if test[i]=='0' and predicts[i]=='1':
                    fp +=1
            if tp >0:
                p = 1.0*tp/(tp+fp)
                r = 1.0*tp/(tp+fn)
                f = 2*p*r/(p+r)
                P.append(p)
                R.append(r)
                F.append(f)
                c+=1
        if len(F)>0:
            return 1.0*sum(P)/c, 1.0*sum(R)/c, 1.0*sum(F)/c
        else:
            
            return 0,0,0
            
            
def main():

    pp = Solution()
    nameMap = {'Statins':'Statins'}#{'ACEInhibitors':'Angiotensin-Converting Enzyme Inhibitors','ADHD':'Clonidine Attention Deficit Disorder with Hyperactivity','Antihistamines':'Histamine H1 Antagonists','AtypicalAntipsychotics':'Antipsychotic Agents','BetaBlockers':'Adrenergic beta-Antagonists','CalciumChannelBlockers':'Calcium Channel Blockers','Estrogens':'Estrogen Replacement Therapy','NSAIDS':'Anti-Inflammatory Agents, Non-Steroidal','Opiods':'Analgesics, Opioid','OralHypoglycemics':'Hypoglycemic Agents','ProtonPumpInhibitors':'Omeprazole','SkeletalMuscleRelaxants':'Muscle Relaxants, Central','Triptans':'Tryptamines', 'UrinaryIncontinence':'Urinary Incontinence'}
    nameindex=[4]#[3,13,4,1,5,6,3,1,1,2,1,4,8,12]
    nameMap1 = sorted(nameMap)
    P = []
    R = []
    F= []
    for i in range(len(nameindex)):
        name = nameMap1[i]
        [p,r,f] = pp.prf(name)
        P.append(p)
        R.append(r)
        F.append(f)
    #p.splitData(name)
    #p.pro_slda_data(name)
    print P
    print R
    print F
main()
    
