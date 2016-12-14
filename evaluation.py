#### evaluation of topic models on information retrieval

import os
import csv
from processData import Process
import processMesh
from searchPubMed import queryPubMed

def process_data(name):
    if not os.path.isdir('./'+name):
        os.mkdir('./'+name)
    f1 = open('./'+name+'/docs.txt','w')
    f2 = open('./'+name+'/pmids_and_refer.txt','w')
    f3 = open('./'+name+'/meshes.txt','w')
    f4 = open('./'+name+'/pmids.txt','w')
    with open('./datasets/'+name+'_processed.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            f1.write(row[1]+' '+row[4]+'\n')
            #print row[1],row[4]
            f2.write(row[0]+' '+row[6]+'\n')
            f3.write(row[5]+'\n')
            f4.write(row[0]+'\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f.close()
    p1 = Process()
    docpath = './'+name
    p1.preprocessforlda(docpath,'docs')
    #processMesh.processmesh(docpath+'/', 'docs')
def get_meshes(name):
    f = open('./'+name+'/pmids.txt','r')
    ids = f.readlines()
    docpath = './'+name
    queryPubMed(docpath,ids)
def process_mesh(name):
    docpath = './'+name+'/'
    processMesh.processmesh(docpath, 'docs')
def build_corr_matrix(name):
    docpath = './'+name+'/'
    ldapath = './'+name+'/resultforlda'
    #for T in range(5,300,5):
    T = 55
    processMesh.compare_dists(docpath,T,ldapath)
def process_pmid(name):
    f = open('./'+name+'/pmids.txt','r')
    ids = f.readlines()
    docpath = './'+name
    queryPubMed(docpath,ids)
def main():
    name = 'UrinaryIncontinence'
    #process_pmid(name)
    #p1 = Process()
    #docpath = './'+name
    #p1.preprocessforlda(docpath,'docs')
    #process_data(name)
    #get_meshes(name)
    #process_mesh(name)
    build_corr_matrix(name)
main()