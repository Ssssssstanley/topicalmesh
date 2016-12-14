## preprocess
from __future__ import division
import nltk
from nltk import bigrams
from nltk import trigrams
from collections import Counter

class Process:
    def __init__(self):
        self.common = ['the','of','and','a','to','in','is','you','that','it',
			'he','was','for','on','are','as','with','his','they','i',
			'at','be','this','have','from','or','one','had','by','word',
			'but','not','what','all','were','we','when','your','can','said',
			'there','use','an','each','which','she','do','how','their','if',
			'will','up','other','about','out','many','then','them','these','so',
			'some','her','would','make','like','him','into','time','has','look',
			'two','more','write','go','see','number','no','way','could','people',
			'my','than','first','water','been','call','who','oil','its','now',
			'find','long','down','day','did','get','come','made','may','part','missing']
        self.wordIndex = dict() #will be a count of each word in the input text
        self.total = 0 #total words
        self.unique = 0 #unique words
        self.uniquewords = []

    def compute_documents(self, data):
        tokens = nltk.word_tokenize(data)
        tokens = [token.translate(None,".!?,;:*+=\)\(\[\]\\\n/'\"").replace('--','') for token in tokens]
        tokens = [token.lower() for token in tokens if len(token) > 2]#remove punctuation, numbers, and newlines
        bi_tokens = bigrams(tokens)
        tri_tokens = trigrams(tokens)
        return[tokens,bi_tokens, tri_tokens]
    def stopwords(self,tokens):
        filtered_tokens = [w for w in tokens if not w in self.common]
        words = Counter(filtered_tokens)
        s1 = sorted(words.items(),key=lambda item:item[0]) #secondary key: sort alphabetically
        s2 = sorted(s1,key=lambda item:item[1], reverse=True) #primary key: sort by count
        return [s2, words]
    
    def preprocessformesh(self,data):
        [tokens,bi_tokens, tri_tokens] = self.compute_documents(data)
        [s2,words] = self.stopwords(tokens)
        return [s2, words]
    
    def preprocessforlda(self,docpath, D):
        docfile = docpath+'/'+str(D)+'.txt'
        print 'Starting process'
        f=open(docfile,'r')
        data = f.read()
        [tokens,bi_tokens, tri_tokens] = self.compute_documents(data)
        [s2,words] = self.stopwords(tokens)


        print 'process WO'
        f = open(docpath+'/WO_'+str(D)+'.txt','w')
        ## remove words with frequency 1
        for item in s2:
            if item[1]>1:
                #print item[0],item[1]
                self.uniquewords.append(item[0])
                f.writelines(item[0]+'\n')
                if self.wordIndex.has_key(item[0]):
                    self.wordIndex[item[0]]=self.wordIndex[item[0]]+1
                else:
                    self.wordIndex[item[0]]=1
        f.close()
        #print len(uniquewords)
        doc_count=0
        print 'process data for lda'
        f=open(docpath+'/dataforlda_'+str(D)+'.txt','w')
        for line in open(docfile,'r'):
            doc_data=line.strip()
            [doc_tokens,bi_tokens, tri_tokens] =self.compute_documents(doc_data)
            [doc_s2, doc_words] = self.stopwords(doc_tokens)
            f.write(str(doc_count)+' ')
            print doc_count
            print doc_data
            doc_count = doc_count+1

            for word in doc_s2:
                if word[0] in self.uniquewords:
                    w = self.uniquewords.index(word[0])
                    f.write(str(w)+':'+str(doc_words[word[0]]) +' ')
            f.write('\n')
        f.close()
        '''f = open(docpath+'/dataforlda_'+str(D)+'.dat','w')
        for line in open(docpath+'/dataforlda_'+str(D)+'.txt','r'):
            data = line.strip().split()
            for i in data:
                f.write(str(i)+' ')
            f.write('\n')
        f.close()'''
        f = open(docpath+'/lda_'+str(D)+'.txt','w')
        for line in file(docpath+'/dataforlda_'+str(D)+'.txt','r'):
            ss = line.strip().split()
            if len(ss) == 0: continue
            doclen = str(len(ss[1:]))
            f.write(doclen+' ')
            for item in ss[1:]:
                f.write(item+' ')
            f.write('\n')
        f.close()
        
        
#process = Process()
#D = 'sifei'
#docpath = './sifei'
#process.preprocessforlda(docpath, D)