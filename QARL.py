import nltk
from nltk import word_tokenize,pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import cPickle
import pickle
import pandas as pd
import numpy as np 
import urllib2
import scipy
import scipy.optimize
import time
import sys
import csv
import re
import gc
import os

# This condition is here since I don't have PyLucene on my Windows system
# if (len(sys.argv) >= 3) and (sys.argv[1] == 'prep') and (int(sys.argv[2]) >= 21):
import lucene
from java.io import File, StringReader
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField
from org.apache.lucene.search.similarities import BM25Similarity 
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, MultiFields, Term
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser, QueryParser
from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.lucene.util import BytesRefIterator, Version


# Q-network learning
import random
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os
import string
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#################################################################################################
# I/O functions
#################################################################################################

def read_input_file(base_dir, filename, max_rows=999999999, use_cols=None, index_col=0, sep=','):
    '''
    Read an input file
    '''
#     print '=> Reading input file %s' % filename
    dataf = pd.read_table('%s/%s' % (base_dir, filename), index_col=index_col, nrows=max_rows, sep=sep)
    if 'correctAnswer' in dataf.columns:
        dataf = dataf[[(ca  in ['A','B','C','D']) for ca in dataf['correctAnswer']]] 
    dataf['ID'] = dataf.index
    return dataf


#################################################################################################
# Parsers
#################################################################################################
                
class WordParser(object):
    '''
    WordParser - base class for parsers
    '''
    def __init__(self, min_word_length=2, max_word_length=25, ascii_conversion=True):
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.ascii_conversion = ascii_conversion
    def filter_words_by_length(self, words):
        return [word for word in words if len(word)>=self.min_word_length and len(word)<=self.max_word_length]
    def convert_ascii(self, text):
        if self.ascii_conversion:
            return AsciiConvertor.convert(text)
        return text
    def parse(self, text, calc_weights=False):
        if calc_weights:
            return text, {}
        else:
            return text
    
class NltkTokenParser(WordParser):
    '''
    NLTK parser, supports tags (noun, verb, etc.)
    '''
    # See nltk.help.upenn_tagset('.*')
    TAG_TO_POS = {'NN': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN, 'NNS': wn.NOUN, 
                  'VB': wn.VERB, 'VBD': wn.VERB, 'VBG' : wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
                  'RB': wn.ADV,  'RBR': wn.ADV , 'RBS' : wn.ADV , 'RP' : wn.ADV, 
                  'JJ': wn.ADJ , 'JJR': wn.ADJ , 'JJS' : wn.ADJ }

    def __init__(self, min_word_length=2, word_func=None, tolower=True, ascii_conversion=True, tuples=[1], ignore_special_words=True,
                 tag_weight_func=lambda tag: 1.0, word_func_requires_tag=True):
        self.word_func = word_func
        self.tolower = tolower
        self.tuples = tuples
        self.ignore_special_words = ignore_special_words
        self.tag_weight_func = tag_weight_func
        self.word_func_requires_tag = word_func_requires_tag
        assert set([1]).issuperset(self.tuples)
        WordParser.__init__(self, min_word_length=min_word_length, ascii_conversion=ascii_conversion)

    def parse(self, text, calc_weights=False):
        text = self.convert_ascii(text)
        tokens = nltk.word_tokenize(text)
        if calc_weights or self.word_func_requires_tag:
            tagged_tokens = nltk.pos_tag(tokens)
        else: # save time - don't use tags
            tagged_tokens = zip(tokens,[None]*len(tokens))
        ##tagged_tokens = nltk.pos_tag(tokens)
        words, weights, self.tags = [], [], []
        for word,tag in tagged_tokens:
            if len(word)>=self.min_word_length and len(word)<=self.max_word_length:
                words.append(word.lower() if self.tolower else word)
                weights.append(self.tag_weight_func(tag) if calc_weights else 0) 
                self.tags.append(tag)
        self.word_weights = {}
        # Filter special words
        if self.ignore_special_words:
            filtered = np.array([SpecialWords.filter1(word) for word in words])
            if np.all(filtered == False): # we're about to filter all the words -> instead, don't filter anything
                filtered = [True]*len(words)
        else:
            filtered = [True]*len(words) # no filtering

        if self.word_func is not None:
            fwords = []
            for word,wgt,fltr,tag in zip(words, weights, filtered, self.tags):
                if fltr:
                    try:
                        fword = str(self.word_func(word, NltkTokenParser.TAG_TO_POS.get(tag,None)))
                    except UnicodeDecodeError:
                        fword = word
#                     fword = self._apply_word_func(word, tag)
                    if type(fword)==list:
                        fwords += fword
                        if calc_weights:
                            for fw in fword:
                                self.word_weights[fw] = np.max([self.word_weights.get(fw,-1.0), wgt])
                    else:
                        fwords.append(fword)
                        if calc_weights:
                            self.word_weights[fword] = np.max([self.word_weights.get(fword,-1.0), wgt])
            words = fwords
        else:
            fwords = []
            for word,wgt,fltr in zip(words, weights, filtered):
                if fltr:
                    fwords.append(word)
                    if calc_weights:
                        self.word_weights[word] = np.max([self.word_weights.get(word,-1.0), wgt])
            words = fwords
        ret_words = []
        if 1 in self.tuples:
            ret_words += words
        if calc_weights:
            return ret_words, self.word_weights
        else:
            return ret_words

class SimpleWordParser(WordParser):
    '''
    SimpleWordParser - supports tuples
    '''
    def __init__(self, stop_regexp='[\-\+\*_\.\:\,\;\?\!\'\"\`\\\/\)\]\}]+ | [\*\:\;\'\"\`\(\[\{]+|[ \t\r\n\?]', 
                 min_word_length=2, word_func=None, tolower=True, ascii_conversion=True, ignore_special_words=True,
                 split_words_regexp=None, # for additional splitting of words, provided that all parts are longer than min_word_length, eg, split_words_regexp='[\-\+\*\/\,\;\:\(\)]' 
                 tuples=[1]):
        self.stop_regexp = re.compile(stop_regexp)
        self.word_func = word_func
        self.tolower = tolower
        self.ignore_special_words = ignore_special_words
        self.split_words_regexp = None if split_words_regexp is None else re.compile(split_words_regexp)
        self.tuples = tuples
        assert set([1,2,3,4]).issuperset(self.tuples)
        WordParser.__init__(self, min_word_length=min_word_length, ascii_conversion=ascii_conversion)
        
    def parse(self, text, calc_weights=False):
        if self.tolower:
            text = text.lower()
        text = ' ' + text.strip() + ' ' # add ' ' at the beginning and at the end so that, eg, a '.' at the end of the text will be removed, and "'''" at the beginning will be removed 
        text = self.convert_ascii(text)
        words = re.split(self.stop_regexp, text)
        if self.split_words_regexp is not None:
            swords = []
            for word in words:
                w_words = re.split(self.split_words_regexp, word)
                if len(w_words) == 1:
                    swords.append(w_words[0])
                else:
                    if np.all([len(w)>=self.min_word_length for w in w_words]):
                        swords += w_words
                    else:
                        swords.append(word) # don't split - some parts are too short
            words = swords
        if self.ignore_special_words:
            words = SpecialWords.filter(words)
        if self.word_func is not None:
            fwords = []
            for word in words:
                try:
                    fword = str(self.word_func(word))
                except UnicodeDecodeError:
                    fword = word
                fwords.append(fword)
            words = fwords
        words = self.filter_words_by_length(words)
        ret_words = []
        if 1 in self.tuples:
            ret_words += words
        if 2 in self.tuples:
            ret_words += ['%s %s'%(words[i],words[i+1]) for i in range(len(words)-1)]
        if 3 in self.tuples:
            ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+2]) for i in range(len(words)-2)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if 4 in self.tuples:
            ret_words += ['%s %s %s %s'%(words[i],words[i+1],words[i+2],words[i+3]) for i in range(len(words)-3)]
            if 3 in self.tuples:
                ret_words += ['%s %s %s'%(words[i],words[i+2],words[i+3]) for i in range(len(words)-3)]
                ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+3]) for i in range(len(words)-3)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+3]) for i in range(len(words)-3)]
                if 3 not in self.tuples:
                    ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if calc_weights:
            return ret_words, {}
        else:
            return ret_words



#################################################################################################
# Parsing & NLP utilities
#################################################################################################
MARK_ANSWER_ALL  = ' <ALL>'
MARK_ANSWER_BOTH = ' <BOTH>'
MARK_ANSWER_NONE = ' <NONE>'
def sub_complex_answers(train):
    '''
    Substitute complex answers like "Both A and B" by the contents of answers A and B,
    "All of the above" by the contents of all answers, and "None of the above" by "". 
    We also mark these substitutions for later use.
    '''
    print 'Substituting complex answers'
    all_re  = re.compile('\s*all of the above\s*')
    both_re = re.compile('\s*both ([a-d]) and ([a-d])[\.]?\s*')
    none_re = re.compile('\s*none of the above\s*')
    for ind,answers in zip(train.index, np.array(train[['answerA','answerB','answerC','answerD']])):
        for ansi,anst in zip(['A','B','C','D'], answers):
            new_ans = None
            all_m = re.match(all_re, anst.lower())
            if all_m is not None:
#                 assert ansi in ['D'], 'Strange... answer%s = %s' % (ansi,anst) # not true in validation set...
                new_ans = '%s and %s and %s%s' % (answers[0], answers[1], answers[2], MARK_ANSWER_ALL)
            else:
                both_m = re.match(both_re, anst.lower())
                if both_m is not None:
                    #assert ansi in ['C','D'], 'Strange... answer%s = %s' % (ansi,anst)
                    both1, both2 = both_m.groups()[0].upper(), both_m.groups()[1].upper()
                    #assert both1!=both2 and both1!=ansi and both2!=ansi
                    new_ans = '%s and %s%s' % (answers[ord(both1)-ord('A')], answers[ord(both2)-ord('A')], MARK_ANSWER_BOTH)
                else:
                    if re.match(none_re, anst.lower()) is not None:
    #                     assert ansi in ['C','D'], 'Strange... answer%s = %s' % (ansi,anst)
                        new_ans = '%s' % MARK_ANSWER_NONE
            if new_ans is not None:
#                 print ' replacing "%s" in #%d by: "%s"' % (anst, ind, new_ans)
                train.set_value(ind, 'answer%s'%ansi, new_ans)

def add_qa_features(train):
    '''
    Add simple features computed from the questions and/or answers
    '''
    parser = SimpleWordParser()
    train['q_which']     = np.array([('which' in qst.lower().split(' ')) for qst in train['question']])
    train['q____']       = np.array([('___' in qst) for qst in train['question']])
    not_words_weights = {'NOT':1, 'EXCEPT':1, 'LEAST':1} #, 'not':0.5, 'except':0.5, 'least':0.5}
#     train['q_not']       = np.array([1*np.any([(w in ['NOT','EXCEPT','LEAST']) for w in qst.split(' ')]) for qst in train['question']])
    train['q_not']       = np.array([np.max([not_words_weights.get(w,0) for w in qst.split(' ')]) for qst in train['question']])
    train['q_num_words'] = np.array([len(parser.parse(qst)) for qst in train['question']])
    train['a_num_words'] = np.array([np.mean([len(parser.parse(ans)) for ans in anss]) for anss in np.array(train[['answerA','answerB','answerC','answerD']])])
    

def prp_binary_dataf(train):
    stemmer = PorterStemmer()
    parser = SimpleWordParser(word_func=stemmer.stem, min_word_length=1, tolower=True, ascii_conversion=True, ignore_special_words=False)
    indices, questions, answers, correct, ans_names, more_cols_vals = [], [], [], [], [], []
    is_all, is_both, is_none, keywords = [], [], [], []
    if 'correctAnswer' in train.columns:
        correct_answer = np.array(train['correctAnswer'])
    else:
        correct_answer = np.zeros(len(train))
    more_cols = [col for col in train.columns if col not in ['question', 'answerA', 'answerB', 'answerC', 'answerD', 'correctAnswer']]
    for idx,(qst,ansA,ansB,ansC,ansD),cor,mcols in zip(train.index, np.array(train[['question', 'answerA', 'answerB', 'answerC', 'answerD']]), 
                                                       correct_answer, np.array(train[more_cols])):
        for ia,(ic,ans) in enumerate(zip(['A','B','C','D'],[ansA, ansB, ansC, ansD])):
            indices.append(idx)
            questions.append(qst)
            a_ans, a_all, a_both, a_none, a_keywords = ans, 0, 0, 0, 0
            if ans.endswith(MARK_ANSWER_ALL):
                a_ans = ans[:-len(MARK_ANSWER_ALL)]
                a_all = 1
            elif ans.endswith(MARK_ANSWER_BOTH):
                a_ans = ans[:-len(MARK_ANSWER_BOTH)]
                a_both = 1
            elif ans.endswith(MARK_ANSWER_NONE):
                a_ans = ans[:-len(MARK_ANSWER_NONE)]
                a_none = 1
            else:
                words = parser.parse(ans)
                if 'both' in words:
                    a_both = 0.5
                if stemmer.stem('investigation') in words:
                    a_keywords = 1
            answers.append(a_ans)
            is_all.append(a_all)
            is_both.append(a_both)
            is_none.append(a_none)
            keywords.append(a_keywords)
            if cor==0:
                correct.append(0) # no 'correctAnswer' column -> set correct=0 for all answers
            else:
                correct.append(1 if ia==(ord(cor)-ord('A')) else 0)
            ans_names.append(ic)
            more_cols_vals.append(mcols)
    pdict = {'ID': indices, 'question': questions, 'answer': answers, 'correct': correct, 'ans_name': ans_names, 
             'is_all': is_all, 'is_both': is_both, 'is_none': is_none} #, 'ans_keywords': keywords}
    for icol,mcol in enumerate(more_cols):
        pdict[mcol] = np.array([vals[icol] for vals in more_cols_vals])
    return pd.DataFrame(pdict)

class AsciiConvertor(object):
    ascii_orig = ['0','1','2','3','4','5','6','7','8','9',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                  'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                  '+','-','=','*','/','\\','_','~','>','<','%','$','#','@','&',
                  '.',',',';',':','!','?',
                  '\'']
    ascii_conv = {138: 's', 140: 'o', 142: 'z', 
                  150: '-', 151: '-', 152: '~', 154: 's', 156: 'o', 158: 'z', 159: 'y', 
                  192: 'a', 193: 'a', 194: 'a', 195: 'a', 196: 'a', 197: 'a', 198: 'a', 199: 'c', 200: 'e', 201: 'e', 202: 'e', 203: 'e', 204: 'i', 205: 'i',
                  206: 'i', 207: 'i', 209: 'n', 210: 'o', 211: 'o', 212: 'o', 213: 'o', 214: 'o', 215: '*', 216: 'o', 217: 'u', 218: 'u', 219: 'u', 220: 'u',
                  221: 'y', 223: 's', 224: 'a', 225: 'a', 226: 'a', 227: 'a', 228: 'a', 229: 'a', 230: 'a', 231: 'c', 232: 'e', 233: 'e', 234: 'e', 235: 'e',
                  236: 'i', 237: 'i', 238: 'i', 239: 'i', 241: 'n', 242: 'o', 243: 'o', 244: 'o', 245: 'o', 246: 'o', 248: 'o', 249: 'u', 250: 'u',
                  250: 'u', 251: 'u', 252: 'u', 253: 'y', 255: 'y' 
                  }
    ascii_mapping = None

    @staticmethod
    def convert(text):
        if AsciiConvertor.ascii_mapping is None:
            print 'Building ascii dict'
            AsciiConvertor.ascii_mapping = [' ']*256
            for c in AsciiConvertor.ascii_orig:
                AsciiConvertor.ascii_mapping[ord(c)] = c
            for oc,c in AsciiConvertor.ascii_conv.iteritems():
                AsciiConvertor.ascii_mapping[oc] = c
        return ''.join(map(lambda c: AsciiConvertor.ascii_mapping[ord(c)], text))

class SpecialWords(object):
    '''
    Stop words
    '''
    ignore_words = None
    my_stopwords = set(['', 'and', 'or', 'the', 'of', 'a', 'an', 'to', 'from',
                        'be', 'is', 'are', 'am', 'was', 'were', 'will', 'would', 
                        'do', 'does', 'did',
                        'have', 'has', 'had', 
                        'can', 'could', 'should', 'ought',
                        'may', 'might',
                        'by', 'in', 'into', 'out', 'on', 'over', 'under', 'for', 'at', 'with', 'about', 'between', 
                        'that', 'this', 'these', 'those', 'there', 'than', 'then', 
                        'we', 'our', 'they', 'their', 'you', 'your', 'he', 'his', 'she', 'her', 'it', 'its', 
                        'which', 'what', 'how', 'where', 'when', 'why', 
                        'not', 'no', 'if', 'maybe', 
                        'more', 'most', 'less', 'least', 'some', 'too', 
                        'best', 'better', 'worst', 'worse',
                        'same', 'as', 'like', 'different', 'other', 'so',
                        ])
     
    
    @staticmethod
    def read_nltk_stopwords(keep_my_words=True):
        # Stopwords taken from nltk's nlp/stopwords/english
        SpecialWords.ignore_words = set([word.strip().lower() for word in 
                                         ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
                                          "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
                                          "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
                                          "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
                                          "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
                                          "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
                                          "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
                                          "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
                                          "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
                                          "don", "should", "now"]])
        if keep_my_words:
            SpecialWords.ignore_words.difference_update(['above','after','again','against','all','any','before','below','between','both','down','during',
                                                         'each','few','further','into','just','more','most','no','not','now','off','once','only','out','over','own',
                                                         'same','through','under','until','up'])
        print '-> Set %d NLTK stopwords (%s my words)' % (len(SpecialWords.ignore_words), 'kept' if keep_my_words else 'did not keep')

    @staticmethod
    def filter(words):
        if SpecialWords.ignore_words is None:
            SpecialWords.read_nltk_stopwords()
#             SpecialWords.ignore_words = SpecialWords.my_stopwords
            
#         print 'filtering: %s' % ' ; '.join(words)
#         print 'filtering: %s' % ' ; '.join([word for word in words if word not in SpecialWords.ignore_words])
        fwords = [word for word in words if word not in SpecialWords.ignore_words]
        if len(fwords) > 0:
            return fwords
        else: # do not filter out ALL the words
            return words

    @staticmethod
    def filter1(word):
        if SpecialWords.ignore_words is None:
            SpecialWords.read_nltk_stopwords()
        return word not in SpecialWords.ignore_words
  
#################################################################################################
# LuceneCorpus
#################################################################################################

class LuceneCorpus(object):
    def __init__(self, index_dir, filenames, parser, similarity=None):
        self._index_dir = index_dir
        self._filenames = filenames
        self._parser = parser
        self._similarity = similarity
        lucene.initVM()
        self._analyzer = WhitespaceAnalyzer(Version.LUCENE_CURRENT)
        self._store = SimpleFSDirectory(File(self._index_dir))
        self._searcher = None

    def prp_index(self):
        '''
        Prepare the index given our "corpus" file(s)
        '''
        print '=> Preparing Lucene index %s' % self._index_dir
        writer = self._get_writer(create=True) #IndexWriter(dir, analyzer, True, IndexWriter.MaxFieldLength(512))
        print '   Currently %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        num_pages, num_sections = 0, 0
        page_name, section_name = None, None
        num_lines = 0
        for ifname,fname in enumerate(self._filenames):
            print '   Adding lines to index from file #%d: %s' % (ifname, fname)
            with open(fname,'rt') as infile:
                for text in infile:
                    # print '%s' % text
                    if len(text)==0:
                        print 'Reached EOF'
                        break # EOF   
                    if text.startswith(CorpusReader.PAGE_NAME_PREFIX):
                        page_name = text[len(CorpusReader.PAGE_NAME_PREFIX):].strip()
                        section_name = None
                        num_pages += 1
                    elif text.startswith(CorpusReader.SECTION_NAME_PREFIX):
                        section_name = text[len(CorpusReader.SECTION_NAME_PREFIX):].strip()
                        num_sections += 1
                    else:
                        print(num_sections)
                        assert (page_name is not None) and (section_name is not None)
                        # section_words = text.split(' ')
                        if self._parser is None:
                            luc_text = text
                        else:
                            section_words = self._parser.parse(text, calc_weights=False) #True)
                            if False:
                                print 'Adding words: %s (weights: %s)' % (section_words, weights)
                            luc_text = ' '.join(section_words)
                        doc = Document()
                        doc.add(Field("text", luc_text, Field.Store.YES, Field.Index.ANALYZED))
                        writer.addDocument(doc)
                    num_lines += 1
                    if num_lines % 100000 == 0:
                        print '    read %d lines so far: %d pages, %d sections' % (num_lines, num_pages, num_sections)

        print '   Finished - %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        writer.close()

    def search(self, words, max_docs, weight_func=lambda n: np.ones(n), score_func=lambda s: s):
        '''
        Search the index for the given words, return total score
        '''
        searcher = self._get_searcher()
        if type(words)==str:
            search_text = words
            search_text = AsciiConvertor.convert(search_text)
            for c in '/+-&|!(){}[]^"~*?:':
                search_text = search_text.replace('%s'%c, '\%s'%c)
        else:
            search_text = ' '.join(words)
        # print 'search_text: %s' % search_text
        query = QueryParser(Version.LUCENE_CURRENT, "text", self._analyzer).parse(search_text)
        hits = searcher.search(query, max_docs)
        # print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)

        score_sum = 0.0
        score_list = []
        weights = weight_func(len(hits.scoreDocs))
        for hit,weight in zip(hits.scoreDocs, weights):
            score_sum += weight * score_func(hit.score)
            score_list.append(weight * score_func(hit.score))
        #     print ' score %.3f , weight %.5f -> %.5f' % (hit.score, weight, weight*hit.score)
        #     print hit.score, hit.doc, hit.toString()
        #     doc = searcher.doc(hit.doc)
        #     print doc.get("text").encode("utf-8")
        # print 'score_sum = %.5f' % score_sum
        return score_sum,score_list, hits 

    def _get_writer(self, analyzer=None, create=False):
        config = IndexWriterConfig(Version.LUCENE_CURRENT, self._analyzer)
        if create:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        if self._similarity is not None:
            config.setSimilarity(self._similarity)
        writer = IndexWriter(self._store, config)
        return writer

    def _get_searcher(self):
        if self._searcher is None:
            self._searcher = IndexSearcher(DirectoryReader.open(self._store))
            if self._similarity is not None:
                self._searcher.setSimilarity(self._similarity)
        return self._searcher


#################################################################################################
# Functions for computing a value (feature) for each answer
#################################################################################################
                
class AnswersFunc(object):
    def __call__(self, question, answers):
        pass
    
class AnswersLuceneSearchFunc(AnswersFunc):
    def __init__(self, lucene_corpus, parser, max_docs, weight_func=lambda n: np.ones(n), score_func=None, norm_scores=True):
        self.lucene_corpus = lucene_corpus
        self.parser = parser
        self.max_docs = max_docs
        self.weight_func = weight_func
        if score_func is None:
            self.score_func = lambda s: s
        else:
            self.score_func = score_func
        self.norm_scores = norm_scores
        
    def __call__(self, question, answer):
        EPSILON = 1E-30
        # print 'question = %s' % question
        if self.parser is None:
            q_words = question
            a_words = answer
        else:
            q_words = self.parser.parse(question, calc_weights=False)
            # print '  -> %s' % ' ; '.join(q_words)
            a_words = self.parser.parse(answer, calc_weights=False)
            # print '  -> %s' % ' ; '.join(a_words)
            search_words = q_words + a_words

            query_score,document_scores,documents= self.lucene_corpus.search(words=search_words, max_docs=self.max_docs, weight_func=self.weight_func, score_func=self.score_func)

        return np.asarray(query_score),np.asarray(document_scores) ,documents 




#################################################################################################
# Functions for Build QA environment 
#################################################################################################

class AnswerChecker(object):
    def __init__(self,scorer,searcher,word2vec):
        self.reward  = [3,-1,1,-3,0] # TT,TF,FF,FT,Query
        self.flag = [0,1]
        self.done = False
        self.gamma = 0.99
        self.scorer = scorer
        self.searcher = searcher
        self.word2vec = word2vec


    def check(self,action,answer):
        if action < 2:
            self.done = True
            if action == 1 and answer == 1:
                return self.done,self.reward[0],self.flag[1]
            if action == 0 and answer == 0:
                return self.done,self.reward[2],self.flag[1]
            if action == 1 and answer == 0:
                return self.done,self.reward[1],self.flag[0]
            if action == 0 and answer == 1:
                return self.done,self.reward[3],self.flag[0]
        else:
            self.done = False
            return self.done,self.reward[4],self.flag[0]

    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    def step(self,question,answer,document):
        doc = self.searcher.doc(document.doc)
        text = doc.get("text").encode("utf-8")
        keys = [x[0] for x in pos_tag(word_tokenize(text)) if x[1] == 'NN']
        for word in list(set(keys))[:100]:
             answer += (' ' + word) if word.isalpha() else ''
        query_score,document_score,_= score_func(question,answer)
        state = np.hstack((query_score,document_score))
        state =  np.reshape(state,(1,len(state)))
        return state,self.word2vec.lookUp(answer)


class RLearner():
    def __init__(self, lr, s_size,h_size):

        self.observations = tf.placeholder(tf.float32, [None,s_size] , name="input_x")
        W1 = tf.get_variable("W1", shape=[s_size, h_size],
                   initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.sigmoid(tf.matmul(self.observations,W1))
        # layer1 = tf.nn.relu(tf.matmul(self.observations,W1))
        W2 = tf.get_variable("W2", shape=[h_size, 1],
                   initializer=tf.contrib.layers.xavier_initializer())
        self.output =  tf.matmul(layer1,W2)

        # self.reward = tf.placeholder(shape=None,dtype=tf.float32)
        # self.action  = tf.placeholder(shape=None,dtype=tf.int32)

        self.nextQ = tf.placeholder(shape=[None,1],dtype=tf.float32)
        # regularizer = tf.nn.l2_loss(weights)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.output))

        # self.indexes = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + self.action
        # self.responsible_outputs = tf.gather(tf.reshape(output, [-1]), self.indexes)
        # # self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward)
        # # self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs) - self.reward)
        # # self.loss = tf.reduce_sum(tf.abs(self.responsible_outputs - self.reward))
        # self.loss = tf.reduce_sum(tf.abs(self.responsible_outputs - self.reward))
        # self.loss = tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

class GloveModel(object):
    def __init__(self, path):
        self.path = path
        self.model = {}

    def load(self):
        print "Loading Glove Model"
        f = open(self.path,'r')
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            self.model[word] = embedding
        print "Done.",len(self.model)," words loaded!"
        return self.model
    
    def lookUp(self,sentence):
        string2vec = np.zeros(len(self.model['a']))
        sentence = "".join(c for c in sentence if c not in string.punctuation)
        for word in sentence.split():
            # print (word)
            if word in self.model:
                string2vec += self.model[word]
        return string2vec


def accuracy_measure(predict,target):
    ture_num = 0.0
    false_num = 0.0
    total_num = len(target)
    
    for (p,t) in zip(predict,target):
        if p == 1 and t == 1:ture_num += 1
        if p == 0 and t == 0:false_num += 1

    # print (ture_num,false_num)
    return 4*ture_num/total_num,4*false_num/(3*total_num)

# ================================================================================================================================
# Main
# ================================================================================================================================

if __name__ == "__main__":

    import sys
    import json   

    with open('SETTINGS.json') as f:
       json_params  = json.load(f)

    # qaEnv = AnswerChecker() 
    # print qaEnv.check(1,1)    
    # print qaEnv.check(1,0)    
    # print qaEnv.check(0,1)    
    # print qaEnv.check(0,0)

    # ------------------------------------------------------------------------------------------------
    # Search Function
    # ------------------------------------------------------------------------------------------------
 
    base_dir       = json_params['BASE_DIR']
    input_dir      = '%s/%s' % (base_dir, json_params['INPUT_DIR'])
    corpus_dir     = '%s/%s' % (base_dir, json_params['CORPUS_DIR'])
    submission_dir = '%s/%s' % (base_dir, json_params['SUBMISSION_DIR'])

    ck12html_corpus = '%s/CK12/OEBPS/ck12.txt' % corpus_dir
    ck12html_para_corpus = '%s/CK12/OEBPS/ck12_paragraphs.txt' % corpus_dir
    ck12text_corpus = '%s/CK12/ck12_text.txt' % corpus_dir
    ck12text_sent_corpus = '%s/CK12/ck12_text_sentences.txt' % corpus_dir
    oer_corpus = '%s/UtahOER/oer_text.txt' % corpus_dir
    saylor_corpus = '%s/Saylor/saylor_text.txt' % corpus_dir
    ai2_corpus = '%s/AI2_data/ai2_corpus.txt' % corpus_dir
    sstack_corpus = '%s/StudyStack/studystack_corpus.txt' % corpus_dir
    sstack_corpus2 = '%s/StudyStack/studystack_corpus2.txt' % corpus_dir
    sstack_corpus3 = '%s/StudyStack/studystack_corpus3.txt' % corpus_dir
    sstack_corpus4 = '%s/StudyStack/studystack_corpus4.txt' % corpus_dir
    quizlet_corpus = '%s/quizlet/quizlet_corpus.txt' % corpus_dir
    simplewiki_corpus2 = '%s/simplewiki/simplewiki_1.0000_0.0500_0_5_True_True_True_corpus.txt' % corpus_dir
    simplewiki_corpus3 = '%s/simplewiki/simplewiki_1.0000_0.1000_0_3_True_True_False_corpus.txt' % corpus_dir
    simplewiki_corpus_pn = '%s/simplewiki/simplewiki_1.0000_0.0100_0_3_True_True_False_pn46669_corpus.txt' % corpus_dir
    wikibooks_corpus = '%s/wikibooks/wikibooks_1.0000_0.0200_0_10_True_True_False_corpus.txt' % corpus_dir
    wiki_corpus3 = '%s/wiki/wiki_1.0000_0.0200_0_5_True_True_False_corpus.txt' % corpus_dir
    wiki_corpus_pn = '%s/wiki/wiki_0.5000_0.1000_0_5_True_True_False_pn46669_corpus.txt' % corpus_dir

    lucene_dir = '%s/lucene_idx7' % corpus_dir
    lucene_parser = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
    lucene_corpus = LuceneCorpus(index_dir=lucene_dir, filenames=[sstack_corpus2, wiki_corpus_pn, simplewiki_corpus_pn,
                                                                              ck12html_para_corpus, oer_corpus], 
                                       parser=lucene_parser, similarity=None)

    params = {'lucene_corpus': lucene_corpus,
                             'parser': lucene_parser,  
                             'max_docs': 2, 'weight_func': lambda n: 1.0/(10.0+np.arange(n)), 'score_func': lambda s: (s+2.0)**3.4, 'norm_scores': True,
                             'recalc': False, 'skip': False, 'lucene': True}
    
    score_func=AnswersLuceneSearchFunc(lucene_corpus=params['lucene_corpus'], parser=params['parser'], 
                                                              max_docs=params['max_docs'], weight_func=params['weight_func'], score_func=params['score_func'], 
                                                              norm_scores=params['norm_scores'])

    searcher = lucene_corpus._get_searcher()

    # question = 'When athletes begin to exercise, their heart rates and respiration rates increase.  At what level of organization does the human body coordinate these functions?'
    # answer   = 'at the tissue level'
    # score,_,test   =score_func(question,answer)
    # print score,test.scoreDocs[:5]
   

    # # path = '/Users/Omega/Project/QuestionAnsweringRL/corpus/glove/'
    # # word2vec = GloveModel(path+'glove.6B.50d.txt')
    # # word2vec.load()

    # for item in test.scoreDocs:
    #     # print item
    #     searcher = lucene_corpus._get_searcher()
    #     doc = searcher.doc(item.doc)
    #     text = doc.get("text").encode("utf-8")
    #     # print text
    #     # print word2vec.lookUp(text)
    #     # print pos_tag(word_tokenize(text))
    #     keys = [x[0] for x in pos_tag(word_tokenize(text)) if x[1] == 'NN']
    #     # print keys,len(keys)
    #     words = list(set(keys))
    #     # print words,len(words)

    #     noun_word_vec = np.zeros(50)
    #     for word in words:
    #         # noun_word_vec += word2vec.lookUp(word)
    #         answer += (' ' + word)

    #     break
    
    # print answer
    # score,_,test   =score_func(question,answer)
    # print score,test.scoreDocs[:5]
    # exit()
    # ------------------------------------------------------------------------------------------------
    # Word Embedding look-up table
    # ------------------------------------------------------------------------------------------------
    path = '/Users/Omega/Project/QuestionAnsweringRL/corpus/glove/'
    word2vec = GloveModel(path+'glove.6B.50d.txt')
    word2vec.load()


    # ------------------------------------------------------------------------------------------------
    # Read input files
    # ------------------------------------------------------------------------------------------------
    train_file      = json_params['TRAINING_FILE']
    validation_file = json_params['VALIDATION_FILE']
    test_file       = json_params['TESTING_FILE']

    print '\n--> Reading input files'
    
    train_sets = read_input_file(input_dir, filename=train_file, sep='\t' if train_file.endswith('.tsv') else ',', max_rows=1000000)
    print 'Read %d train questions' % len(train_sets) 
    sub_complex_answers(train_sets)
    train_tf = prp_binary_dataf(train_sets)
    question_num = len(train_tf)


    valid_sets = read_input_file(input_dir, filename=validation_file, sep='\t' if validation_file.endswith('.tsv') else ',', max_rows=1000000)
    print 'Read %d validation questions' % len(valid_sets) 
    sub_complex_answers(valid_sets)

    test_sets = read_input_file(input_dir, filename=test_file, sep='\t' if test_file.endswith('.tsv') else ',', max_rows=1000000)
    print 'Read %d test questions' % len(test_sets) 
    sub_complex_answers(test_sets)


    questionList,answerList,targetList,featureList= [],[],[],[]
    for (index,qa) in train_tf.iterrows():
        questionList.append(qa['question'])
        answerList.append(qa['answer'])
        targetList.append(qa['correct']) 
        featureList.append(np.array([qa['is_all'] , qa['is_both'], qa['is_none']]))


    # ------------------------------------------------------------------------------------------------
    # Reinfrocement learning grpah
    # ------------------------------------------------------------------------------------------------
    tf.reset_default_graph() #Clear the Tensorflow graph.

    myLearner= RLearner(lr=0.0001,s_size=params['max_docs']+51,h_size = 200) #Load the agent.
    gamma = 0.99
    total_episodes = 20#Set total number of episodes to train agent on.
    e = 0.1#Set the chance of taking a random action.
    print e,params['max_docs']

    seed(1)   
    set_random_seed(1)    

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        qaEnv = AnswerChecker(score_func,searcher,word2vec) 
        jList = []
        acc_list = []

        for i in range(total_episodes):
            ep_correct= 0 #Set scoreboard to 0.
            ep_reward = 0 #Set scoreboard to 0.
            predictList = []
            
            
            print "Train Phase"
            for qaid in range(question_num):
            # for qaid in range(0):   
                question = questionList[qaid]
                answer   = answerList[qaid]
                target      = targetList[qaid]
                # qa_word_vec = word2vec.lookUp(question+answer)
                # query_word_vec = np.zeros(len(qa_word_vec)) 
                query_score,document_score,document = score_func(question,answer)
                qaEnv.document = document
                qaEnv.state_transition = 0

                state_history,action_history,reward_history = [],[],[]
                # state = np.zeros((1,params['max_docs']+4))

                # state = np.hstack((qa_word_vec,query_word_vec,query_score,document_score,featureList[qaid]))
                state = np.hstack((query_score,document_score))
                state =  np.reshape(state,(1,len(state)))
                # print state,len(state)
                # continue
                done = False
                j = 0

                ACTION= {
                                    0 : np.zeros(50), #"Label as False",
                                    1 : np.ones(50), #"Label as True",
                                    2 : np.arange(50), #"Query expansion",
                                    }

                while j < params['max_docs']:
                    qa_loss = 0
                    action_value = []
                    for a,action in ACTION.items():
                        state_action = np.hstack((state[0],action))
                        state_action =np.reshape(state_action,(1,len(state_action)))
                        output= sess.run(myLearner.output,feed_dict={myLearner.observations:state_action})
                        action_value.append(output)

                    chosen_action = np.argmax(action_value)

                    if np.random.rand(1) < e:
                        chosen_action = np.random.randint(len(ACTION))


                    done,reward,crroect = qaEnv.check(chosen_action,target) 
                    # print done,reward,crroect 
                    # state_history.append(state)
                    # exit()

                    next_state,next_action = qaEnv.step(question,answer,document.scoreDocs[j]) 
                    action_value = []
                    for a,action in ACTION.items():
                        state_action = np.hstack((next_state[0],action))
                        state_action =np.reshape(state_action,(1,len(state_action)))
                        output= sess.run(myLearner.output,feed_dict={myLearner.observations:state_action})
                        action_value.append(output)
                    next_output  = np.max(action_value)
                    targetQ = gamma*next_output + reward
                    # targetQ = next_output + reward
                    # print next_output,reward,targetQ


                    state_action = np.hstack((state[0],ACTION[chosen_action]))
                    state_action =np.reshape(state_action,(1,len(state_action)))
                    feed_dict={myLearner.observations:state_action,myLearner.nextQ:np.reshape(targetQ,(-1,1))}  
                    _,loss,output= sess.run([myLearner.update,myLearner.loss,myLearner.output],feed_dict=feed_dict)

                    state = next_state
                    ACTION[2] = next_action 
                    j+=1

                    if done:
                        # ep_correct += reward
                        ep_reward += reward
                        predictList.append(chosen_action) 
                        e = 1./((i/50) + 10)
                        break

                    if j == (params['max_docs']-1):
                        action = np.random.randint(len(ACTION)-1)
                        done,reward,crroect = qaEnv.check(action,target) 
                        ep_correct += crroect
                        predictList.append(action) 
                        break

            print 'the reward is %.2f in episodes %d' %(ep_reward,i)  
            if len(predictList) == len(targetList) :
                ture_acc,false_acc = accuracy_measure(predictList,targetList)
                # print len(predictList),len(predictList)
                print confusion_matrix(targetList,predictList)
                print f1_score(targetList,predictList)
                # print  'the precision for ture is %d percent and the accuracy for false is %d percent' %(ture_acc*100,false_acc*100)
                # print  'the F1 score is %d in episodes %d' %(f1_score(predictList,targetList,average='binary'),i) 
            else :
                print 'No full answer'


            # print "Validation Phase"
            qa_hits = 0
            valid_pred = []
            for _,qa in valid_sets.iterrows():
                question = qa['question']
                answers = np.array([qa['answerA'] , qa['answerB'], qa['answerC'], qa['answerD'] ])
                target = ord(qa['correctAnswer']) - ord("A")

                true_score = []
                for answer in answers:
                    query_score,document_score,_= score_func(question,answer)
                # state = np.hstack((qa_word_vec,query_word_vec,query_score,document_score,featureList[qaid]))
                    state_action= np.hstack((query_score,document_score,np.ones(50)))
                    state_action =  np.reshape(state_action,(1,len(state_action)))
                    output= sess.run(myLearner.output,feed_dict={myLearner.observations:state_action})
                    # print output
                    true_score.append(output)

                prediction = np.argmax(true_score)
                # prediction = np.random.randint(4)
                valid_pred.append([prediction,target])

                if prediction == target:
                    qa_hits += 1.0

            valid_acc = qa_hits/len(valid_sets)
            print 'the accuracy is %.2f in episodes %d on validation set' %(valid_acc,i) 


            # print "Test Phase"
            qa_hits = 0
            test_pred = []
            for _,qa in test_sets.iterrows():
                question = qa['question']
                answers = np.array([qa['answerA'] , qa['answerB'], qa['answerC'], qa['answerD'] ])
                target = ord(qa['correctAnswer']) - ord("A")

                true_score = []
                for answer in answers:
                    query_score,document_score,_= score_func(question,answer)
                # state = np.hstack((qa_word_vec,query_word_vec,query_score,document_score,featureList[qaid]))
                    state_action= np.hstack((query_score,document_score,np.ones(50)))
                    state_action =  np.reshape(state_action,(1,len(state_action)))
                    output= sess.run(myLearner.output,feed_dict={myLearner.observations:state_action})
                    true_score.append(output)

                prediction = np.argmax(true_score)
                # prediction = np.random.randint(4)
                test_pred.append([prediction,target])

                if prediction == target:
                    qa_hits += 1.0

            test_acc = qa_hits/len(test_sets)
            print 'the accuracy is %.2f in episodes %d on test set' %( test_acc,i) 
            acc_list.append([valid_acc,test_acc])

    print 'Done.'
    # print type(acc_list)
    # print acc_list
    timestr = time.strftime("%H%M%S")
    with open(timestr+"-test.txt", "wb") as fp:
        pickle.dump([acc_list,valid_pred,test_pred], fp)







