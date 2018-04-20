import pandas as pd
import numpy as np
from os import listdir
from os import path
import math
import csv

PROCESSED_DIR = '../Data_EvolvingRelationships_Chaturvedi_AAAI2017/processedText/'
FULLY_ANNOTATED_DIR = '../Data_EvolvingRelationships_Chaturvedi_AAAI2016/fullyAnnotatedSequences/'
PARIALLY_ANNOTATED_DIR = '../Data_EvolvingRelationships_Chaturvedi_AAAI2016/partiallyAnnotatedSequences/'
PROCESSED_SUFFIX = '.processed'
SENTENCE_SUFFIX = '.sent'

class Corpus:

    def __init__(self, verbose=False):
        self.passages = []
        f_names = listdir(PROCESSED_DIR)
        for f_name in f_names:
            if PROCESSED_SUFFIX == f_name[-len(PROCESSED_SUFFIX):]:
                f_name = f_name[:-len(PROCESSED_SUFFIX)]
                if verbose:
                    print('Processing %s ...' % f_name)
                self.passages.append(Passage(f_name, verbose))
                if verbose:
                    print('Processing %s finishes.' % f_name)

    

'''
    The passage class contains information about a sparknote summary of a passage
'''
class Passage:

    def __init__(self, author_name, verbose=False):
        self.author_name = author_name
        self.df = pd.read_csv(PROCESSED_DIR + author_name + PROCESSED_SUFFIX, sep='\t', quoting=csv.QUOTE_NONE)
        # making sure that all tokens have been read
        tid = self.df['tokenId'].values
        for idx in range(tid.shape[0]):
            assert(tid[idx] == idx)
        self.basic_info(verbose)
        self.create_sequences()
    
    def basic_info(self, verbose=False):
        # basic information about the passage
        self.num_sentences = np.max(self.df['sentenceID'].values) + 1
        self.num_characters = np.max(self.df['correctedCharId'].values) + 1
        if verbose:
            print('There are %d sentences, %d characters' % (self.num_sentences, self.num_characters))

    def create_sequences(self):
        # creating sentences
        self.sentences = [Sentence(self.df[self.df.sentenceID == sentence_idx])
                          for sentence_idx in range(self.num_sentences)]
        
        # creating a dictionary of sequences according to the interaction between character pairs
        self.pair_set = set()
        for sentence in self.sentences:
            self.pair_set = self.pair_set | sentence.pair_set
        self.sequence_dict = dict([(char_pair, []) for char_pair in self.pair_set])
        for sentence in self.sentences:
            for char_pair in sentence.pair_set:
                self.sequence_dict[char_pair].append(sentence)
        for char_pair in self.sequence_dict:
            self.sequence_dict[char_pair] = Sequence(self.sequence_dict[char_pair], char_pair, self)

'''
    A sequence of sentences
'''

class Sequence:

    def __init__(self, sentence_seq, char_pair, passage):
        self.sentence_seq, self.char_pair = sentence_seq, char_pair
        self.seq_length = len(self.sentence_seq)
        self.passage = passage
        c1, c2 = self.char_pair
        self.file_name = passage.author_name + ('.%d.%d' % self.char_pair)
        self.get_annotated_labels()

    def get_annotated_labels(self):
        # check wether there are annotations for this sequence
        in_file = None
        annotated_dir = FULLY_ANNOTATED_DIR + self.file_name + SENTENCE_SUFFIX
        if not path.isfile(annotated_dir):
            annotated_dir = PARIALLY_ANNOTATED_DIR + self.file_name + SENTENCE_SUFFIX
            if not path.isfile(annotated_dir):
                return
    
        self.labeled = True
        seqeuence_df = pd.read_csv(annotated_dir, sep=':::', engine='python')
        self.labels = [None if str(l) == 'nan'
                  else (1 if 'p' == l[0] else 0)
                  for l in seqeuence_df['manualLabel'].values]
        
        assert(len(self.labels) == len(self.sentence_seq))

'''
    A sentence class that contains necessary information for a "sentence" in a passage
    Major information is saved in the self.df field
'''
class Sentence:

    def __init__(self, df):
        self.df = df
        self.get_sentence_form()
        self.extract_char_ids()
        self.sentenceID = df['sentenceID'].values[0]
        self.num_tokens = self.df.shape[0]
    
    def extract_char_ids(self):
        # extracting character ids
        self.char_ids = list(set(self.df['correctedCharId'].values))
        if -1 in self.char_ids:
            self.char_ids.remove(-1)
        
        # extracting character pairs
        self.pair_set = set()
        # create char pairs
        for i in range(len(self.char_ids)):
            for j in range(i + 1, len(self.char_ids)):
                char_pair = (self.char_ids[i], self.char_ids[j]) \
                                if self.char_ids[j] > self.char_ids[i] \
                                else (self.char_ids[j], self.char_ids[i])
                self.pair_set.add(char_pair)

        # make sure that for any pair, the one with smaller char id comes first
        for char_pair in self.pair_set:
            c1, c2 = char_pair
            assert(c1 < c2)
    
    def get_sentence_form(self):
        self.token_list = self.df['originalWord'].values
        self.sentence_form = ' '.join(self.token_list)

    def __str__(self):
        string_representation = str(self.df) + '\n' \
                                + 'Complete sentence: ' + self.sentence_form + '\n' \
                                + 'Characters(id) involved: ' + ', '.join([str(x) for x in self.char_ids]) + '\n' \
                                + 'Character pairs: ' + ', '.join([str(x) for x in self.pair_set]) + '\n'
        string_representation += 'In this sentence, the word representing the characters are\n'
        for char_id in self.char_ids:
            df_tmp = self.df[self.df.correctedCharId == char_id]
            string_representation += str(char_id) + ': ' + ' '.join(df_tmp['originalWord'].values) + '\n'
        string_representation += '-------------------'
        return string_representation

if __name__ == '__main__':
    corpus = Corpus(verbose=True)
    author_name = 'albee.dream'
    passage = Passage(author_name)
