import pandas as pd
import numpy as np

'''
    The passage class contains information about a sparknote summary of a passage
'''
class Passage:

    def __init__(self, dir):
        self.df = pd.read_csv(dir, sep='\t')
        self.basic_info()
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
            self.sequence_dict[char_pair] = Sequence(self.sequence_dict[char_pair], char_pair)
            print(self.sequence_dict[char_pair].seq_length)

'''
    A sequence of sentences
'''

class Sequence:

    def __init__(self, sentence_seq, char_pair):
        self.sentence_seq, self.char_pair = sentence_seq, char_pair
        self.seq_length = len(self.sentence_seq)


'''
    A sentence class that contains necessary information for a "sentence" in a passage
    Major information is saved in the self.df field
'''
class Sentence:

    def __init__(self, df):
        self.df = df
        self.sentenceID = df['sentenceID'].values[0]
        self.get_sentence_form()
        self.extract_char_ids()
    
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
                                + 'Character pairs: ' + ', '.join([str(x) for x in self.pair_set])
        return string_representation



if __name__ == '__main__':
    dir = '../Data_EvolvingRelationships_Chaturvedi_AAAI2017/processedText/alighieri.inferno.processed'
    passage = Passage(dir)
