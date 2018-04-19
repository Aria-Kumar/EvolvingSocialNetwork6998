import gensim
import logging
import os
import sys


class WordEmbedding():

    def __init__(self, load_from=None, training_sentences_file=None, save_to=None):
        """
        :param load_from: If not None, load a pretrained model from this path
        :param training_sentences_file: If none, use the sentences provided. Otherwise, load the sentences from this path.
        :param save: If not None, save a model to this path
        """

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.data_path = 'Data_EvolvingRelationships_Chaturvedi_AAAI2017/originalText/'


        if load_from:
            self.model = gensim.models.Word2Vec.load(load_from)

        else:
            self.threads = 10
            self.n_epochs = 10
            # Size of the embedded vectors
            self.embedding_size = 300

            if training_sentences_file is None:
                print('reading sentences from', self.data_path)
                self.sentences = self.read_default_sentences()
                print('read {} lines'.format(len(self.sentences)))

            else:
                print('reading sentences from', self.data_path)
                self.sentences = self.read_sentences_from_corpus(training_sentences_file)
                print('read {} lines'.format(len(self.sentences)))


            # Create bigrams for common phrases in the model
            bigram_transformer = gensim.models.Phrases(self.sentences)

            # Initialize the model. Ensure that words must occur 4 times before they are used in the vocabulary
            self.model = gensim.models.Word2Vec(min_count=4, size=self.embedding_size, workers=self.threads)
            self.model.build_vocab(bigram_transformer[self.sentences])
            self.model.train(bigram_transformer[self.sentences], total_examples=self.model.corpus_count, epochs=self.n_epochs)

            print('trained', self.model)

        if save_to:
            print('saving to', save_to)
            self.model.save(save_to)

    def read_default_sentences(self):

        # Read in data from sentences
        sentences = []
        for filename in os.listdir(self.data_path):
            file = open(self.data_path + filename, 'r')

            for line in file.readlines():
                line = line.strip().lower().split(' ')
                if line != '':
                    sentences.append(line)

        return sentences

    def read_sentences_from_corpus(self, sentences_file_path):

        try:
            file = open(sentences_file_path, 'r')
        except IOError:
            print('Error. Could not load file at {}. Aborting'.format(sentences_file_path))
            sys.exit(1)

        sentences = []

        for line in file.readlines():
            line = line.strip().lower().split(' ')
            if line != '':
                sentences.append(line)

        return sentences

    def vector(self, word):
        """Return the vector for the given word"""
        return self.model.wv[word]



if __name__ == '__main__':

    # Create a word embedding from the downloaded sparknotes corpus
    WordEmbedding(save_to='models/sparknotes.w2v', training_sentences_file='corpus_data/sparknotes.txt')
