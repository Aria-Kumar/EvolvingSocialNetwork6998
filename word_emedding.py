import gensim
import logging
import os

class WordEmbedding():

    def __init__(self, load=False, training_sentences=None, save=True):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.data_path = 'Data_EvolvingRelationships_Chaturvedi_AAAI2017/originalText/'
        self.model_path = 'models/sparknotes.w2v'

        if load:
            self.model = gensim.models.Word2Vec.load(self.model_path)

        else:
            self.threads = 10
            self.n_epochs = 10
            # Size of the embedded vectors
            self.embedding_size = 300

            if training_sentences is None:
                print('reading sentences from', self.data_path)
                self.sentences = self.read_default_sentences()
                print('read {} lines'.format(len(self.sentences)))

            # Create bigrams for common phrases in the model
            bigram_transformer = gensim.models.Phrases(self.sentences)

            # Initialize the model. Ensure that words must occur 4 times before they are used in the vocabulary
            self.model = gensim.models.Word2Vec(min_count=4, size=self.embedding_size, workers=self.threads)
            self.model.build_vocab(bigram_transformer[self.sentences])
            self.model.train(bigram_transformer[self.sentences], total_examples=self.model.corpus_count, epochs=self.n_epochs)

            print('trained', self.model)

        if save:
            print('saving to', self.model_path)
            self.model.save(self.model_path)

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


if __name__ == '__main__':

    WordEmbedding()
