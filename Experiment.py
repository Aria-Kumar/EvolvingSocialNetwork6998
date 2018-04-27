from Corpus import Corpus
from HMM import Seqs
from sklearn.metrics import f1_score
import random
import pandas as pd

# arverage the a list of result dictionaries
def average_dictionary(dictionaries):
    df = pd.DataFrame(dictionaries)
    return df.mean(), df.std()

class Experiment:

    def __init__(self, verbose=False):
        print('Loading corpus ...')
        self.corpus = Corpus(verbose=verbose)
        self.corpus.create_data()
        self.X_seqs, self.y_seqs = self.corpus.X_seqs, self.corpus.y_seqs
        self.seq_count = len(self.X_seqs)
        for seq_idx in range(self.seq_count):
            assert(len(self.X_seqs[seq_idx]) == len(self.y_seqs[seq_idx]))
        self.feature_dim = len(self.X_seqs[0][0])
    
    def cv(self, fold):
        results_agg = []
        for fold_idx in range(fold):
            print('cross validation for %d fold.' % (fold_idx + 1))
            results = self.experiment_once()
            results_agg += results
        return results_agg

    def experiment_once(self):
        self.create_train_test_split()
        results = []
        num_repeats = 20
        for _ in range(num_repeats):
            result = self.train_test()
            results.append(result)
        return results

    def create_train_test_split(self):
        shuffle_order = [idx for idx in range(len(self.X_seqs))]
        random.shuffle(shuffle_order)
        self.train_size = int(0.9 * self.seq_count)
        self.train_ind, self.test_ind = shuffle_order[:self.train_size], shuffle_order[self.train_size:]
        self.X_seqs_train, self.y_seqs_train = ([self.X_seqs[idx] for idx in self.train_ind],
                                                [self.y_seqs[idx] for idx in self.train_ind])
        self.X_seqs_test, self.y_seqs_test = ([self.X_seqs[idx] for idx in self.test_ind],
                                              [self.y_seqs[idx] for idx in self.test_ind])

    def train_test(self):
        self.clf = Seqs(self.feature_dim, order=2)
        self.clf.fit(self.X_seqs_train, self.y_seqs_train)
        return self.clf.evaluate(self.X_seqs_test, self.y_seqs_test)


if __name__ == '__main__':
    fold = 10
    experiment = Experiment()
    results = experiment.cv(fold)
    mean, standard_dev = average_dictionary(results)
    print('Mean for cross validation')
    print(mean)
    print('Standard Deviation')
    print(standard_dev)


