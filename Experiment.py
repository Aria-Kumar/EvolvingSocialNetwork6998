from Corpus import Corpus
from HMM import Seqs
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
import pandas as pd

# arverage the a list of result dictionaries
def average_dictionary(dictionaries, sd=False):
    df = pd.DataFrame(dictionaries)
    if sd:
        return df.mean(), df.std()
    else:
        return df.mean()

def mask_y_seqs(y_seqs, inds):
    new_y_seqs = [y_seq[:] for y_seq in y_seqs]
    for (seq_idx, sent_idx) in inds:
        assert (y_seqs[seq_idx][sent_idx] is not None)
        new_y_seqs[seq_idx][sent_idx] = None
    return new_y_seqs

def extract_labels(y_seqs, inds):
    result = [y_seqs[seq_idx][sent_idx] for (seq_idx, sent_idx) in inds]
    for r in result:
        assert(r is not None)
    return result

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
        fold_inds = self.create_train_test_split(fold)
        results = []
        for fold_idx in range(fold):
            print('Cross validating fold %d.' % fold_idx)
            fold_ind = fold_inds[fold_idx]
            result = self.train_test(fold_ind)
            results.append(result)
        return results
    
    
    def create_train_test_split(self, fold):
        seq_sent_agg = []
        for seq_idx in range(self.seq_count):
            for sent_idx in range(len(self.y_seqs[seq_idx])):
                if self.y_seqs[seq_idx][sent_idx] is not None:
                    seq_sent_agg.append((seq_idx, sent_idx))
        random.shuffle(seq_sent_agg)
        cutoffs = [int(float(len(seq_sent_agg)) / fold * fold_idx)
                   for fold_idx in range(fold)]
        cutoffs.append(len(seq_sent_agg))
        fold_inds = [seq_sent_agg[cutoffs[fold_idx]:cutoffs[fold_idx + 1]]
                     for fold_idx in range(fold)]
        return fold_inds

    def train_test(self, fold_ind):
        train_y_seqs = mask_y_seqs(self.y_seqs, fold_ind)
        num_repeats = 50
        results = []
        for _ in range(num_repeats):
            self.clf = Seqs(self.feature_dim, order=2)
            self.clf.fit(self.X_seqs, train_y_seqs)
            pred_y_seqs = self.clf.predict(self.X_seqs, train_y_seqs)
            y_test, y_pred = [extract_labels(y, fold_ind)
                              for y in [self.y_seqs, pred_y_seqs]]
            result = ({'macro_f1': f1_score(y_test, y_pred, average='macro'),
                      'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
                      'accuracy': accuracy_score(y_test, y_pred)})
            # print(classification_report(y_test, y_pred))
            results.append(result)
        return average_dictionary(results)
    
    # ==================== cross validation by sequence ====================
    def _cv(self, fold):
        results_agg = []
        for fold_idx in range(fold):
            print('cross validation for %d fold.' % (fold_idx + 1))
            results = self.experiment_once()
            results_agg += results
        return results_agg

    def _experiment_once(self):
        self._create_train_test_split()
        results = []
        num_repeats = 10
        for _ in range(num_repeats):
            result = self._train_test()
            results.append(result)
        return results

    def _create_train_test_split(self):
        shuffle_order = [idx for idx in range(len(self.X_seqs))]
        random.shuffle(shuffle_order)
        self.train_size = int(0.9 * self.seq_count)
        self.train_ind, self.test_ind = shuffle_order[:self.train_size], shuffle_order[self.train_size:]
        self.X_seqs_train, self.y_seqs_train = ([self.X_seqs[idx] for idx in self.train_ind],
                                                [self.y_seqs[idx] for idx in self.train_ind])
        self.X_seqs_test, self.y_seqs_test = ([self.X_seqs[idx] for idx in self.test_ind],
                                              [self.y_seqs[idx] for idx in self.test_ind])

    def _train_test(self):
        self.clf = Seqs(self.feature_dim, order=2)
        self.clf.fit(self.X_seqs_train, self.y_seqs_train)
        return self.clf.evaluate(self.X_seqs_test, self.y_seqs_test)


if __name__ == '__main__':
    fold = 10
    experiment = Experiment()
    results = experiment.cv(fold)
    mean, standard_dev = average_dictionary(results, sd=True)
    print('Mean for cross validation')
    print(mean)
    print('Standard Deviation')
    print(standard_dev)


