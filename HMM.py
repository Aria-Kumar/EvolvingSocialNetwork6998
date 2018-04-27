from Perceptron import Custom_Perceptron
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

class Seqs:

    # initialize the sequences
    def __init__(self, feature_dim, order):
        self.feature_dim = feature_dim
        self.state_dim = 2 ** (order + 1) - 2
        self.order = order
        self.dimension = self.feature_dim + self.state_dim
        self.clf = Custom_Perceptron(self.dimension)

    # loading the features and labels
    def load_data(self, feature_X, y):
        self.seq_count = len(feature_X)
        self.seq_length = [len(x) for x in feature_X]
        self.feature_X, self.y = feature_X, y
        for seq_idx in range(self.seq_count):
            assert(len(self.feature_X[seq_idx]) == len(self.y[seq_idx]))
        self.create_states_for_seqs()
        self.set_vectors_for_states()

    # creating state for each of the sequences
    def create_states_for_seqs(self):
        self.seqs_states_array = [self.create_states_for_seq(y) for y in self.y]

    def set_vectors_for_states(self):
        for seq_idx in range(self.seq_count):
            for position_idx in range(self.seq_length[seq_idx]):
                for state in self.seqs_states_array[seq_idx][position_idx]:
                    state.set_vector(self.feature_X[seq_idx][position_idx], self.state_dim)
    
    # creating an array of states for each sentence
    # return an array of array of states
    def create_states_for_seq(self, partial_label):
        seq_length = len(partial_label)
        states_seq = []
        for position_idx in range(seq_length):
            states_array = [[]]
            position_idx2 = position_idx
            while position_idx - position_idx2 <= self.order and position_idx2 >= 0:
                if partial_label[position_idx2] is None:
                    states_array = [s[:] + [True] for s in states_array] + [s[:] + [False] for s in states_array]
                else:
                    states_array = [s[:] + [partial_label[position_idx2]] for s in states_array]
                position_idx2 -= 1
            states_array = [State(np.array(x)) for x in states_array]
            states_seq.append(states_array)
        return states_seq

    def decode_seqs(self):
        self.clear()
        self.seqs_decode_results, self.picked_states = [], []
        
        # decode all the sequences
        for seq_idx in range(self.seq_count):
            
            seq_decode_results = []
            # the starting of the sequence
            for state in self.seqs_states_array[seq_idx][0]:
                state.prev = None
                state.eval(self.clf)
                state.accum_score = state.score

            for position_idx in range(1, self.seq_length[seq_idx]):
                cur_states = self.seqs_states_array[seq_idx][position_idx]
                prev_states = self.seqs_states_array[seq_idx][position_idx - 1]
                
                # finding the best path compatible with the current state
                for cur_state in cur_states:
                    cur_state.eval(self.clf)
                    best_prev_state, best_accum_score = None, -float('inf')
                    for prev_state in prev_states:
                        if cur_state.compatible(prev_state):
                            if prev_state.accum_score > best_accum_score:
                                best_accum_score = prev_state.accum_score
                                best_prev_state = prev_state
                    cur_state.prev= best_prev_state
                    cur_state.accum_score = best_accum_score + cur_state.score

            # collecting the decoding results
            # first finding the best last state
            best_state, best_score = None, -float('inf')
            for state in self.seqs_states_array[seq_idx][-1]:
                if state.accum_score > best_score:
                    best_state, best_score = state, state.accum_score

            # tracing back and decode
            cur = best_state
            while(cur is not None):
                self.picked_states.append(cur)
                seq_decode_results = [cur.positve] + seq_decode_results
                cur = cur.prev
            
            assert(len(seq_decode_results) == self.seq_length[seq_idx])
            # saving all the best states for the sake of training
            self.seqs_decode_results.append(seq_decode_results)
        
        return self.seqs_decode_results

    # train for one iteration after decoding
    def train_once(self):
        for state in self.picked_states:
            state.train_perceptron(self.clf)

    # clear all the history of decoding
    # prepare for new rounds of inference
    def clear(self):
        for seq_idx in range(self.seq_count):
            for position_idx in range(self.seq_length[seq_idx]):
                for state in self.seqs_states_array[seq_idx][position_idx]:
                    state.clear()

    # the actual training algorithm
    # decode - training loop
    def fit(self, X, y, num_iterations=100):
        self.load_data(X, y)
        for _ in range(num_iterations):
            self.decode_seqs()
            self.train_once()

    # given feature vector, partially observed y, predict
    def predict(self, X, y=None):
        if y is None:
            y = [[None] * len(x) for x in X]
        self.load_data(X, y)
        return self.decode_seqs()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test, y_pred = flatten_rm_none(y_test, y_pred)
        f1_macro, f1_weighted, acc = (f1_score(y_test, y_pred, average='macro'),
                                      f1_score(y_test, y_pred, average='weighted'),
                                      accuracy_score(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        return {'macro_f1_score': f1_macro,
                'weighted_f1_score': f1_weighted,
                'accuracy': acc}

# remove the labels that is None in the test set and flatten
def flatten_rm_none(y_test, y_pred):
    _y_test, _y_pred = [], []
    seq_count = len(y_test)
    for seq_idx in range(seq_count):
        sentence_count = len(y_test[seq_idx])
        for sentence_idx in range(sentence_count):
            if y_test[seq_idx][sentence_idx] is not None:
                _y_test.append(y_test[seq_idx][sentence_idx])
                _y_pred.append(y_pred[seq_idx][sentence_idx])
    for idx in range(len(_y_test)):
        assert(_y_test[idx] is not None and _y_pred[idx] is not None)
    return _y_test, _y_pred

def eval_state(clf, state):
    return clf.score(state.x, state.positve)

def encode2idx(np_encode):
    accum = 2 ** (len(np_encode) - 1) - 2
    for idx in range(len(np_encode) - 1):
        accum += np_encode[idx + 1] * (2 ** idx)
    return accum

class State:

    def __init__(self, np_encode):
        self.state = np_encode
        self.positve = (self.state[0] == 1)
        self.order = len(np_encode)
        self.idx = encode2idx(np_encode)
        self.idxes = [encode2idx(np_encode[:idx + 1]) for idx in range(1, self.order)]
        self.x = None
        self.prev, self.score, self.accum_score = None, None, None
    
    # check whether previous state can transition to the current state
    def compatible(self, prev_state):
        if self.order != prev_state.order + 1 and self.order != prev_state.order:
            print('state encode length error')
            assert(False)
        if self.order == prev_state.order:
            return np.array_equal(prev_state.state[:-1], self.state[1:])
        else:
            return np.array_equal(prev_state.state[:], self.state[1:])

    # generating the x vector by concatenating the feature vector and the state encode
    def set_vector(self, feature_vec, state_dim):
        state_vec = np.zeros(state_dim)
        state_vec[self.idxes] = 1
        self.x = np.concatenate((feature_vec, state_vec))

    # clear the previous state and score after the vertibi algorithm
    def clear(self):
        self.prev, self.score, self.accum_score = None, None, None

    # using the clf (which is perceptron) to evaluate the current state
    def eval(self, clf):
        self.score = clf.score(self.x, self.positve)

    # if this state is in the
    # train the perceptron with its x and y
    def train_perceptron(self, p):
        p.fit_one_datapoint(self.x, self.positve)

    def __str__(self):
        return 'encode: ' + str(self.state) + '. idxes: ' + str(self.idxes) + '\n x: ' + str(self.x)



if __name__ == '__main__':
    debug = 'test_case4'
    print('Debug content %s ' % debug)
    
    # test whether the encoding scheme (and state constructor) works
    if debug == 'encode scheme':
        np_encode = np.array([0, 0, 1, 1])
        s = State(np_encode)
        print(s)

    # test whether the constructor works
    if debug == 'Seqs':
        np.random.seed(0)
        feature_dim, order = 5, 2
        num_iterations = 10
        seqs = Seqs(feature_dim, order)
        X = np.random.normal(size=(1, 5, feature_dim))
        y = [[None, True, False, True, True]]
        
        seqs.load_data(X, y)
        for seq_states_array in seqs.seqs_states_array:
            for states in seq_states_array:
                for state in states:
                    print(state)

    # test_case1, testing whether perceptron can fit each individual "emission"
    if debug == 'test_case1':
        
        # initialize data and parameters
        np.random.seed(1)
        feature_dim, order = 5, 0
        num_iterations = 10
        seqs = Seqs(feature_dim, order)
        X = np.array([np.eye(feature_dim)])
        y = [[False, False, False, True, True]]
        # training
        seqs.fit(X, y, num_iterations=100)
        r = seqs.predict(X)
        print(r)

    # test_case2, testing whether perceptron can fit local trends
    if debug == 'test_case2':
        
        # initialize data and parameters
        array_length, feature_dim, order = 100, 1, 1
        X = np.random.normal(size=(1, array_length, feature_dim))
        y = [[idx % 2 == 0 for idx in range(array_length)]]
        # training
        seqs = Seqs(feature_dim, order)
        seqs.fit(X, y, num_iterations=1000)
        r = seqs.predict(X)
        print(len([idx for idx in range(array_length - 1) if r[0][idx + 1] != r[0][idx]]) / (len(r[0]) - 1))

    # test_case3, testing whether perceptron can fit longer trends
    if debug == 'test_case3':
        
        # initialize data and parameters
        array_length, feature_dim, order = 100, 1, 2
        X = np.random.normal(size=(1, array_length, feature_dim))
        y = [[idx % 3 != 0 for idx in range(array_length)]]
        # training
        seqs = Seqs(feature_dim, order)
        seqs.fit(X, y, num_iterations=500)
        r = seqs.predict(X)
        print(r)

    # test_case4, testing whether the algorithm performs correctly when there are multiple sequences
    if debug == 'test_case4':
        num_seq, feature_dim, order, seq_length = 50, 500, 2, 10
        X = np.random.normal(size=(num_seq, seq_length, feature_dim))
        y = np.random.random((num_seq, seq_length)) > 0.5
        seqs = Seqs(feature_dim, order)
        seqs.fit(X, y, num_iterations=100)
        r = seqs.predict(X)
        print(np.sum(r == y) / (num_seq * seq_length))
