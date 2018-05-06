from Corpus import *
import nltk
from nltk.corpus import stopwords
import pickle as pkl
from


stopwords_set = set(stopwords.words('english'))
word2feature_vector = pkl.load(open('../resource/word2feature_vec.pkl', 'rb'))
word_dim = len(word2feature_vector['good_a'])
word_embed_dir = 'models/final_sparknotes.w2v'
emb = WordEmbedding(load_from=word_embed_dir)

def pos2pos(pos):
    if pos[0] == 'V':
        return 'v'
    if pos[0] == 'N':
        return 'n'
    return 'a'

'''
    Util functions
'''
# return a list of tokenids of words that a character is the deprel ('nsubj' or 'dobj' or 'iobj')
def char_deprel_set(sentence, char_id, deprel, debug=False):
    token_id_set = set()
    
    # the dataframe containing all the tokens (in the sentence) and their info
    df = sentence.df

    # code logics ...
    # retrieving the data in the form of lists that is convenient to deal with
    correctedCharIds = df['correctedCharId'].values
    deprels = df['deprel'].values
    tokenIds = df['tokenId'].values
    offset = tokenIds[0]
    headTokenIds = df['headTokenId'].values

    for token_idx in range(sentence.num_tokens):
        if correctedCharIds[token_idx] == char_id:
            cur_token_idx = token_idx
            while deprels[cur_token_idx] == 'conj':
                cur_token_idx = headTokenIds[cur_token_idx] - offset
                if cur_token_idx == -1:
                    break
            if cur_token_idx == -1:
                continue
            if deprels[cur_token_idx] == deprel and headTokenIds[cur_token_idx] != -1:
                token_id_set.add(df.ix[cur_token_idx + offset]['headTokenId'])

    if debug:
        print('extracted words are: '
              + ' '.join([df.ix[token_idx]['originalWord'] for token_idx in token_id_set]))
    return token_id_set

def append_wordset2feature_vector(feature_vec, word_set):
    if len(word_set) == 0:
        feature_vec += [0 for _ in range(word_dim)]
        return
    agg = np.zeros(word_dim, dtype='float')
    i = 0
    for word in word_set:
        if word2feature_vector.get(word) is None:
            continue
        agg = agg + np.array(word2feature_vector[word], dtype='float')
        i += 1
    if i == 0:
        feature_vec += [0 for _ in range(word_dim)]
        return
    agg = agg /  i
    feature_vec += agg.tolist()

# given a sentence and char pair, return a list of tokenIds for which char1/2 is nsubj/dobj
def get_general_deprel_sets(sentence, char_pair):
    character_id1, character_id2 = char_pair
    char1_sub_set = (char_deprel_set(sentence, character_id1, 'nsubj')
                     | char_deprel_set(sentence, character_id1, 'nmod:agent'))
    char2_sub_set = (char_deprel_set(sentence, character_id2, 'nsubj')
                     | char_deprel_set(sentence, character_id2, 'nmod:agent'))
    char1_obj_set = (char_deprel_set(sentence, character_id1, 'dobj')
                     | char_deprel_set(sentence, character_id1, 'nsubjpass'))
    char2_obj_set = (char_deprel_set(sentence, character_id2, 'dobj')
                     | char_deprel_set(sentence, character_id2, 'nsubjpass'))
    return char1_sub_set, char2_sub_set, char1_obj_set, char2_obj_set

def get_original_word(sentence, token_idx_set):
    return [sentence.df.ix[tokenId]['lemma'] + '_'
            + pos2pos(sentence.df.ix[tokenId]['pos'])
            for tokenId in token_idx_set]

'''
    Core Implementations
'''
# implement the “are team” feature
# returns a boolean
def are_team(sentence, char_pair):
    char1_sub_set, char2_sub_set, char1_obj_set, char2_obj_set \
        = get_general_deprel_sets(sentence, char_pair)
    return len((char1_sub_set & char2_sub_set) | (char1_obj_set & char2_obj_set)) != 0


def _act_together(sentence, char_pair):
    char1_sub_set, char2_sub_set, char1_obj_set, char2_obj_set = get_general_deprel_sets(sentence, char_pair)
    token_idx_set = (char1_sub_set & char2_obj_set) | (char2_sub_set & char1_obj_set)
    return token_idx_set

def _surrogate_act_together(sentence, char_pair):
    correctedCharIds = sentence.df['correctedCharId'].values
    for c in correctedCharIds:
        if c != -1 and c not in char_pair:
            return set()
    char1_sub_set, char2_sub_set, char1_obj_set, char2_obj_set = get_general_deprel_sets(sentence, char_pair)
    token_idx_set = char1_sub_set | char2_sub_set | char1_obj_set | char2_obj_set \
        - ((char1_sub_set & char2_obj_set) | (char2_sub_set & char1_obj_set)) \
        - ((char1_sub_set & char1_obj_set) | (char2_sub_set & char2_obj_set))

    return token_idx_set

def _adverbs_used(sentence, char_pair):
    token_idx_set = set()
    verbs_idx = _act_together(sentence, char_pair) | _surrogate_act_together(sentence, char_pair)
    df = sentence.df
    deprels = df['deprel'].values
    tokenIds = df['tokenId'].values
    headTokenIds = df['headTokenId'].values
    for token_idx in range(sentence.num_tokens):
        headTokenId = headTokenIds[token_idx]
        if headTokenId in verbs_idx and deprels[token_idx] == 'advmod':
            token_idx_set.add(tokenIds[token_idx])
    return token_idx_set

def _in_between(sentence, char_pair):
    # getting the char ids of interest
    character_id1, character_id2 = char_pair
    
    # the dataframe containing all the tokens (in the sentence) and their info
    df = sentence.df
    
    # the raw text of the original sentence
    sentence_form = sentence.sentence_form
    
    # code logics ...
    # retrieving the data in the form of lists that is convenient to deal with
    correctedCharIds = df['correctedCharId'].values
    deprels = df['deprel'].values
    tokenIds = df['tokenId'].values
    headTokenIds = df['headTokenId'].values

    start, start_charId = 0, None
    for token_idx in range(sentence.num_tokens):
        if correctedCharIds[token_idx] in char_pair:
            start, start_charId = token_idx, correctedCharIds[token_idx]
            break
    if start_charId is None:
        return set()
    for token_idx in reversed(range(sentence.num_tokens)):
        if correctedCharIds[token_idx] in char_pair and correctedCharIds[token_idx] != start_charId:
            end, end_charId = token_idx, correctedCharIds[token_idx]
            break
    if end_charId is None:
        return set()
    token_idx_set = set([tokenIds[token_idx] for token_idx in range(start + 1, end)])
    return token_idx_set

'''
    Wrappers
'''
def act_together(sentence, char_pair):
    return get_original_word(sentence, _act_together(sentence, char_pair))

def surrogate_act_together(sentence, char_pair):
    return get_original_word(sentence, _surrogate_act_together(sentence, char_pair))

def adverbs_used(sentence, char_pair):
    return get_original_word(sentence, _adverbs_used(sentence, char_pair))

def in_between(sentence, char_pair):
    wordlist = get_original_word(sentence, _in_between(sentence, char_pair))
    return [w for w in wordlist if w.split('_')[0] not in stopwords_set]

def get_embed(sentence, char_pair):
    wordlist = (act_together(sentence, char_pair)
                + surrogate_act_together(sentence, char_pair)
                + adverbs_used(sentence, char_pair)
                + in_between_words(sentence, char_pair))





def get_feature_vec(sentence, char_pair):
    feature_vec = []

    if are_team(sentence, char_pair):
        feature_vec += [1]
    else:
        feature_vec += [0]

    act_together_words = act_together(sentence, char_pair)
    append_wordset2feature_vector(feature_vec, act_together_words)
    surrogate_act_together_words = surrogate_act_together(sentence, char_pair)
    append_wordset2feature_vector(feature_vec, surrogate_act_together_words)
    adverb_used_words = adverbs_used(sentence, char_pair)
    append_wordset2feature_vector(feature_vec, adverb_used_words)
    in_between_words = in_between(sentence, char_pair)
    append_wordset2feature_vector(feature_vec, in_between_words)
    # only connotation value is used for lexical features
    feature_vec = feature_vec[:-6] + feature_vec[-4:-2]


    return feature_vec

if __name__ == '__main__':
    # reading the texts and picking a sentence
    author_name = 'albee.dream' # file name
    # author_name = 'agee.death' # file names
    passage = Passage(author_name)
    sequence_dict = passage.sequence_dict
    # char_pair = [char_pair for char_pair in sequence_dict][0] # which character pair to pick
    char_pair = (1,3)
    sentence_array = sequence_dict[char_pair].sentence_seq # sentence array is
    sentence = sentence_array[5] # getting the first sentence of the sequence
    print('The sentence of interest')
    print(sentence)
    print('The character pair of interest')
    print(char_pair)
    print(char_deprel_set(sentence, 2, 'dobj', debug=True))
    are_team_feature = are_team(sentence, char_pair)
    print(are_team_feature)
    act_together_words = act_together(sentence, char_pair)
    print(act_together_words)
    surrogate_act_together_words = surrogate_act_together(sentence, char_pair)
    print(surrogate_act_together_words)
    adverb_used_words = adverbs_used(sentence, char_pair)
    print(adverb_used_words)
    in_between_words = in_between(sentence, char_pair)
    print(in_between_words)
    feature_vector = get_feature_vec(sentence, char_pair)
    print(feature_vector)
