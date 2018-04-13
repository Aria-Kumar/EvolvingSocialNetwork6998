from Corpus import *

# implement the “are team” feature
# returns a boolean
def are_team(sentence, char_pair):
    
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
    
    # according to paper, sharing the same verbs as subj/dobj, but does not seem to work
    char1_sub_set, char2_sub_set, char1_obj_set, char2_obj_set = set(), set(), set(), set()
    
    for token_idx in range(sentence.num_tokens):
        if correctedCharIds[token_idx] not in [character_id1, character_id2]:
            continue
        if correctedCharIds[token_idx] == character_id1:
            if deprels[token_idx] == 'nsubj':
                char1_sub_set.add(headTokenIds[token_idx])
            elif deprels[token_idx] == 'dobj':
                char1_obj_set.add(headTokenIds[token_idx])
        elif correctedCharIds[token_idx] == character_id2:
            if deprels[token_idx] == 'nsubj':
                char2_sub_set.add(headTokenIds[token_idx])
            elif deprels[token_idx] == 'dobj':
                char2_obj_set.add(headTokenIds[token_idx])
    
        # using conj dependency relation
        headTokenId = headTokenIds[token_idx]
        if deprels[token_idx] == 'conj':
            if headTokenId != -1 and df.ix[headTokenId]['correctedCharId'] in [character_id1, character_id2]:
                    return True

    return len(char1_sub_set & char2_sub_set) != 0 or len(char1_obj_set & char2_obj_set) != 0

if __name__ == '__main__':
    # reading the texts and picking a sentence
    author_name = 'albee.dream' # file name
    passage = Passage(author_name)
    sequence_dict = passage.sequence_dict
    # char_pair = [char_pair for char_pair in sequence_dict][0] # which character pair to pick
    char_pair = (1,3)
    sentence_array = sequence_dict[char_pair].sentence_seq # sentence array is
    sentence = sentence_array[6] # getting the first sentence of the sequence
    print('The sentence of interest')
    print(sentence)
    print('The character pair of interest')
    print(char_pair)
    act_together_feature = are_team(sentence, char_pair)
    print(act_together_feature)


