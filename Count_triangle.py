import pandas as pd
from os import listdir

def run(processed_file_dir):
    df = pd.read_csv(processed_file_dir, sep='\t')
    row_count = df.shape[0]
    r = 0
    last = 0
    edge_set = set()
    char_appearance = set()
    nodes = set()
    while r < row_count:
        sentence_id = df['sentenceID'][r]
        if sentence_id != last:
            last = sentence_id
            for char1 in char_appearance:
                for char2 in char_appearance:
                    if char1 != char2:
                        edge_set.add((char1, char2))
                        edge_set.add((char2, char1))
            char_appearance = set()
        c = df['correctedCharId'][r]
        if c != -1:
            char_appearance.add(c)
            nodes.add(c)
        r += 1
    triangle_count = 0
    for c1 in nodes:
        for c2 in nodes:
            for c3 in nodes:
                if len(set([c1, c2, c3])) == 3:
                    if (c1, c2) in edge_set and (c2, c3) in edge_set and (c1, c3) in edge_set:
                        triangle_count += 1
    return int(triangle_count / 6)

if __name__ == '__main__':

    processed_data_dir = '../Data_EvolvingRelationships_Chaturvedi_AAAI2017/processedText/'
    # run(processed_data_dir + 'cooper.last.processed')
    # exit(0)
    
    f_names = listdir(processed_data_dir)
    triangle_count = 0
    file_postfix = '.processed'
    for f in f_names:
        if len(f) > len(file_postfix) and f[-len(file_postfix):] == file_postfix:
            try:
                triangle_count += run(processed_data_dir + f)
            except:
                print(f)
                pass
    print(triangle_count)
