from subprocess import call

import os, sys
import time
import xml.etree.ElementTree as ET
import copy
import pandas as pd
from collections import Counter


data_dir = '../Data_EvolvingRelationships_Chaturvedi_AAAI2017/processedText/'
frames_dir = './frames/'
script = './semafor/semafor-semantic-parser/release/fnParserDriver.sh'
compressed_dir = './compressed/'

def compress_summaries():

    all_summaries = []

    for file in sorted(os.listdir(data_dir)):
        lines = open(data_dir + file, 'r').readlines()
        lines = [line.strip() for line in lines if line.strip() != ''] + ['\n']
        # print(i, len(lines), file)

        all_summaries += lines + ['\n']

    out = open('all.txt', 'w')

    out.writelines(all_summaries)


def compress_from_processed_text():

    for file in sorted(os.listdir(data_dir)):

        sentences = []


        df = pd.read_csv(data_dir + file, sep='\t')

        original_words = df['originalWord']

        i = 0
        sentence = ''
        for j in range(len(original_words)):

            if df['sentenceID'][j] != i:
                sentences.append(copy.deepcopy(sentence) + '\n\n')
                sentence = ''
                i += 1

            sentence += (original_words[j] + ' ')

        out = open(compressed_dir + file, 'w')

        out.writelines(sentences)


def generate_frames():
    # Get the frames that have already been generated - .out
    previously_generated_frames = set([file[:-4] for file in os.listdir(frames_dir)])

    for file in sorted(set(os.listdir(compressed_dir)) - previously_generated_frames):
        s = time.time()

        call(['ls', '-l'])

        print('parsing', file)
        try:
            call([script, compressed_dir + file, frames_dir + file + '.out'])
        except Exception:
            print('Error, exception at ', file)

        print('done in', time.time() - s)

def parse_frames():
    names = []

    for file in os.listdir(frames_dir):
        print(file)
        try:
            root = ET.parse(frames_dir + file).getroot()
            sentences = root.findall('.//sentences/')
        except:
            continue

        for sentence in sentences:
            aS = sentence.findall('.//annotationSet')
            print('set')



            for annot in aS:
                name = annot.get('frameName')
                print('  ' + name)

                for label in annot.findall('.//label'):
                    ID, name, start, end = label.get('ID'), label.get('name'), label.get('start'), label.get('end')
                    print('\t', ID, name, start, end)
                    # annot.add((ID, name, start, end))
                    names.append(name)

    return names

def most_common_frames():
    names = Counter()

    for name in parse_frames():
        names[name] += 1

    print(names)



if __name__ == '__main__':

    # Make sure to create a folder 'compressed/' before running this
    # compress_from_processed_text()

    generate_frames()
    # parse_frames()
    most_common_frames()