from subprocess import call

import os
import time


data_dir = './Data_EvolvingRelationships_Chaturvedi_AAAI2017/originalText/'

i = 0

all_summaries = []

for file in sorted(os.listdir(data_dir)):
    lines = open(data_dir + file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip() != ''] + ['\n']
    # print(i, len(lines), file)


    all_summaries += lines + ['\n']

out = open('all.txt', 'w')

out.writelines(all_summaries)

script = './semafor/semafor-semantic-parser/release/fnParserDriver.sh'

for file in sorted(os.listdir(data_dir)):
    s = time.time()

    call(['ls', '-l'])

    print('parsing', file)
    try:
        call([script, data_dir + file, data_dir + file + '.out'])
    except Exception:
        print('Error, exception at ', file)
    print('done in', time.time() - s)