from bs4 import BeautifulSoup
from urllib.request import urlopen
import sys
import re
import time

verbose = True


def get_sparknotes_novel_urls():
    """
    Scrape the SparkNotes directory of novels for each novels' url
    """

    novel_urls = []

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        # Iterate through the directory
        soup = make_soup('http://www.sparknotes.com/lit/index_{}.html'.format(letter))
        column = soup.find('div', class_='col-mid')
        if column is None:
            sys.stderr.write('Could not find list of books for letter \'' + letter + '\' Perhaps Sparknotes has '
                                                                                     'changed since this script was '
                                                                                     'written?')

        for entry in column.find_all('div', class_='entry'):

            entry_url = str(entry.find('a')['href'])

            # Do some cleaning
            if entry_url == '#':
                continue

            if entry_url[-1] != '/':
                entry_url += '/'

            print(entry_url)

            novel_urls.append(entry_url)
`
    return novel_urls


def make_soup(url):
    """
    Create a beautiful soup instance from the given url
    """
    try:
        response = urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
    except IOError:
        # If the page doesn't exist, return None
        return None

    return soup


def get_sparknotes_summaries(urls):
    """
    Iterate through each books' summary page and extract the text
    """

    summaries = []

    for i in range(len(urls)):

        if verbose and i % 10 == 0:
            sys.stdout.write('\rDownload Progress:\t{:.2}%'.format(100 * float(i) / float(len(urls))))
            sys.stdout.flush()

        url = urls[i]

        # Open the summary webpage
        summary_url = url + 'summary/'

        soup = make_soup(summary_url)

        # Make sure it was able to find a summary page
        if soup is None:
            continue

        text = ''
        studyguide = soup.find('div', id='plotoverview', class_='studyGuideText')

        if studyguide is None:
            print('No plot overview at ' + summary_url)
            continue

        for p in studyguide.find_all('p'):
            text += re.sub('\s+', ' ', p.text).strip()

        summaries.append(text)

    return summaries


def save_summaries(summaries):
    """
    Save each text on a new line
    """
    with open('corpus_data/sparknotes.txt', 'w') as file:
        for text in summaries:
            file.write(text + '\n')


if __name__ == '__main__':
    """
    Default usage: download and save all sparknotes summaries to corpus/ folder.
    """

    start = time.time()

    print('Compiling summary URL\'s...')
    urls = get_sparknotes_novel_urls()

    print('Compiling summary texts...')
    print('Found {} plot summaries'.format(len(urls)))

    summaries = get_sparknotes_summaries(urls)

    print('Done in ', time.time() - start, 's')

    save_summaries(summaries)