from bs4 import BeautifulSoup
import urllib
import sys
import re


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

            entry_url = entry.find('a')['href']
            if entry_url != '#':
                novel_urls.append(entry_url)

    return novel_urls


def make_soup(url):
    """
    Create a beautiful soup instance from the given url
    """
    try:
        response = urllib.urlopen(url)
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

    for url in urls:

        # Open the summary webpage
        summary_url = url + 'summary/'

        soup = make_soup(summary_url)

        # Make sure it was able to find a summary page
        if soup is None:
            continue

        text = ''
        studyguide = soup.find('div', id='plotoverview', class_='studyGuideText')
        for p in studyguide.find_all('p'):
            text += re.sub('\s+', ' ', p.text.encode('ascii', 'ignore')).strip()

        print(text)

    return summaries


if __name__ == '__main__':

    print('Compiling summary URL\'s...')
    urls = get_sparknotes_novel_urls()

    print('Compiling summary texts...')
    print('Found {} plot summeries'.format(len(urls)))

    get_sparknotes_summaries(urls)