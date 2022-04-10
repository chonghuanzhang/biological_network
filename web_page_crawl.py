import requests


url = 'https://www.genome.jp/kegg/annotation/br01800.html'
response = requests.get(url).text


import pandas as pd
tables = pd.read_html(url)

tables2 = pd.read_html('https://en.wikipedia.org/wiki/Minnesota')


# use jupyter
def reduce_text(text):
    stop_pos = text.find(' ')
    return text[1:stop_pos]

reduce_text(text)