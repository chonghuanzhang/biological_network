import requests
import pandas as pd
import signal
import pickle
import os

def timeout(func, input, time_out):
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_out)
    try:
        output = func(input)
    except TimeoutError:
        print('The function', str(func), 'takes longer than', time_out, 'seconds. The function is terminated.')
        output = False
    finally:
        signal.alarm(0)
    return output


# find longest subcommon sequence
def get_lcs(x, y):
    min_len = min(len(x), len(y))
    for i in range(min_len, 0, -1):
        x_subs = []
        y_subs = []

        for j in range(min_len):
            x_sub = x[j:i+j]
            if len(x_sub) == i:
                x_subs.append(x_sub)

            y_sub = y[j:i + j]
            if len(y_sub) == i:
                y_subs.append(y_sub)

        commons = [x_sub for x_sub in x_subs for y_sub in y_subs if x_sub == y_sub]
        if commons:
            break

    return commons


def lcs(commons, aaseq):
    commons_new = []
    for common in commons:
        commons_new +=get_lcs(common, aaseq)
    print(commons_new)

    return commons_new


def binarize_aaseq(aaseq):

    aatypes = set('ARNDCQEGHILKMFPSTWYV')
    df = pd.DataFrame(columns=aatypes)
    for aa in aaseq:
        data = {aatype: 0 for aatype in aatypes}
        for aatype in aatypes:
            if aa == aatype:
                data.update({aa: 1})
                row = pd.Series(data)
                df = df.append(row, ignore_index=True)
                break
    return df


def get_enzyme(enzyme):
    web = 'http://rest.kegg.jp/get/' + enzyme
    return requests.get(web).text


def get_aaseq(enzyme):

    def get_sequence(gene):

        web = 'http://rest.kegg.jp/get/' + gene + '/aaseq'
        response = requests.get(web).text
        aaseq = response[response.find('\n')+1::]
        return  "".join(aaseq.split('\n'))

    try:
        gene_text = enzyme[enzyme.find('GENES') + 12::]
    except Exception:
        return False

    gene = gene_text.split('\n')
    genes = []
    genes.append(gene[0])

    for g in gene[1::]:
        if g.startswith('            '):
            genes.append(g[12::])
        else:
            break

    parsed_genes = []
    for g in genes:
        parts = g.split(' ')
        init = str.lower(parts[0])
        if len(parts) == 2:
            parsed_genes.append(init + parts[1])
        else:
            for part in parts[1::]:
                parsed_genes.append(init + part)

    for idx, g in enumerate(parsed_genes):
        if g.find('(') != -1:
            parsed_genes[idx] = g[:g.find('(')]


    sequences = {}
    for gene in parsed_genes:
        print(gene)
        aaseq = timeout(get_sequence, gene, time_out=3)
        if aaseq != False and aaseq != '':
            sequences.update({gene: binarize_aaseq(aaseq)})
        else:
            print('aaseq of ', gene, ' is not found.')

    return sequences


if __name__ == "__main__":

    address = '/home/cz332/programming/aaseqCrawler/'
    exist = os.listdir(address)
    exist = [i[0:-4] for i in exist]

    with open('/home/cz332/programming/kegg_database/original_database_file/kegg_enzyme.txt', 'rt') as f:
    #with open('kegg_enzyme.txt', 'rt') as f:
        enzymes = f.read()
    enzymes = enzymes.split('\n')
    enzymes = enzymes[0:-1]
    enzyme_tot = []
    for i in enzymes:
        enzyme = i[3:i.find('\t')]
        enzyme_tot.append(enzyme)

    enzyme_tot = enzyme_tot
    enzyme_aaseq_dict = {}
    execuate = []
    for enzyme in enzyme_tot:
        if enzyme not in exist:
            execuate.append(enzyme)

    for enzyme in execuate:
        print('\n')
        print(enzyme)
        enzyme_text = timeout(get_enzyme, enzyme, time_out=15)
        if enzyme_text != '' or enzyme_text != False:
            sequences = get_aaseq(enzyme_text)
            if sequences != False:
                with open(address + '%s.pkl'%enzyme, 'wb') as f:
                    pickle.dump(sequences, f)


    """
    commons = [sequences[0]]
    for aaseq in sequences:
        commons = lcs(commons, aaseq)
        if commons == []:
            break

    all_commons = []
    for aaseq1 in sequences:
        for aaseq2 in sequences:
            if aaseq1 != aaseq2:
                commons = get_lcs(aaseq1, aaseq2)
                all_commons += commons
    """



