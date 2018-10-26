"""
Script to merge
"""

import fire
from pathlib import Path
import csv
import numpy as np
from elasticsearch_util import AnoIterable


def get_anos_regioes():
    regioes = np.arange(1, 25, 1)
    anos_treino = [2014]
    anos_validacao = np.arange(1997, 2013, 1)
    anos_teste = [2013]
    return anos_treino, anos_validacao, anos_teste, regioes


def write_file(file_path, text_iter, num_tokens):
    total_num_tokens = 0
    print(f'Writing to {file_path}...')
    j = 0
    with open(file_path, 'w', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for i, text in enumerate(text_iter):
            j += 1
            writer.writerow([text])

            # calculate approximate length based on tokens
            total_num_tokens += len(text.split())
            if total_num_tokens > num_tokens:
                break
            if i % 10000 == 0:
                print('Processed {:,} documents. Total # tokens: {:,}.'.format(i, total_num_tokens))
    print('{}. # documents: {:,}. # tokens: {:,}.'.format(
        file_path, j, total_num_tokens))


def main(output, num_tokens=100000000):

    output = Path(output)
    assert output.exists(), f'Error: {output} does not exist.'
    output.mkdir(exist_ok=True)

    train_path = output.joinpath('train.csv')
    valid_path = output.joinpath('valid.csv')
    test_path = output.joinpath('test.csv')

    anos_treino, anos_validacao, anos_teste, regioes = get_anos_regioes()

    write_file(train_path, AnoIterable(anos_treino, regioes), num_tokens)
    write_file(valid_path, AnoIterable(anos_validacao, regioes), num_tokens/10)
    write_file(test_path, AnoIterable(anos_teste, regioes), num_tokens / 10)


if __name__ == '__main__': fire.Fire(main)
