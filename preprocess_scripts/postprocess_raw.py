import os
import csv
import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--lang", help="target language")
args = parser.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

splits = ['dev', 'tst-COMMON', 'tst-HE', 'train']
data_dir = os.path.join(root, 'data', 'mustc', f'en-{args.lang}')
seg_dir = os.path.join(data_dir, 'segment')


def replace(output, origin):
    output_list = output.split(',')
    origin_list = origin.split(' ')
    output_index = [idx for idx, word in enumerate(output_list) if word != '']
    if len(output_index) != len(origin_list):
        return None
    for idx, word in enumerate(origin_list):
        output_list[output_index[idx]] = word
    return ','.join(output_list)
    
def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def load_df_from_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def main():
    for split in splits:
        output_file = os.path.join(seg_dir, split + '_align', split + '_word_align.tsv')
        origin_file = os.path.join(data_dir, split + '_raw.tsv')
        output_table = load_df_from_tsv(output_file)
        origin_table = load_df_from_tsv(origin_file)
        concat_table = pd.merge(output_table, origin_table, on='id')
        concat_dict = list(concat_table.T.to_dict().values())
        pbar = tqdm.tqdm(range(len(concat_dict)))
        final_dict = []
        for value in concat_dict:
            pbar.update()
            new_text = replace(value['word_text'], value['src_text'])
            if new_text is not None:
                value['word_text'] = new_text
                final_dict.append(value)
        final_table = pd.DataFrame.from_dict(final_dict)
        save_df_to_tsv(final_table, os.path.join(data_dir, split + '_raw_seg.tsv'))

if __name__ == '__main__':
    main()
