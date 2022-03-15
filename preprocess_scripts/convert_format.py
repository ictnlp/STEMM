import os
import csv
import tqdm
import textgrid
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--lang", help="target language")
args = parser.parse_args()

WORD_TABLE_COLUMNS = ['id', 'word_time', 'word_text']

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

seg_path = os.path.join(root, 'data', 'mustc', f'en-{args.lang}', 'segment')
splits = ['dev', 'tst-COMMON', 'tst-HE', 'train']

def convert(path):
    tg = textgrid.TextGrid.fromFile(path)

    word_time = [tg[0][j].maxTime for j in range(len(tg[0]))]
    word_text = [tg[0][j].mark for j in range(len(tg[0]))]
    word_time = ','.join(map(str, word_time))
    word_text = ','.join(word_text)

    return word_time, word_text


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


def main():
    for split in splits:
        word_table = {c: [] for c in WORD_TABLE_COLUMNS}
        split_path = os.path.join(seg_path, split + '_align')
        speaker_dirs = os.listdir(split_path)
        speaker_dirs = list(filter(lambda x:str.isdigit(x), speaker_dirs))
        speaker_dirs.sort(key=lambda x:int(x))
        pbar = tqdm.tqdm(range(len(speaker_dirs)))
        for speaker in speaker_dirs:
            pbar.update()
            speaker_dir = os.path.join(split_path, speaker)
            if os.path.isdir(speaker_dir):
                align_files = os.listdir(speaker_dir)
                align_files.sort(key=lambda x:int(x.split('.')[0].split('_')[-1]))
                for align_file in align_files:
                    word_time, word_text = convert(os.path.join(speaker_dir, align_file))
                    index = align_file.split('.')[0]
                    word_table['id'].append(index)
                    word_table['word_time'].append(word_time)
                    word_table['word_text'].append(word_text)
                    
        word_df = pd.DataFrame.from_dict(word_table)
        save_df_to_tsv(word_df, os.path.join(split_path, split + '_word_align.tsv'))
                

if __name__ == '__main__':
    main()