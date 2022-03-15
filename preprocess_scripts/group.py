import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", help="target language")
args = parser.parse_args()

splits = ['dev', 'tst-COMMON', 'tst-HE', 'train']
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seg_path = os.path.join(root, 'data', 'mustc', f'en-{args.lang}', 'segment')

for split in splits:
    split_path = os.path.join(seg_path, split)
    for f in os.listdir(split_path):
        if f.startswith('ted'):
            speaker = f.split('_')[1]
            speaker_dir = os.path.join(split_path, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            shutil.move(os.path.join(split_path, f), speaker_dir)