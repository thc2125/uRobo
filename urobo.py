import argparse

from pathlib import Path

from preprocess import preprocess
from t2s.speech_generator import 
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--nn_model')
    parser.add_argument('--preprocess_path', type=Path)
    parser.add_argument('--speakers', nargs='+')
    parser.add_argument('--train')
    args = parser.parse_args()

    data_dir = args.data_dir

    if args.preprocess_path:
        preprocess_path = args.preprocess_path
        preprocess_args['orig_dirpath']=
        if args.speakers:
            speakers = args.speakers
            preprocess_args['speakers'] = speakers
        preprocess(preprocess_path, data_dir)

    if 

