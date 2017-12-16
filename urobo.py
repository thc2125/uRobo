import argparse

from pathlib import Path

import target_feat_predicter.nn_utils
import utils

from preprocess import preprocess
from concatenative_model.concat_nn import NNConcatenator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A concatenative speech'
                                                 + ' synthesis system with'
                                                 + ' an end-to-end pipeline'
                                                 + ' from data alignment to'
                                                 + ' neural model training to'
                                                 + ' final synthesis.')
    parser.add_argument('data_dir_path', 
                        type=Path,
                        help='The directory containing pre-processed data')
    parser.add_argument('-n',
                        '--target_feature_model',
                        help='The model to use for predicting target features'
                             + ' or if training, the location/filename of the'
                             + ' new model.')
    parser.add_argument('-m',
                        '--mono',
                        action='store_true',
                        help='Whether to use monophones instead of overlapping'
                             + ' mono-di-tri-phones in the synthesizer')
    parser.add_argument('-t',
                        '--text',
                        help='The text to synthesize into a wav file.')
    parser.add_argument('-o',
                        '--output_file',
                        type=Path,
                        help='The output file in which to store the wav.')

    preprocessing_group = parser.add_argument_group('preprocessing_arguments')
    preprocessing_group.add_argument('-p',
                        '--preprocess_path', 
                        type=Path,
                        help='The path to the data to be pre-processed')
    preprocessing_group.add_argument('-d',
                        '--duration_limit',
                        help='The limit (in hours) of how'
                             + ' much audio data to preprocess')
    preprocessing_group.add_argument('-g',
                        '--gender',
                        help='The gender of the speakers to collect from the'
                             + ' raw corpus.')
    preprocessing_group.add_argument('-s', 
                                     '--speakers', 
                                     nargs='+',
                                     help='The speakers to pull from the raw corpus.')
    preprocessing_group.add_argument('--mono_di_tri_phones',
                                     help='A json file listing monophones,'
                                          + ' diphones, and triphones to use'
                                          + ' when preprocessing the data.')

    training_group = parser.add_argument_group('target_feature_training_arguments')
    training_group.add_argument('-c',
                                '--test_corpus',
                                type=Path,
                                help='A directory to a pre-processed corpus'
                                     + ' against which to evaluate the trained model')
    training_group.add_argument('-e', 
                                '--epochs',
                                help='The number of epochs over which to train'
                                     + ' the model')

    args = parser.parse_args()

    #0. Align the data
    # TODO
    #1. Pre-process data if required
    if args.preprocess_path:
        preprocess_args = {}
        preprocess_args['processed_dirpath'] = args.data_dir_path
        preprocess_args['orig_dirpath']=args.preprocess_path
        if args.duration_limit:
            preprocess_args['duration_limit'] = args.duration_limit
        if args.gender:
            preprocess_args['gender'] = args.gender
        if args.speakers:
            preprocess_args['speakers'] = args.speakers
        preprocess(**preprocess_args)

    #2. Train the target feature prediction model
    if args.test_corpus:
        train_args = {}
        train_args['train_corpus_dir'] = args.data_dir_path
        train_args['test_corpus_dir'] = args.test_corpus
        if args.target_feature_model:
            train_args['model_path'] = args.target_feature_model
        if args.epochs:
            train_args['epochs'] = args.epochs
        trained_model_path = nn_utils.train_nn_model(**train_args)
        # If the user hasn't specified a model, use the default created by 
        # this function
        if not args.target_feature_model:
            args.target_feature_model = trained_model_path

    #3. Run the synthesizer
    if args.text and args.target_feature_model:
        print("Building a concatenative synthesizer.")
        concat_args = {}
        concat_args['data_dir'] = args.data_dir_path
        concat_args['target_predicter_model_path'] = (args.target_feature_model)
        if args.mono:
            concat_args['mono'] = (args.mono)

        c = NNConcatenator(**concat_args)
        synth_args = {}
        synth_args['text'] = args.text
        if args.output_file:
            synth_args['output_path']=args.output_file
        c.synthesize(**synth_args)
