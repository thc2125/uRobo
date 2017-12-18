import argparse

from pathlib import Path

import alignment
from target_feat_predicter import nn_utils
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


    asr_group = parser.add_argument_group('ASR Group')
    asr_group.add_argument('-k',
                           '--kaldi_dir',
                           type=Path,
                           help='The directory holding Kaldi')

    asr_training_group = parser.add_argument_group('ASR Training Group')
    asr_training_group.add_argument('-r',
                                 '--raw_data_dir',
                                 type=Path,
                                 help='The directory for Kaldi to download raw'
                                      + ' LibriSpeech data into')

    alignment_group = parser.add_argument_group('Alignment Arguments')
    alignment_group.add_argument('-D',
                                 '--data_to_align',
                                 help='The directory holding the specific data'
                                      + ' to align')
    alignment_group.add_argument('-a',
                                 '--asr_model',
                                 help='The alignment model from Kaldi to use')

    preprocessing_group = parser.add_argument_group('Preprocessing Arguments')
    preprocessing_group.add_argument('-K',
                                     '--kaldi_data_dir', 
                                     type=Path,
                                     help='The path to the kaldi data to be pre-processed')
    preprocessing_group.add_argument('-L',
                                     '--kaldi_language_model_dir', 
                                     type=Path,
                                     help='The path to the kaldi language model')

    preprocessing_group.add_argument('-P',
                                     '--processed_dir', 
                                     type=Path,
                                     help='The directory to hold pre-processed data')

    preprocessing_group.add_argument('-d',
                                     '--duration_limit',
                                     type=utils.get_hours,
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
    preprocessing_group.add_argument('-M',
                                     '--mono_di_tri_phones',
                                     type=Path,
                                     help='A json file listing monophones,'
                                          + ' diphones, and triphones to use'
                                          + ' when preprocessing the data.')
    preprocessing_group.add_argument('-S',
                                     '--skip_audio',
                                     action='store_true',
                                     help='Skip copying audio files if they already exist.')

    training_group = parser.add_argument_group('Target Feature Prediction Training Arguments')
    training_group.add_argument('-T',
                                '--train_corpus',
                                type=Path,
                                help='A directory to a pre-processed corpus'
                                     + ' against which to evaluate the trained model')
    training_group.add_argument('-E',
                                '--test_corpus',
                                type=Path,
                                help='A directory to a pre-processed corpus'
                                     + ' against which to evaluate the trained model')
    training_group.add_argument('-f',
                                '--final_model',
                                type=Path,
                                help='The name/location of the final target'
                                     + ' feature prediction model.')
    training_group.add_argument('-e', 
                                '--epochs',
                                type=int,
                                help='The number of epochs over which to train'
                                     + ' the model')

    synthesis_group = parser.add_argument_group('Synthesizer Arguments')
    synthesis_group.add_argument('-A',
                                 '--audio_data_dir', 
                                 type=Path,
                                 help='The directory that (will) contain pre-processed'
                                      + ' data. This directory will store the results of'
                                      + ' pre-processing and be used in synthesis.')

    synthesis_group.add_argument('-n',
                        '--target_feature_model',
                        help='The model to use for predicting target features'
                             + ' or if training, the location/filename of the'
                             + ' new model.')
    synthesis_group.add_argument('-m',
                        '--mono',
                        action='store_true',
                        help='Whether to use monophones instead of overlapping'
                             + ' mono-di-tri-phones in the synthesizer')
    synthesis_group.add_argument('-t',
                        '--text',
                        help='The text to synthesize into a wav file.')
    synthesis_group.add_argument('-o',
                        '--output_file',
                        type=Path,
                        help='The output file in which to store the wav.')




    args = parser.parse_args()

    #0. Train an ASR / Align the data
    if args.kaldi_dir:
        if args.raw_data_dir:
            train_asr_args = {}
            train_asr_args['kaldi_path'] = args.kaldi_dir
            train_asr_args['raw_data_dir'] = args.raw_data_dir
            alignment.train_asr(**train_asr_args)
        
        if args.data_to_align:
            align_args = {}
            align_args['kaldi_path'] = args.kaldi_dir
            align_args['data_to_align'] = args.data_to_align
            # If we haven't defined a Kaldi data directory,
            # we can do that here.
            if not args.kaldi_data_dir:
                args.kaldi_data_dir=args.data_to_align
            if args.asr_model:
                align_args['asr_model'] = args.asr_model
            alignment.align(**align_args)

    #1. Pre-process data
    if args.kaldi_data_dir and args.kaldi_language_model_dir and args.processed_dir:
        preprocess_args = {}
        preprocess_args['kaldi_data_dirpath']=args.kaldi_data_dir
        preprocess_args['kaldi_lm_dirpath']=args.kaldi_language_model_dir
        preprocess_args['processed_dirpath'] = args.processed_dir
        # We can also set the target feature predicter training corpus if it hasn't
        # already been set
        if not args.train_corpus:
            args.train_corpus = args.processed_dir

        if args.duration_limit:
            preprocess_args['duration_limit'] = args.duration_limit
        if args.gender:
            preprocess_args['gender'] = args.gender
        if args.speakers:
            preprocess_args['speakers'] = args.speakers
        if args.mono_di_tri_phones:
            preprocess_args['mono_di_tri_phones'] = utils.load_json(args.mono_di_tri_phones)
        if args.skip_audio:
            preprocess_args['skip_audio'] = args.skip_audio
        preprocess(**preprocess_args)

    #2. Train the target feature prediction model
    if args.train_corpus and args.test_corpus:
        train_args = {}
        train_args['train_corpus_dir'] = args.train_corpus
        train_args['test_corpus_dir'] = args.test_corpus
        if args.final_model:
            train_args['model_path'] = args.final_model
        if args.epochs:
            train_args['epochs'] = args.epochs
        trained_model_path = nn_utils.train_nn_model(**train_args)
        # If the user hasn't specified a model, use the default created by 
        # this function
        if not args.target_feature_model:
            args.target_feature_model = trained_model_path
        # If the user hasn't specified a data path for the synthesizer, use the
        # training data path
        if not args.audio_data_dir:
            args.data_dir = args.train_corpus

    #3. Run the synthesizer i.e. the "Decode" stage
    if args.text and args.target_feature_model and args.audio_data_dir:
        print("Building a concatenative synthesizer.")
        concat_args = {}
        concat_args['data_dir'] = args.audio_data_dir
        concat_args['target_predicter_model_path'] = (args.target_feature_model)
        if args.mono:
            concat_args['mono'] = (args.mono)

        c = NNConcatenator(**concat_args)
        synth_args = {}
        synth_args['text'] = args.text
        if args.output_file:
            synth_args['output_path']=args.output_file
        c.synthesize(**synth_args)
