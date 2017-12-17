import numpy as np

from pathlib import Path

from target_feat_predicter.nn import NN

# Define constants
utterance_phone_file = 'utt_mono_di_tri_phones.npy'
utterance_phone_feats_file = 'spkr_ind_mono_di_tri_phonestarget_feats_normalized.npy'

def train_nn_model(train_corpus_dir, 
                   test_corpus_dir,
                   model_path=Path('./models/model.h5'), 
                   epochs=20):
    '''A function that goes through the steps of training a neural target 
    feature prediction model.
    Keyword Arguments:

    train_corpus_dir -- a Path to a directory containing the training corpus data
    test_corpus_dir  -- a Path to a directory containing the test corpus data
    model_path       -- a filepath to save the newly trained model
    epochs           -- the number of epochs over which to train the model
    '''

    if not model_path.parent.exists():
        model_path.mkdir(parents=True)

    (utt_len,
     train_phones, 
     train_features, 
     test_phones, 
     test_features) = prepare_data(train_corpus_dir, test_corpus_dir)

    num_phones = np.amax(train_phones) + 1
    num_features = train_features.shape[-1]

    nn = NN(utt_len, num_phones, num_features, model_path)

    nn.train(train_phones, train_features, epochs)

    print(nn.evaluate(test_phones, test_features))

    return model_path

def prepare_data(train_corpus_dir, test_corpus_dir):
    '''A function to load data for ingestion into the neural network
    Keyword Arguments
    train_corpus_dir -- a Path to the training corpus, containing the 
                        utterance_phone_file and utterance_phone_feats_file 
                        (numpy files)
    test_corpus_dir -- a Path to the test corpus, containing the 
                        utterance_phone_file and utterance_phone_feats_file 
                        (numpy files)
    '''

    train_phones_unpad = (np.load(str(train_corpus_dir / utterance_phone_file)))
    train_features_unpad = (np.load(str(train_corpus_dir / utterance_phone_feats_file)))

    test_phones_unpad = (np.load(str(test_corpus_dir / utterance_phone_file)))
    test_features_unpad = (np.load(str(test_corpus_dir / utterance_phone_feats_file)))

    utt_len = max(train_phones_unpad.shape[-1], test_phones_unpad.shape[-1])

    train_padding = utt_len - train_phones_unpad.shape[-1]
    test_padding = utt_len - test_phones_unpad.shape[-1]

    train_phones = np.pad(train_phones_unpad, [(0,0), (0,train_padding)], mode='constant')
    train_features = np.pad(train_features_unpad, [(0,0), (0,train_padding), (0,0)], mode='constant')

    test_phones = np.pad(test_phones_unpad, [(0,0), (0,test_padding)], mode='constant')
    test_features = np.pad(test_features_unpad, [(0,0), (0,test_padding), (0,0)], mode='constant')

    return utt_len, train_phones, train_features, test_phones, test_features
