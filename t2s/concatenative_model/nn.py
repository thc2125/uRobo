import argparse

from pathlib import Path

import numpy as np


class NN():

    def __init__(self, utt_len=0, num_phones=0, num_features=0, model_dir=0, model_path=None):
        if model_path:
            from keras.models import load_model
            self.model=load_model(str(model_path))
        else:
            self.model=self._init_model(utt_len, num_phones, num_features)
            self.utt_len = utt_len
            self.num_phones = num_phones
            self.model_dir = model_dir

    def _init_model(self, utt_len, num_phones, num_features):
        from keras.models import Model, load_model
        from keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, TimeDistributed
        from keras.optimizers import SGD

        phone_input = Input(shape=(utt_len,), dtype='int32', name='phone_input')

        embeddings = Embedding(output_dim=50, 
                               input_dim=num_phones, 
                               input_length=utt_len, 
                               mask_zero=True)(phone_input)

        context_embeddings1 = Bidirectional(LSTM(67, return_sequences=True))(embeddings)
        context_embeddings2 = Bidirectional(LSTM(57, return_sequences=True))(context_embeddings1)
        context_embeddings3 = Bidirectional(LSTM(46, return_sequences=True))(context_embeddings2)

        features = TimeDistributed(Dense(num_features, activation='relu'))(context_embeddings3)

        model = Model(inputs=phone_input, outputs=features)
        optimizer = SGD()
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def train(self, utterance_phones, features, epochs=20):
        from keras.callbacks import ModelCheckpoint
        checkpoints_dirpath = Path(self.model_dir / 'checkpoints')
        if not checkpoints_dirpath.exists():
            checkpoints_dirpath.mkdir(parents=True)
        checkpointer = ModelCheckpoint(filepath=str(checkpoints_dirpath
                                                    / '{epoch:02d}.hdf5'))
        self.model.fit(utterance_phones, features, callbacks=[checkpointer], epochs=epochs)
        self.model.save(str(self.model_dir / 'model.h5'))

    def get_input_length(self):
        return self.model.get_layer(index=0).input_shape[1]

    def evaluate(self, utterance_phones, gold_features):
        return self.model.evaluate(utterance_phones, gold_features)

    def predict(self,utterance_phones):
        return self.model.predict(utterance_phones)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('train_corpus_dir', type=Path)
    parser.add_argument('test_corpus_dir', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-e', '--epochs', type=int)


    args=parser.parse_args() 

    train_corpus_dir = args.train_corpus_dir
    test_corpus_dir = args.test_corpus_dir

    model_dir = args.model_dir
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    utterance_phone_file = 'utt2mono_di_tri_phones.npy'
    utterance_phone_feats_file = 'target_feats_normalized.npy'

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

    num_phones = np.amax(train_phones) + 1
    num_features = train_features.shape[-1]
    nn = NN(utt_len, num_phones, num_features, model_dir)

    if args.epochs:
        nn.train(train_phones, train_features, args.epochs)
    else:
        nn.train(train_phones, train_features)


    print(nn.evaluate(test_phones, test_features))

