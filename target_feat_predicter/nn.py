import argparse

from pathlib import Path

import numpy as np


class NN():

    def __init__(self, 
                 utt_len=0, 
                 num_phones=0, 
                 num_features=0, 
                 model_path='model.h5', 
                 existing_model_path=None):
        if existing_model_path:
            from keras.models import load_model
            self.model=load_model(str(existing_model_path))
        else:
            self.model=self._init_model(utt_len, num_phones, num_features)
            self.utt_len = utt_len
            self.num_phones = num_phones
            self.model_path = model_path

    def _init_model(self, utt_len, num_phones, num_features):
        from keras.models import Model, load_model
        from keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, TimeDistributed
        from keras.optimizers import SGD, RMSprop

        phone_input = Input(shape=(utt_len,), dtype='int32', name='phone_input')

        embeddings = Embedding(output_dim=32, 
                               input_dim=num_phones, 
                               input_length=utt_len, 
                               mask_zero=True)(phone_input)

        context_embeddings1 = Bidirectional(LSTM(67, return_sequences=True))(embeddings)
        context_embeddings2 = Bidirectional(LSTM(57, return_sequences=True))(context_embeddings1)
        context_embeddings3 = Bidirectional(LSTM(46, return_sequences=True))(context_embeddings2)

        features = TimeDistributed(Dense(num_features, activation='tanh'))(context_embeddings3)

        model = Model(inputs=phone_input, outputs=features)
        optimizer = RMSprop()
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def train(self, utterance_phones, features, epochs=20):
        from keras.callbacks import ModelCheckpoint
        checkpoints_dirpath = Path(self.model_path.parent / 'checkpoints')
        if not checkpoints_dirpath.exists():
            checkpoints_dirpath.mkdir(parents=True)
        checkpointer = ModelCheckpoint(filepath=str(checkpoints_dirpath
                                                    / '{epoch:02d}.hdf5'))
        self.model.fit(utterance_phones, features, callbacks=[checkpointer], epochs=epochs)
        self.model.save(str(self.model_path))

    def get_input_length(self):
        return self.model.get_layer(index=0).input_shape[1]

    def evaluate(self, utterance_phones, gold_features):
        return self.model.evaluate(utterance_phones, gold_features)

    def predict(self,utterance_phones):
        return self.model.predict(utterance_phones)
