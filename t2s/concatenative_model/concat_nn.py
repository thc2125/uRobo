import utils
from t2s.concatenative_model.nn import NN

import numpy as np

class NNConcatenator():

    def __init__(self, 
                 word2phones,
                 phone2idx,
                 idx2phone,
                 utterance2idx,
                 idx2utterance,
                 utterance_phones,
                 utterance_feats,
                 phone2units,
                 alignments):
        self.target_predicter = NN(model_path)
        self.phone2idx = phone2idx
        self.idx2phone = idx2phone
        self.word2phones = word2phones
        self.utterance2idx = utterance2idx
        self.idx2utterance = idx2utterance
        self.utterance_phones = utterance_phones
        self.utterance_feats = utterance_feats
        self.phone2units = phone2units
        self.alignments = alignments

    def train(self, ):
        self.target_predicter.train()

    def generate(self, text):

        # Append the silence index to the beginning and the end
        # To indicate the start and stop of the target utterance
        phone_idxs = [1] + self._text2phones(text) + [1]

        candidate_units = self._get_candidates(phone_idxs)


        # Weights for the calculation of target cost
        # c_t_weights = numpy.array([3, 2, 1])

        # Weights for the calculation of concatenation cost
        # Get the target features for each phone
        target_feats = self._get_target_feats(phone_idxs)

        # W
        for candidate in candidate_units[1]:
            c_t = np.sum(np.absolute(np.subtract(target_feats, candidate_feats)))
            c_c = 
            candidate_costs[0][candidate] = 


        for idx in range(len(phone_idxs)):


            prev_candidate = -1
            for candidate in candidate[units]:
                candidate_target_feats = np.array(self.utterance_feats[candidate[0]][candidate[1]])
                # TODO: Bring in new feats
                candidate_cost_feats = candidate_target_feats
                c_t = np.sum(np.absolute(np.subtract(target_feats, candidate_feats)))
                if prev_candidate < 0:
                    c_c = 0
                else:
                    c_c = 

    def _text2phones(self, text):
        #TODO: Get a better model for t2p (another neural model?)
        phone_idxs = []
        for word in text.split():
            phones = self.word2phones[word]
            phone_idxs += [self.phone2idx[phone] for phone in phones]
        return phone_idxs

    def _get_candidates(self, phone_idxs):
        # There needs to be a nil silence candidate at the beginning and end
        candidates=[[(0,0)]]
        #
        for phone_idx in phone_idxs[1:-1]:
            candidates.append(self.phone2units[phone_idx])
        candidates.append[[(0,0)]]
        return candidates

    def _get_target_feats(self, phone_idxs):
        return self.target_predicter.predict(phone_idxs)

    def _concatenate(self):

