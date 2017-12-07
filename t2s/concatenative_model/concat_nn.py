import scipy.io.wavfile

from t2s.concatenative_model.nn import NN

import numpy as np

# TODO: Experiment with silences. Do we remove them in pre-processing? Does
# adding additional silence context at the beginning and end coerce the model
# to be more smooth?

class NNConcatenator():
    fs=16000
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
        self.utterance_target_feats = utterance_target_feats
        self.utterance_concat_feats = utterance_concat_feats
        self.phone2units = phone2units
        self.alignments = alignments

    def train(self, ):
        self.target_predicter.train()

    def generate(self, text, output_path=Path('./synth.wav')):

        candidates, target_feats = self._get_unit_context(text)

        # Initialize the Viterbi matrix
        cost_matrix = [[]]
        back_matrix = [[]]
        beg_sil_target_feats=target_feats[0]
        for sil_candidate_unit in candidates[0]:
            # For this round, we're only looking for the target cost
            # These are silence candidates
            sil_candidate_unit_feats = (
                np.array(
                    utterance_target_feats[sil_candidate_unit[0]][sil_candidate_unit[1]]))
            # Sum of the absolute difference of all the features
            # TODO: Add weights for each feature?
            c_t = np.sum(np.fabs(np.diff(sil_candidate_unit_feats, beg_sil_target_feats)))
            cost_matrix[0].append(c_t)

        # Now run the DP algorithm for the remaining candidates
        for idx in range(1, len(candidates[1:])):
            back_matrix.append([])
            candidate = candidates[idx]
            target_unit_feats = target_feats[idx]
            for candidate_unit in candidate:
                candidate_unit_feats = np.array(
                    utterance_target_feats[candidate_unit[0]][candidate_unit[1]])
                c_t = np.sum(np.fabs(np.diff(candidate_unit_feats,
                                             target_unit_feats)))
                # Get the concatenation cost from previous units
                c_c = float('inf')
                c_c_idx = 0
                # Reset our features to exclude duration
                candidate_unit_feats = candidate_unit_feats[1:]
                for prev_candidate_unit_idx in len(candidate_units[idx-1]):
                    # TODO: Experiment with different features for
                    # concatenation?
                    prev_candidate_unit = candidate_units[idx-1][prev_candidate_unit_idx]

                    prev_candidate_unit_feats = np.array(
                            utterance_target_feats[prev_candidate_unit[0]][prev_candidate_unit[1]][1:])
                    cand_c_c = np.sum(np.fabs(np.diff(candidate_unit_feats,
                                                      prev_candidate_unit_feats)))
                    if cand_c_c < c_c:
                        c_c = cand_c_c
                        c_c_idx = prev_candidate_unit_idx
                C = c_t + c_c
                back_matrix[idx].append(c_c_idx)
        # Now find the minimum of the last row
        final_C = float('inf')
        final_C_idx = 0
        for last_candidate_unit_idx in range(len(cost_matrix[-1])):
            cand_final_C = cost_matrix[last_candidate_unit_idx]
            if cand_final_C < final_C:
                final_C = cand_final_C
                final_C_idx = last_candidate_unit_idx
        candidate_unit_idx = final_c_idx
        final_units = []
        curr_idx = -1
        while(curr_idx > -(len(back_matrix))):
            final_units.append(candidate[curr_idx][candidate_unit_idx])
            curr_idx -= 1

        final_units.reverse() 
        concatenation = self._concatenate(final_units)
        wavfile.write(str(output_path), self.fs, concatenation)
        return concatenation




    def _get_unit_context(self, text):
        # TODO: Go straight from word to feature context?
        phones = self._get_phones(text)
        target_feats = self.target_predicter.predict(phones)
        candidates = self._get_candidates(phones)
        return candidate_units, 

    def _get_phones(self, text):
        #TODO: Get a better model for t2p (another neural model?)
        # Start with the silence candidates
        phone_idxs = [self.phone2units['SIL']]
        for word in text.split():
            phones = self.word2phones[word.upper()]
            phone_idxs += [self.phone2idx[phone] for phone in phones]
        # End with a final silence candidate
        phone_idxs += [self.phone2units['SIL']]
        return phone_idxs

    def _get_candidates(self, phone_idxs):
        candidates = []
        for phone_idx in phone_idxs:
            candidates.append(self.phone2units[phone_idx])
        return candidates

    def _get_target_feats(self, phone_idxs):
        return self.target_predicter.predict(phone_idxs)

    def _concatenate(self, final_units):
        units = []
        for unit_idxs in final_units:
            utterance = idx2utterance[unit_idxs[0]]
            utterance_dirs = preprocess.get_utterance_dirs(utterance)
            unit_start, unit_end = alignments[utterance][unit_idxs[1]]
            utterance_wav = wavfile.read(str(audio_dirpath
                                             / utterance_dirs 
                                             / (utterance + '.wav')))
            units += utterance_wav[int(unit_start*self.fs):int(unit_end*self.fs)]
        concatenation = np.concatenate(units)
        return concatenation
