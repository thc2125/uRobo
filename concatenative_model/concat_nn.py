from collections import defaultdict
from pathlib import Path

from scipy.io import wavfile

import preprocess
import utils

from target_feat_predicter.nn import NN

import numpy as np

# TODO: Experiment with silences. Do we remove them in pre-processing? Does
# adding additional silence context at the beginning and end coerce the model
# to be more smooth?

class NNConcatenator():
    '''Creates a concatenative speech synthesizer using a neural network for 
    target prediction'''

    fs=16000
    def __init__(self, 
                 data_dir,
                 target_predicter_model_path,
                 mono=False):
        ''' Initializes the synthesizer
        Keyword Arguments
        data_dir -- directory to the audio data and appropriate data structures
        target_predicter_model_path -- the path to the neural network model for prediction
        '''
        (self.idx2vocabulary, 
         self.vocabulary2idx, 
         self.idx2phones, 
         self.phones2idx, 
         self.idx2mono_di_tri_phones, 
         self.mono_di_tri_phones2idx, 
         self.utt2words,
         self.utt2phones, 
         self.utt2alignments, 
         self.utt2diphones,
         self.utt2diphone_alignments,
         self.utt2triphones,
         self.utt2triphone_alignments,
         self.utt2mono_di_tri_phones,
         self.utt2mono_di_tri_alignments) = utils.load_data(data_dir)

        self.word2phones = utils.load_json(data_dir / (utils.lexicon_filename + '.json'))

        self.phone2units, self.mono_di_tri_phone2units = self._get_phone2units()
        self.target_predicter = NN(existing_model_path=target_predicter_model_path)

        self.data_dir = data_dir
        
        (self.utt2mono_di_tri_target_feats, 
         self.spkr2mono_di_tri_target_feats_mean, 
         self.spkr2mono_di_tri_target_feats_std,
         self.utt2phones_target_feats,
         self.spkr2phones_target_feats_mean,
         self.spkr2phones_target_feats_std) = utils.load_target_feats(data_dir, mono=mono)

        '''
        (self.utt2concat_feats, 
         self.concat_feats_mean, 
         self.concat_feats_std) = utils.load_concat_feats(data_dir)
        '''
        

    def synthesize(self, text, output_path=Path('./synth.wav')):
        # Get the phone sequence of the text
        # The current version of the synthesizer uses triphones with 1-phone 
        # overlap, with diphone and monophone backoff
        phone_sequence = self._get_phone_sequence(text)

        # Get the candidates and a final phone sequence (in case no units 
        # exist in the data for a phone unit-TODO)
        phone_sequence, candidates = self._get_candidates(phone_sequence)

        # Get the target features for the phones
        # Note that the neural network outputs values that ostensibly have 
        # been feature scaled
        phone_target_feats_fs = self._get_target_feats(phone_sequence)[0]

        # Begin viterbi.
        # Initialize the Viterbi matrices
        print("Initializing viterbi matrix.")
        cost_matrix = [[]]
        back_matrix = [[]]
        beg_target_feats=phone_target_feats_fs[0]
        print("Going through " + str(len(candidates[0])) + " phone 0"
              + " candidates.")
        for candidate_unit_idx in range(len(candidates[0])):
            # For this round, we're only looking for the target cost
            candidate_unit = candidates[0][candidate_unit_idx]
            if candidate_unit[0] == 'phone':
                utt2target_feats=self.utt2phones_target_feats
                spkr2target_feats_mean = self.spkr2phones_target_feats_mean
                spkr2target_feats_std = self.spkr2phones_target_feats_std
            elif candidate_unit[0] == 'mono_di_tri':
                utt2target_feats=self.utt2mono_di_tri_target_feats
                spkr2target_feats_mean = self.spkr2mono_di_tri_target_feats_mean
                spkr2target_feats_std = self.spkr2mono_di_tri_target_feats_std


            candidate_unit_feats = (
                np.array(
                    utt2target_feats[candidate_unit[1]][candidate_unit[2]]))
            # Scale the features according to the speaker's mean and std
            spkr = candidate_unit[1].split('-')[0]
            candidate_unit_target_feats_mean = np.array(
                    spkr2target_feats_mean[spkr])
            candidate_unit_target_feats_std = np.array(
                    spkr2target_feats_std[spkr])
            candidate_unit_feats_fs = ((candidate_unit_feats
                                        - candidate_unit_target_feats_mean)
                                       / candidate_unit_target_feats_std)
            # Sum of the absolute difference of all the features
            # TODO: Add weights for each feature?
            c_t = np.sum(np.fabs(np.subtract(candidate_unit_feats_fs, beg_target_feats)))
            cost_matrix[0].append(c_t)
            back_matrix[0].append(candidate_unit_idx)
        # Now run the DP algorithm for the remaining candidates
        for idx in range(1, len(candidates)):
            cost_matrix.append([])
            back_matrix.append([])
            candidate = candidates[idx]
            print("Going through " + str(len(candidate)) + " phone " + str(idx)
                  + " candidates.")
            unit_target_feats_fs = phone_target_feats_fs[idx]
            for candidate_unit in candidate:
                if candidate_unit[0] == 'phone':
                    utt2target_feats=self.utt2phones_target_feats
                    spkr2target_feats_mean = self.spkr2phones_target_feats_mean
                    spkr2target_feats_std = self.spkr2phones_target_feats_std
                elif candidate_unit[0] == 'mono_di_tri':
                    utt2target_feats=self.utt2mono_di_tri_target_feats
                    spkr2target_feats_mean = self.spkr2mono_di_tri_target_feats_mean
                    spkr2target_feats_std = self.spkr2mono_di_tri_target_feats_std

                candidate_unit_target_feats = np.array(
                    utt2target_feats[candidate_unit[1]][candidate_unit[2]])
                # Scale the features according to the speaker's mean and std
                spkr = candidate_unit[1].split('-')[0]
                candidate_unit_target_feats_mean = np.array(
                        spkr2target_feats_mean[spkr])
                candidate_unit_target_feats_std = np.array(
                        spkr2target_feats_std[spkr])

                candidate_unit_target_feats_fs = ((candidate_unit_target_feats
                                                   - candidate_unit_target_feats_mean)
                                                  / candidate_unit_target_feats_std)

                # TODO: Change to match RNN Paper
                c_t = np.sum(np.fabs(np.subtract(candidate_unit_target_feats_fs,
                                                 unit_target_feats_fs)))
                # Get the concatenation cost from previous units
                c_c = float('inf')
                prev_idx = 0
                # Reset our features to exclude duration and initial f_0
                candidate_unit_concat_feats = candidate_unit_target_feats[1::2]
                for prev_candidate_unit_idx in range(len(candidates[idx-1])):
                    # TODO: Experiment with different features for
                    # concatenation?
                    prev_candidate_unit = candidates[idx-1][prev_candidate_unit_idx]

                    #print(self.utt2phones[prev_candidate_unit[0]][prev_candidate_unit[1]])
                    if prev_candidate_unit[0] == 'phone':
                        prev_utt2target_feats=self.utt2phones_target_feats
                        prev_spkr2target_feats_mean = self.spkr2phones_target_feats_mean
                        prev_spkr2target_feats_std = self.spkr2phones_target_feats_std
                    elif prev_candidate_unit[0] == 'mono_di_tri':
                        prev_utt2target_feats=self.utt2mono_di_tri_target_feats
                        prev_spkr2target_feats_mean = self.spkr2mono_di_tri_target_feats_mean
                        prev_spkr2target_feats_std = self.spkr2mono_di_tri_target_feats_std
                    prev_candidate_unit_concat_feats = np.array(
                            prev_utt2target_feats[prev_candidate_unit[1]]
                                                 [prev_candidate_unit[2]])[2:]
                    # We don't want to scale these for the initial subtraction. 
                    # We're concerned here with
                    # absolute relation of the two units to each other.
                    '''
                    prev_spkr = prev_candidate_unit[0]
                    prev_candidate_unit_concat_feats_mean = np.array(
                            self.spkr2target_feats_mean[prev_spkr])[2:]
                    prev_candidate_unit_concat_feats_std = np.array(
                            self.spkr2target_feats_std[prev_spkr])[2:]
              
                    prev_candidate_unit_concat_feats_fs = ((prev_candidate_unit_concat_feats
                                                            - prev_candidate_unit_concat_feats_mean)
                                                           / prev_candidate_unit_concat_feats_std)
                    curr_c_c = np.sum(np.fabs(np.subtract(candidate_unit_concat_feats_fs,
                                                          prev_candidate_unit_concat_feats_fs)))
                    '''
                    # If the two units overlap, it's a zero cost: we like overlaps!
                    # Moreover if they're actually sequential, it is also zero cost
                    if ((prev_candidate_unit[0] == candidate_unit[0]) and 
                        (prev_candidate_unit[1] == candidate_unit[1]) and 
                        (prev_candidate_unit[2] == candidate_unit[2]-1)):
                        curr_c_c = 0
                    else:
                        curr_c_c_unscaled = (np.fabs(np.subtract(candidate_unit_concat_feats,
                                                                 prev_candidate_unit_concat_feats)))
                        curr_c_c = np.sum(np.fabs((curr_c_c_unscaled-candidate_unit_target_feats_mean[1::2])
                                           / candidate_unit_target_feats_std[1::2]))
                    if curr_c_c < c_c:
                        c_c = curr_c_c
                        prev_idx = prev_candidate_unit_idx

                C = c_t + c_c + cost_matrix[idx-1][prev_idx]
                cost_matrix[idx].append(C)
                back_matrix[idx].append(prev_idx)
        candidate_lens = [len(candidates) for candidates in cost_matrix]
        act_candidate_lens = [len(candidates) for candidates in candidates]
        # Now find the minimum of the last row
        final_C = float('inf')
        final_unit_idx = 0
        final_units = []
        for candidate_unit_idx in range(len(cost_matrix[-1])):
            cand_final_C = cost_matrix[-1][candidate_unit_idx]
            if cand_final_C < 0:
                print(cand_final_c)
            if cand_final_C < final_C:
                final_C = cand_final_C
                final_unit_idx = candidate_unit_idx
        print("FINAL COST: " + str(final_C))
        phone_idx = -1
        back_idx = final_unit_idx
        # Now go through the back table
        while(phone_idx >= -(len(back_matrix))):
            #print((phone_idx, back_idx))
            final_units.append(candidates[phone_idx][back_idx])
            back_idx = back_matrix[phone_idx][back_idx]
            phone_idx -= 1

        final_units.reverse() 
        final_units_phones = [self.utt2mono_di_tri_phones[utterance][idx] 
                              if pt=='mono_di_tri'
                              else self.utt2phones[utterance][idx] 
                              for pt, utterance, idx in final_units]
        print("FINAL PHONES: " + str(final_units_phones))
        #print(phone_sequence)
        concatenation = self._concatenate(final_units)
        wavfile.write(str(output_path), self.fs, concatenation)
        return concatenation

    def _get_phone2units(self):
        phone2units = defaultdict(list)
        mono_di_tri_phone2units = defaultdict(list)
        for utterance in self.utt2phones:
            # Get phone level candidates
            for unit_idx in range(len(self.utt2phones[utterance])):
                phone = self.utt2phones[utterance][unit_idx]
                phone2units[phone].append(('phone', utterance, unit_idx))
            # Get mono_di_tri_phone_level_candidates
            for unit_idx in range(len(self.utt2mono_di_tri_phones[utterance])):
                mono_di_tri_phone = (
                        tuple(self.utt2mono_di_tri_phones[utterance][unit_idx]))
                mono_di_tri_phone2units[mono_di_tri_phone].append(
                        ('mono_di_tri', utterance, unit_idx))
        return phone2units, mono_di_tri_phone2units

    def _get_phone_sequence(self, text):
        phones = self._get_phones(text)
        phone_sequence = self._get_mono_di_triphones_from_phones(phones)
        return phone_sequence

    def _get_phones(self, text):
        #TODO: Get a better model for t2p (another neural model?)
        # Start with the silence phone
        phones = []

        for word in text.split():
            phones += self.word2phones[word.upper()]
        # Optionally end with a final silence. In larger corpora this takes a 
        # a very long time to synthesize as it searches through all of the 
        # silences i.e. a goodly portion of the data
        #phones += ['SIL']
        return phones

    def _get_mono_di_triphones_from_phones(self, phones):
        idx = 0
        mono_di_tri_phones = []
        while idx < len(phones):
            # First try to get a triphone
            if idx + 2 < len(phones):
                triphone = tuple(phones[idx:idx+3])
                if (triphone in self.mono_di_tri_phones2idx
                    and len(self.mono_di_tri_phone2units[triphone]) > 1):
                    mono_di_tri_phones.append(triphone)
                    idx += 2
                    continue
            # We weren't able to get a usable triphone, fall back to diphone
            if idx + 1 < len(phones):
                diphone = tuple(phones[idx:idx+2])
                if (diphone in self.mono_di_tri_phones2idx
                    and len(self.mono_di_tri_phone2units[diphone]) > 1):
                    mono_di_tri_phones.append(diphone)
                    idx += 1
                    continue

            # If we couldn't get a usable diphone, get the monophone and move on
            # Note that this monophone underlaps the previous phone tuple
            monophone = tuple(phones[idx:idx+1])
            if (monophone in self.mono_di_tri_phones2idx
                    and (len(self.mono_di_tri_phone2units[monophone]) > 1 or
                         len(self.phone2units[monophone[0]]) > 1)):
                mono_di_tri_phones.append(monophone)
                idx += 1

            else:
                raise KeyError('Bad phone: ' + str(monophone))
        return mono_di_tri_phones



    def _get_candidates(self, phone_sequence):
        # TODO: implement backoff
        candidates = []
        for phones in phone_sequence:
            #print(str(phones) + " " + str(phones in self.mono_di_tri_phone2units))
            if len(self.mono_di_tri_phone2units[phones])>0:
                #print("MISS")
                candidates.append(self.mono_di_tri_phone2units[phones])
            elif len(phones)==1 and len(self.phone2units[phones[0]]) > 0:
                #print("HIT!")
                candidates.append(self.phone2units[phones[0]])
            else:
                raise KeyError('No candidate units for phones ' + str(phones))
        return phone_sequence, candidates

    def _get_target_feats(self, phone_sequence):
        idx_sequence = self._phone_seq2idxs(phone_sequence)
        return self.target_predicter.predict(idx_sequence)

    def _phone_seq2idxs(self, phone_seq):
        idx_shape = self.target_predicter.get_input_length()
        #print(idx_shape)
        idx_sequence = np.pad([self.mono_di_tri_phones2idx[phones] 
                               for phones in phone_seq], 
                              (0, idx_shape-len(phone_seq)),
                              mode='constant')
        return idx_sequence.reshape(1, idx_shape)

    def _concatenate(self, final_units):
        unit_wavs = []
        join_phones = False
        for phone_type, utterance, unit_idx in final_units:
            #print((utterance, unit_idx))
            utterance_dirs = preprocess.get_utterance_dirs(utterance)
            if phone_type == 'phone':
                utt2alignments=self.utt2alignments
                alignments = [utt2alignments[utterance][unit_idx]]
            elif phone_type == 'mono_di_tri':
                utt2alignments=self.utt2mono_di_tri_alignments
                alignments = utt2alignments[utterance][unit_idx]

            _, utterance_wav = wavfile.read(str(self.data_dir
                                            / utterance_dirs 
                                            / (utterance + '.wav')))

            phones = []
            for phone in alignments:
                phone_start = phone[0]
                phone_end = phone[1]
                phones.append(utterance_wav[phone_start:phone_end])

            if join_phones:
                unit_wavs[-1] = utils.join(unit_wavs[-1], phones[0])
                '''
                else:
                    unit_wavs[-1] = utils.cross_fade(unit_wavs[-1], phones[1])
                '''
                if len(phones) > 1:
                    unit_wavs += phones[1:]
            else:
                unit_wavs += phones

            if len(alignments) == 1:
                join_phones = False
            else:
                join_phones = True

            #print((unit_start, unit_end))

            #print(utterance_wav.shape)
            #print(unit_wav)
        #print(unit_wavs)
        concatenation = np.concatenate(unit_wavs)
        return concatenation
