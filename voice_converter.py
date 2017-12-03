import json

from pathlib import Path
from scipy.spatial.distance import euclidean

import pysptk
from pydub import AudioSegment

from t2s.dummy_synthesizer import Dummy

class VoiceConverter():
    transcript_filename = 'trans.txt'
    target_utterance_dirname = 'target_utterances'
    source_utterance_dirname = 'source_utterances'

    # Frame size and overlap measured in ms
    frame_size = 25
    frame_overlap = 5

    def __init__(self, 
                 data_dirpath, 
                 model=None,
                 t2s=Dummy):

        self.data_dirpath=data_dirpath
        self.t2s = t2s
        if model:
            self.model = model
        else:
            self.model = self._generate_model()

    def _generate_model(self):
        self._generate_source_utterances()
        self._generate_exemplars()

    def _generate_source_utterances(self):
        self.utterance2txt = self._generate_transcript()
        for utterance_id, utterance in utterance2txt.items():
            source_utterance = self._generate_source_utterance(utterance)
            source_utterance.export(str(self.data_dirpath 
                                        / source_utterance_dirname 
                                        / (utterance_id + '.wav')), 
                                    format = 'wav')

    def _generate_transcript(self):
        # Assumes the transcript only applies to the target
        utterance2txt = {}
        with (self.data_dirpath / self.transcript_filename).open() as transcript_file:
            for line in transcript:
                split_line = line.split()
                utterance_id = split_line()[0]
                utterance2txt[utterance_id] = ' '.join(split_line[1:])
        return utterance2txt

    def _generate_source_utterance(self, utterance):
        '''Synthesizes the utterance using the voice converter's speech model,
        returning a pydub wav object.

        Keyword Arguments:
        utterance -- a text string to turn into audio
        '''
        source_utt = self.t2s.synthesize(utterance)
        return source_utt

    def _generate_exemplars(self):

        for utterance_id in utterance2txt:
            # Source and target utterances have the same filenames but live in
            # different directories.
            source_utterance = AudioSegment.from_wav(str(self.data_dirpath 
                                        / source_utterance_dirname 
                                        / (utterance_id + '.wav')))
            target_utterance = AudioSegment.from_wav(str(self.data_dirpath 
                                        / target_utterance_dirname 
                                        / (utterance_id + '.wav')))
            source_frames = self._generate_frames(source_utterance)
            target_frames = self._generate_frames(target_utterance)

            # Now we need mfccs for the frames for dtw and CUTE
            source_mfccs, source_f0s = _get_mfccs(source_frames)
            target_mfccs, target_f0s = _get_mfccs(target_frames)
            # We want to apply DTW to these resulting frames
            # in order to start building our exemplars
            frame_alignments = fastdtw.fastdtw(source_mfccs, target_mfccs, dist=euclidean)
            

    def _generate_frames(utterance):
        frame_idxs = []
        curr_idx = 0
        '''
        while (curr_idx + frame_size) < len(utterance):
            frame_idxs.append(curr_idx)
            curr_idx += self.frame_overlap
        '''
        frames = []
        
        for frame_idx in range(0, len(utterance)-frame_size, 5):
            frames.append(utterance[frame_idx, frame_idx + 25])

        return frames

    def _get_mfccs_f0(frames)
        frame_mfccs = []
        frame_f0s = []
        for frame in frames:
            frame_mfccs.append(pysptk.mfcc(np.array(frame.get_array_of_samples())))
            frame_f0s.append(pysptk.rapt(np.array(frame.get_array_of_samples()))
        return np.stack(frame_mfccs), np.stack(frame_f0s)
            
    def synthesize(query):
        pass
