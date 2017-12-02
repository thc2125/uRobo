import json

from pathlib import Path

class VoiceConverter():
    transcript_filename = 'text'
    target_utterance_dirname = 'target_utterances'
    source_utterance_dirname = 'source_utterances'

    def __init__(self, 
                 target_dirpath, 
                 model=None,
                 t2s=Tacotron('default_model')):

        self.target_dirpath=target_dirpath
        self.t2s = t2s
        if model:
            self.model=model
        else:
            self._generate_model()

    def _generate_model(self):
        self._generate_source_utterances()
        self._generate_exemplars()

    def _generate_source_utterances(self):
        utterance2txt = self._generate_transcript()
        for utterance_id, utterance in utterance2txt.items():
            source_utterance = self._generate_source_utterance(utterance)
            source_utterance.export(str(self.target_dirpath 
                                        / source_utterance_dirname 
                                        / (utterance_id + '.wav')), 
                                    format = 'wav')

    def _generate_exemplars(self):
        
        for source_utterance, target_utterance in utterance2txt:

    def _generate_source_utterance(self, utterance):
        '''Synthesizes the utterance using the voice converter's speech model,
        returning a pydub wav object.

        Keyword Arguments:
        utterance -- a text string to turn into audio
        '''
        source_utt = self.t2s.synthesize(utterance)
        return source_utt

    def _generate_transcript(self):
        # Assumes the transcript only applies to the target
        utterance2txt = {}
        with (self.target_dirpath / self.transcript_filename).open() as transcript_file:
            for line in transcript:
                split_line = line.split()
                utterance_id = split_line()[0]
                utterance2txt[utterance_id] = ''.join(split_line[1:])
        return utterance2txt

    def synthesize(query):
        pass
