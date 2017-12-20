#!/usr/bin/python3

import re

from pathlib import Path

from pydub import AudioSegment
from pydub.playback import play

from concatenative_model.concat_nn import NNConcatenator


alice = NNConcatenator(Path('data/train-clean-spk39/'), 
                       Path('models/train-clean-10-f-lin-20e/model.h5'))
                       mono=True)

bad_key = re.compile('Bad Phone: ')
no_candidates = re.compile('No candidate units for phones ')
while True:
    synth_text = input('What would you like me to say? ')
    if synth_text == 'quit()':
        raise SystemExit
    try:
        _, wav = alice.synthesize(synth_text)
        play(AudioSegment.from_wav(str(wav)))
    except KeyError as k:
        if bad_key.match(str(k)) or no_candidates.match(str(k)):
            phone = str(k).split()[-1]
            print("Sorry, my voice can't produce n-phone " + phone)

        else:
            word = str(k).split()[-1]
            print("Sorry, I don't know how to pronounce " + word)
