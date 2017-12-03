# Reimplementation of concatenative model from
# "Recent Advances in Google Real-time HMM-driven Unit Selection Synthesizer"

import wave # Used to concatenate the final wav files.


# First, gather the phonemes and prosodic features of the target audio
    # stored in u, the target unit sequence and f, the unit's prosodic
    # features. Both lists should be of length K
    u, f = text_to_target(text)


    # Next, 

# Pre-processing that must be done:
# 2. Get alignments (audio-to-phoneme) (Kaldi, librispeech 100 hr)
# 3. Use python wave module to split up the files! (or pydub)
# 3. Compute prosodic features and create a final tsv or json with your data:
#    file
text_to_target(text):
    '''Target phoneme and feature generation function

    Keyword Arguments:
    text -- a string of words from which to generate phonemes and features
    '''
    # First, convert text into phonemes
    phonemes = []
    for word in text.split():
        # Set word to out of vocabulary if it's not there
        if word not in lexicon:
            word = '<oov>'
            # TODO: How do you handle OOV words
            continue
        phonemes += lexicon[word]
    # Let u be the sequence of units
    u = []

    features = phonemes_to_features(phonemes)
    return phonemes, features

# TODO: Get a lexicon (word-to-phoneme) (Kaldi, librispeech lm)
text_to_phonemes(text):
    '''Uses a lexicon to get phonemes from a given text input
    
    Keyword Arguments:
    text -- a string of words from which to generate phonemes and features
    '''
    pass
