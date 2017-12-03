#from tacotron_synthesizer import Tacotron
from t2s.dummy_synthesizer import Dummy

def synthesize(text, synthesizer=Dummy()):
    '''Speech generation function

    Keyword Arguments:
    text -- a string of words from which to generate audio
    '''
    
    return synthesizer.synthesize(text)

