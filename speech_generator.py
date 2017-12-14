#from tacotron_synthesizer import Tacotron
from t2s.concatenative_model.concat_nn import NNConcatenator

def synthesize(text, data_dir, model_path, Synthesizer=NNConcatenator):
    '''Speech generation function

    Keyword Arguments:
    text -- a string of words from which to generate audio
    '''
    synthesizer = Synthesizer(data_dir, model_path)
    return synthesizer.synthesize(text)

