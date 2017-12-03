from tacotron.synthesizer import Synthesizer

class Tacotron():
    model = '/tmp/tacotron-20170720/model.ckpt'

    def __init__(self):
        self.tt = Synthesizer()
        self.tt.load(model)

    def synthesize(self,text):
        return self.tt.synthesize(text)
