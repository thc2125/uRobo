from pydub import AudioSegment

from subprocess import run

class Dummy():

    voice=None
    tmpfilename = 'tmp_audio.wav'

    def synthesize(self, text):
        with open(self.tmpfilename) as tmpfile:
            command = ['espeak', text]
            if voice:
                command += ['-v', voice]
            run(command, stdout=tmpfile)

        return AudioSegment.from_wav(tmpfile)

