uRobo: The Voice That Sounds Like You

IMPORTANT NOTES:
1. uRobo should be run from the uRobo root directory
2. The below prerequisites have already been satisfied on this instance and
   are here for anyone trying to get uRobo working on another
   PC/Cluster/Instance


PREREQUISITES:
   KALDI:
   uRobo needs the most recent version of Kaldi to handle the ASR training and 
   alignment stages. The output of ASR training is used for alignment which is
   then fed into preprocessing for the Text-to-Speech synthesizer. If you
   already have preprocessed data (i.e. the kind provided in the 'data'
   directory), Kaldi is not strictly required. Otherwise, Kaldi must be func-
   tioning properly.
   
   All of the currently existing pre-processed data in 'data/' was
   analyzed/aligned/etc. by Kaldi's Librispeech example script. 'urobo.py' allows
   a user to replicate this process.
     NOTE: a change has been made to the LibriSpeech 'run.sh' and 'cmd.sh'
     scripts in this instance. Therefore, if copying to another pc/computer, 
     it is best to clone Kaldi from: 
     https://github.com/thc2125/kaldi.git
     From there, follow the install instructions provided by the Kaldi
     documentation
   
   PYTHON3 REQUIREMENTS:
   These can be found in requirements.txt. They can be installed with
   pip3 install -r requirements.txt

USAGE:
