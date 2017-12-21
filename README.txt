uRobo: The Voice That Sounds Like You

IMPORTANT NOTES:
1. uRobo scripts should be run from the uRobo root directory
2. The python3 package requirements may themselves have binary or os-level
   package requirements. Running 'pip3 install -r requirements.txt --user' should
   indicate which if any binary packages are missing.

PREREQUISITES:
   PYTHON3:
   uRobo makes extensive use of the Python standard library and requires
   Python >= 3.5.2 to function

   Moreover pip3 is required to install the additional Python dependencies
   (see more info below)

   FFMPEG:
   FFMPEG is required for playback/audio conversion and copying. On debian
   based systems, 

   sudo apt-get install ffmpeg

   should suffice to install it.

   KALDI:
   uRobo needs the most recent version of Kaldi to handle the ASR training and 
   alignment stages. The output of ASR training is used for alignment which is
   then fed into preprocessing for the Text-to-Speech synthesizer. If you
   already have preprocessed data (i.e. the kind provided in the 'data'
   directory), Kaldi is not strictly required. Otherwise, Kaldi must be func-
   tioning properly.

   This can be done from the 'kaldi' folder provided (see the 'README.txt' in
   the parent folder).
   
   All of the currently existing pre-processed data in 'data/' was
   analyzed/aligned/etc. by Kaldi's Librispeech example script. 'urobo.py' allows
   a user to replicate this process.
     NOTE: a change has been made to the LibriSpeech 'run.sh' and 'cmd.sh'
     scripts in this instance. Therefore, if copying to another pc/computer, 
     it is best to clone Kaldi from: 
     https://github.com/thc2125/kaldi.git
     From there, follow the install instructions provided by the Kaldi
     documentation
   
   PYTHON3 PACKAGE REQUIREMENTS:
       These can be found in requirements.txt. They can be installed with the
       command:

           pip3 install -r requirements.txt --user

       There may be an issue with pysptk requiring numpy before installation
       is allowed. If so, install numpy first:
           pip3 install numpy --user
       Then try the command again.

QUICKSTART:
    All the below commands are expected to be run from the 'uRobo' root directory.

    EXPERIMENTS:
    To re-create the experiments used in the paper, the following command:

        python3 experiments/experiment_generator.py

    Or failing that:

        python3 -m experiments.experiment_generator

    This will synthesize audio for each line in the file
    'experiments/experiment_text.txt'

    The path of each audio file looks like:
    'experiments/audio/<line #>-<synthesizer>.wav'

    Where each 'synthesizer' is one of:
        train-clean-10-f (the 10 hour mixed female speaker synthesizer)
        train-clean-spk39 (the single speaker synthesizer)
        train-clean-spk39_mono (the single speaker synthesizer that only
                                uses monophones)

    Note that audio from the 10 hour mixed female speaker synthesizer takes
    a while to produce, as the Viterbi algorithm must go through significantly
    more candidate units than the other models.

    INTERACTIVE COMMAND LINE:
        To interactively play queried audio from the single speaker model,
        call:
            python3 t2s_train-clean-spk39.py

        The program will then ask you to type something for it to say.

        On some Linux systems, the final phone or two is cut off. The matching
        audio file, however, can still be found in the uRobo root directory as
        'synth.wav'. 

        To quit the program, type 'quit()' or '<CTRL>+C'

        NOTE: If there is no unit representation for a phone in the units data
              or a word does not have a phonetic representation in the
              lexicon, an error message will appear. A user can then input 
              another text string. The program will continue until a user
              quits.

              Punctuation is not currently stripped and is not present in
              the source data. Any punctuation (aside from apostrophes) will 
              cause a lexicon error. 

    The pre-trained data can be used to export individual audio files with the 
    following command:

            python3 urobo.py -A data/train-clean-spk39 \
                             -n models/train-clean-10-f-lin-20e/model.h5 \
                             -t 'hello my name is alice' \
                             -o 'hello.wav'

    That will produce audio file 'hello.wav' wherein speaker 39 says 'Hello my
    name is Alice.' More on this usage later in the README.


FOLDER STRUCTURE:
    Preprocessed data for uRobo is meant to live in the 'data' folder. 

    Trained target feature prediction models can be found in the 'models'
    directory. 

    Kaldi data and alignment models live in the 'kaldi' directory (NOT the 
    'uRobo' directory). All uRobo specific scripts must be run from 'uRobo'!


USAGE OF 'urobo.py':

    The command line interface for the entire system is the script 'urobo.py'

    This script ostensibly controls the entire end-to-end process from Kaldi's 
    initial librispeech download/training and alignment, to preprocessing, to
    target feature training, to final concatenative synthesis.

    Although ostensibly all 4 stages could be run with a single command, this
    is not recommended. Kaldi is a third party program and is prone to errors
    during the long training process for LibriSpeech data. Of other note: the 
    data in the 'librispeech' folder was trained on a different machine, and
    therefore may not work correctly externally. The 'scp' files which are
    used in alignment and other kaldi scripts have specific paths given the
    machine they were created on.

    That said, to begin Kaldi's ASR training/LibriSpeech download:
        
        python3 urobo.py -k <KALDI_ROOT_DIR> -r <RAW_DATA_DIR>

        e.g. 
            python3 urobo.py -k ../kaldi -r ../librispeech

    To align the data once an alignment model has been trained:

        python3 urobo.py -k <KALDI_ROOT_DIR> -D <DATA_TO_ALIGN> -a <ASR_MODEL>

        e.g.
            python3 urobo.py -k ../kaldi -D train-clean-100 -a tri6b
          
    NOTE: The 'DATA_TO_ALIGN' and 'ASR_MODEL' directories are relative to
          'kaldi/egs/librispeech/s5/'. They should not be qualified with their 
          absolute path.

    Once data has been aligned, preprocess it for uRobo with:
   
        python3 urobo.py -K <KALDI_DATA_DIR> -L <KALDI_LANGUAGE_MODEL_DIR> \
                         -P <PROCESSED_DIR> \
                         [-d <DURATION_LIMIT> -g <GENDER> -s <SPEAKERS> \
                          -M <MONO_DI_TRI_PHONES>]

        So to preprocess speaker 40 from the train-clean-100 data set using the
        n-phone list from 10 hours of female speakers:
      
            python3 urobo.py \
                    -K ../kaldi/egs/librispeech/s5/data/train-clean-100 \
                    -L ../kaldi/egs/librispeech/s5/data/lang \
                    -P data/train-clean-spk40 \
                    -s 40 \
                    -M data/train-clean-10-f/mono_di_tri_phones.json

        Or to preprocess 5 hours of utterances spoken by males from the 
        train-clean-100 data set:
             python3 urobo.py \
                    -K ../kaldi/egs/librispeech/s5/data/train-clean-100 \
                    -L ../kaldi/egs/librispeech/s5/data/lang \
                    -P data/train-clean-5-m \
                    -d 5 \
                    -g m
    
       This will also produce a new n-phone list based on this new speaker
       data.

    With pre-processed data, you can train a target feature prediction model.
  
        python3 urobo.py -T <TRAIN_CORPUS> \
                         -E <TEST_CORPUS> \
                         -f <FINAL_MODEL> \
                         [-e <EPOCHS>]

        e.g. 
            python3 urobo.py -T data/train-clean-5-m \
                             -E data/train-clean-spk40 \
                             -f models/train-clean-5-m-50e/model.h5 \
                             -e 50

    With a target feature prediction model and pre-processed data, you can now
    transform text to speech.
        python3 urobo.py -A <AUDIO_DATA_DIR> \
                         -n <TARGET_FEATURE_MODEL> \
                         -t <TEXT> \
                         -o <OUTPUT_FILE> \
                         [-m]
        e.g.
            python3 urobo.py -A data/train-clean-spk40 \
                             -n models/train-clean-5-m-50e/model.h5 \
                             -t 'hello my name is harry' \
                             -o 'hello.wav'

    The '-m' flag signifies to only use monophones for concatenation. This is
    not recommended. 


