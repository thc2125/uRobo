import re
import subprocess

from pathlib import Path


cmd='run.pl'
libri_speech_sub_dir = Path('egs' / 'librispeech' / 's5')


def train_asr(kaldi_path, raw_data_dir=Path('data/kaldi/')):
    '''Train an automatic speech recognition system over the librispeech
    corpus.
    Keyword Arguments:
    raw_data_dir -- where to store data downloaded by Kaldi, pathlike object
    kaldi_path -- the main Kaldi directory, pathlike object
    '''

    # This process will download the librispeech data, then train several ASR
    # models which can be used to align the data
    abs_raw_data_dir = raw_data_dir.resolve()
    abs_kaldi_path = kaldi_path.resolve()

    subprocess.run(['./run.sh', str(abs_raw_data_dir)],
                   cwd=str(abs_kaldi_path / libri_speech_sub_dir))


def align(kaldi_path, 
          data_to_align='train_clean_100',
          asr_model='nnet_6a_clean_460_gpu'):
    '''
    Keyword Arguments:
    kaldi_path -- the main Kaldi directory; pathlike object
    data_to_align -- where mfccs/duration/etc. data created by Kaldi is stored
                     this would have been one of the folders created during
                     train_asr
    asr_model -- the acoustic model to perform the alignment
    '''
    # Run the aligment script
    #absolute_raw_data_dir = Path(raw_data_dir.resolve())
    abs_kaldi_path = kaldi_path.resolve()
    aligned_data_dir = (asr_model + '_ali_' + data_to_align)
    subprocess.run([str(Path('steps' / 'align_si.sh')),
                    '--cmd',
                    cmd,
                    str(Path('data' / data_to_align)), # Where mfccs,etc. live
                    str('data' / 'lang'), # Where language model info lives
                    str('exp' / asr_model), # Where to find the alignment model
                    str('exp' / aligned_data_dir)], 
                    # Where to store final alignments
                    cwd=str(abs_kaldi_path / libri_speech_sub_dir))

    # We now have alignments in the 'aligned_data_dir' directory
    # We need to put them in a format usable by the uRobo preprocessing script
    # Code inspired by Kaldi tutorial by Eleanor Chodroff, Northwestern University
    alignment_data_patt = re.compile('ali\..*\.gz')
    for i in Path(abs_kaldi_path / libri_speech_sub_dir / 'exp' / aligned_data_dir).iterdir():
        if alignment_data_patt.match(file_path.name):
            subprocess.run([str(abs_kaldi_path / 'src' / 'bin' / 'ali-to-phones'),
                            '--ctm-output',
                            'exp'/asr_model/'final.mdl',
                            'ark:"gunzip -c ' + str(i) + '|"-> ' + str(i.parent / i.stem) '.ctm'],
                           cwd=abs_kaldi_path / libri_speech_sub_dir)
    # Compile all the ctms into a single text file
    ctm_patt = re.compile('.*\.ctm')
    # We want the alignments file to live with the rest of our data from Kaldi
    # so we only have to preprocess from one place
    alignments_file = abs_kaldi_path / libri_speech_sub_dir / 'data' / data_to_align / 'alignments'
    with alignments_file.open('w'):
        for i in Path(abs_kaldi_path / libri_speech_sub_dir / 'exp' / aligned_data_dir).iterdir():
            with i.open():
                lines = i.readlines()
                alignments_file.writelines(lines)
 
    return 
