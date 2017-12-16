import subprocess

def align(data_dir='/data/aligned', orig_data_dir, kaldi_path):
    ''' Much code borrowed from Kaldi Librispeech example
    Keyword Arguments:
    data_dir -- where to store pre-processed data
    orig_data_dir -- where the original data to be aligned lives
    kaldi_path -- the main Kaldi directory
    '''

    data_url='www.openslr.org/resources/12'
    lm_url='www.openslr.org/resources/11'

    train_cmd="queue.pl"
    decode_cmd="queue.pl"
    mkgraph_cmd="queue.pl"

train_cmd="queue.pl --mem 2G"

    
