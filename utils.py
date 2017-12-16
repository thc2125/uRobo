import json
import math
import numpy as np

from collections import defaultdict
from pathlib import Path

import pysptk

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

phones_filename = 'phones.txt'
utterances_filename = 'utterances'
vocabulary_filename = 'vocab'
alignments_filename = 'alignments'
lexicon_filename = str(Path('phones'/'align_lexicon.txt'))
spk2gender_filename = 'spk2gender'
transcriptions_filename = 'text'
utterance_duration_filename = 'utt2dur'

utt2phones_filename = 'utt2phone'
utt2diphones_filename = 'utt2diphones'
utt2triphones_filename = 'utt2triphones'
utt2mono_di_tri_phones_filename = 'utt2mono_di_tri_phones'

utt2alignments_filename = 'utt2alignments'
utt2diphone_alignments_filename = 'utt2diphone_alignments'
utt2triphone_alignments_filename = 'utt2triphone_alignments'
utt2mono_di_tri_alignments_filename = 'utt2mono_di_tri_alignments'

utt_prefix = 'utt_'

words_filename = 'words'

utt2_id = 'utt2'

mono_di_tri_phones_filename = 'mono_di_tri_phones'

target_feats_suffix = 'target_feats'

mean_suffix='_mean'
std_suffix='_std'

spkr2mean_prefix = 'spkr2mean_'
spkr2std_prefix = 'spkr2std_'
spkr_independent_prefix = 'spkr_ind_'
normalized_suffix = '_normalized'
utt2concat_feats_filename = 'utt2concat_feats'


def load_data(processed_dirpath):
    with (processed_dirpath / (vocabulary_filename + '.json')).open() as vocab_file:
        idx2vocabulary = json.load(vocab_file)
    vocabulary2idx = {idx2vocabulary[idx]: idx 
                      for idx in range(len(idx2vocabulary))}

    with (processed_dirpath / (phones_filename + '.json')).open() as phones_file:
        idx2phones = json.load(phones_file)
    phones2idx = {idx2phones[idx]: idx 
                      for idx in range(len(idx2phones))}
    
    with (processed_dirpath / (mono_di_tri_phones_filename + '.json')).open() as mono_di_tri_phones_file:
        idx2mono_di_tri_phones = json.load(mono_di_tri_phones_file)
    mono_di_tri_phones2idx = {tuple(idx2mono_di_tri_phones[idx]): idx 
                              for idx in range(len(idx2mono_di_tri_phones))}


    with (processed_dirpath / (transcriptions_filename + '.json')).open() as transcriptions_file:
        utt2words = json.load(transcriptions_file)
    with (processed_dirpath / (utt2phones_filename + '.json')).open() as utt2phones_file:
        utt2phones=json.load(utt2phones_file)
    with (processed_dirpath / (utt2alignments_filename + '.json')).open() as utt2alignments_file:
        utt2alignments=json.load(utt2alignments_file)

    utt2diphones = load_json(processed_dirpath / (utt2diphones_filename + '.json'))
    utt2diphone_alignments = load_json(processed_dirpath / (utt2diphone_alignments_filename + '.json'))

    utt2triphones = load_json(processed_dirpath / (utt2triphones_filename + '.json'))
    utt2triphone_alignments = load_json(processed_dirpath / (utt2triphone_alignments_filename + '.json'))


    with (processed_dirpath / (utt2mono_di_tri_phones_filename + '.json')).open() as utt2mono_di_tri_phones_file:
        utt2mono_di_tri_phones=json.load(utt2mono_di_tri_phones_file)
    with (processed_dirpath / (utt2mono_di_tri_alignments_filename + '.json')).open() as utt2mono_di_tri_alignments_file:
        utt2mono_di_tri_alignments=json.load(utt2mono_di_tri_alignments_file)

    return (idx2vocabulary, 
            vocabulary2idx, 
            idx2phones, 
            phones2idx, 
            idx2mono_di_tri_phones, 
            mono_di_tri_phones2idx, 
            utt2words,
            utt2phones, 
            utt2alignments, 
            utt2diphones,
            utt2diphone_alignments,
            utt2triphones,
            utt2triphone_alignments,
            utt2mono_di_tri_phones,
            utt2mono_di_tri_alignments)

def save_json(data, json_filepath):
    with json_filepath.open('w') as jsonfile:
         json.dump(data, jsonfile, indent=4)

def load_json(json_filepath):
    with json_filepath.open() as jsonfile:
        return json.load(jsonfile)

def load_target_feats(data_dir, mono=False):
    phone_type = mono_di_tri_phones_filename if not mono else phones_filename 
    utt2target_feats = load_json(
            (data_dir 
             / (utt2_id
                + phone_type
                + target_feats_suffix
                + '.json')))

    spkr2target_feats_mean = load_json(
            (data_dir 
             / (spkr2mean_prefix
                + phone_type
                + target_feats_suffix
                + '.json')))
    spkr2target_feats_std = load_json(
            (data_dir 
             / (spkr2std_prefix
                + phone_type
                + target_feats_suffix
                + '.json')))

    utt2phonestargets_feats = load_json(
            (data_dir 
             / (utt2_id
                + phones_filename
                + target_feats_suffix
                + '.json')))

    spkr2phonestarget_feats_mean = load_json(
            (data_dir 
             / (spkr2mean_prefix
                + phones_filename
                + target_feats_suffix
                + '.json')))

    spkr2phonestarget_feats_std = load_json(
            (data_dir 
             / (spkr2std_prefix
                + phones_filename
                + target_feats_suffix
                + '.json')))



    return (utt2target_feats,
            spkr2target_feats_mean,
            spkr2target_feats_std,
            utt2phonestargets_feats,
            spkr2phonestarget_feats_mean,
            spkr2phonestarget_feats_std)

def load_concat_feats(data_dir):
    '''
    utt2concat_feats = load_json((data_dir / (utt2concat_feats_filename + '.json')))
    concat_feats_mean = np.load(str(data_dir / (concat_feats_mean_filename + '.npy')))
    concat_feats_std = np.load(str(data_dir / (concat_feats_std_filename + '.npy')))
    '''
    utt2concat_feats_raw = load_json((data_dir / (utt2target_feats_filename + '.json')))
    utt2concat_feats = [[feats[1:len(feats)] 
                         for feats in utt_feats] 
                        for utt_feats in utt2concat_feats_raw]
    concat_feats_mean = np.load(str(data_dir / (target_feats_mean_filename + '.npy')))[1:]
    concat_feats_std = np.load(str(data_dir / (target_feats_std_filename + '.npy')))[1:]

    return utt2concat_feats, concat_feats_mean, concat_feats_std

def get_word2phones(lexicon_path):
    word2phones = {}
    dupes = 0
    with lexicon_path.open() as lexicon_file:
        for line in lexicon_file:
            split_line = line.split()
            word = split_line[0]
            phones=split_line[2:-1]
            if word in word2phones:
                #print("DUPLICATE!! " + word)
                dupes+=1
            word2phones[word]=phones
    print(dupes)
    return word2phones

def get_phone2idx(phones_path):
    phone2idx = {}
    idx2phone = {}
    with phones_path.open() as phones_file:
        for line in phones_path:
            split_line = line.split()
            phone = split_line[0]
            idx = split_line[1]
            phone2idx[phone]=idx
            idx2phone[idx]=phone
    return phone2idx, idx2phone

def join(samples1, samples2, fade_function=lambda i,t: (1-math.log(i,t),math.log(i,t)) if i>0 else (1, 0)):
    new_samples = []
    join_len = len(min(samples1, samples2, key=len))
    for sample_idx in range(join_len):
        fade_factor = fade_function(sample_idx, join_len)
        '''
        print(samples1[sample_idx])
        print(samples2[sample_idx])
        print(fade_factor)
        print(math.floor(fade_factor[0] * samples1[sample_idx]))
        print(math.ceil(fade_factor[1] * samples2[sample_idx]))
        print()
        '''
        new_samples.append(math.floor(fade_factor[0] * samples1[sample_idx])
                           + math.ceil(fade_factor[1] * samples2[sample_idx]))
    #new_samples = np.concatenate([samples1,samples2])
    return np.array(new_samples, dtype=np.int16)
