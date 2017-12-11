from collections import defaultdict
from pathlib import Path

import numpy as np
import json
import pysptk

phones_filename = 'phones'
utterances_filename = 'utterances'
vocabulary_filename = 'vocab'
alignments_filename = 'alignments'
lexicon_filename = 'align_lexicon'
spk2gender_filename = 'spk2gender'
transcriptions_filename = 'text'
utterance_duration_filename = 'utt2dur'
utt2phones_filename = 'utt2phone'
utt2alignments_filename = 'utt2alignments'
mono_di_tri_phones_filename = 'mono_di_tri_phones'
utt2mono_di_tri_phones_filename = 'utt2mono_di_tri_phones'
utt2mono_di_tri_alignments_filename = 'utt2mono_di_tri_alignments'
utt2target_feats_filename = 'utt2target_feats'
target_feats_filename = 'target_feats'
utt2concat_feats_filename = 'utt2concat_feats'
concat_feats_filename = 'concat_feats'
target_feats_mean_filename = 'target_feats_mean'
concat_feats_mean_filename  = 'concat_feats_mean'
target_feats_std_filename = 'target_feats_std'
concat_feats_std_filename  = 'concat_feats_std'

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
            utt2mono_di_tri_phones,
            utt2mono_di_tri_alignments)

def save_json(data, json_filepath):
    with json_filepath.open('w') as jsonfile:
         json.dump(data, jsonfile, indent=4)

def load_json(json_filepath):
    with json_filepath.open() as jsonfile:
        return json.load(jsonfile)

def load_target_feats(data_dir):
    print(type(data_dir))
    utt2target_feats = load_json((data_dir / (utt2target_feats_filename + '.json')))
    target_feats_mean = np.load(str(data_dir / (target_feats_mean_filename + '.npy')))
    target_feats_std = np.load(str(data_dir / (target_feats_std_filename + '.npy')))
    return utt2target_feats, target_feats_mean, target_feats_std

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

def get_unitdicts(alignments_path):
    # TODO: Parametrize fs
    fs=16000

    utterance2idx = {}
    idx2utterance = []
    utterance_phones = []
    utterance_feats = []
    phone2units = defaultdict(list)
    alignments = []

    with alignments_path.open() as alignments_file:
        for line in alignments_file:
            split_line = line.split()
            utterance = split_line[0]
            phone_idx = int(split_line[4])

            spk = '-'.join(utterance.split('-')[0:-1])
            utt_alignments = (int(float(split_line[2])*fs), int(float(split_line[2])*fs + float(split_line[3])*fs))

            if utterance not in utterance2idx:
                utterance2idx[utterance] = len(utterance_phones)
                idx2utterance.append(utterance)
                utterance_phones.append([])
                alignments.append([])

            utterance_idx = utterance2idx[utterance]

            unit_idx = len(utterance_phones[utterance_idx])
            utterance_phones[utterance_idx].append(phone_idx)

            phone2units[phone_idx].append((utterance_idx, unit_idx))

            alignments[utterance_idx].append(utt_alignments)

        return (utterance2idx,
                idx2utterance,
                utterance_phones,
                utterance_feats,
                phone2unit,
                alignments)


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


