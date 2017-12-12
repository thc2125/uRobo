import argparse
import json
import math
import os
import random
import shutil

from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pysptk

from pydub import AudioSegment
from scipy.io import wavfile

import utils

# Define constants
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

male = 'm'
female = 'f'

fs=16000

def preprocess(orig_dirpath, processed_dirpath, duration_limit=float('inf'), gender=None, nphones=None):

    if not processed_dirpath.exists():
        processed_dirpath.mkdir(parents=True)

    utterances = copy_base_data(orig_dirpath, 
                                processed_dirpath, 
                                duration_limit=duration_limit,
                                gender=gender,
                                nphones=nphones)
    data = utils.load_data(processed_dirpath)

    process_data(processed_dirpath, sorted(list(utterances)), *data)

def copy_base_data(orig_dirpath, 
                   processed_dirpath, 
                   duration_limit=None,
                   gender=None,
                   nphones=None):
    spk2gender = get_spk2gender((orig_dirpath / (spk2gender_filename + '.txt')))
    # JSONize spk2gender
    with (processed_dirpath / (spk2gender_filename + '.json')).open('w') as spk2gender_file:
        json.dump(spk2gender, spk2gender_file, indent=4)


    idx2phones, phones2idx = get_phones((orig_dirpath / (phones_filename + '.txt')))
    # JSONize phone list
    with (processed_dirpath / (phones_filename + '.json')).open('w') as phones_file:
        json.dump(idx2phones, phones_file, indent=4)

    idx2vocab, vocab2idx = get_vocab((orig_dirpath / (vocabulary_filename + '.txt')))
    # JSONIze vocabulary
    with (processed_dirpath / (vocabulary_filename + '.json')).open('w') as vocabulary_file:
        json.dump(idx2vocab, vocabulary_file, indent=4)

    word2phones = get_lexicon((orig_dirpath / (lexicon_filename + '.txt')))
    # JSONIze lexicon
    with (processed_dirpath / (lexicon_filename + '.json')).open('w') as lexicon_file:
        json.dump(word2phones, lexicon_file, indent=4)

    utt2words_all = get_transcriptions((orig_dirpath / (transcriptions_filename +
        '.txt')))

    utt2phones_all, utt2alignments_all = get_alignments(
            orig_dirpath / (alignments_filename + '.txt'), 
            idx2phones)

    utterances, spks, utt2dur, curr_duration = get_utterances(
            (orig_dirpath / (utterance_duration_filename + '.txt')),
            gender,
            duration_limit,
            spk2gender, 
            word2phones,
            utt2words_all,
            utt2alignments_all)
    # JSONIze utterances
    utils.save_json(sorted(list(utterances)), processed_dirpath / (utterances_filename + '.json'))
    utils.save_json(utt2dur, processed_dirpath / (utterance_duration_filename + '.json'))

    # Create new transcripts and alignments
    utt2words = {utterance: words
                for utterance, words in utt2words_all.items() 
                if utterance in utterances}
    utils.save_json(utt2words, processed_dirpath / (transcriptions_filename + '.json'))

    utt2phones = {utterance: phones 
                  for utterance, phones in utt2phones_all.items()
                  if utterance in utterances}
    #jsonize 
    utils.save_json(utt2phones, processed_dirpath / (utt2phones_filename + '.json'))


    utt2alignments = {utterance: alignments
                      for utterance, alignments in utt2alignments_all.items() 
                      if utterance in utterances}
    with (processed_dirpath / (utt2alignments_filename + '.json')).open('w') as utt2alignments_file:
        json.dump(utt2alignments, utt2alignments_file, indent=4)

    if nphones:
        idx2mono_di_tri_phones=[tuple(phones) for phones in nphones]
    else:

        # Create metrics of diphones
        diphone_counts = get_nphones(utt2phones, 2)
        idx2diphones = [diphone 
                        for diphone, count in sorted(diphone_counts.items()) 
                        if ((count / len(diphone_counts)) > .01)]

        # Create metrics of triphones
        triphone_counts = get_nphones(utt2phones, 3)
        idx2triphones = [triphone 
                         for triphone, count in sorted(triphone_counts.items()) 
                         if ((count / len(triphone_counts)) > .01)]

        idx2mono_di_tri_phones = [tuple([phone]) for phone in idx2phones] + idx2diphones + idx2triphones
    with (processed_dirpath / (mono_di_tri_phones_filename
                               + '.json')).open('w') as mono_di_tri_phones_file:
        json.dump(idx2mono_di_tri_phones, mono_di_tri_phones_file, indent=4)

    mono_di_tri_phones2idx = {idx2mono_di_tri_phones[idx] : idx 
                             for idx in range(len(idx2mono_di_tri_phones))}

    utt2mono_di_tri_phones, utt2mono_di_tri_alignments = (
            get_utterance2mono_di_tri_phones(mono_di_tri_phones2idx,
                                             utt2phones, 
                                             utt2alignments))
    with (processed_dirpath / (utt2mono_di_tri_phones_filename
                               + '.json')).open('w') as utt2mono_di_tri_phones_file:
        json.dump(utt2mono_di_tri_phones, utt2mono_di_tri_phones_file, indent=4)

    with (processed_dirpath / (utt2mono_di_tri_alignments_filename
                               + '.json')).open('w') as utt2mono_di_tri_alignments_file:
        json.dump(utt2mono_di_tri_alignments, utt2mono_di_tri_alignments_file, indent=4)

    
    # Copy and convert audio from flac to wav
    copy_and_convert_utterances(utterances, orig_dirpath, processed_dirpath)

    return utterances


def process_data(processed_dirpath,
                 utterances,
                 idx2vocabulary, 
                 vocabulary2idx, 
                 idx2phones, 
                 phones2idx, 
                 idx2mono_di_tri_phones, 
                 mono_di_tri_phones2idx, 
                 utt2words,
                 utt2phones, 
                 utt2alignments, 
                 utt2mono_di_tri_phones,
                 utt2mono_di_tri_alignments):

    # Get numpy arrays representing both the transcripts and the 
    # phonetic transcripts
    word_list = []
    phones_list = []
    maxuttlen = max([len(words) for words in utt2words.values()])
    maxphonelen = max([len(phones) for phones in utt2phones.values()])

    for idx in range(len(utterances)):
        word_list.append(np.pad([vocabulary2idx[word] 
                                 for word in utt2words[utterances[idx]]], 
                                (0, maxuttlen - len(utt2words[utterances[idx]])), 
                                'constant'))
        phones_list.append(np.pad([mono_di_tri_phones2idx[tuple(phone)] 
                                   for phone in utt2mono_di_tri_phones[utterances[idx]]],
                                  (0, maxphonelen - len(utt2mono_di_tri_phones[utterances[idx]])),
                                  'constant'))

    np_word_list=np.stack(word_list)
    np.save(str(processed_dirpath / (transcriptions_filename + '.npy')), 
            word_list, 
            allow_pickle=False)
    np_phones_list=np.stack(phones_list)
    np.save(str(processed_dirpath / (utt2mono_di_tri_phones_filename + '.npy')), 
            phones_list, 
            allow_pickle=False)

    # Get the features of each phone group in each utterance
    utterance_wavs = get_utterance_wavs(processed_dirpath, utterances)

    generate_target_feats(utterances, 
                          utterance_wavs, 
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          maxphonelen)

    '''generate_concat_feats(utterances, 
                          utterance_wavs, 
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          maxphonelen)'''

def generate_target_feats(utterances, 
                          utterance_wavs, 
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          maxphonelen):
    print("HERE!")
    # A dictionary from utterance to features
    utt2target_feats=defaultdict(list)
    # A sorted list of utterance features (sorted 
    utt_target_feats = []
    for utterance in utterances:
        utt_target_feats.append([])
        for alignment in utt2mono_di_tri_alignments[utterance]:
            target_feats = get_target_feats(utterance_wavs[utterance], alignment)
            utt_target_feats[-1].append(target_feats)
            utt2target_feats[utterance].append(target_feats)

    utils.save_json(utt2target_feats, processed_dirpath / (utt2target_feats_filename + '.json'))
    num_target_feats = len(list(utt2target_feats.values())[0][0])

    np_target_feats_flattened = np.array([feats for utterance in utt_target_feats for feats in utterance])
    np_target_feats_mean = np.mean(np_target_feats_flattened, axis=0)
    np_target_feats_std = np.std(np_target_feats_flattened, axis=0)
    np_target_feats_normalized = np.asarray([np.pad((np.array(feats)
                                                     - np_target_feats_mean)
                                                     / np_target_feats_std, 
                                                    ((0,maxphonelen-len(feats)),(0,0)),
                                                    mode='constant')
                                             for feats in utt_target_feats])

    np.save(str(processed_dirpath / (target_feats_mean_filename + '.npy')),
            np_target_feats_mean,
            allow_pickle=False)
    np.save(str(processed_dirpath / (target_feats_std_filename + '.npy')),
            np_target_feats_std,
            allow_pickle=False)
    np.save(str(processed_dirpath / (target_feats_filename + '_normalized.npy')),
            np_target_feats_normalized,
            allow_pickle=False)

def generate_concat_feats(utterances, 
                          utterance_wavs, 
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          maxphonelen):
    utt2concat_feats=defaultdict(list)
    utt_concat_feats = []
    for utterance in utterances:
        utt_concat_feats.append([])
        for alignment in utt2mono_di_tri_alignments[utterance]:
            concat_feats = get_concat_feats(utterance_wavs[utterance], alignment)
            utt_concat_feats[-1].append(concat_feats)
            utt2concat_feats[utterance].append(concat_feats)

    utils.save_json(utt2concat_feats, processed_dirpath / (utt2concat_feats_filename + '.json'))

    num_concat_feats = len(list(utt2concat_feats.values())[0][0])


    np.save(str(processed_dirpath / (concat_feats_mean_filename + '.npy')), np_concat_feats_mean, allow_pickle=False)
    np.save(str(processed_dirpath / (concat_feats_std_filename + '.npy')), np_concat_feats_std, allow_pickle=False)
    np.save(str(processed_dirpath / (concat_feats_filename + '_normalized.npy')), np_concat_feats_normalized, allow_pickle=False)


def get_target_feats(utterance_wav, alignments):
 
    #phone_start = int(alignments[0] * fs)
    first_phone_start = alignments[0][0]
    first_phone_end = alignments[0][1]

    #print("START: " + str(phone_start))
    #phone_end = int(alignments[1] * fs)
    #print("END: " + str(phone_end))
    last_phone_start = alignments[-1][0]
    last_phone_end = alignments[-1][1]
    #print(phone_start)
    #print(phone_end)
    #print(utterance_wav)
    #print(len(utterance_wav))
    duration = last_phone_end - first_phone_start
    first_phone_samples = utterance_wav[first_phone_start:first_phone_end]
    last_phone_samples = utterance_wav[last_phone_start:last_phone_end]
    all_phone_samples = utterance_wav[first_phone_start:last_phone_end]
    #phone_test = utterance_wav[phone_start]
    '''
    try:
        phone_test = utterance_wav[phone_start]
    except:
        print("Outof bounds!!")
        print(utterance)
        print(phone_start)
        print(phone_end)
        print(alignments)
        return (0, 0, 0, 0)
    '''
    #print(duration)

    #print(phone_samples)
    #try:
    '''
    print(first_phone_start)
    print(first_phone_end)
    print(last_phone_start)
    print(last_phone_end)
    '''
    f_0_init = np.mean(pysptk.swipe(first_phone_samples.astype(np.float64), 
                                    fs=fs, 
                                    hopsize=100, 
                                    otype='f0'))
    #print(f_0_init)
    f_0_end = np.mean(pysptk.swipe(last_phone_samples.astype(np.float64), 
                                   fs=fs, 
                                   hopsize=100, 
                                   otype='f0'))
    #print(f_0_end)
    #except IndexError:
        # For "Index Error: Out of bounds on buffer access (axis 0)


    #mfcc = pysptk.mfcc(samples)

    #pitch = pysptk.swipe(phone_samples.astype(np.float64), fs=fs, hopsize=100, otype='pitch')

    #excitation = pysptk.excite(pitch)
    #excitation_mu = np.mean(excitation)
    #excitation_std = np.std(excitation)
    #print()
    energy = np.sum(np.square(all_phone_samples)) / duration
    return duration, f_0_init, f_0_end, energy

def get_concat_feats(utterance_wav, alignments):
    #phone_start = int(alignments[0] * fs)
    first_phone_start = alignments[0][0]
    first_phone_end = alignments[0][1]

    #print("START: " + str(phone_start))
    #phone_end = int(alignments[1] * fs)
    #print("END: " + str(phone_end))
    last_phone_start = alignments[-1][0]
    last_phone_end = alignments[-1][1]
    first_phone_samples = utterance_wav[first_phone_start:first_phone_end]
    last_phone_samples = utterance_wav[last_phone_start:last_phone_end]
    all_phone_samples = utterance_wav[first_phone_start:last_phone_end]

    duration = last_phone_end - first_phone_start

    #first_phone_mfccs = pysptk.mfcc(first_phone_samples)
    #last_phone_mfccs =  pysptk.mfcc(last_phone_samples)

    f_0_init = np.mean(pysptk.swipe(first_phone_samples.astype(np.float64), 
                                    fs=fs, 
                                    hopsize=100, 
                                    otype='f0'))
    f_0_end = np.mean(pysptk.swipe(last_phone_samples.astype(np.float64), 
                                   fs=fs, 
                                   hopsize=100, 
                                   otype='f0'))

    energy = np.sum(np.square(all_phone_samples)) / duration
    return f_0_init, f_0_end, energy
    #return first_phone_mfccs.tolist(), last_phone_mfccs.tolist(), f_0_init, f_0_end, energy


def get_utterance_wavs(processed_dirpath, utterances):
    utterance_wavs = {}
    for utterance in utterances:
        utterance_dirs = get_utterance_dirs(utterance)
        _, utterance_wav = wavfile.read(str(processed_dirpath 
                                         / utterance_dirs 
                                         / (utterance + '.wav')))

        utterance_wavs[utterance]=utterance_wav

    return utterance_wavs

def copy_id_values(orig_filepath, copied_filepath, idnts):
    with orig_filepath.open() as orig_file:
      with copied_filepath.open('w') as copied_file:
        idnts2values = {}
        for line in orig_file:
            split_line = line.split()
            idnt = split_line[0]
            if idnt in idnts:
                idnts2values[idnt] = split_line[1:]
                copied_file.write(line)
    return idnts2values

def get_idnt_value(filepath, lists=False, idnts_type=str, values_type=str):
    idnt2value = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            if not lists:
                idnt2value[idnts_type(split_line[0])] = values_type(
                    split_line[1])
            else:
                idnt2value[idnts_type(split_line[0])] = [values_type(value)
                                                         for value in split_line[1:]]
    return idnt2value

def get_spk2gender(filepath):
    spk2gender = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            spk2gender[split_line[0]]=split_line[1]
    return spk2gender


def get_phones(filepath):
    idx2phones = []
    phones2idx = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            phones2idx[split_line[0]]=len(idx2phones)
            idx2phones.append(split_line[0])
    return idx2phones, phones2idx

def get_vocab(filepath):
    idx2vocab = []
    vocab2idx = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            vocab2idx[split_line[0]]=len(idx2vocab)
            idx2vocab.append(split_line[0])
    return idx2vocab, vocab2idx

def get_lexicon(filepath):
    word2phones = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            word2phones[split_line[0]] = split_line[2:]
    return word2phones

def get_transcriptions(filepath):
    utt2words = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            utt2words[split_line[0]] = split_line[1:]
    return utt2words

def get_alignments(alignments_filepath, idx2phones):
    utt2phones = defaultdict(list) 
    utt2alignments = defaultdict(list)
    with alignments_filepath.open() as alignments_file:
        for line in alignments_file:
            split_line = line.split()
            utterance = split_line[0]
            utt2phones[utterance].append(idx2phones[int(split_line[-1])])
            start_sample = int(float(split_line[2])*fs)
            end_sample =  start_sample + int(float(split_line[3])*fs)
            utt2alignments[utterance].append([start_sample, end_sample])

    return utt2phones, utt2alignments

def get_utterances(utterance_duration_filepath,
                   gender,
                   duration_limit,
                   spk2gender,
                   word2phones,
                   utt2words,
                   utt2alignments):
    utt2dur = {}  
    # Get utterances and durations, filtering by gender
    with utterance_duration_filepath.open() as utterance_duration_file:
        for line in utterance_duration_file:
            split_line = line.split()
            utterance=split_line[0]
            spk = '-'.join(utterance.split('-')[0:-1])
            curr_duration = split_line[1]
            if ((gender and spk2gender[spk] == gender) or
                 not gender):
                utt2dur[utterance]=float(curr_duration)


    curr_duration = 0
    utterances = set()
    spks = set()
    # Get a random selection of utterances <= a duration limit
    shuffled_utterances = (list(utt2dur.keys()))
    random.shuffle(shuffled_utterances)
    for utterance in shuffled_utterances:
        lex_word = True
        # Ensure we have a phonization of all words
        for word in utt2words[utterance]:
            if word not in word2phones:
                lex_word = False
                break
        # Ensure we have:
        # 1. Alignment
        # 2. A phonization
        if (utterance in utt2alignments and 
            lex_word):
            # Ensure that we haven't hit our max duration
            if ((duration_limit and 
                 (curr_duration + utt2dur[utterance]) 
                  <= duration_limit) or
                not duration_limit):
                utterances.add(utterance)
                spks.add('-'.join(utterance.split('-')[0:-1]))
                curr_duration += utt2dur[utterance]
    utt2dur = {utterance: duration 
                          for utterance, duration in utt2dur.items()
                          if utterance in utterances}
    return utterances, spks, utt2dur, curr_duration

def copy_and_convert_utterances(utterances, orig_dirpath, processed_dirpath):
    for utterance in utterances:
        utterance_dirnames = get_utterance_dirs(utterance)
        flac_file = orig_dirpath / utterance_dirnames / (utterance + '.flac')
        flac = AudioSegment.from_file(str(flac_file), 'flac')
        new_dir = (processed_dirpath / utterance_dirnames)
        if not new_dir.exists():
            new_dir.mkdir(parents=True)

        flac.export(str(new_dir / (utterance + '.wav')),
                   format='wav')

def get_utterance_dirs(utterance):
    # TODO: Fix utterance_dirs to be tuple
    utterance_dirs = os.path.join(*(utterance.split('-')[0:-1]))
    return utterance_dirs

def get_nphones(utt2phones, n):
    nphone_counts = Counter()
    for phones in utt2phones.values():
        idx = 0
        while idx+n < len(phones):
            nphone = tuple(phones[idx:idx+n])
            nphone_counts[nphone] += 1
            idx += 1
    return nphone_counts

def get_utterance2mono_di_tri_phones(mono_di_tri_phones2idx,
                                     utt2phones, 
                                     utt2alignments):
    utt2mono_di_tri_phones = {}
    utt2mono_di_tri_alignments = {}
    for utterance in utt2phones.keys():
        utterance_mono_di_tri_phones, utterance_mono_di_tri_alignments = (
                get_mono_di_tri_phones_alignments_from_utterance(
                    mono_di_tri_phones2idx,
                    utt2phones[utterance],
                    utt2alignments[utterance]))
        utt2mono_di_tri_phones[utterance] = utterance_mono_di_tri_phones
        utt2mono_di_tri_alignments[utterance] = utterance_mono_di_tri_alignments
    return utt2mono_di_tri_phones, utt2mono_di_tri_alignments


def get_mono_di_tri_phones_alignments_from_utterance(nphone2idx, 
                                                     utterance_phones,
                                                     utterance_alignments):
    idx = 0
    utterance_mono_di_tri_phones = []
    utterance_mono_di_tri_alignments = []
    while idx < len(utterance_phones):
        if idx + 2 < len(utterance_phones):
            triphone = tuple(utterance_phones[idx:idx+3])
            trialignments = tuple(utterance_alignments[idx:idx+3])
            if triphone in nphone2idx:
                utterance_mono_di_tri_phones.append(triphone)
                utterance_mono_di_tri_alignments.append(trialignments)
                idx += 2
                continue
        if idx + 1 < len(utterance_phones):
            diphone = tuple(utterance_phones[idx:idx+2])
            dialignments = tuple(utterance_alignments[idx:idx+2])
            if diphone in nphone2idx:
                utterance_mono_di_tri_phones.append(diphone)
                utterance_mono_di_tri_alignments.append(dialignments)
                idx += 1
                continue

        # We go ahead and add the overlapping monophone for (arguably) better smoothing
        monophone = tuple(utterance_phones[idx:idx+1])
        monoalignments = tuple(utterance_alignments[idx:idx+1])
        if monophone in nphone2idx:
            utterance_mono_di_tri_phones.append(monophone)
            utterance_mono_di_tri_alignments.append(monoalignments)
            idx += 1

        else:
            raise KeyError('Bad phone: ' + str(monophone))
    return utterance_mono_di_tri_phones, utterance_mono_di_tri_alignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dur_limit', type=lambda d: int(d)*60*60*fs)
    parser.add_argument('--gender', type=str)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('normalized_dir', type=Path)
    parser.add_argument('-n', '--nphones', type=Path)

    args = parser.parse_args()
    main_args = {'orig_dirpath':args.data_dir, 'processed_dirpath':args.normalized_dir}
    if args.dur_limit:
        main_args['duration_limit']=args.dur_limit
    if args.gender:
        main_args['gender']=args.gender
    if args.nphones:
        main_args['nphones']=utils.load_json(args.nphones)
    preprocess(**main_args)
