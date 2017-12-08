import argparse
import json
import math
import os
import random
import shutil

from collections import defaultdict
from pathlib import Path

import numpy as np
import pysptk

from pydub import AudioSegment
from scipy.io import wavfile

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

target_feats_filename = 'target_feats'
concat_feats_filename = 'concat_feats'

male = 'm'
female = 'f'

fs=16000

def preprocess(orig_dirpath, processed_dirpath, duration=float('inf'), gender=None):

    if not processed_dirpath.exists():
        processed_dirpath.mkdir(parents=True)

    utterances = copy_base_data(orig_dirpath, 
                                processed_dirpath, 
                                duration=duration,
                                gender=gender)

    process_data(processed_dirpath, utterances)

def copy_base_data(orig_dirpath, 
                   processed_dirpath, 
                   duration=None,
                   gender=None):

    spk2gender = get_gender((orig_dirpath / (spk2gender_filename + '.txt')))
    idx2phones, phones2idx = get_phones((orig_dirpath / (phones_filename + '.txt')))
    idx2vocab, vocab2idx = get_vocab((orig_dirpath / (vocabulary_filename + '.txt')))
    word2phones = get_lexicon((orig_dirpath / (lexicon_filename + '.txt')))

    utt2text = get_transcript((orig_dirpath / (transcriptions_filename +
        '.txt')), lists=True)

    utt2phones_all, utt2alignments_all = get_alignments(
            orig_dirpath / (alignments_filename + '.txt'), 
            idx2phones)

    utterances, spks, utterance2duration, curr_duration = get_utterances(
            (orig_dirpath / (utterance_duration_filename + '.txt')),
            spk2gender, 
            gender,
            duration,
            lexicon,
            transcriptions,
            utt2alignments_all)


    # Copy and convert audio from flac to wav
    # TODO: Uncomment this!
    #copy_and_convert_utterances(utterances, orig_dirpath, processed_dirpath)

    # JSONize phone list
    with (processed_dirpath / (phones_filename + '.json')).open('w') as phones_file:
        json.dump(idx2phones, phones_file, indent=4)

    # JSONIze vocabulary
    with (processed_dirpath / (vocabulary_filename + '.json')).open('w') as vocabulary_file:
        json.dump(idx2vocab, vocabulary_file, indent=4)

    # JSONIze lexicon
    with (processed_dirpath / (lexicon_filename + '.json')).open('w') as lexicon_file:
        json.dump(word2phones, lexicon_file, indent=4)

    # JSONIze utterances
    with (processed_dirpath / (utterances_filename + '.json')).open('w') as utterance_file:
        json.dump(list(utterances), utterance_file, indent=4)

    utt2text = {utterance: text 
                for utterance, text in utt2text.items() 
                if utterance in utterances_set}

    with (processed_dirpath / (transcriptions_filename + '.json')).open('w') as transcriptions_file:
        json.dump(utt2text, transcriptions_file, indent=4)

    # JSONize spk2gender
    with (processed_dirpath / (spk2gender_filename + '.json')).open('w') as spk2gender_file:
        json.dump(spk2gender, spk2gender_file, indent=4)

    # Get new alignments
    utt2phones, utt2alignments = get_alignment_data(processed_dirpath / (alignments_filename + '.txt'), phones)
    #jsonize 
    with (processed_dirpath / (utt2phones_filename + '.json')).open('w') as utt2phones_file:
        json.dump(utt2phones, utt2phones_file, indent=4)

    with (processed_dirpath / (utt2alignments_filename + '.json')).open('w') as utt2alignments_file:
        json.dump(utt2alignments, utt2alignments_file, indent=4)

    return utterances

def process_data(processed_dirpath, utterances):
    with (processed_dirpath / (utt2phones_filename + '.json')).open() as utt2phones_file:
        utt2phones=json.load(utt2phones_file)
    with (processed_dirpath / (utt2alignments_filename + '.json')).open() as utt2alignments_file:
        utt2alignments=json.load(utt2alignments_file)

    with (processed_dirpath / (transcriptions_filename + '.json')).open() as transcriptions_file:
        transcriptions = json.load(transcriptions_file)

    with (processed_dirpath / (phones_filename + '.json')).open() as phones_file:
        idx2phones = json.load(phones_file)

    phones2idx = {idx2phones[idx]: idx 
                      for idx in range(len(idx2phones))}
    with (processed_dirpath / (vocabulary_filename + '.json')).open() as vocab_file:
        idx2vocabulary = json.load(vocab_file)

    vocabulary2idx = {idx2vocabulary[idx]: idx 
                      for idx in range(len(idx2vocabulary))}

    word_list = []
    phones_list = []
    maxuttlen = max([len(utterance) for utterance in transcriptions.values()])
    maxphonelen = max([len(phones) for phones in utt2alignments.values()])

    for idx in range(len(utterances)):
        word_list.append(np.pad([vocabulary2idx[word] for word in transcriptions[utterances[idx]]],(0, maxuttlen - len(transcriptions[utterances[idx]])), 'constant'))
        phones_list.append(np.pad([phones2idx[phone] for phone in utt2phones[utterances[idx]]],(0, maxphonelen - len(utt2phones[utterances[idx]])), 'constant'))

    np_word_list=np.stack(word_list)
    np.save(str(processed_dirpath / (transcriptions_filename + '.npy')), word_list, allow_pickle=False)
    np_phones_list=np.stack(phones_list)
    np.save(str(processed_dirpath / (utt2phones_filename + '.npy')), phones_list, allow_pickle=False)


    # Get the features of each utterance
    print("UTTS")
    print(utterances)
    utterance_wavs = get_utterance_wavs(processed_dirpath, utterances)
    utterance2target_feats=[]
    utterance2concat_feats=[]

    for utterance in utterances:
        utterance2target_feats.append([])
        utterance2concat_feats.append([])
        for alignment in utt2alignments[utterance]:
            target_feats = get_target_feats(utterance_wavs[utterance], alignment)
            # For now, just get rid of duration.
            concat_feats = target_feats[1:]
            #concat_feats = get_concat_feats(utterance_wavs[utterance], alignment)
            utterance2target_feats[-1].append(target_feats)
            utterance2concat_feats[-1].append(concat_feats)

    pad_utterance2target_feats = [np.pad(feats, ((0, maxphonelen-len(feats)), (0, 0)), 'constant')
                                  for feats in utterance2target_feats]

    pad_utterance2concat_feats = [np.pad(feats, ((0, maxphonelen-len(feats)), (0,0)), 'constant')
                                  for feats in utterance2concat_feats]
    np_utterance2target_feats = np.stack(pad_utterance2target_feats)
    np.save(str(processed_dirpath / target_feats_filename), np_utterance2target_feats, allow_pickle=False)
    np_utterance2concat_feats = np.stack(pad_utterance2concat_feats)
    np.save(str(processed_dirpath / concat_feats_filename), np_utterance2concat_feats, allow_pickle=False)

    '''
    np_utterance2concat_feats = np.stack(utterance2concat_feats)
    np.save(str(processed_dirpath / concat_feats_filename), np_utterance2concat_feats, allow_pickle=False)
    '''
def get_target_feats(utterance_wav, alignments):
    #phone_start = int(alignments[0] * fs)
    phone_start = alignments[0]
    #print("START: " + str(phone_start))
    #phone_end = int(alignments[1] * fs)
    #print("END: " + str(phone_end))
    phone_end = alignments[1]
    #print(phone_start)
    #print(phone_end)
    #print(utterance_wav)
    #print(len(utterance_wav))
    duration = phone_end - phone_start
    phone_samples = utterance_wav[phone_start:phone_end]
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
    f_0 = pysptk.swipe(phone_samples.astype(np.float64), fs=fs, hopsize=100, otype='f0')
    f_0_init = np.mean(f_0[:math.ceil(f_0.size/3)])
    f_0_end = np.mean(f_0[2*math.floor(f_0.size/3):])

    #mfcc = pysptk.mfcc(samples)

    #pitch = pysptk.swipe(phone_samples.astype(np.float64), fs=fs, hopsize=100, otype='pitch')

    #excitation = pysptk.excite(pitch)
    #excitation_mu = np.mean(excitation)
    #excitation_std = np.std(excitation)
    #print()
    energy = np.sum(np.square(phone_samples)) / duration
    return duration, f_0_init, f_0_end, energy


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
            word2phones[split_line[0]] = split_line[1:]
    return word2phones

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
                   utt2text,
                   utt2alignments):
    #print(duration)
    utterance2duration = {}  
    # Get utterances and durations, filtering by gender
    with utterance_duration_filepath.open() as utterance_duration_file:
        for line in utterance_duration_file:
            split_line = line.split()
            utterance=split_line[0]
            spk = '-'.join(utterance.split('-')[0:-1])
            curr_duration = split_line[1]
            if ((gender and spk2gender[spk] == gender) or
                 not gender):
                utterance2duration[utterance]=float(curr_duration)


    curr_duration = 0
    utterances = set()
    spks = set()
    # Get a random selection of utterances <= a duration limit
    shuffled_utterances = (list(utterance2duration.keys()))
    random.shuffle(shuffled_utterances)
    for utterance in shuffled_utterances:
        lex_word = True
        # Ensure we have a phonization of all words
        for word in transcriptions[utterance]:
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
                 (curr_duration + utterance2duration[utterance]) 
                  <= duration_limit) or
                not duration_limit):
                utterances.add(utterance)
                spks.add('-'.join(utterance.split('-')[0:-1]))
                curr_duration += utterance2duration[utterance]
    utterance2duration = {utterance: duration 
                          for utterance, duration in utterance2duration.items()
                          if utterance in utterances}
    return utterances, spks, utterance2duration, curr_duration

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

def get_tri_di_phones_alignments_from_utterance(nphone2idx, 
                                                utterance_phones,
                                                utterance_alignments):
    '''
    final_phones -- a dict containing indices for triphones and diphones recognized by the model
    '''
    idx = 0
    utterance_tri_di_mono_phones = []
    utterance_tri_di_mono_alignments = []
    while idx < len(utt2phones[utterance]):
        triphone = tuple(utterance_phones[idx:idx+3])
        trialignments = tuple(utterance_alignments[idx:idx+3])
        if triphone in nphone2idx:
            utterance_tri_di_mono_phones.append(triphone)
            utterance_tri_di_mono_alignments.append(trialignments)
            idx += 2
        else:
            diphone = tuple(utterancephones[idx:idx+2])
            dialignments = tuple(utterance_alignments[idx:idx+2])
            if diphone in nphone2idx:
                utterance_tri_di_mono_phones.append(diphone)
                utterance_tri_di_mono_alignments.append(dialignments)
                idx += 1
            else:
                monophone = tuple(utterancephones[idx+1:idx+2])
                monoalignments = tuple(utterance_alignments[idx:idx+1])
                if monophone in nphone2idx:
                    utterance_tri_di_mono_phones.append(monophone)
                    utterance_tri_di_mono_alignments.append(monoalignments)
    return utterance_tri_di_mono_phones, utterance_tri_di_mono_alignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dur_limit', type=lambda d: int(d)*60*60*fs)
    parser.add_argument('--gender', type=str)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('normalized_dir', type=Path)

    args = parser.parse_args()
    main_args = {'orig_dirpath':args.data_dir, 'processed_dirpath':args.normalized_dir}
    if args.dur_limit:
        main_args['duration']=args.dur_limit
    if args.gender:
        main_args['gender']=args.gender

    preprocess(**main_args)
