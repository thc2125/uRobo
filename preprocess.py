# Python script allowing for preprocessing of data for uRobo
# Written by Tyrus Cukavac
# thc2125


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
import soundfile as sf

#from pydub import AudioSegment
from scipy.io import wavfile

import utils

# Define constants
phones_filename = 'phones'
utterances_filename = 'utterances'
vocabulary_filename = 'words'
alignments_filename = 'alignments'
audiofiles_filename = 'wav.scp'
lexicon_filename = str(Path('phones', 'align_lexicon.txt'))
spk2gender_filename = 'spk2gender'
transcriptions_filename = 'text'
utterance_duration_filename = 'utt2dur'

word2phones_filename = 'word2phones'

utt2flac_filename = 'utt2flac'
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

male = 'm'
female = 'f'

fs=16000

def preprocess(kaldi_data_dirpath, 
               kaldi_lm_dirpath,
               processed_dirpath, 
               duration_limit=float('inf'), 
               gender=None, 
               mono_di_tri_phones=None,
               speakers=None,
               skip_audio=False):
    '''Preprocess from Kaldi audio data
    Keyword Arguments:
    kaldi_data_dirpath -- Path to Kaldi extracted data
    kaldi_lm_dirpath -- Path to Kaldi Language Model ('data/lang')
    processed_dirpath -- Output Path; where to place pre-processed files
    duration_limit -- how much audio to get (in samples)
    gender -- the gender of speakers to extract
    mono_di_tri_phones -- a list of mono/di/triphones
    speakers -- a list of speakers to extract
    skip_audio -- whether to skip copying audio (deprecated, could result in
                  unstable behavior) 
    '''

    if not processed_dirpath.exists():
        processed_dirpath.mkdir(parents=True)

    utterances = copy_base_data(kaldi_data_dirpath, 
                                kaldi_lm_dirpath,
                                processed_dirpath, 
                                duration_limit=duration_limit,
                                gender=gender,
                                mono_di_tri_phones=mono_di_tri_phones,
                                speakers=speakers,
                                skip_audio=skip_audio)
    data = utils.load_data(processed_dirpath)

    process_data(processed_dirpath, sorted(list(utterances)), *data)

def copy_base_data(kaldi_data_dirpath, 
                   kaldi_lm_dirpath,
                   processed_dirpath, 
                   duration_limit=None,
                   gender=None,
                   mono_di_tri_phones=None,
                   speakers=None,
                   skip_audio=False):
    ''' Copies the base Kaldi processed data, converting to JSON format and 
    filtering based on gender, speaker, and total duration
    '''
    print("Copying base Kaldi/audio data and converting to json.")
    utt2flac = get_utt2flac((kaldi_data_dirpath / audiofiles_filename))
    #JSONize utt2flac for posterity
    utils.save_json(utt2flac, processed_dirpath / (utt2flac_filename + '.json'))

    spk2gender = get_spk2gender((kaldi_data_dirpath / (spk2gender_filename)))
    # JSONize spk2gender
    with (processed_dirpath / (spk2gender_filename + '.json')).open('w') as spk2gender_file:
        json.dump(spk2gender, spk2gender_file, indent=4)


    idx2phones, phones2idx = get_phones((kaldi_lm_dirpath / (phones_filename + '.txt')))
    # JSONize phone list
    with (processed_dirpath / (phones_filename + '.json')).open('w') as phones_file:
        json.dump(idx2phones, phones_file, indent=4)

    idx2vocab, vocab2idx = get_vocab((kaldi_lm_dirpath / (vocabulary_filename + '.txt')))
    # JSONIze vocabulary
    with (processed_dirpath / (vocabulary_filename + '.json')).open('w') as vocabulary_file:
        json.dump(idx2vocab, vocabulary_file, indent=4)

    word2phones = get_lexicon((kaldi_lm_dirpath / (lexicon_filename)))
    # JSONIze lexicon
    utils.save_json(word2phones, (processed_dirpath / (word2phones_filename + '.json')))

    utt2words_all = get_transcriptions(kaldi_data_dirpath / transcriptions_filename)

    utt2phones_all, utt2alignments_all = get_alignments(
            kaldi_data_dirpath / (alignments_filename), 
            idx2phones)

    # Get the final list of utterances we will be copying
    utterances, spks, utt2dur, curr_duration = get_utterances(
            (kaldi_data_dirpath / (utterance_duration_filename)),
            gender,
            speakers,
            duration_limit,
            spk2gender, 
            word2phones,
            utt2words_all,
            utt2alignments_all)
    # Copy and convert audio from flac to wav
    if not skip_audio:
        print("Converting audio files from flac to wav.")
        copy_and_convert_utterances(utterances, utt2dur, utt2flac, processed_dirpath)

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


    # Create metrics of diphones
    utt2diphones, utt2diphone_alignments, diphone_counts = get_nphones(utt2phones, utt2alignments, 2)
    idx2diphones = [diphone 
                    for diphone, count in sorted(diphone_counts.items()) 
                    if ((count / len(diphone_counts)) > .001)]

    utils.save_json(utt2diphones, 
                    processed_dirpath / (utt2diphones_filename + '.json'))
    utils.save_json(utt2diphone_alignments, 
                    processed_dirpath / (utt2diphone_alignments_filename + '.json'))


    # Create metrics of triphones
    utt2triphones, utt2triphone_alignments, triphone_counts = get_nphones(utt2phones, utt2alignments, 3)
    idx2triphones = [triphone 
                     for triphone, count in sorted(triphone_counts.items()) 
                     if ((count / len(triphone_counts)) > .01)]


    utils.save_json(utt2triphones, processed_dirpath / (utt2triphones_filename + '.json'))
    utils.save_json(utt2triphone_alignments, 
                    processed_dirpath / (utt2triphone_alignments_filename + '.json'))


    # Use or create indexes of mono_di_tri_phones
    if mono_di_tri_phones:
        idx2mono_di_tri_phones=[tuple(phones) for phones in mono_di_tri_phones]
    else:
        idx2mono_di_tri_phones = [tuple([phone]) for phone in idx2phones] + idx2diphones + idx2triphones

    with (processed_dirpath / (mono_di_tri_phones_filename
                               + '.json')).open('w') as mono_di_tri_phones_file:
        json.dump(idx2mono_di_tri_phones, mono_di_tri_phones_file, indent=4)

    mono_di_tri_phones2idx = {idx2mono_di_tri_phones[idx] : idx 
                              for idx in range(len(idx2mono_di_tri_phones))}

    # Split the utterances using the list of n-phones
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
                 utt2diphones,
                 utt2diphone_alignments,
                 utt2triphones,
                 utt2triphone_alignments,
                 utt2mono_di_tri_phones,
                 utt2mono_di_tri_alignments):
    '''Process the data, extracting speaker independent target features for 
    mono-di-tri-phones and monophones
    '''

    print("Processing data and collecting features.")
    # Get numpy arrays representing both the transcripts and the 
    # phonetic transcripts
    word_list = []
    phones_list = []
    mono_di_tri_phones_list = []
    maxuttlen = len(max(utt2words.values(), key=len))
    maxphonelen = len(max(utt2phones.values(), key=len))
    max_mono_di_tri_phonelen = len(max(utt2mono_di_tri_phones.values(), key=len))
    print("Generating word/phonetic transcripts as numpy arrays.")
    for idx in range(len(utterances)):
        word_list.append(np.pad([vocabulary2idx[word]
                                 for word in utt2words[utterances[idx]]], 
                                (0, maxuttlen - len(utt2words[utterances[idx]])), 
                                'constant'))
        #print(utt2phones[utterances[idx]])
        phones_list.append(np.pad([phones2idx[phone] 
                                   for phone in utt2phones[utterances[idx]]],
                                  (0, maxphonelen - len(utt2phones[utterances[idx]])),
                                  'constant'))
        mono_di_tri_phones_list.append(np.pad([mono_di_tri_phones2idx[tuple(phone)] 
                                               for phone in utt2mono_di_tri_phones[utterances[idx]]],
                                              (0, max_mono_di_tri_phonelen - len(utt2mono_di_tri_phones[utterances[idx]])),
                                              'constant'))
        #phones_mono_di_tri_phones_list = phones_list + mono_di_tri_phones_list

    np_word_list=np.stack(word_list)
    np.save(str(processed_dirpath / (utt_prefix + words_filename + '.npy')), 
            np_word_list, 
            allow_pickle=False)

    np_phones_list=np.stack(phones_list)
    np.save(str(processed_dirpath / (utt_prefix + phones_filename + '.npy')),
            np_phones_list,
            allow_pickle=False)

    np_mono_di_tri_phones_list=np.stack(mono_di_tri_phones_list)
    np.save(str(processed_dirpath / (utt_prefix + mono_di_tri_phones_filename + '.npy')), 
            np_mono_di_tri_phones_list,
            allow_pickle=False)

    # Get the features of each phone group in each utterance
    # First load all the utterance samples into memory
    utterance_wavs = get_utterance_wavs(processed_dirpath, utterances)

    # Now generate target features for phones
    print("Generating phone level features.")
    generate_target_feats(utterances, 
                          utterance_wavs,
                          utt2alignments,
                          processed_dirpath,
                          phones_filename,
                          maxphonelen)

    # Now generate target features for mono_di_tri_phones
    print("Generating mono-di-tri-phone level features.")
    generate_target_feats(utterances, 
                          utterance_wavs,
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          mono_di_tri_phones_filename,
                          max_mono_di_tri_phonelen)

    '''generate_concat_feats(utterances, 
                          utterance_wavs, 
                          utt2mono_di_tri_alignments,
                          processed_dirpath,
                          maxphonelen)'''

def generate_target_feats(utterances, 
                          utterance_wavs, 
                          utt2alignments,
                          processed_dirpath,
                          phone_type,
                          maxphonelen):
    '''Generates target features for the provided utterances'''
    # A dictionary from utterance to features
    utt2target_feats=defaultdict(list)
    # A sorted list of utterance features (sorted 
    utt_target_feats = []
    for utterance in utterances:
        utt_target_feats.append([])
        for alignment in utt2alignments[utterance]:
            if 'phones' == phone_type:
                target_feats = get_target_feats(utterance,utterance_wavs[utterance], tuple([alignment]))
            else:
                target_feats = get_target_feats(utterance,utterance_wavs[utterance], alignment)
            utt_target_feats[-1].append(target_feats)
            utt2target_feats[utterance].append(target_feats)

    utils.save_json(utt2target_feats, 
                    processed_dirpath 
                    / (utt2_id 
                       + phone_type 
                       + target_feats_suffix 
                       + '.json'))

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

    np.save(str(processed_dirpath 
                / (utt_prefix 
                   + target_feats_suffix
                   + mean_suffix 
                   + '.npy')),
            np_target_feats_mean,
            allow_pickle=False)

    np.save(str(processed_dirpath 
                / (utt_prefix 
                   + phone_type
                   + target_feats_suffix
                   + std_suffix 
                   + '.npy')),
            np_target_feats_std,
            allow_pickle=False)

    np.save(str(processed_dirpath / (phone_type + '_normalized.npy')),
            np_target_feats_normalized,
            allow_pickle=False)

    # Given features standardixed by the whole mean and std, now let's make 
    # speaker independent standardized features.
    get_spk_independent_feats(utterances,
                              utt2target_feats,
                              processed_dirpath,
                              phone_type,
                              target_feats_suffix,
                              maxphonelen)

def get_target_feats(utterance, utterance_wav, alignments):
    '''Generates target features for an utterance (in this case duration, 
    initial phone fundamental frequency, final phone fundamental frequency, and
    energy'''
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

    #try:
    '''
    print(first_phone_start)
    print(first_phone_end)
    print(last_phone_start)
    print(last_phone_end)
    '''
    try:
        f_0_init = np.mean(pysptk.swipe(first_phone_samples.astype(np.float64), 
                                    fs=fs, 
                                    hopsize=100, 
                                    otype='f0'))
        #print(f_0_init)
        f_0_end = np.mean(pysptk.swipe(last_phone_samples.astype(np.float64), 
                                   fs=fs, 
                                   hopsize=100, 
                                   otype='f0'))
    except:
        print("Outof bounds!!")
        print(utterance)
        print(first_phone_start)
        print(first_phone_end)
        print(last_phone_start)
        print(last_phone_end)
        print(alignments)
        raise Exception("Out of bounds")
        return (0, 0, 0, 0)

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

def get_utterance_wavs(processed_dirpath, utterances):
    ''' Pull the actual samples of all utterances from wav files and store them
    in a dictionary'''
    utterance_wavs = {}
    for utterance in utterances:
        utterance_dirs = get_utterance_dirs(utterance)
        _, utterance_wav = wavfile.read(str(processed_dirpath 
                                         / utterance_dirs 
                                         / (utterance + '.wav')))

        utterance_wavs[utterance]=utterance_wav

    return utterance_wavs

def copy_id_values(orig_filepath, copied_filepath, idnts):
    '''Read in a Kaldi text file and create a dictionary from an id to 
    its values'''
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
    '''Read in a Kaldi text file and create a dictionary from an id to 
    its value (note there should be only one'''

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

def get_utt2flac(filepath):
    '''Read in a Kaldi wav.scp and get the original flac path'''
    utt2flac = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            # Under the assumption that this is a wav.scp file in a Kaldi
            # data directory under 'data'
            utt2flac[split_line[0]]=str(Path(filepath.parent.parent.parent.resolve(), split_line[-2]))
    return utt2flac


def get_spk2gender(filepath):
    '''Read in a Kaldi spk2gender file and create an equivalent dictionary'''
    spk2gender = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            spk2gender[split_line[0]]=split_line[1]
    return spk2gender


def get_phones(filepath):
    '''Read in a kaldi phones.txt file and create a list of phones as well as
    a dictionary from a phone to its given index'''
    idx2phones = []
    phones2idx = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            phones2idx[split_line[0]]=len(idx2phones)
            idx2phones.append(split_line[0])
    return idx2phones, phones2idx

def get_vocab(filepath):
    '''Read in a Kaldi vocabulary file and create a list of vocabulary words 
    and a dictionary from a word to its index'''
    idx2vocab = []
    vocab2idx = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            vocab2idx[split_line[0]]=int(split_line[1])
            if len(idx2vocab) != int(split_line[1]):
                print(split_line[0])
                print(split_line[1])
                raise Exception("The vocab file is corrupt and missing words")
            idx2vocab.append(split_line[0])
    return idx2vocab, vocab2idx

def get_lexicon(filepath):
    '''Read in a kaldi lexicon file and create a dictionary from a word to the 
    phones that compose it'''
    word2phones = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            word2phones[split_line[0]] = split_line[2:]
    return word2phones

def get_transcriptions(filepath):
    '''Read in a kaldi transcript and create a dictionary from an utterance to 
    the words that compose it.'''
    utt2words = {}
    with filepath.open() as open_file:
        for line in open_file:
            split_line = line.split()
            utt2words[split_line[0]] = split_line[1:]
    return utt2words

def get_alignments(alignments_filepath, idx2phones):
    '''Read in a Kaldi CTM alignments file (generated by ali-to-phones) and
    turn it into two dictionares: from utterance to list of phones and from 
    utterance to list of alignments'''
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
                   speakers,
                   duration_limit,
                   spk2gender,
                   word2phones,
                   utt2words,
                   utt2alignments):
    '''Filter out utterances by gender, duration, and speaker to get a final 
    list of utterances to be used in the preprocessed data'''
    utt2dur = {}
    # Get utterances and durations, filtering by gender and speaker
    with utterance_duration_filepath.open() as utterance_duration_file:
        for line in utterance_duration_file:
            split_line = line.split()
            utterance=split_line[0]
            spk = '-'.join(utterance.split('-')[0:-1])
            spkr = utterance.split('-')[0]
            curr_duration = split_line[1]
            if ((speakers and spkr in speakers) or
                not speakers):
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

def copy_and_convert_utterances(utterances, utt2dur, utt2flac, processed_dirpath):
    '''Given a list of utterances and their original flac filepaths, convert 
    them to wav format and copy to the new preprocessed directory.'''
    to_remove = []
    for utterance in utterances:
        utterance_dirnames = get_utterance_dirs(utterance)
        flac_file = utt2flac[utterance]
        flac, sr =sf.read(str(flac_file))
        if len(flac) != round(fs * utt2dur[utterance]):
            # There's an issue where some flac files are not being read 
            # properly (presumably an FFMPEG issue)
            print("IMPORTED SAMPLES FROM " + utterance + " DO NOT MATCH!")
            print(str(len(flac)) + '!=' +  str((fs * utt2dur[utterance])))
            to_remove.append(utterance)
            continue
        new_dir = (processed_dirpath / utterance_dirnames)
        if not new_dir.exists():
            new_dir.mkdir(parents=True)

        sf.write(str(new_dir / (utterance + '.wav')),
                 flac,
                 sr)
    for utt_to_remove in to_remove:
        print("Removing " + utt_to_remove + " from utterances")
        utterances.remove(utt_to_remove)
        utt2dur.pop(utt_to_remove)

def get_utterance_dirs(utterance):
    '''Given an utterance name, determine its parent and grandparent directory'''
    # TODO: Fix utterance_dirs to be tuple
    utterance_dirs = os.path.join(*(utterance.split('-')[0:-1]))
    return utterance_dirs

def get_nphones(utt2phones, utt2alignments, n):
    '''Given a dictionary of utterances to phones, convert them to nphones
    (mono/di/triphones)'''
    nphone_counts = Counter()
    utt2nphones = defaultdict(list)
    utt2nphone_alignments = defaultdict(list)
    for utterance, phones in utt2phones.items():
        idx = 0
        while idx+n < len(phones):
            nphone = tuple(phones[idx:idx+n])
            nphone_alignments = utt2alignments[utterance][idx:idx+n]
            nphone_counts[nphone] += 1
            utt2nphones[utterance].append(nphone)
            utt2nphone_alignments[utterance].append(nphone_alignments)
            idx += 1
    return utt2nphones, utt2nphone_alignments, nphone_counts

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
    '''Given an utterance, convert it to a list list of mono/di/triphones'''
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

def get_spk_independent_feats(utterances,
                              utt2feats, 
                              processed_dirpath, 
                              phone_type,
                              feats_filename,
                              maxphonelen):
    '''Standardize features according to speaker mean and std such that
    feat'=(feat-spk2mean[spk])/spk2std[spk]'''
    # First get target features for each speaker's utterances
    spkr2utt_feats = {}
    for utterance in utt2feats:
        spkr = utterance.split('-')[0]
        if spkr not in spkr2utt_feats:
            spkr2utt_feats[spkr] = []
        spkr2utt_feats[spkr].append(utt2feats[utterance])

    # Now get mean and std for each speaker
    spkr2mean={}
    spkr2std={}
    for spkr in spkr2utt_feats:
        spkr_utterances = spkr2utt_feats[spkr]
        #print(spkr_utt2feats.values())
        flattened = np.array([feats
                              for utterance in spkr_utterances
                                  for feats in utterance])
        #print("TYPE: " + str(flattened.dtype))
        mean = (np.mean(flattened, axis=0)).tolist()
        std = (np.std(flattened, axis=0)).tolist()
        spkr2mean[spkr] = mean
        spkr2std[spkr] = std
    #JSONize
    utils.save_json(spkr2mean, ((processed_dirpath 
                                 / (spkr2mean_prefix
                                    + phone_type
                                    + feats_filename + '.json'))))
    utils.save_json(spkr2std, ((processed_dirpath 
                                 / (spkr2std_prefix 
                                    + phone_type
                                    + feats_filename + '.json'))))

    # Now produce a numpy structure to store the normalized data,
    # for use in the neural network
    utterance_normalized_feats = []
    for utterance in utterances:
        spkr = utterance.split('-')[0]
        feats = utt2feats[utterance]
        normalized_feats = np.pad((np.array(feats)
                                   - np.array(spkr2mean[spkr]))
                                  / np.array(spkr2std[spkr]),
                                  ((0,maxphonelen-len(feats)),(0,0)),
                                  mode='constant')

        utterance_normalized_feats.append(normalized_feats)
    np_utterance_normalized_feats = np.asarray(utterance_normalized_feats)
    np.save(str(processed_dirpath / (spkr_independent_prefix + phone_type + feats_filename
                                     + normalized_suffix + '.npy')),
            np_utterance_normalized_feats)

'''
Old command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dur_limit', type=lambda d: int(d)*60*60*fs)
    parser.add_argument('--gender', type=str)
    parser.add_argument('--speakers', nargs='+', type=str)
    parser.add_argument('kaldi_data_dir', type=Path)
    parser.add_argument('alignment_dir', type=Path)
    parser.add_argument('processed_dir', type=Path)
    parser.add_argument('-m', '--mono_di_tri_phones', type=Path)

    args = parser.parse_args()
    main_args = {'kaldi_data_dirpath':args.data_dir,
                 'processed_dirpath':args.normalized_dir,
                 'aligned_dirpath':args.alignment_dir}
    if args.dur_limit:
        main_args['duration_limit']=args.dur_limit
    if args.gender:
        main_args['gender']=args.gender
    if args.mono_di_tri_phones:
        main_args['mono_di_tri_phones']=utils.load_json(args.mono_di_tri_phones)
    if args.speakers:
        main_args['speakers'] = set(args.speakers)
    preprocess(**main_args)
'''
