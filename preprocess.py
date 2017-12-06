import argparse
import os
import random
import shutil

import json

from pathlib import Path

# Define constants
phones_filename = 'phones'
utterances_filename = 'utterances'
vocabulary_filename = 'vocab'
alignments_filename = 'alignments'
lexicon_filename = 'align_lexicon'
spk2gender_filename = 'spk2gender'
transcriptions_filename = 'text'
utterance_duration_filename = 'utt2dur'

male = 'm'
female = 'f'

fs=16000

def preprocess(orig_dirpath, processed_dirpath, duration=None, gender=None):

    if not processed_dirpath.exists():
        processed_dirpath.mkdir(parents=True)

    utterances = copy_base_data(orig_dirpath, 
                                processed_dirpath, 
                                duration=None,
                                gender=None)

    process_data(processed_dirpath, utterances)

def copy_base_data(orig_dirpath, 
                   processed_dirpath, 
                   duration=None,
                   gender=None):

    spk2gender = get_idnt_value((orig_dirpath / spk2gender_filename + '.txt'))
    phones = get_idnt((orig_dirpath / phones_filename + '.txt'))
    vocabulary = get_idnt((orig_dirpath / vocabulary_filename + '.txt'))
    lexicon = get_idnt_values((orig_dirpath / lexicon_filename + '.txt'), lists=True)

    utterances, spks, utterance2duration, curr_duration = get_utterances(
            (orig_dirpath / utterance_duration_filename),
            spk2gender, 
            gender,
            duration)

    transcriptions = get_idnt_value((orig_dirpath / transcriptions_filename +
        '.txt'), lists=True)

    # Copy and convert audio from flac to wav
    copy_and_convert_utterances(utterances)

    # Copy phone list
    shutil.copy(str(orig_dirpath / phones_filename + '.txt'), 
                str(processed_dirpath / phones_filename + '.txt'))
    # JSONize phone list
    with (processed_dirpath / phones_filename + '.json').open as phones_file:
        json.dump(phones, phones_file, indent=4)

    # Copy vocabulary
    shutil.copy(str(orig_dirpath / vocabulary_filename + '.txt'), 
                str(processed_dirpath / vocabulary_filename + '.txt'))
    # JSONIze vocabulary
    with (processed_dirpath / vocabulary_filename + '.json').open as
    vocabulary_file:
        json.dump(vocabulary, vocabulary_file, indent=4)

    # Copy lexicon
    shutil.copy(str(orig_dirpath / lexicon_filename), 
                str(processed_dirpath / lexicon_filename))
    # JSONIze lexicon
    with (processed_dirpath / lexicon_filename + '.json').open as lexicon_file:
        json.dump(lexicon, lexicon_file, indent=4)

    # JSONIze utterances
    with (processed_dirpath / utterances_filename + '.json').open as utterance_file:
        json.dump(utterances, utterance_file, indent=4)

    # Copy transcriptions
    copy_id_values(orig_dirpath / transcriptions_filename + '.txt', 
                   processed_dirpath / transcriptions_filename,
                   utterances)


    # Copy spk2gender
    shutil.copy(str(orig_dirpath / spk2gender_filename + '.txt'),
                str(processed_dirpath / spk2gender_filename + '.txt'))
    # JSONize spk2gender
    with (processed_dirpath / spk2gender_filename + '.json').open as spk2gender_file:
        json.dump(spk2gender, spk2gender_file, indent=4)



    # Copy alignments
    copy_id_values(orig_dirpath / alignments_filename + '.txt', 
                   processed_dirpath / alignments_filename,
                   utterances)


def process_data(processed_dirpath, utterances):
    utterance2phones, utterance2alignments = get_alignment_data(
            processed_dirpath / alignments_filename)

    transcriptions = get_idnt_value(processed_dirpath / transcriptions_filename,
                                    lists=True)

    idx2phones = get_idnt((processed_dirpath / phones_filename + '.txt'))
    phones2idx = {idx2phones[idx]: idx 
                      for idx in range(len(idx2vocabulary))}

    idx2vocabulary = get_idnt((processed_dirpath / vocabulary_filename + '.txt'))
    vocabulary2idx = {idx2vocabulary[idx]: idx 
                      for idx in range(len(idx2vocabulary))}

    # Get the features of each utterance
    utterance_wavs = get_utterance_wavs(processed_dirpath, utterances)
    utterance2target_feats=defaultdict(list)
    utterance2concat_feats=defaultdict(list)
    for utterance in utterance:
        target_feats = get_target_feats(utterance_wavs[utterance])
        concat_feats = get_concat_feats(utterance_wavs[utterance])
        utterance2target_feats[utterance].append(target_feats)
        utterance2concat_feats[utterance].append(concat_feats)

    # Save the necessary data as numpy arrays




def get_alignment_data(alignments_file):
    utterance2phones = defaultdict(list) 
    utterance2alignments = defaultdict(list)
    with alignments_file.open():
        for line in alignments_file:
            split_line = line.split()
            utterance = split_line[0]
            utterance2phones[utterance].append(split_line[-1])
            utterance2alignments[utterance].append((int(unit_idx) for phone_idx in
                    split_line[2:4]))

    return utterance2phones, utterance2alignments

def get_utterance_wavs(processed_dirpath, utterances):
    utterance_wavs = {}
    for utterance in utterances:
        utterance_dirs = get_utterance_dirs(utterance)
        utterance_wav = wavfile.read(str(processed_dirpath 
                                         / utterance_dirs 
                                         / (utterance + '.wav')))

        utterance_wavs[utterance]=utterance_wav

    return utterance_wavs


def copy_id_values(orig_filepath, copied_filepath, idnts):
    with orig_filepath.open() as orig_file:
      with copied_filepath.open() as copied_file:
        for line in orig_file:
            split_line = line.split()
            idnt = split_line[0]
            if idnt in idnts:
                idnts2values[idnt] = split_line[1:]
                copied_file.write(line)

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

def get_idnt(filepath, idnt_type=str):
    idnts = []
    with filepath.open() as open_file:
        for line in open_file:
            idnts.add(idnt_type(line))
    return idnts


def get_utterances(utterance_duration_filepath,
                   spk2gender,
                   gender,
                   duration):

    utterance2duration = {}  
    # Get utterances and durations, filtering by gender
    with utterance_duration_filepath.open() as utterance_duration_file:
        for line in utterance_duration_file:
            split_line = line.split()
            utterance=split_line[0]
            spk = '-'.join(utterance.split('-')[0:-1])
            duration = split_line[1]
            if ((gender and spk2gender[spk] == gender) or
                 not gender):
                utterance2duration[utterance]=duration


    curr_duration=0
    utterances = set() 
    spks = set()
    # Get a random selection of utterances <= a duration limit
    for utterance in random.shuffle(list(utterance2duration.keys())):
        if duration and (curr_duration + utterance2duration[utterance]) <= duration:
            utterances.add(utterance)
            spks.add('-'.join(utterance.split('-')[0:-1]))
            curr_duration += utterance2duration[utterance]
    return utterances, spks, utterance2duration, curr_duration

def copy_and_convert_utterances(utterances, orig_dirpath, processed_dirpath):
    for utterance in utterances:
        utterance_dirnames = get_utterance_dirs(utterance))
    flac_file = orig_dirpath / utterance_dirnames / (utterance + '.flac')
    flac = AudioSegment.from_file(str(flac_file), 'flac')
    flac.export(str(all_dir
                    / (flac_file.stem + '.wav')),
                format='wav')

def get_utterance_dirs(utterance):
    # TODO: Fix utterance_dirs to be tuple
    utterance_dirs = os.path.join(utterance.split('-')[0:-1])
    return utterance_dirs

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
