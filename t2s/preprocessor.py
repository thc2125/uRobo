import argparse
import math
import os
import shutil

from collections import defaultdict
from pathlib import Path

import numpy as np
import pysptk

from pydub import AudioSegment
from scipy.io import wavfile

phones_filename = 'phones.txt'
vocabulary_filename = 'vocab.txt'
alignments_filename = 'alignments.txt'
lexicon_filename = 'align_lexicon.txt'

male = 'm'
female = 'f'
all_dirname = 'all'

audio_rate = 16000
fs = audio_rate

def preprocess(data_dir, normalized_dir, dur_limit=(fs * 60 * 60 * 10)):
    spk2gender = convert_and_gender(data_dir, normalized_dir)
    idx2phone = [] 
    phone2idx = {}
    with (normalized_dir / 'all' / phones_filename).open('r') as phones_file:
        for line in phones_file:
            phone2idx[line.split()[0]] = len(idx2phone)
            idx2phone.append(line.split()[0])
            
    word2phones = {}
    with (normalized_dir / 'all' / lexicon_filename).open('r') as lexicon_file:
        for line in lexicon_file:
            split_line = line.split()
            word = split_line[0]
            phones = split_line[1:]
            word2phones[word] = phones

    # Preload utterance samples
    utterance_wavs = {}
    utterance_phones = []
    utterance_phone_feats = []
    utterance_phone_duration = []
    utterance_phone_f_0 = []
    utterance_phone_excitation = []
    utterance2idx = {}
    with (normalized_dir / 'all' / alignments_filename).open('r') as alignments_file:
        for line in alignments_file:
            split_line = line.split()
            utterance = split_line[0]
            if utterance not in utterance_wavs:
                utterance_dirs = os.path.join(*utterance.split('-')[:2])
                # Sample rate is assumed to be consistent
                _, utterance_wav = wavfile.read(str(normalized_dir / 'all' / utterance_dirs / (utterance + '.wav')))
                utterance_wavs[utterance] = utterance_wav
                utterance2idx[utterance] = len(utterance_phones)
                utterance_phones.append([])
                utterance_phone_feats.append([])
                '''
                utterance_phone_f_0.append([])
                utterance_phone_excitation.append([])
                utterance_phone_duration.append([])
                '''

    with (normalized_dir / 'all' / alignments_filename).open('r') as alignments_file:
        for line in alignments_file:
        # Alignments assumed to be in milliseconds
            split_line = line.split()
            utterance = split_line[0]
            utterance_idx = utterance2idx[utterance]
            spk = '-'.join(utterance.split('-')[0:-1])
            alignments = (float(split_line[2]), float(split_line[2]) + float(split_line[3]))

            phone = int(split_line[4])
            utterance_phones[utterance_idx].append(phone)

            duration, f_0_init, f_0_end, energy = get_feats(utterance_wavs[utterance], utterance, alignments)
            utterance_phone_feats[utterance_idx].append(np.array([duration, f_0_init, f_0_init, energy]))
            '''
            utterance_phone_duration[utterance_idx].append(f_0)
            utterance_phone_f_0[utterance_idx].append(excitation)
            utterance_phone_excitation[utterance_idx].append(excitation)
            '''
    max_utt_len = len(max(utterance_phones, key=len))
    pad_utterance_phones = [np.lib.pad(phones, (0, max_utt_len-len(phones)), 'constant', constant_values=0)
                            for phones in utterance_phones]
    pad_utterance_phone_feats = [feats + ([np.zeros(len(feats[0]))] * (max_utt_len -len(feats)))
                                 for feats in utterance_phone_feats]
    '''
    pad_utterance_phone_f_0 = [f_0s + ([f_0s[0].dtype.type(0)] * (max_utt_len -len(f_0s)))
                               for f_0s in utterance_phone_f_0]
    pad_utterance_phone_excitation = [excitations + ([excitations[0].dtype.type(0)] * (max_utt_len -len(excitations)))
                                      for excitations in utterance_phone_excitation]
    pad_utterance_phone_duration = [durations + ([durations[0].dtype.type(0)] * (max_utt_len -len(durations)))
                                    for durations in utterance_phone_duration]
    '''
    np_phones = np.stack(pad_utterance_phones)
    np_phone_feats = np.stack(pad_utterance_phone_feats)
    '''
    np_phone_f_0 = np.stack(pad_utterance_phone_f_0)
    print(np.max(np_phone_f_0))
    np_phone_duration = np.stack(pad_utterance_phone_duration)
    np_phone_excitation = np.stack(pad_utterance_phone_excitation)
    '''
    for i in range(0,10):
        print(np_phones2str(np_phones[i], idx2phone))

    np.save(str(normalized_dir / all_dirname / 'np_phones'), np_phones, allow_pickle=False)
    np.save(str(normalized_dir / all_dirname / 'np_phone_feats'), np_phone_feats, allow_pickle=False)

    '''
    np.save(str(normalized_dir / all_dirname / 'np_phone_f_0'), np_phone_f_0, allow_pickle=False)
    np.save(str(normalized_dir / all_dirname / 'np_phone_excitation'), np_phone_excitation, allow_pickle=False)
    np.save(str(normalized_dir / all_dirname / 'np_phone_duration'), np_phone_duration, allow_pickle=False)
    '''
    male_utterance_phones = []
    male_utterance_phone_feats = []
    '''
    male_utterance_phone_duration = []
    male_utterance_phone_f_0 = []
    male_utterance_phone_excitation = []
    '''
    male_dur = 0

    female_utterance_phones = []
    female_utterance_phone_feats = []
    '''
    female_utterance_phone_duration = []
    female_utterance_phone_f_0 = []
    female_utterance_phone_excitation = []
    '''
    female_dur = 0

    for utterance, idx in utterance2idx.items():
        spk = '-'.join(utterance.split('-')[0:-1])
        if spk in spk2gender and spk2gender[spk]==male and male_dur < dur_limit:
            male_utterance_phones.append(pad_utterance_phones[idx])
            male_utterance_phone_feats.append(pad_utterance_phone_feats[idx])
            '''
            male_utterance_phone_duration.append(pad_utterance_phone_duration[idx])
            male_utterance_phone_f_0.append(pad_utterance_phone_f_0[idx])
            male_utterance_phone_excitation.append(pad_utterance_phone_excitation[idx])
            '''
            male_dur += len(utterance_wavs[utterance])
        if spk in spk2gender and spk2gender[spk]==female and female_dur < dur_limit:
            female_utterance_phones.append(pad_utterance_phones[idx])
            female_utterance_phone_feats.append(pad_utterance_phone_feats[idx])
            '''
            female_utterance_phone_duration.append(pad_utterance_phone_duration[idx])
            female_utterance_phone_f_0.append(pad_utterance_phone_f_0[idx])
            female_utterance_phone_excitation.append(pad_utterance_phone_excitation[idx])
            '''
            female_dur += len(utterance_wavs[utterance])

    np_male_phones = np.stack(male_utterance_phones)
    np_male_phone_feats = np.stack(male_utterance_phone_feats)
    '''
    np_male_phone_duration = np.stack(male_utterance_phone_duration)
    np_male_phone_f_0 = np.stack(male_utterance_phone_f_0)
    np_male_phone_excitation = np.stack(male_utterance_phone_excitation)
    '''
    np_female_phones = np.stack(female_utterance_phones)
    np_female_phone_feats = np.stack(female_utterance_phone_feats)
    '''
    np_female_phone_duration = np.stack(female_utterance_phone_duration)
    np_female_phone_f_0 = np.stack(female_utterance_phone_f_0)
    np_female_phone_excitation = np.stack(female_utterance_phone_excitation)
    '''
    np.save(str(normalized_dir / male / 'np_phones'), np_male_phones, allow_pickle=False)
    np.save(str(normalized_dir / male /'np_phone_feats'), np_male_phone_feats, allow_pickle=False)
    '''
    np.save(str(normalized_dir / male /'np_phone_duration'), np_male_phone_duration, allow_pickle=False)
    np.save(str(normalized_dir / male /'np_phone_f_0'), np_male_phone_f_0, allow_pickle=False)
    np.save(str(normalized_dir / male /'np_phone_excitation'), np_male_phone_excitation, allow_pickle=False)
    '''
    np.save(str(normalized_dir / female / 'np_phones'), np_female_phones, allow_pickle=False)
    np.save(str(normalized_dir / female /'np_phone_feats'), np_female_phone_feats, allow_pickle=False)
    '''
    np.save(str(normalized_dir / female /'np_phone_duration'), np_female_phone_duration, allow_pickle=False)
    np.save(str(normalized_dir / female /'np_phone_f_0'), np_female_phone_f_0, allow_pickle=False)
    np.save(str(normalized_dir / female /'np_phone_excitation'), np_female_phone_excitation, allow_pickle=False)
    '''

def np_phones2str(np_phones, idx2phone):
    phonestr = ''
    for phone_idx in np_phones:
        phonestr += idx2phone[phone_idx] + ' '
    return phonestr

def get_feats(utterance_wav, utterance, alignments):
    #print(fs)
    #print(utterance)
    #print(len(utterance_wav))
    phone_start = int(alignments[0] * fs)
    #print("START: " + str(phone_start))
    phone_end = int(alignments[1] * fs)
    #print("END: " + str(phone_end))
    duration = phone_end - phone_start
    phone_samples = utterance_wav[phone_start:phone_end]
    try:
        phone_test = utterance_wav[phone_start]
    except:
        print("Outof bounds!!")
        print(utterance)
        print(phone_start)
        print(phone_end)
        print(alignments)
        return (0, 0, 0, 0)

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

def convert_and_gender(data_dir, normalized_dir):
    spk2gender = get_spk2gender(data_dir)
    flac2wav(data_dir, normalized_dir, spk2gender)
    dup_txt(data_dir, normalized_dir, spk2gender)
    return spk2gender

def get_spk2gender(data_dir):
    spk2gender = {}
    with (data_dir / 'spk2gender').open() as spk2gender_file:
        for line in spk2gender_file:
            split_line = line.split()
            spk2gender[split_line[0]] = split_line[1]
    return spk2gender

def flac2wav(data_dir, normalized_dir, spk2gender):
    flac_files = sorted(Path(data_dir).glob('**/*.flac'))

    for flac_file in flac_files:
        all_dir = Path(normalized_dir) / all_dirname / flac_file.relative_to(data_dir).parent

        if not all_dir.exists():
            all_dir.mkdir(parents=True)

        flac = AudioSegment.from_file(str(flac_file), 'flac')
        flac.export(str(all_dir
                        / (flac_file.stem + '.wav')),
                    format='wav')
        spk = '-'.join(flac_file.name.split('-')[0:-1])
        if spk in spk2gender:
            if spk2gender[spk] == male:
                m_dir = Path(normalized_dir) / male / flac_file.relative_to(data_dir).parent
                if not m_dir.exists():
                    m_dir.mkdir(parents=True)
                flac.export(str(m_dir
                                / (flac_file.stem + '.wav')),
                            format='wav')

            elif spk2gender[spk] == female:
                f_dir = Path(normalized_dir) / female / flac_file.relative_to(data_dir).parent
                if not f_dir.exists():
                    f_dir.mkdir(parents=True)
                flac.export(str(f_dir
                                / (flac_file.stem + '.wav')),
                            format='wav')
        else:
                m_dir = Path(normalized_dir) / male / flac_file.relative_to(data_dir).parent
                if not m_dir.exists():
                    m_dir.mkdir(parents=True)
                flac.export(str(m_dir
                                / (flac_file.stem + '.wav')),
                            format='wav')

                f_dir = Path(normalized_dir) / female / flac_file.relative_to(data_dir).parent
                if not f_dir.exists():
                    f_dir.mkdir(parents=True)
                flac.export(str(f_dir
                                / (flac_file.stem + '.wav')),
                            format='wav')


def dup_txt(data_dir, normalized_dir, spk2gender):
    txt_files = sorted(Path(data_dir).glob('**/*.txt'))
    for txt_file in txt_files:
        all_dir = Path(normalized_dir) / all_dirname / txt_file.relative_to(data_dir).parent
        m_dir = Path(normalized_dir) / male / txt_file.relative_to(data_dir).parent
        f_dir = Path(normalized_dir) / female / txt_file.relative_to(data_dir).parent

        if not all_dir.exists():
            all_dir.mkdir(parents=True)
        shutil.copy(str(txt_file), str(all_dir / txt_file.name))   
        spk = txt_file.name.split('.')[0]

        if spk in spk2gender:
            if spk2gender[spk] == male:
                if not m_dir.exists():
                    m_dir.mkdir(parents=True)
                shutil.copy(str(txt_file), str(m_dir / txt_file.name))   

            if spk2gender[spk] == female:
                if not f_dir.exists():
                    f_dir.mkdir(parents=True)
                shutil.copy(str(txt_file), str(f_dir / txt_file.name))   

        else:
            if not m_dir.exists():
                m_dir.mkdir(parents=True)
            shutil.copy(str(txt_file), str(m_dir / txt_file.name))   

            if not f_dir.exists():
                f_dir.mkdir(parents=True)
            shutil.copy(str(txt_file), str(f_dir / txt_file.name))   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dur_limit', type=lambda d: int(d)*60*60*fs)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('normalized_dir', type=Path)
    args = parser.parse_args()
    main_args = {'data_dir':args.data_dir, 'normalized_dir':args.normalized_dir}
    if args.dur_limit:
        main_args['dur_limit']=args.dur_limit
    preprocess(**main_args)
