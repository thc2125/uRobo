from collections import defaultdict
from pathlib import Path

import pysptk

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

'''
def get_concat_feats(utterance_wav, utterance, alignments):
    fs = 16000
    phone_start = int(alignments[0] * fs)
    phone_end = int(alignments[1] * fs)

    phone_samples = utterance_wav[phone_start:phone_end]
'''
