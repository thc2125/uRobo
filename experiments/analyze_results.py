import argparse
import csv
import re

from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

voices = {'c': 'Control (Human)', 't10f':'10 hours, female','ts39':'Single Speaker, Female','ts39_m':'Single Speaker, Female, monophones'}
t10f = re.compile('t10f')
ts39 = re.compile('ts39')
ts39_m = re.compile('ts39_m')

sent_lens = {5:9, 6:9, 7:11, 8:18, 9:20}
def analyze(results_file):
    workers = Counter()
    humanity_scores = defaultdict(list)
    intelligibility_scores = defaultdict(list)
    smoothness_scores = defaultdict(list)

    word_humanity = defaultdict(list)
    word_int = defaultdict(list)
    word_smooth = defaultdict(list)

    sent_humanity = defaultdict(list)
    sent_int = defaultdict(list)
    sent_smooth = defaultdict(list)

    with results_file.open() as results:
        reader = csv.DictReader(results)
        for row in reader:
            audio_file_comps = row['Input.audio_name'].split('_')
            utterance = int(audio_file_comps[0])
            if len(audio_file_comps) == 4:
                voice = '_'.join(audio_file_comps[-2:])
            else:
                voice = audio_file_comps[-1]
            humanity_scores[voice].append(int(row['Answer.Human']))
            intelligibility_scores[voice].append(int(row['Answer.Intelligible']))
            smoothness_scores[voice].append(int(row['Answer.Smoothness']))
            workers[row['WorkerId']] += 1

            if voice == 'ts39':
                # Word-level
                if utterance < 5:
                    word_humanity[utterance+1].append(int(row['Answer.Human']))
                    word_int[utterance+1].append(int(row['Answer.Intelligible']))
                    word_smooth[utterance+1].append(int(row['Answer.Smoothness']))

                else: 
                    sent_humanity[sent_lens[utterance]].append(int(row['Answer.Human']))
                    sent_int[sent_lens[utterance]].append(int(row['Answer.Intelligible']))
                    sent_smooth[sent_lens[utterance]].append(int(row['Answer.Smoothness']))

    print(len(workers))
    print(str(sum(workers.values())/len(workers)))
    print(sorted(workers.values(), reverse=True))
    # Code for the figure inspired by the matplotlib bar chart demo
    # https://matplotlib.org/gallery/statistics/barchart_demo.html
    humanity_means = {voice: float(np.mean(scores)) for voice, scores in humanity_scores.items()}
    humanity_stds = {voice: float(np.std(scores)) for voice, scores in humanity_scores.items()}

    intelligibility_means = {voice: float(np.mean(scores)) for voice, scores in intelligibility_scores.items()}
    intelligibility_stds = {voice: float(np.std(scores)) for voice, scores in intelligibility_scores.items()}

    smoothness_means = {voice: float(np.mean(scores)) for voice, scores in smoothness_scores.items()}
    smoothness_stds = {voice: float(np.std(scores)) for voice, scores in smoothness_scores.items()}


    xlabels = sorted(humanity_means.keys())
    print(xlabels)
    yhumanity = [mean for _, mean in sorted(humanity_means.items())]
    yintelligibility = [mean for _, mean in sorted(intelligibility_means.items())]
    ysmoothness = [mean for _, mean in sorted(smoothness_means.items())]

    errhumanity = [std for _, std in sorted(humanity_stds.items())]
    errintelligibility = [std for _, std in sorted(intelligibility_stds.items())]
    errsmoothness = [std for _, std in sorted(smoothness_stds.items())]


    index = np.arange(len(xlabels))
    bar_width = 0.25

    fig, ax = plt.subplots()
    humanity = ax.bar(index+bar_width, yhumanity, bar_width, label='Humanity', yerr=errhumanity)

    intelligibility = ax.bar(index, yintelligibility, bar_width, label='Intelligibility', yerr=errintelligibility)
    smoothness = ax.bar(index+(2*bar_width), ysmoothness, bar_width, label='Smoothness', yerr=errsmoothness)

    ax.set_title('Voice Rankings by Mechanical Turk Workers')
    ax.set_xticks(index + (2*bar_width / 3))
    ax.set_xlabel('Voice')
    ax.set_ylabel('Mean Score')
    ax.set_xticklabels(xlabels)
    ax.legend()

    fig.tight_layout()
    plt.savefig('experiments/results/all_voices_results.png')
    #plt.show()

    # Let's do words
    fig, ax = plt.subplots()

    syllables = np.array(sorted(word_humanity.keys()))
    word_hum = np.array([sum(hum)/len(word_humanity) for comp, hum in sorted(word_humanity.items())])
    word_int = np.array([sum(hum)/len(word_int) for comp, hum in sorted(word_int.items())])
    word_smooth = np.array([sum(hum)/len(word_smooth) for comp, hum in sorted(word_smooth.items())])

    hum, intel, smooth = ax.plot(syllables, word_hum, syllables, word_int, syllables, word_smooth)
    plt.setp(hum, label='Humanity')
    plt.setp(intel, label='Intelligibility')
    plt.setp(smooth, label='Smoothness')

    ax.set_xticks(syllables)
    ax.set_title('Performance of ts39 over words with increasing complexity')
    ax.set_xlabel('# of Syllables')
    ax.set_ylabel('Mean Score')
    ax.set_xticklabels(sorted(word_humanity.keys()))
    ax.legend()
    fig.tight_layout()
    plt.savefig('experiments/results/ts39_word_results.png')
    plt.show()

    # Let's do sentences
    fig, ax = plt.subplots()

    words = np.array(sorted(sent_humanity.keys()))
    sent_hum = np.array([sum(hum)/len(sent_humanity) for comp, hum in sorted(sent_humanity.items())])
    sent_int = np.array([sum(hum)/len(sent_int) for comp, hum in sorted(sent_int.items())])
    sent_smooth = np.array([sum(hum)/len(sent_smooth) for comp, hum in sorted(sent_smooth.items())])

    hum, intel, smooth = ax.plot(words, sent_hum, words, sent_int, words, sent_smooth)
    plt.setp(hum, label='Humanity')
    plt.setp(intel, label='Intelligibility')
    plt.setp(smooth, label='Smoothness')

    ax.set_xticks(words)
    ax.set_title('Performance of ts39 over sentences with increasing complexity')
    ax.set_xlabel('# of Words')
    ax.set_ylabel('Mean Score')
    ax.set_xticklabels(sorted(sent_humanity.keys()))
    ax.legend()
    fig.tight_layout()
    plt.savefig('experiments/results/ts39_sent_results.png')
    plt.show()
   
    

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('results_file', type=Path)
   args = parser.parse_args()
   analyze(args.results_file) 
