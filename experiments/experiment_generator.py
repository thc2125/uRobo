import argparse
import itertools

from pathlib import Path

from concatenative_model.concat_nn import NNConcatenator

# Test over the 10 hour, multi-speaker data set and the ~30 minute, 1-speaker 
# data set
data_dirs = [Path('data/train-clean-spk39'), Path('data/train-clean-10-f')]
#data_dirs = [Path('data/train-clean-spk39')]
#data_dirs = [Path('data/train-clean-10-f')]

model = Path('models/train-clean-10-f-lin-20e/model.h5')

synthesizers = []
for data_dir in data_dirs:
    # Test when using monophones (True) and when using triphones (False)
    #for mono in (True,False):
    for mono in (True,False):
        if mono and data_dir.name == 'train-clean-10-f':
            continue
        print("Building " 
              + data_dir.name 
              + ("_mono" if mono else '')
              + " synthesizer")
        synthesizers.append(NNConcatenator(
            data_dir=data_dir,
            target_predicter_model_path=model,
            mono=mono))

output_path = Path('experiments','audio')
if not output_path.exists():
    output_path.mkdir()

for synthesizer in synthesizers:
    with open('experiments/experiment_text.txt') as experiment_text_file:
        line_no = 0
        for line in experiment_text_file:
            filename = str(line_no)
            print("Synthesizing '" 
                  + line 
                  + "' with " 
                  + synthesizer.data_dir.name 
                  + ("_mono" if synthesizer.mono else ''))
            filename += '-' + synthesizer.data_dir.name
            filename += '_mono' if synthesizer.mono else ''
            filename += '.wav'
            synthesizer.synthesize(text=line, 
                                   output_path=output_path / filename)
            line_no += 1
