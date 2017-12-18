import argparse
import itertools

from pathlib import Path

from concatenative_model.concat_nn import NNConcatenator

# Test over the 10 hour, multi-speaker data set and the ~30 minute, 1-speaker 
# data set
data_dirs = [Path('data/train-clean-10-f'), Path('data/train-clean-spk39')]
model = Path('models/train-clean-10-f/model.h5')

synthesizers = []
for data_dir in data_dirs:
    # Test when using monophones (True) and when using triphones (False)
    for mono in (True,False):
        synthesizers.append(NNConcatenator(
            data_dir=data_dir,
            target_predicter_model_path=model,
            mono=mono))

output_path = Path('experiments','audio')
if not output_path.exists():
    output_path.mkdir()

with open('experiment_text.txt') as experiment_text_file:
    line_no = 0
    for line in experiment_text_file:
        filename = line_no
        for synthesizer in synthesizers:
            filename += synthesizer.data_dir.name
            filename += '_mono' if synthesizer.mono else ''
            filename += '.wav'
            synthesizer.synthesize(text=line, 
                                   output_path=output_path / filename)
        line_no += 1
