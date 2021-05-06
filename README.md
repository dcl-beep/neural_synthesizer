# Raw Audio Classifier

### Example Usage

```python

from raw_audio_music_preprocessing import RawAudioDataPreProcess
from raw_audio_music_classifier_training import RawAudioClassifier

# make sure to set data directories in raw_audio_music_config.py !

# make sure your working directory has a similar structure:
#
# current_dir/
#				data/ music.wav, ...
#				output/
#						train/ ...
#						test/

raw = RawAudioDataPreProcess()

# this step is required if you have large audio files
# you want to slice up for training/testing data
# only need to run once
raw.run_slicing()

# run preprocessing -> pickles data for easy future loading 
raw.preprocess_and_pickle()

raw = RawAudioClassifier()
raw.setup_model()
# name it something that's easy to remember... :)
raw.train_model(model_name='m5_30ep_22050_10_real2')

# have fun!

```