#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    raw_audio_music_preprocessing.py

"""

import os
import pickle
from glob import iglob
import numpy as np
from pydub import AudioSegment


import os
import pickle
from glob import iglob
import numpy as np
import librosa
import soundfile as sf

from raw_audio_music_config import params
from raw_audio_music_config import music_class_ids


class RawAudioDataPreProcess(object):
    """
        #=================
        # data prep for raw audio classifier
        #================
    """
    def __init__(self, parameters=params, class_ids=music_class_ids):
        self._p= parameters
        self._class_ids = class_ids

        self._dataaudiodir = self._p['DATA_AUDIO_DIR']
        self._targetsr = self._p['TARGET_SR']
        self._audiolen = self._p['AUDIO_LENGTH']
        self._traindir = self._p['OUTPUT_DIR_TRAIN']
        self._testdir = self._p['OUTPUT_DIR_TEST']
        self._inputdir = self._p['INPUT']
        self._outputdir = self._p['OUTPUT']
        self._slicemin = self._p['SLICE_MIN_LEN']
        self._dreamout = self._p['FORDEEPDREAM']

    def get_slices_from_single_wav(self, class_abbrev, slices=150):
        """ given a class abbreviation referring to a big long wavfile
            open up file
            slice it into pieces of length [4s,14s]
            save as a single dat
        """
        wavfile = f"{ self._inputdir }/{class_abbrev}.wav"
        wavAudio = AudioSegment.from_wav(wavfile)

        #Export all of the individual chunks as wav files
        beginning=0
        
        for i in range(slices):
            # get slice length, slice id, slice idx
            l = (np.random.random()+1)* self._slicemin
            rand_id = str(l).replace('.','')[:8]
            end = beginning + self._slicemin
            
            audioSlice = wavAudio[beginning:end]
            beginning = end
            slice_filename = f'{ self._outputdir }/{class_abbrev}_{rand_id}.wav'
            
            print(f"exporting {slice_filename}")
            
            #Exports to a wav file in the current path.
            audioSlice.export(slice_filename, format="wav")

        print('........done')

    def run_slicing(self):
        for id in os.listdir( self._inputdir ):
            name = id.replace('.wav','')
            self.get_slices_from_single_wav(name)

    def extract_class_id_music(self, wav_filename):
        if 'ce' in wav_filename:
            return self._class_ids.get('ce')
        elif 'bn' in wav_filename:
            return self._class_ids.get('bn')
        elif 'up' in wav_filename:
            return self._class_ids.get('up')
        elif 'sc' in wav_filename:
            return self._class_ids.get('sc')
        elif 'rh' in wav_filename:
            return self._class_ids.get('rh')
        else:
            return self._class_ids.get('unlabelled')


    def read_audio_from_filename(self, filename, target_sr):
        """ given filename, loads wav file, given sample rate
            returns audio (np.array)
        """
        audio, _ = librosa.load(filename, sr=target_sr, mono=True)
        audio = audio.reshape(-1, 1)
        return audio

    def preprocess_and_pickle(self):
        """ runs preprocessing on the data slices
            stores each audio sample in a pickle container along with
            corresponding target label

        """
        j=0

        for wav_filename in os.listdir( self._dataaudiodir ):
            print(f"Working on {wav_filename}")
            class_id = self.extract_class_id_music(wav_filename)
            audio_buf = self.read_audio_from_filename(f"{self._dataaudiodir}/{wav_filename}", target_sr=self._targetsr)
            audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
            original_length = len(audio_buf)
            print(j, wav_filename, original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
            if original_length < self._audiolen:
                audio_buf = np.concatenate((audio_buf, np.zeros(shape=(self._audiolen - original_length, 1))))
                print('PAD New length =', len(audio_buf))
            elif original_length > self._audiolen:
                audio_buf = audio_buf[0:self._audiolen]
                print('CUT New length =', len(audio_buf))

            output_folder = self._traindir
            if j // 30 == 0:
                output_folder = self._testdir

            output_filename = os.path.join(output_folder, str(j) + '.pkl')

            out = {'class_id': class_id,
                   'audio': audio_buf,
                   'sr': self._targetsr}
            with open(output_filename, 'wb') as w:
                pickle.dump(out, w)
                
            j+=1

    def preprocess_original(self, filename):
        """ to run fresh new data through deepdream, it first has to be processed
            just as the training data was.
        """
        filenamefull = './snippets/' + filename
        audio_buf = self.read_audio_from_filename(filenamefull, target_sr=self._targetsr)
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        if original_length < self._audiolen:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(self._audiolen - original_length, 1))))
            print('PAD New length =', len(audio_buf))
        elif original_length > self._audiolen:
            audio_buf = audio_buf[0:self._audiolen]
            print('CUT New length =', len(audio_buf))
            
        sf.write(self._dreamout+filename, audio_buf, self._targetsr, subtype='FLOAT')


if __name__ == '__main__':

    raw = RawAudioDataPreProcess()

    # only need to run once
    raw.run_slicing()

    # run preprocessing
    raw.preprocess_and_pickle()