#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	raw_audio_music_config.py


	contains parameters used by data preprocessing and classifier training

"""
import os

outdir = './output_22050_10'

params = {

	'DATA_AUDIO_DIR' : './data_sliced',
	'TARGET_SR' : 22050,
	'OUTPUT_DIR' : outdir,
	'OUTPUT_DIR_TRAIN' : os.path.join(outdir, 'train'),
	'OUTPUT_DIR_TEST' : os.path.join(outdir, 'test'),
	'AUDIO_LENGTH' : 10000,
	'INPUT' : 'audio',
	'OUTPUT' : 'data_sliced',
	'SLICE_MIN_LEN' : 10000,
	'SLICE_MAX_LEN' : 10000,
	'FORDEEPDREAM' : './presnips/'

}

music_class_ids = {
    'ce' : 0,
    'bn' : 1,
    'up' : 2,
    'sc' : 3,
    'rh' : 4
}

