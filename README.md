# Multi-Stage-Multi-Target-WaveNet-for-the-ICASSP-2024-Auditory-EEG-Challenge-2024
This is the code implementation for Multi-Stage-Multi-Target-WaveNet for the ICASSP 2024 Auditory EEG Challenge.Our solution (2nd Place) for the ICASSP 2024 Signal Processing Grand Challenge - Auditory EEG Decoding Challenge (Task 2).

##  preprocessrawdata 
This is the code used to process raw data to generate wav files. In the official processing, the generated envelope and mel spectrum parts are removed, and only the 16kHz WAV file is saved.If you want to try this model,you need to run  genwav.py and splitwav.py to generate the corresponding 16k wav data.

## train.py
This is the training code.

## test.py
This is the code used to generate the final submission file.
