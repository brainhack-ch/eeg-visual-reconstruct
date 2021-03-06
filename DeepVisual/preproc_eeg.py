#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:26:22 2019

@author: loriette
"""
import os
import numpy as np
import mne
import pandas as pd
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

raw=mne.io.read_raw_edf('/home/loriette/Documents/Brainhack geneva/DATA/S01/Session1/NeoRec_2018-10-04_13-28-04.edf')

print(raw)
raw.ch_names;
data = pd.read_csv("/home/loriette/Documents/Brainhack geneva/DATA/S01/Session1/ann.csv")
loc= pd.read_csv("/home/loriette/Documents/Brainhack geneva/DATA/sensorsPosition.csv",header=None)
loc=np.array(loc)
print(loc)

new_data=[data.labels[0] , data.onsets_start[0] , 3];
I=0;
for i in range(0,len(data.onsets_end)):
    
    while(data.onsets_start[i]+I+3) < data.onsets_end[i]:
        new_data=np.vstack((new_data,[data.labels[i] , data.onsets_start[i]+I , 3]))
        I=I+1
    I=0;
        
        
new_data=np.delete(new_data,obj=0,axis=0)
my_annot = mne.Annotations(onset=new_data[:,1],
                           duration=new_data[:,2],
                           description=new_data[:,0])
raw.set_annotations(my_annot)

montage=mne.channels.read_montage('/home/loriette/Documents/Brainhack geneva/DATA/sensorsPosition.txt')
new_names=['Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6','Ft10','T7','C3','Cz','C4','T8','Tp9','Cp5',  'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Fpz', 'Af7', 'Af3', 'Af4', 'Af8',  'F5', 'F1', 'F2', 'F6', 'Ft7', 'Fc3', 'Fcz', 'Fc4', 'Ft8', 'C5', 'C1', 'C2', 'C6', 'Tp7', 'Cp3', 'Cpz', 'Cp4', 'Tp8', 'P5', 'P1', 'P2', 'P6', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'Aff1h', 'Aff2h', 'F9', 'F10', 'Ffc5h', 'Ffc1h', 'Ffc2h', 'Ffc6h', 'Ftt7h', 'Fcc3h', 'Fcc4h', 'Ftt8h', 'Ccp5h', 'Ccp1h', 'Ccp2h', 'Ccp6h', 'Tpp7h', 'Cpp3h', 'Cpp4h', 'Tpp8h', 'P9', 'P10', 'Ppo9h', 'Ppo1h', 'Ppo2h', 'Ppo10h', 'Po9', 'Po10', 'I1', 'Oi1h', 'Oi2h', 'I2', 'Afp1', 'Afp2', 'Aff5h', 'Aff6h', 'Fft9h', 'Fft7h', 'Ffc3h', 'Ffc4h', 'Fft8h', 'Fft10h', 'Ftt9h', 'Fcc5h', 'Fcc1h', 'Fcc2h', 'Fcc6h', 'Ftt10h', 'Ttp7h',  'Ccp3h', 'Ccp4h', 'Ttp8h', 'Tpp9h', 'Cpp5h', 'Cpp1h', 'Cpp2h', 'Cpp6h', 'Tpp10h', 'Ppo5h', 'Ppo6h', 'Poo9h', 'Poo1',  'Poo2', 'Poo10h', 'Diff 2', 'Diff 3','Diff 4']
old_names=raw.ch_names;
for i in range(0,len(new_names)):
    raw.rename_channels({old_names[i]: new_names[i]})


raw.ch_names
raw.set_montage(montage)
raw.info['bads']=['Diff 2' , 'Diff 3' ,'Diff 4']
print(raw.annotations)
events, _ = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, tmin=0, tmax=3, preload=True,baseline=None)

filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)

ica = ICA(n_components=40, random_state=97)
ica.fit(filt_raw)

raw.load_data()
ica.plot_sources(raw)

ica.plot_components()

raw.plot_sensors()


# define frequencies of interest (log-spaced)
freqs = range(1,201)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
