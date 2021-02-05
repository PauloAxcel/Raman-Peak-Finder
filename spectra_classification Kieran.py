# -*- coding: utf-8 -*-
"""spectra_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xHSwj49F4jksW5diiEUi_Kb_FKxyf8cV

# Classifying Spectra
---
Let's see what we can do!

- ~~Load data~~
- ~~Average~~
- Baseline
- Remove noise
- Smooth
- Lorentz
"""

pip install -q rampy

import rampy as rp
import numpy as np

def load_reference_csv(fname):
    """
    Load a csv file of the form
    name, peak_1, peak_2, ...
    params:
        fname: Name of the file to read data from
    returns:
        List of tuples of the form [(name, [peak_1, peak_2, ... ])]
    """
    with open(fname, 'r') as data_file:
        data = []
        for line in data_file.readlines():
            line = line.replace("\n", '')
            line = line.split(",")
            line = list(filter(lambda x: x != '', line))
            name = line[0]
            peaks = list(map(lambda x: int(x), line[1:]))
            peaks.sort()
            data.append((name, peaks))

    return data

def load_sample_data(fname):
    """
    Load a csv file of the form
    X, Y, Wave Intensity
    params:
        fname: Name of the file to read data from
    returns:
        List of tuples of the form [(wave, intensity)]
    """
    with open(fname, 'r') as data_file:
        data = {}
        data_file.readline()
        for line in data_file.readlines():            
            line = line.replace("\n", '')
            line = line.split("\t")
            try:
                data[float(line[2])].append(float(line[3]))
            except Exception as e:
                data[float(line[2])] = [float(line[3])]

    tmp = []
    for k in data.keys():
        tmp.append((k, sum(data[k])/len(data[k])))

    data = tmp
    return data

ref_data = load_reference_csv("Saliva Reference.csv")
sample_data = load_sample_data("Raman Example 2.txt")

ref_data.sort(key=lambda x: x[1][0])
print(ref_data)
print(sample_data)

avg_wave_num = list(map(lambda x: x[0], sample_data))
avg_intensity = list(map(lambda x: x[1], sample_data))

import matplotlib.pyplot as plt

plt.plot(avg_wave_num, avg_intensity)
plt.show()

smoothed_intensity = rp.smooth(np.asarray(avg_wave_num), np.asarray(avg_intensity), method="whittaker", Lambda=100**0.5)
baseline_intensity = rp.baseline(np.asarray(avg_wave_num), smoothed_intensity, bir=np.array([[100.,200.],[500.,600.]]), method="als") # NOTE: bir is not used for als, but dummy values required for the function to work!
plt.figure(figsize=(9,9/1.618))
plt.plot(avg_wave_num, baseline_intensity[0], lw=0.5)
plt.show()

baseline_intensity2 = rp.baseline(np.asarray(avg_wave_num), np.asarray(avg_intensity), bir=np.array([[100.,200.],[500.,600.]]), method="als")
plt.figure(figsize=(9,9/1.618))
plt.plot(avg_wave_num, baseline_intensity2[0],lw=0.5)

class PeakRange():
    
    def __init__(self, peaks, tau):
        self.peaks = peaks
        self.tau = tau
        self.min = peaks[0]
        self.max = peaks[len(peaks)-1]
        self.ranges = None

        if self.max != self.min:
            self.generate_ranges()
        else:
            self.ranges = [[self.min]]

    def generate_ranges(self):
        # TODO: Make more efficient with convolution!
        prev = self.peaks[0]
        ranges = []

        for idx, p in enumerate(self.peaks[1:]):
            # idx lags one behind which is perfect
            if p <= self.peaks[idx]+tau:
                pass
            else:
                ranges.append([prev, self.peaks[idx]]) if prev != self.peaks[idx] else ranges.append([prev])
                prev = p
            idx += 1
            if idx == len(self.peaks)-1:
                ranges.append([prev, p]) if prev != p else ranges.append([p])
            
            self.ranges = ranges

    
    def check_for_existence(self, wave_num):
        # TODO: Early stopping
        assert self.ranges != None, "ERROR: Range is None!"

        for p_range in self.ranges:
            if len(p_range) == 1:
                if p_range[0]-tau < wave_num < p_range[0]+tau:
                    return True
            else:
                if p_range[0]-tau < wave_num < p_range[1]+tau:
                    return True
        return False

# Extract the ranges
tau = 5.0
peak_id = []
for name, vals in ref_data:
    print(name)
    print(vals)
    peak_id.append((name, PeakRange(vals, tau)))

for wave_num, _ in sample_data:
    for name, peak in peak_id:
        if peak.check_for_existence(wave_num):
            print("Wave Number: {} belongs to: {}".format(wave_num, name))