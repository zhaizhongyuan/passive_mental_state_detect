from scipy.io import loadmat
import numpy as np
import pandas as pd
import csv
import pickle
import os
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
import scipy.io

import networkx as nx
import matplotlib.pyplot as plt
import io
# import community as community_louvain
import matplotlib.cm as cm
from collections import defaultdict


def get_EEG_data(root, file):
    """
    Args:
      root: address of the root data
      file: the file to be read
    Returns:
      a pandas dataframe given the .dat data file
    """
    # Extract the data from one of these files.
    data_path = os.path.join(root, file)
    mat = scipy.io.loadmat(data_path)
    data = pd.DataFrame.from_dict(mat["o"]["data"][0, 0])

    # Limit the data to the 7 valid EEG leads.
    dat = data.filter(list(range(3, 17)))
    dat.columns = list(range(1, 15))
    dat = dat.filter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17], axis=1)
    labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    dat.columns = labels

    # print(dat)

    # Extract only 7 channels
    dat = dat[['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4']]

    return dat


def channel_processing(dat_ch, exp, channel):
    """
    NFFT: the number of data points used in each block for the FFT; m = 1024 fast discrete Fourier transform (DFT)
    Fs: sampling frequency of Fs = 128 Hz
    """
    fs = 128
    n_fft = 1024
    n_overlap = 896
    # Compute ST-DFT using 8-second blackman window, 1-second time step.
    # Zxx has frequencies as rows in ascending order, and times as columns in ascending order
    f, t, Zxx = scipy.signal.stft(dat_ch, fs=fs, window=np.blackman(n_fft), nperseg=n_fft, noverlap=n_overlap,
                                  nfft=n_fft)

    # Plot spectrogram
    plt.figure()
    plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    plt.colorbar(use_gridspec=True)
    plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' spectrogram by 8-second blackman window')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman.png')
    plt.show()

    # Discard DC component on 1st row
    Zxx = Zxx[1::, :]
    n_freq, _ = Zxx.shape

    # Bin frequencies into 0.5Hz apart
    bin_width = 0.5
    # Number of frequency components to bin
    freq_step = int(bin_width / (fs / n_fft))  # Type cast from float to int
    # Binning
    Zxx_s = Zxx[0:0 + freq_step, :]
    Zxx_s_bin = np.mean(Zxx_s, 0)
    Zxx_temp = Zxx_s_bin
    for i in range(freq_step, n_freq, freq_step):
        Zxx_s = Zxx[i:i + freq_step, :]
        Zxx_s_bin = np.mean(Zxx_s, 0)
        Zxx_temp = np.vstack((Zxx_temp, Zxx_s_bin))
    Zxx = Zxx_temp

    # ST-IDFT to reconstruct time-domain signal
    n_fft = n_fft // freq_step
    n_overlap = 128
    _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft, noverlap=n_overlap,
                                      nfft=n_fft)
    plt.figure()
    # Plot spectrogram
    plt.figure()
    plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    plt.colorbar(use_gridspec=True)
    plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' binning frequencies into 0.5Hz apart')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_0.5_bin.png')
    plt.show()

    # Restrict the constituent frequencies upto 18Hz
    # Set all upper frequencies' values to 0
    upper_freq = 18
    Zxx[int(upper_freq / bin_width)::, :] = np.finfo(float).eps
    _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft, noverlap=n_overlap,
                                   nfft=n_fft)
    plt.figure()
    plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    plt.colorbar(use_gridspec=True)
    plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' cutting off at 18Hz')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_cutoff_18.png')
    plt.show()

    # Temporally smooth by 8-second running average
    n_window = 8
    _, n_times = Zxx.shape
    # Smoothing
    Zxx_s = Zxx[:, 0:n_window]
    Zxx_s_bin = np.mean(Zxx_s, 1)
    Zxx_s_bin = np.reshape(Zxx_s_bin, (Zxx_s_bin.shape[0], 1))
    Zxx_temp = Zxx_s_bin
    for i in range(n_window, n_times):
        Zxx_s = Zxx[:, i - (n_window - 1):i]
        Zxx_s_bin = np.mean(Zxx_s, 1)
        Zxx_s_bin = np.reshape(Zxx_s_bin, (Zxx_s_bin.shape[0], 1))
        Zxx_temp = np.hstack((Zxx_temp, Zxx_s_bin))
    # Directly copy the front elements in total fewer than a window length back to the average matrix.
    # A moving average on them can be tried as well.
    Zxx_temp = np.hstack((Zxx[:, 0:n_window - 1], Zxx_temp))
    Zxx = Zxx_temp

    _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft,
                                   noverlap=n_overlap, nfft=n_fft)
    plt.figure()
    plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    plt.colorbar(use_gridspec=True)
    plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' running an 8-second moving average')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_avg.png')
    plt.show()

    # Remove the upper frequency part
    Zxx = Zxx[0:int(upper_freq / bin_width), :]
    # Remove 0th second
    Zxx = Zxx[:, 1::]
    return Zxx


def exp_processing(dat_exp, exp):
    # Extract the first 30-minute excerpt
    hz = 128  # Sampling frequency
    time = 35 * 60  # Seconds
    dat_exp = dat_exp[0:hz * time, :]

    channel = 0
    dat_ch = dat_exp[:, channel]
    Zxx_temp = channel_processing(dat_ch, exp, 0)

    # _, n_channel = dat_exp.shape
    # for channel in range(1, n_channel):
    #     dat_ch = dat_exp[:, channel]
    #     dat_ch = channel_processing(dat_ch, exp, channel)
    #     dat_temp = np.vstack((dat_temp, dat_ch))

    Zxx = 10*np.log10(np.abs(Zxx_temp))
    return Zxx


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.figsize"] = (9.6, 7.2)
    # Locate file path
    data_root = './EEG_Data'

    """
    Pre-processing
    """
    # n_EEG_file = 24
    # for EEG_file_number in range(n_EEG_file):
    EEG_file_number = 0
    filename = 'eeg_record' + str(EEG_file_number) + '.mat'
    # Load pandas dict
    dat_pd_exp = get_EEG_data(data_root, filename)
    # Convert pandas dict to numpy array
    dat_np_exp = dat_pd_exp.to_numpy()  # (308868, 7)
    Zxx_exp = exp_processing(dat_np_exp, 0)
    print(Zxx_exp.shape)








