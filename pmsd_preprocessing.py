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
import matplotlib.pyplot as plt
import io


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
    # Spectrogram
    Sxx = np.abs(Zxx)**2

    # Plot spectrogram
    # plt.figure()
    # plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    # plt.colorbar(use_gridspec=True)
    # plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' spectrogram by 8-second blackman window')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman.png')
    # plt.show()

    # Discard DC component on 1st row
    Sxx = Sxx[1::, :]
    n_freq, _ = Sxx.shape

    # Bin frequencies into 0.5Hz apart
    bin_width = 0.5
    # Number of frequency components to bin
    freq_step = int(bin_width / (fs / n_fft))  # Type cast from float to int
    # Binning
    Sxx_s = Sxx[0:0 + freq_step, :]
    Sxx_s_bin = np.mean(Sxx_s, 0)
    Sxx_temp = Sxx_s_bin
    for i in range(freq_step, n_freq, freq_step):
        Sxx_s = Sxx[i:i + freq_step, :]
        Sxx_s_bin = np.mean(Sxx_s, 0)
        Sxx_temp = np.vstack((Sxx_temp, Sxx_s_bin))
    Sxx = Sxx_temp

    # # ST-IDFT to reconstruct time-domain signal
    # n_fft = n_fft // freq_step
    # n_overlap = 128
    # _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft, noverlap=n_overlap,
    #                                   nfft=n_fft)
    # plt.figure()
    # plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    # plt.colorbar(use_gridspec=True)
    # plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' binning frequencies into 0.5Hz apart')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_0.5_bin.png')
    # plt.show()

    # Restrict the constituent frequencies upto 18Hz
    # Set all upper frequencies' values to 0
    upper_freq = 18
    # Zxx[int(upper_freq / bin_width)::, :] = np.finfo(float).eps
    # _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft, noverlap=n_overlap,
    #                                nfft=n_fft)
    # plt.figure()
    # plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    # plt.colorbar(use_gridspec=True)
    # plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' cutting off at 18Hz')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_cutoff_18.png')
    # plt.show()

    # Temporally smooth by 8-second running average
    n_window = 8
    _, n_times = Sxx.shape
    # Smoothing
    Sxx_s = Sxx[:, 0:n_window]
    Sxx_s_bin = np.mean(Sxx_s, 1)
    Sxx_s_bin = np.reshape(Sxx_s_bin, (Sxx_s_bin.shape[0], 1))
    Sxx_temp = Sxx_s_bin
    for i in range(n_window, n_times):
        Sxx_s = Sxx[:, i - (n_window - 1):i]
        Sxx_s_bin = np.mean(Sxx_s, 1)
        Sxx_s_bin = np.reshape(Sxx_s_bin, (Sxx_s_bin.shape[0], 1))
        Sxx_temp = np.hstack((Sxx_temp, Sxx_s_bin))
    # Directly copy the front elements in total fewer than a window length back to the average matrix.
    # A moving average on them can be tried as well.
    Sxx_temp = np.hstack((Sxx[:, 0:n_window - 1], Sxx_temp))
    Sxx = Sxx_temp

    # _, dat_ch = scipy.signal.istft(Zxx, fs=fs, window=np.blackman(n_fft), nperseg=n_fft,
    #                                noverlap=n_overlap, nfft=n_fft)
    # plt.figure()
    # plt.specgram(dat_ch, Fs=fs, NFFT=n_fft, noverlap=n_overlap, window=np.blackman(n_fft))
    # plt.colorbar(use_gridspec=True)
    # plt.title('Exp.' + str(exp) + ' Ch.' + str(channel) + ' running an 8-second moving average')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (Seconds)')
    # plt.savefig('./plots/Exp3Ch2_spec_blackman_avg.png')
    # plt.show()

    # Remove the upper frequency part
    Sxx = Sxx[0:int(upper_freq / bin_width), :]
    # Remove 0th second
    Sxx = Sxx[:, 1::]
    return Sxx


def exp_processing(dat_exp, exp):
    # Extract the first 30-minute excerpt
    hz = 128  # Sampling frequency
    time = 30 * 60  # Seconds
    dat_exp = dat_exp[0:hz * time, :]

    channel = 0
    dat_ch = dat_exp[:, channel]
    Sxx_temp = channel_processing(dat_ch, exp, 0)

    _, n_channel = dat_exp.shape
    for channel in range(1, n_channel):
        dat_ch = dat_exp[:, channel]
        Sxx_ch = channel_processing(dat_ch, exp, channel)
        Sxx_temp = np.vstack((Sxx_temp, Sxx_ch))

    # Covert power values into decibel form
    Sxx = 10*np.log10(Sxx_temp)

    # Normalize feature vectors at each second
    Sxx = (Sxx - Sxx.mean(axis=0)) / Sxx.std(axis=0)
    return Sxx


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.figsize"] = (9.6, 7.2)
    # Locate file path
    data_root = './EEG_Data'

    """
    Pre-processing
    """
    EEG_file_number = 0
    filename = 'eeg_record' + str(EEG_file_number) + '.mat'
    # Load pandas dict
    dat_pd_exp = get_EEG_data(data_root, filename)
    # Convert pandas dict to numpy array
    dat_np_exp = dat_pd_exp.to_numpy()  # (308868, 7)
    # Pre-process an entire experiment
    Sxx_exp = exp_processing(dat_np_exp, 0)  # (252, 1800)
    print(Sxx_exp.shape)

    # Focused during first 10 minutes, unfocused during next 10 minutes, drowsy during final 15 minutes
    Sxx_focused = Sxx_exp[:, 0:600]
    print(Sxx_focused.shape)
    Sxx_unfocused = Sxx_exp[:, 600:1200]
    print(Sxx_unfocused.shape)
    Sxx_drowsy = Sxx_exp[:, 1200:1800]
    print(Sxx_drowsy.shape)
    print("Experiment {} processed.".format(EEG_file_number))
    n_EEG_file = 23
    for EEG_file_number in range(1, n_EEG_file):
        filename = 'eeg_record' + str(EEG_file_number) + '.mat'
        # Load pandas dict
        dat_pd_exp = get_EEG_data(data_root, filename)
        # Convert pandas dict to numpy array
        dat_np_exp = dat_pd_exp.to_numpy()  # (308868, 7)
        # Pre-process an entire experiment
        Sxx_exp = exp_processing(dat_np_exp, 0)  # (252, 1800)
        print(Sxx_exp.shape)

        # Focused during first 10 minutes, unfocused during next 10 minutes, drowsy during final 15 minutes
        Sxx_focused = np.hstack((Sxx_focused, Sxx_exp[:, 0:600]))
        print(Sxx_focused.shape)
        Sxx_unfocused = np.hstack((Sxx_unfocused, Sxx_exp[:, 600:1200]))
        print(Sxx_unfocused.shape)
        Sxx_drowsy = np.hstack((Sxx_drowsy, Sxx_exp[:, 1200:1800]))
        print(Sxx_drowsy.shape)
        print("Experiment {} processed.".format(EEG_file_number))

    print(Sxx_focused.shape)
    print(Sxx_unfocused.shape)
    print(Sxx_drowsy.shape)

    # Save processed features
    data_dir = "./Data"
    pickle.dump(Sxx_focused, open(os.path.join(data_dir, "focused_pre" + ".pkl"), "wb"))
    print("Focused features dumped.")
    pickle.dump(Sxx_unfocused, open(os.path.join(data_dir, "unfocused_pre" + ".pkl"), "wb"))
    print("Unfocused features dumped.")
    pickle.dump(Sxx_drowsy, open(os.path.join(data_dir, "drowsy_pre" + ".pkl"), "wb"))
    print("Drowsy features dumped.")
