import os
import pickle
import numpy as np
import pandas as pd
import pingouin as pg

# Load processed features
data_dir = "./Data"
Sxx_focused = pickle.load(open(os.path.join(data_dir, "focused_pre" + ".pkl"), "rb"))
print("Focused features loaded. {}".format(Sxx_focused.shape))
Sxx_unfocused = pickle.load(open(os.path.join(data_dir, "unfocused_pre" + ".pkl"), "rb"))
print("Unfocused features loaded. {}".format(Sxx_unfocused.shape))
Sxx_drowsy = pickle.load(open(os.path.join(data_dir, "drowsy_pre" + ".pkl"), "rb"))
print("Drowsy features loaded. {}".format(Sxx_drowsy.shape))

n_feature, n_time_stamps = Sxx_focused.shape
# Horizontally stack three power matrices together, so that each row is the power density for a frequency
Sxx_all = np.hstack((Sxx_focused, Sxx_unfocused, Sxx_drowsy))

frequency_list = np.arange(0.5, 18.1, 0.5)
icc_list = np.zeros(n_feature)
time_stamp_list = np.tile(np.arange(n_time_stamps), 3)
mental_state_list = ['F'] * n_time_stamps + ['U'] * n_time_stamps + ['D'] * n_time_stamps
for n in range(n_feature):
    df = pd.DataFrame({'time_stamp': time_stamp_list,
                       'mental_state': mental_state_list,
                       'spectral_power': Sxx_all[n, :]})

    icc = pg.intraclass_corr(data=df, targets='time_stamp', raters='mental_state', ratings='spectral_power')
    icc_list[n] = icc['ICC'][2]
    print('{}Hz component processed.'.format(frequency_list[n % 36]))

# pickle.dump(icc_list, open(os.path.join(data_dir, "icc_list" + ".pkl"), "wb"))
# print('ICC list dumped')

# # Load processed features
# data_dir = "./Data"
# icc_list = pickle.load(open(os.path.join(data_dir, "icc_list" + ".pkl"), "rb"))

# ICC measures agreement between classes,
# whereas the power spectra across mental states are desired to be as disagreeing as possible
comp_icc_list = 1 - icc_list
des_ind_list = np.flip(np.argsort(comp_icc_list))
comp_icc_list_sorted = comp_icc_list[des_ind_list]
# print(comp_icc_list_sorted)

n_channel = 7
n_feature_ch = n_feature // n_channel
channel_list = ['F3'] * n_feature_ch + ['F4'] * n_feature_ch + ['Fz'] * n_feature_ch + ['C3'] * n_feature_ch\
               + ['C4'] * n_feature_ch + ['Cz'] * n_feature_ch + ['Pz'] * n_feature_ch
channel_list_sorted = [channel_list[i] for i in des_ind_list]
# print(channel_list_sorted)

meta_frequency_list = np.tile(frequency_list, n_channel)
meta_frequency_list_sorted = meta_frequency_list[des_ind_list]
# print(meta_frequency_list_sorted)

Sxx_focused_sorted = Sxx_focused[des_ind_list, :]
Sxx_unfocused_sorted = Sxx_unfocused[des_ind_list, :]
Sxx_drowsy_sorted = Sxx_drowsy[des_ind_list, :]

# Take the first 25 most significant features
Sxx_focused_rd = Sxx_focused_sorted[0:25, :]
Sxx_unfocused_rd = Sxx_unfocused_sorted[0:25, :]
Sxx_drowsy_rd = Sxx_drowsy_sorted[0:25, :]

pickle.dump(Sxx_focused_rd, open(os.path.join(data_dir, "focused_rd" + ".pkl"), "wb"))
print("Dimensionality-reduced focused features dumped.")
pickle.dump(Sxx_unfocused_rd, open(os.path.join(data_dir, "unfocused_rd" + ".pkl"), "wb"))
print("Dimensionality-reduced unfocused features dumped.")
pickle.dump(Sxx_drowsy_rd, open(os.path.join(data_dir, "drowsy_rd" + ".pkl"), "wb"))
print("Dimensionality-reduced drowsy features dumped.")
