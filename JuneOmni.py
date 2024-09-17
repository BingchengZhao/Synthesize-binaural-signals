# %% Import vaae_audio_toolbox
import sys
sys.path.append('/usr/local/lib/python3.11/site-packages')
import scipy
import vaae_audio_toolbox as vaae
import numpy as np
import os 
import time 
import socket
import struct
import turntableToolboxMIT as tT
import shutil
import matplotlib.pyplot as plt


#### Set, check, reset and test audio output

# %% Check audio devices:

vaae.acquisition.list_audio_devices()

# %% Reset audio devices:

vaae.acquisition.reset_audio_devices()

# %% Set Audio devices - pick the input and output from the list returned by 'list_audio_devices()' 

vaae.acquisition.set_audio_device(6,6)
# vaae.acquisition.set_audio_device(0)


#### Create Measurement sessions and IRs

# %% Create new IR session 

# session = vaae.acquisition.IrSession(measuremement_name='jag-pilot', config_file_name='jaguar-config.json')
# session = vaae.acquisition.IrSession(measuremement_name='Test3', config_file_name='JuneEigenmike.json')
session = vaae.acquisition.IrSession(measuremement_name='Test-Omni', config_file_name='JuneOmni.json')


 # %% Test sound sources
session.test_sources()

# %% Create new sweep recording:
session.record_sweep()


# %% Load exsiting IR session from output/<IR_SESSION_NAME>

# distance = 3.25
# session_name_prefix = 'Farfield_{}m'.format(distance)
# session_name = '{}-{:03}deg'.format(session_name_prefix, 15)
# session_name = 'Nearfield_{}m'.format(distance)
session = 'Farfield_3.25m-090deg'
session = vaae.acquisition.IrSession(load_measurement=session)

# %% Create IR from existing instance of IR session.
# This passes the raw recording, inverse sweep and parameters to the ImpulseResponse class.

ir = vaae.acquisition.ImpulseResponse(session)

# %% View IR

ir.view_ir(1)
plt.xlim(0,700)
plt.grid()
# print(np.argmax(np.abs(ir.ir[0,0])))
# # print(np.argmax(ir.ir[0,1]))
# peak_index = np.argmax(np.abs(ir.ir[0,0]))
# plt.axvline(x=peak_index, color='b', linestyle='--')
# plt.text(peak_index, 0.5, f'{peak_index}', color='blue', fontsize=10, verticalalignment='bottom')

# %% Save IR

ir.save_ir('wav','npy')

#### Calculate metrics

#%%
# test_ir = ir.ir[0,0]
# clarity_oct = vaae.metrics.clarity(test_ir)
# clarity_avg = vaae.metrics.clarity(test_ir, oct_band_avg= [2,5])

#%%
# test_ir = ir.ir[0,0]
# definition_oct = vaae.metrics.definition(test_ir)
# definition_avg = vaae.metrics.definition(test_ir, oct_band_avg= [2,5])

#%%
# test_ir = ir.ir[0]
# iacc_oct = vaae.metrics.iacc(test_ir[0], test_ir[1],[0,80,0,80], 48000)
# iacc_avg = vaae.metrics.iacc(test_ir[0], test_ir[1],[0,80,0,80], 48000, [3,7])

#%% 

# test_ir = ir.ir[1]
# itd = vaae.metrics.itd(test_ir[0], test_ir[1],[0,3,0,3], 48000)
# ild = vaae.metrics.ild(test_ir[0], test_ir[1],[0,3,0,3], 48000)


# %%
# Create experiment folder for each each experiment

folder_name = 'JuneListRoom_Omni'
Secordary_folder_name1 = 'Farfield_horizontal_3.25m'
Secordary_folder_name2 = 'Nearfield_horizontal_1.00m'
Secordary_folder_name3 = 'Narfield_nonhorizontal(ElE45DEG)_1.00m'

base_path = '/Users/bogdanbacila/Dev/vaae-audio-toolbox/output'
first_folder_path = os.path.join(base_path, folder_name)
secordary_folder_path1 = os.path.join(first_folder_path,Secordary_folder_name1)
secordary_folder_path2 = os.path.join(first_folder_path,Secordary_folder_name2)
secordary_folder_path3 = os.path.join(first_folder_path,Secordary_folder_name3)
os.makedirs(secordary_folder_path1, exist_ok = True)
os.makedirs(secordary_folder_path2, exist_ok = True)
os.makedirs(secordary_folder_path3, exist_ok = True)

print(f"Directories '{first_folder_path}' created successfully.")

# %%
"""
Experiment of Omnidirection microphone

"""
distance = 3.25
session_name = 'Farfield_{}m'.format(distance)

# get raw recording
session = vaae.acquisition.IrSession(measuremement_name=session_name, config_file_name='JuneOmni.json')
# session.test_sources()
session.record_sweep()
print(f"experiment finished.")

# %%
"""

# Noise floor measurements

"""
degree = 360
distance = 3.25
session_name_prefix = 'Farfield_{}m'.format(distance) + 'Noisefloor' + 'Gain15'

session_name = '{}-{:03}deg'.format(session_name_prefix, 360-degree)
session = vaae.acquisition.IrSession(measuremement_name=session_name, config_file_name='JuneOmni.json')
session.record_sweep()
print(f"Noisefloor experiment finished.")
# %%
# Post-procee raw recordings

session = vaae.acquisition.IrSession(load_measurement=session_name)
ir = vaae.acquisition.ImpulseResponse(session)

# ensure the accuracy of the IR
ir.view_ir(1)
plt.xlim(0,2000)
plt.title(f' IR for all channels')
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.grid()
plt.show()
# input("press enter to continue")
    
ir.save_ir('wav','npy')
print('All IR conversion finished')
# %%
# create a new folder where every measurement files should be stored.
folder_name = 'JuneListRoom_Omni'
Secordary_folder_name1 = 'Farfield_horizontal_3.25m'
Secordary_folder_name2 = 'Nearfield_horizontal_1.00m'
Secordary_folder_name3 = 'Narfield_nonhorizontal(ElE45DEG)_1.00m'
base_path = '/Users/bogdanbacila/Dev/vaae-audio-toolbox/output'
first_folder_path = os.path.join(base_path, folder_name)
secordary_folder_path1 = os.path.join(first_folder_path,Secordary_folder_name1)
secordary_folder_path2 = os.path.join(first_folder_path,Secordary_folder_name2)
secordary_folder_path3 = os.path.join(first_folder_path,Secordary_folder_name3)

session_name = 'Farfield_{}m'.format(distance)
current_folder = base_path + '/'+'IR_{}'.format(session_name)
target_dir = secordary_folder_path1
# print(current_folder)

# move data from current to a subfolder in the new folder.
shutil.move(current_folder,target_dir)

# %%
# second post - process
distance = 3.25
session_name = 'Farfield_{}m'.format(distance)

session = vaae.acquisition.IrSession(load_measurement=session_name)
ir = vaae.acquisition.ImpulseResponse(session)

ir.view_ir(1)
plt.xlim(0,700)
plt.grid()
print(np.argmax(np.abs(ir.ir[0,0])))
peak_index = np.argmax(np.abs(ir.ir[0,0]))
# plt.axvline(x=peak_index, color='b', linestyle='--')
# plt.text(peak_index, 0.5, f'{peak_index}', color='blue', fontsize=10, verticalalignment='bottom')

ir.save_ir('wav','npy')
# %%
