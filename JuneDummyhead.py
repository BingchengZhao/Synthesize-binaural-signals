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
session = vaae.acquisition.IrSession(measuremement_name='Test3', config_file_name='JuneDummyhead.json')


 # %% Test sound sources
session.test_sources()

# %% Create new sweep recording:
session.record_sweep()


# %% Load exsiting IR session from output/<IR_SESSION_NAME>

session = vaae.acquisition.IrSession(load_measurement='Test3')

# %% Create IR from existing instance of IR session.
# This passes the raw recording, inverse sweep and parameters to the ImpulseResponse class.

ir = vaae.acquisition.ImpulseResponse(session)

# %% View IR

ir.view_ir(1)

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
degree = np.arange(0, 360, 15)
folder_name = 'JuneListRoom_KU100'
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

Turntable_IP = '192.168.1.34'
Turntable_Port = 6667
# po = tT.ET250_3D( Turntable_IP , Turntable_Port, 'set zero')

def move_and_wait(target_angle, waiting_time=1, debug=False):
    """
    Move the turntable to a target angle and wait until it has been reached.

    """

    current_angle = tT.ET250_3D( Turntable_IP , Turntable_Port, 'read angle')
    angular_movement = target_angle - current_angle

    if angular_movement > 0:
        tT.ET250_3D( Turntable_IP , Turntable_Port, 'move forward', target_angle-current_angle)
        if debug:
             print('moving forward.')
    elif angular_movement < 0:
        tT.ET250_3D( Turntable_IP , Turntable_Port, 'move backward', np.abs(target_angle-current_angle))
        if debug:
            print('moving backward.')
    else:
        if debug:
            print('already there.')
        return current_angle
    while tT.ET250_3D( Turntable_IP , Turntable_Port, 'read angle') != target_angle:
        time.sleep(waiting_time)
    # time.sleep(2)
    if debug:
        return tT.ET250_3D( Turntable_IP , Turntable_Port, 'read angle')
    else:
        return

# %%
move_and_wait(0)
    
# %%
# current_degree = tT.ET250_3D( Turntable_IP , Turntable_Port, 'read angle')
# tT.ET250_3D( Turntable_IP , Turntable_Port, 'move forward', -current_degree)

# for i, angle in enumerate(degree):
#     move_and_wait(angle)
    # time.sleep(3)


# print(po)

# %%
"""
Experiment of Dummyhead

"""
# For loop
degree = np.arange(0, 360, 15)
distance = 3.25
session_name_prefix = 'Farfield_{}m'.format(distance)

for i, angle in enumerate(degree):
    session_name = '{}-{:03}deg'.format(session_name_prefix, 360-degree[i])
    print(session_name)             
    # create session, run measurement, save data
    move_and_wait(angle)
    # avoid the affect of the sound of the turntable
    time.sleep(1)
    # get raw recording
    session = vaae.acquisition.IrSession(measuremement_name=session_name, config_file_name='JuneDummyhead.json')
    # session.test_sources()
    session.record_sweep()

    # session = vaae.acquisition.IrSession(load_measurement=session_name)

    # get the impulse response of raw recordings
    # ir = vaae.acquisition.ImpulseResponse(session)
    # ensure the accuracy of the IR
    # ir.view_ir(1)
    # input("print enter to continue to the next experiment")
    print(f"{i+1} experiment finished.")
    time.sleep(2)

# %%
"""

# Noise floor measurements

"""
degree = 360
distance = 3.25
session_name_prefix = 'Farfield_{}m'.format(distance) + 'Noisefloor' + 'Gain15'

session_name = '{}-{:03}deg'.format(session_name_prefix, 360-degree)
session = vaae.acquisition.IrSession(measuremement_name=session_name, config_file_name='JuneDummyhead.json')
session.record_sweep()
print(f"Noisefloor experiment finished.")

# %%
move_and_wait(0)

# %%
# Post-procee raw recordings
for i, angle in enumerate(degree):
    session_name = '{}-{:03}deg'.format(session_name_prefix, 360-angle)
    session = vaae.acquisition.IrSession(load_measurement=session_name)
    ir = vaae.acquisition.ImpulseResponse(session)

    # Visualize the every IR
    # plt.figure()
    # plt.plot(ir.ir[0, 0], label = 'channel 1 for left ear')
    # plt.plot(ir.ir[0, 1], label = 'channel 2 for right ear')
    ir.view_ir(1)
    plt.xlim(0,2000)
    # plt.legend()
    plt.title(f'{i+1} IR for all channels')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.grid()
    plt.show()
    # input("press enter to continue")
    
    ir.save_ir('wav','npy')
    print(f"{i+1} IR conversion finished.")
print('All IR conversion finished')

# %%
# create a new folder where every measurement files should be stored.
folder_name = 'JuneListRoom_KU100'
Secordary_folder_name1 = 'Farfield_horizontal_3.25m'
Secordary_folder_name2 = 'Nearfield_horizontal_1.00m'
Secordary_folder_name3 = 'Narfield_nonhorizontal(ElE45DEG)_1.00m'

base_path = '/Users/bogdanbacila/Dev/vaae-audio-toolbox/output'
first_folder_path = os.path.join(base_path, folder_name)
secordary_folder_path1 = os.path.join(first_folder_path,Secordary_folder_name1)
secordary_folder_path2 = os.path.join(first_folder_path,Secordary_folder_name2)
secordary_folder_path3 = os.path.join(first_folder_path,Secordary_folder_name3)

for i, angle in enumerate(degree):
    session_name = '{}-{:03}deg'.format(session_name_prefix, 360-degree[i])
    current_folder = base_path + '/'+'IR_{}'.format(session_name)
    target_dir = secordary_folder_path1
    # print(current_folder)
    # move data from current to a subfolder in the new folder.
    shutil.move(current_folder,target_dir)


# %%
# Second Post-procee raw recordings
degree = np.arange(0, 360, 15)
distance = 3.25
session_name_prefix = 'Farfield_{}m'.format(distance)

for i, angle in enumerate(degree):
    session_name = '{}-{:03}deg'.format(session_name_prefix, 360-angle)
    session = vaae.acquisition.IrSession(load_measurement=session_name)
    ir = vaae.acquisition.ImpulseResponse(session)

    # Visualize the every IR
    # plt.figure()
    # plt.plot(ir.ir[0, 0], label = 'channel 1 for left ear')
    # plt.plot(ir.ir[0, 1], label = 'channel 2 for right ear')
    ir.view_ir(1)
    plt.xlim(0,700)
    
    plt.title(f'{i+1} IR for all channels')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.grid()
    print(np.argmax(ir.ir[0,0]))
    peak_index = np.argmax(ir.ir[0,0])
    plt.axvline(x=peak_index, color='b', linestyle='--')
    plt.text(peak_index, 0.5, f'{peak_index}', color='blue', fontsize=10, verticalalignment='bottom')
    plt.show()
    # input("press enter to continue")
    
    ir.save_ir('wav','npy')
    print(f"{i+1} IR conversion finished.")
print('All IR conversion finished')

# %%
# create a new folder where every measurement files should be stored.
folder_name = 'JuneListRoom_KU100'
Secordary_folder_name1 = 'Farfield_horizontal_3.25m_Gain0'
Secordary_folder_name2 = 'Nearfield_horizontal_1.00m'
Secordary_folder_name3 = 'Narfield_nonhorizontal(ElE45DEG)_1.00m_Gain10'

base_path = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output'
first_folder_path = os.path.join(base_path, folder_name)
secordary_folder_path1 = os.path.join(first_folder_path,Secordary_folder_name1)
secordary_folder_path2 = os.path.join(first_folder_path,Secordary_folder_name2)
secordary_folder_path3 = os.path.join(first_folder_path,Secordary_folder_name3)

for i, angle in enumerate(degree):
    session_name = '{}-{:03}deg'.format(session_name_prefix, 360-degree[i])
    current_folder = base_path + '/'+'IR_{}'.format(session_name)
    target_dir = secordary_folder_path1
    # print(current_folder)
    # move data from current to a subfolder in the new folder.
    shutil.move(current_folder,target_dir)
# %%
