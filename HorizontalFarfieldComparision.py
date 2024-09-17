''' Jacob Hollebon, Institute of Sound and Vibration Research 2021'''
#%%
# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
from pysofaconventions import *
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import sofa
import matplotlib.pyplot as plt
import soundfile as sf
import matplotlib.gridspec as gridspec
# from scipy.special import sph_harm
from scipy.linalg import pinv
from scipy.linalg import svd
from scipy import signal

plt.style.use('default')
def sphHarmReal(n, m, azi, col, normalisation, CSphase=False):
    '''Calculate real spherical harmonic of order n, degree m sampled at azi and col
    
    Arguments --
    n: (Ambisonics) order of the spherical harmonic
    m: (Ambisonics) degree of the spherical harmonic
    azi: Azimuthal sampling position in radians, between 0-2pi. 0, 90, 180 and 270 degrees point forwards, left, back, right respectively
    col: Colattitude sampling position in radians, between 0-pi, 0, 90, 180 degrees point up (+ve z), straight x-y plane, down (-ve z)
    normalisation: String, either 'SN3D' or 'N3D' 
    '''
    from math import factorial
    from scipy.special import lpmv as legendre
    
    # Calculate the normalisation term
    if m==0:
        delta = 1
    else:
        delta = 0
        
    norm = np.sqrt( ((2-delta)/(4*np.pi)) * factorial(n-abs(m)) / factorial(n+abs(m)) )
    if normalisation in ['N3D', 'n3d']:
        norm *= np.sqrt(2*n + 1)
    
    # Calculate the azimuthal dependance
    if m<0:
        trig = np.sin(-m * azi)
    else:
        trig = np.cos(m * azi)
        
    # Calculate the colattitude depndance, just abs(m) needed here
    # This function includes the CS phase by default, we do not want this as per normal
    # ambisonics conventions, but can include it if desired
    leg = legendre(abs(m), n, np.cos(col))
    if not CSphase:
        leg *= (-1)**abs(m) # JH - should this be absm or just m?
    
    #Build the SH
    Ynm = norm * leg * trig
    
    return Ynm

def CalculateComplexErrorAndPlot(pf_analytical, pf_measured, fs=48000*4,ylower=-40,yupper=40,title='Complex Error'):
    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    ComplexError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')

    for i in range(ComplexError.shape[0]):
        for ii in range(ComplexError.shape[1]):
            ComplexError[i,ii,:]= (np.abs(pf_analytical[i,ii,:]-pf_measured[i,ii,:])**2 / np.abs(pf_measured[i,ii,:])**2)

    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(title,fontsize='xx-large')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(ComplexError.shape[0]):
            axs[ii].semilogx(freqs, 10*np.log10(ComplexError[i, ii, :]), label=f'Complex error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)',fontsize='xx-large')
        axs[ii].set_ylabel('Error (dB)',fontsize='xx-large')
        axs[ii].legend(fontsize='xx-large')
        axs[ii].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def CalculateMagnitudeErrorAndPlot(pf_analytical, pf_measured, fs=48000*4,ylower=-20,yupper=20,title='Magnitude Error'):
    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    MagnitudeError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')

    for i in range(MagnitudeError .shape[0]):
        for ii in range(MagnitudeError .shape[1]):
            MagnitudeError [i,ii,:]= 10*np.log10((np.abs(pf_analytical[i, ii, :]) )** 2 / (np.abs(pf_measured[i, ii, :]) )** 2)
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(title,fontsize = 'xx-large')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(MagnitudeError.shape[0]):
            axs[ii].semilogx(freqs, MagnitudeError[i, ii, :], label=f'Magnitude error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)',fontsize = 'xx-large')
        axs[ii].set_ylabel('Error(dB)',fontsize = 'xx-large')
        axs[ii].legend(fontsize='xx-large')
        axs[ii].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def CalculatePhaseErrorAndPlot(pf_analytical, pf_measured, fs=48000*4,ylower=-180,yupper=40,title='Phase Error'):

    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    PhaseError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')
    for i in range(PhaseError .shape[0]):
        for ii in range(PhaseError .shape[1]):
            PhaseError[i,ii,:]= np.angle(pf_analytical[i,ii,:] * pf_measured[i,ii,:].conjugate())
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(title, fontsize='xx-large')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(PhaseError.shape[0]):
            axs[ii].semilogx(freqs, np.unwrap(PhaseError[i, ii, :]), label=f'Phase error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)',fontsize='xx-large')
        axs[ii].set_ylabel('Error(rad)',fontsize='xx-large')
        axs[ii].legend(fontsize='xx-large')
        axs[ii].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def CalculateILD(Pf,fs):
    # ILD = np.zeros((Pf.shape[0],Pf.shape[2]),dtype = 'float64')
    # for i in range(Pf.shape[0]):
    #     ILD[i,:] = 10*np.log10(np.abs(Pf[i,0,:])**2/np.abs(Pf[i,1,:])**2)
    # Calculate the ILD in 1/3 octave band

    fc = 10**3 * (2 ** (np.arange(-18, 14)/3))
    fupper = fc * 2**(1/6)
    flower = fc / 2**(1/6)
    ILD_1_3_octave = np.zeros((Pf.shape[0], len(fc)), dtype='float64')
    
    freqs = np.fft.rfftfreq(Pf.shape[2], 1/fs)
    for i in range(Pf.shape[0]):
        for j, (fl, fu) in enumerate(zip(flower, fupper)):
            band_indices = np.where((freqs >= fl) & (freqs <= fu))[0]
            if len(band_indices) > 0:
                right_energy = np.sum(np.abs(Pf[i, 1, band_indices])**2)
                left_energy = np.sum(np.abs(Pf[i, 0, band_indices])**2)
                ILD_1_3_octave[i, j] = 10 * np.log10(right_energy / left_energy)

    return fc, ILD_1_3_octave

def CalculateITD(Pt,fs,lowcut,highcut,order):
    # band pass filter 20hz - 1000hz
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    Pt_filtered = np.zeros_like(Pt)

    for i in range(Pt.shape[0]):
        for ii in range(Pt.shape[1]):
            Pt_filtered[i,ii,:] = signal.lfilter(b,a,Pt[i,ii,:]) 
    # 
    ITD = []
    for i in range(Pt_filtered.shape[0]):
        corr = signal.correlate(Pt_filtered[i,1,:], Pt_filtered[i,0,:], mode='full')
        lag = np.argmax(corr) - len(Pt_filtered[i,0,:]) + 1
        ITD.append(lag / fs)
    
    return np.array(ITD)
#%%
"""
------Read 3D HRIR dataset online------
original: Right ear: 0
          Left ear: 1
          elevation:-90 - 90
"""
#Far field 3.25m 
HRIRO3D_path = "/Users/katakuri/Desktop/Msc project/Experiment/HRIR/HRIR_Farfield/3928297/HRIR_L2702.sofa"
HRIRO3D = sofa.Database.open(HRIRO3D_path)
#HRIRO3D_source_positions : (azimuth,elevation,r)
HRIRO3D_source_positions = HRIRO3D.Source.Position.get_values(system = 'spherical')
HRIRO3D_receiber_positions =HRIRO3D.Receiver.Position.get_values()

# Degree to Radians
HRIRO3D_source_positions = HRIRO3D_source_positions.astype(np.float64)
for i in range(HRIRO3D_source_positions.shape[0]):
    HRIRO3D_source_positions[i,0] = (HRIRO3D_source_positions[i,0]/180) * np.pi
    HRIRO3D_source_positions[i,1] = ((90-HRIRO3D_source_positions[i,1])/180) * np.pi

# %%
"""
------ Calculate Y matrix ------

"""
order = 35
Y = np.zeros((HRIRO3D_source_positions.shape[0],(order+1)**2),dtype = 'float64')
index = 0
for i in range (HRIRO3D_source_positions.shape[0]):
    index = 0
    for n in range(order + 1):
        for m in range(-n,n+1):
            Y[i, index] = sphHarmReal(n, m, HRIRO3D_source_positions[i, 0], HRIRO3D_source_positions[i, 1], normalisation='N3D')
            index += 1
con_num = np.linalg.cond(Y)
print(f'Y matrix: order: {order} con_num: {con_num}')
# %%
"""

------ Calculate the H(f,theta,phi) matrix
H matrix : (Q,Z) Q: sampling points  Z:frequency points

"""
nfft = HRIRO3D.Dimensions.N
rfftn = HRIRO3D.Dimensions.N // 2 + 1
fs = 48000
HRIRO3D_data_RightEar=np.zeros((HRIRO3D_source_positions.shape[0], HRIRO3D.Dimensions.N),dtype='float64')
HRIRO3D_data_LeftEar=np.zeros((HRIRO3D_source_positions.shape[0], HRIRO3D.Dimensions.N),dtype='float64')

for i in range(HRIRO3D_source_positions.shape[0]):
    HRIRO3D_data_RightEar[i,:]=HRIRO3D.Data.IR.get_values(indices={"M":i, "R":1, "E":0}) 
    HRIRO3D_data_LeftEar[i,:]=HRIRO3D.Data.IR.get_values(indices={"M":i, "R":0, "E":0}) 


HRTFO3D_R = np.zeros((HRIRO3D_source_positions.shape[0],rfftn),dtype = 'complex')
HRTFO3D_L = np.zeros((HRIRO3D_source_positions.shape[0],rfftn),dtype = 'complex')
for i in range(HRIRO3D_data_RightEar.shape[0]):
    HRTFO3D_R[i,:] = np.fft.rfft(HRIRO3D_data_RightEar[i,:])
    HRTFO3D_L[i,:] = np.fft.rfft(HRIRO3D_data_LeftEar[i,:])

freqs = np.fft.rfftfreq(nfft,1/fs)

#%%
"""
------ Calculate the Hnm(f) matix
Hnm(f) : ((order+1)**2 , Z)

"""

Y_pinv = np.linalg.pinv(Y)

Hnm_R = np.dot(Y_pinv, HRTFO3D_R)
Hnm_L = np.dot(Y_pinv, HRTFO3D_L)
# %%
"""

-----Recover the HRTF at Horizontal and non-horizontal

"""
Azimuth = -np.arange(0,210,30)/180 *np.pi

Elevation_hor = 90/180 * np.pi
new_position_hor = np.zeros((Azimuth.shape[0],2))

for i in range(Azimuth.shape[0]):
    new_position_hor[i, :] = [Azimuth[i], Elevation_hor]

#%%

# Calculate New Y matrix 
Y_new_hor = np.zeros((new_position_hor.shape[0], (order + 1)**2), dtype='float64')

for i in range(new_position_hor.shape[0]):
    index = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            Y_new_hor[i, index] = sphHarmReal(n, m, new_position_hor[i, 0], new_position_hor[i, 1], normalisation='N3D')
            index += 1

# Calculate the New HRTF at new position
HRTF_R_new_hor = np.dot(Y_new_hor, Hnm_R)
HRTF_L_new_hor = np.dot(Y_new_hor, Hnm_L)

# %%
# Calculate the HRIR(24,2,128)
HRIRO3D_new_hor = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,HRIRO3D_data_RightEar.shape[1]),dtype='float64')
for i in range((new_position_hor.shape[0])):
    
    HRIRO3D_new_hor[i,0,:] = np.fft.irfft(HRTF_L_new_hor[i,:])
    HRIRO3D_new_hor[i,1,:] = np.fft.irfft(HRTF_R_new_hor[i,:])

# %%
# samples = np.arange(0,128)        
# for i in range(HRIRO3D_new_hor.shape[0]):
#     plt.figure()
#     plt.plot(samples,HRIRO3D_new_hor[i,0,:], label = 'right')
#     plt.plot(samples,HRIRO3D_new_hor[i,1,:], label = 'left')
#     plt.title(f'DEG: {Azimuth[i]/np.pi*180}')
   
#     plt.xlim(0,60)
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
# %%
"""
---- Read omni data----
"""
#Far field
file_path_1 = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output/JuneListRoom_Omni/Farfield_horizontal_3.25m_Gain0/IR_Farfield_3.25m/IRs/IR_Farfield_3.25m_1s1r_S1.wav'

OmniIR_data, fs = sf.read(file_path_1, dtype = 'float64')
time1 = np.arange(len(OmniIR_data)) / fs

#%%
"""
------Convolve HRIRO3D dataset with my omni IR data 
"""

Truncate_len = 2**14
All_HRIRO3D_newhor_Convolved = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,Truncate_len),dtype='float64')

for i in np.arange(All_HRIRO3D_newhor_Convolved.shape[0]):
    for receiver in np.arange(HRIRO3D.Dimensions.R):
       
        # Convolve omni data and HRIR dataset
        Convolved_signal = np.convolve(HRIRO3D_new_hor[i, receiver, :], OmniIR_data, mode='full')

        # s = np.arange(len(Convolved_signal))  
        # plt.plot(s,Convolved_signal) 
        # plt.grid()
        # plt.show()

        # Truncate the signal after convolution
        All_HRIRO3D_newhor_Convolved[i, receiver, :] = Convolved_signal[:Truncate_len]

        # time = np.arange(Truncate_len) 
        # plt.figure()
        # plt.plot(time, All_HRIRO3D_newhor_Convolved[i, receiver, :])
        # #  plt.xlim(0,0.03)
        # plt.ylim(-0.01,0.01)
        # plt.grid()

#%%
"""
------Read and convolve HRIRO2D dataset with my omni data------

""" 
#Far field
HRIRO2D_path = "/Users/katakuri/Desktop/Msc project/Experiment/HRIR/HRIR_Farfield/3928297/HRIR_CIRC360.sofa"

HRIRO2D = sofa.Database.open(HRIRO2D_path)
HRIRO2D_source_positions = HRIRO2D.Source.Position.get_values(system = 'spherical')

azimuth_index = []
step = 30
for i, coordinate in enumerate(HRIRO2D_source_positions):
    if coordinate[0] % step == 0 and 180 <= coordinate[0] <= 360:
        azimuth_index.append(i)
measurement_index = np.flip(azimuth_index)
measurement_index = np.insert(measurement_index,0,0)
#%%
# measurement_index = [0,30,60,90,120,150,180]

emitter = 0
Truncate_len = 2**14
All_HRIRO2D_Convolved = np.zeros((new_position_hor.shape[0], HRIRO2D.Dimensions.R, Truncate_len),dtype='float64') 

for i,measurement in enumerate(measurement_index):
    print(measurement)
    for receiver in np.arange(HRIRO2D.Dimensions.R):
       
        HRIRO_data_SingleEar=HRIRO2D.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}) 
        fs = HRIRO2D.Data.SamplingRate.get_values(indices={"M":measurement})
        # Convolve omni data and HRIR dataset
        Convolved_signal = np.convolve(HRIRO_data_SingleEar, OmniIR_data, mode = 'full')
        
        # Truncate the signal after convolution
        All_HRIRO2D_Convolved[i, receiver,:] = Convolved_signal[0:Truncate_len] 

# %%
"""
------Interpolation using FFT and IFFT------

"""
upsample = 4
nfft = Truncate_len * upsample
rfft_len = nfft // 2 + 1

All_FRO2D_Convolved = np.zeros((len(measurement_index), HRIRO2D.Dimensions.R,All_HRIRO2D_Convolved.shape[2]//2+1),dtype = 'complex')
All_HRIRO2D_Convolved_Interpolated = np.zeros((len(measurement_index), HRIRO2D.Dimensions.R,nfft), dtype = 'float64')
All_FRO2D_Convolved_Interpolated = np.zeros((len(measurement_index), HRIRO2D.Dimensions.R,rfft_len), dtype = 'complex')

All_FRO3D_newhor_Convolved = np.zeros_like(All_FRO2D_Convolved)
All_HRIRO3D_newhor_Convolved_Interpolated = np.zeros_like(All_HRIRO2D_Convolved_Interpolated)
All_FRO3D_newhor_Convolved_Interpolated = np.zeros_like(All_FRO2D_Convolved_Interpolated)

for i in range(All_FRO2D_Convolved .shape[0]):
    for ii in range(All_FRO2D_Convolved .shape[1]):
        
        # rfft
        All_FRO2D_Convolved[i,ii,:] = np.fft.rfft(All_HRIRO2D_Convolved[i,ii,:])
        All_FRO3D_newhor_Convolved[i,ii,:] = np.fft.rfft(All_HRIRO3D_newhor_Convolved[i,ii,:])

        # irfft
        All_HRIRO2D_Convolved_Interpolated[i,ii,:] = np.fft.irfft(All_FRO2D_Convolved[i,ii,:], n =nfft) * upsample
        All_HRIRO3D_newhor_Convolved_Interpolated[i,ii,:]= np.fft.irfft(All_FRO3D_newhor_Convolved[i,ii,:], n =nfft)* upsample

        # rfft  get the FR after interpolation
        All_FRO2D_Convolved_Interpolated[i,ii,:] = np.fft.rfft(All_HRIRO2D_Convolved_Interpolated[i,ii,:])
        All_FRO3D_newhor_Convolved_Interpolated[i,ii,:] = np.fft.rfft(All_HRIRO3D_newhor_Convolved_Interpolated[i,ii,:])
#%%
# Show IR before interpolation and after interpolation        
t2 = np.arange(All_HRIRO2D_Convolved_Interpolated.shape[2])/(48000*4)
t = np.arange(All_HRIRO2D_Convolved.shape[2])/48000
fig, ax = plt.subplots()
ax.plot(t2, All_HRIRO2D_Convolved_Interpolated[0,0,:], 'k.',label = 'After interpolation')
ax.plot(t, All_HRIRO2D_Convolved[0,0,:], 'o',label = 'Before Interpolation')
ax.set_xlim(0.0096, 0.0105)
# ax.set_xlim(0, 0.0105)
plt.title('Impulse response')
plt.legend()
plt.grid()
# %%
"""
----- Read dummy head data and Truncate------
"""
degree = np.arange(30,210,30)
degree = np.insert(degree, 0, 360)
# print(degree)

distance = 3.25
ele = 0
session_name_prefix = 'Farfield_{}m'.format(distance) 

All_KU100HRIR = np.zeros((len(degree),2,Truncate_len),dtype='float64')

for i, angle in enumerate(degree):
   
    session_name = '{}-{:03}deg'.format(session_name_prefix, angle)
    KU100_file_path = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output/JuneListRoom_KU100/Farfield_horizontal_3.25m_Gain0/IR_' + session_name + '/IRs/IR_' + session_name +'_1s2r_S1.wav'

    data_KU100, fs = sf.read(KU100_file_path,dtype = 'float64')
    
    All_KU100HRIR[i,0,:] = data_KU100[0:Truncate_len, 0]
    All_KU100HRIR[i,1,:] = data_KU100[0:Truncate_len, 1]

# %%
"""
----- Do the Interpolation to the KU100 measurement------
"""
All_KU100FR = np.zeros((All_KU100HRIR.shape[0],All_KU100HRIR.shape[1],All_KU100HRIR.shape[2]//2+1),dtype='complex')
All_KU100IR_Interpolated = np.zeros((All_KU100HRIR.shape[0],All_KU100HRIR.shape[1],nfft),dtype='float64')
All_KU100FR_Interpolated = np.zeros((All_KU100HRIR.shape[0],All_KU100HRIR.shape[1],rfft_len),dtype='complex')

for i in range (All_KU100FR .shape[0]):
    for ii in range(All_KU100FR .shape[1]):
    # Truncate the KU100 data then use FFT to interpolation
        All_KU100FR [i,ii,:]= np.fft.rfft(All_KU100HRIR[i,ii,:])
    # 
        All_KU100IR_Interpolated[i,ii,:]= np.fft.irfft(All_KU100FR[i,ii,:], n =nfft)*upsample
    # 
        All_KU100FR_Interpolated[i,ii,:]= np.fft.rfft(All_KU100IR_Interpolated[i,ii,:])

# %%
"""
------ Time alignment to the 2D and 3D HRIRO data------
"""
# Manually timealignment need to -10 samples#haven't do time-alignment to the 3D interpolated dataset
shift_amount2D = np.argmax(np.abs(All_HRIRO2D_Convolved_Interpolated[0,1,:])) - \
                np.argmax(np.abs(All_KU100IR_Interpolated[0,1,:]))-11
shift_amount3D = np.argmax(np.abs(All_HRIRO3D_newhor_Convolved_Interpolated[0,1,:])) - \
                np.argmax(np.abs(All_KU100IR_Interpolated[0,1,:]))-11

All_HRIRO2D_Convolved_Interpolated_Shifted = np.zeros_like(All_HRIRO2D_Convolved_Interpolated)
All_HRIRO3D_newhor_Convolved_Interpolated_Shifted = np.zeros_like(All_HRIRO3D_newhor_Convolved_Interpolated)

All_FRO2D_Convolved_Interpolated_Shifted = np.zeros_like(All_FRO2D_Convolved_Interpolated)
All_FRO3D_newhor_Convolved_Interpolated_Shifted = np.zeros_like(All_FRO3D_newhor_Convolved_Interpolated)

for i in range(All_HRIRO2D_Convolved_Interpolated.shape[0]):
    for ii in range(All_HRIRO2D_Convolved_Interpolated.shape[1]):
        # Shift amount
        All_HRIRO2D_Convolved_Interpolated_Shifted[i,ii,:len(All_HRIRO2D_Convolved_Interpolated[i,ii,:])-shift_amount2D] = \
        All_HRIRO2D_Convolved_Interpolated[i,ii,shift_amount2D:]

        All_HRIRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:len(All_HRIRO3D_newhor_Convolved_Interpolated[i,ii,:])-shift_amount3D] = \
        All_HRIRO3D_newhor_Convolved_Interpolated[i,ii,shift_amount3D:]

        All_FRO2D_Convolved_Interpolated_Shifted[i,ii,:] = np.fft.rfft(All_HRIRO2D_Convolved_Interpolated_Shifted[i,ii,:])
        All_FRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:] = np.fft.rfft(All_HRIRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:])


# %%
"""
------plot IR-----
"""
fs_interpolated = fs * upsample
IRtime = np.arange(All_KU100IR_Interpolated.shape[2])/fs_interpolated

plt.figure(figsize=(24, 40))
plt.suptitle('IRs of KU100 measurements & HRIR2D & Interpolated HRIR3D: Distance: 3.25m Elevation: 0DEG Azimuth:0-180DEG',fontsize = 'large')

for i in range(All_KU100IR_Interpolated.shape[0]):
    for ii in range(All_KU100IR_Interpolated.shape[1]):
        ax = plt.subplot(All_KU100IR_Interpolated.shape[0], 2, i*2+ii+1)
        if ii == 0:
            plt.plot(IRtime,All_HRIRO2D_Convolved_Interpolated_Shifted[i,ii,:],label = f'IR of HRIR2D dataset(L) at {30*i} Deg')
            plt.plot(IRtime,All_KU100IR_Interpolated[i,ii,:],label = f'IR of KU100 measurement(L) at {30*i} Deg')
            plt.plot(IRtime,All_HRIRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:],label = f'IR of HRIR3D dataset(L) at {30*i} Deg')
        else:
            plt.plot(IRtime,All_HRIRO2D_Convolved_Interpolated_Shifted[i,ii,:],label = f'IR of HRIR2D dataset(R) at {30*i} Deg')
            plt.plot(IRtime,All_KU100IR_Interpolated[i,ii,:],label = f'IR of KU100 measurement(R) at {30*i} Deg')
            plt.plot(IRtime,All_HRIRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:],label = f'IR of HRIR3D dataset(R) at {30*i} Deg')
            
        plt.xlim(0.009, 0.011)
        plt.ylim(-0.3, 0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend(fontsize='large') 

plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
#%%
fs_interpolated = fs * upsample
IRtime = np.arange(All_KU100IR_Interpolated.shape[2])/fs_interpolated
position = 6

labels= [[['Koln2D(L)', 'KU100(L)','Koln3D(L)'], ['Koln2D(R)', 'KU100(R)','Koln3D(R)']] for _ in range(7)]
plt.figure(figsize=(20, 10))
plt.suptitle(f'IRs of KU100 measurements & Koln 2D & Interpolated Koln 3D: Turntable angle: {30*position}Deg',fontsize='xx-large')
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.plot(IRtime,All_KU100IR_Interpolated[position,i,:],label = f'{labels[0][i][1]} at {30*position} Deg')
    plt.plot(IRtime,All_HRIRO2D_Convolved_Interpolated_Shifted[position,i,:],label = f'{labels[0][i][0]} at {30*position} Deg')
    plt.plot(IRtime,All_HRIRO3D_newhor_Convolved_Interpolated_Shifted[position,i,:],label = f'{labels[0][i][2]} at {30*position} Deg')
    plt.xlim(0.008, 0.011)
    plt.ylim(-0.3, 0.3)
    plt.xlabel('Time (s)',fontsize='xx-large')
    plt.ylabel('Amplitude',fontsize='xx-large')
    plt.grid(True)
    plt.legend(fontsize='xx-large') 
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
# %%
"""

------ Plot all FR for right and left ear for each position ------

"""
plt.figure(figsize=(12, 20))
plt.suptitle('FRs of KU100 measurements & HRIR2D & Interpolated HRIR3D: Distance: 3.25m Elevation: 0DEG Azimuth:0-180DEG')

freqs = np.fft.rfftfreq(len(All_KU100IR_Interpolated[0,0,:]),1/fs_interpolated)

for i in range(All_KU100FR_Interpolated.shape[0]):
    for ii in range(All_KU100FR_Interpolated.shape[1]):
        ax = plt.subplot(All_KU100IR_Interpolated.shape[0], 2, i*2+ii+1)
        if ii == 0:
            plt.semilogx(freqs,20*np.log10(np.abs(All_FRO2D_Convolved_Interpolated_Shifted[i,ii,:])),label = f'FR of HRIR2D dataset(R) at {30*i} Deg')
            plt.semilogx(freqs,20*np.log10(np.abs(All_KU100FR_Interpolated[i,ii,:])),label = f'FR of KU100 measurement(R) at {30*i} Deg')
            plt.semilogx(freqs,20*np.log10(np.abs(All_FRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:])),label = f'FR of HRIR3D dataset(R) at {30*i} Deg')
        else:
            plt.semilogx(freqs,20*np.log10(np.abs(All_FRO2D_Convolved_Interpolated_Shifted[i,ii,:])),label = f'FR of HRIR2D dataset(L) at {30*i} Deg')
            plt.semilogx(freqs,20*np.log10(np.abs(All_KU100FR_Interpolated[i,ii,:])),label = f'FR of KU100 measurement(L) at {30*i} Deg')
            plt.semilogx(freqs,20*np.log10(np.abs(All_FRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:])),label = f'FR of HRIR3D dataset(R) at {30*i} Deg')

        plt.xlim(20,20000)
        plt.ylim(-60,20)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        plt.legend(fontsize='small') 
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
#%%
freqs = np.fft.rfftfreq(len(All_KU100IR_Interpolated[0,0,:]),1/fs_interpolated)
position = 6
plt.figure(figsize=(20, 10))
plt.suptitle(f'MRs of KU100 measurements & Koln 2D & Interpolated Koln 3D: Turntable angle: {30*position}Deg',fontsize='xx-large')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs,20*np.log10(np.abs(All_KU100FR_Interpolated[position,i,:])),label = f'{labels[0][i][1]} at {30*position} Deg')
    plt.semilogx(freqs,20*np.log10(np.abs(All_FRO2D_Convolved_Interpolated_Shifted[position,i,:])),label = f'{labels[0][i][0]} at {30*position} Deg')
    plt.semilogx(freqs,20*np.log10(np.abs(All_FRO3D_newhor_Convolved_Interpolated_Shifted[position,i,:])),label = f'{labels[0][i][2]} at {30*position} Deg')
    plt.xlim(20, 20000)
    plt.ylim(-60, 20)
    plt.xlabel('Frequency (Hz)',fontsize='xx-large')
    plt.ylabel('Amplitude (dB)',fontsize='xx-large')
    plt.legend(fontsize='xx-large') 
    plt.grid()
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
# %%
"""

------ Plot all PR for right and left ear for each position ------

"""

plt.figure(figsize=(12, 20))
plt.suptitle('PRs of KU100 measurements & HRIR92D & Interpolated HRIR3D: Distance: 3.25m Elevation: 0DEG Azimuth:0-180DEG')

for i in range(All_KU100FR_Interpolated.shape[0]):
    for ii in range(All_KU100FR_Interpolated.shape[1]):
        ax = plt.subplot(All_KU100IR_Interpolated.shape[0], 2, i*2+ii+1)
        if ii == 0:
            plt.semilogx(freqs,np.unwrap(np.angle(All_FRO2D_Convolved_Interpolated_Shifted[i,ii,:])),label = f'PR of HRIR2D dataset(R) at {30*i} Deg')
            plt.semilogx(freqs,np.unwrap(np.angle(All_KU100FR_Interpolated[i,ii,:])),label = f'PR of KU100 measurement(R) at {30*i} Deg')
            plt.semilogx(freqs,np.unwrap(np.angle(All_FRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:])),label = f'PR of HRIR3D dataset(R) at {30*i} Deg')
        else:
            plt.semilogx(freqs,np.unwrap(np.angle(All_FRO2D_Convolved_Interpolated_Shifted[i,ii,:])),label = f'PR of HRIR2D dataset(L) at {30*i} Deg')
            plt.semilogx(freqs,np.unwrap(np.angle(All_KU100FR_Interpolated[i,ii,:])),label = f'PR of KU100 measurement(L) at {30*i} Deg')
            plt.semilogx(freqs,np.unwrap(np.angle(All_FRO3D_newhor_Convolved_Interpolated_Shifted[i,ii,:])),label = f'PR of HRIR3D dataset(R) at {30*i} Deg')

        plt.xlim(20,20000)
        plt.ylim(-1000,50)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (Rad)')
        plt.grid(True)
        plt.legend(fontsize='small') 
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
#%%
position=6
plt.figure(figsize=(20, 10))
plt.suptitle(f'PRs of KU100 measurements & Koln 2D & Interpolated Koln 3D: Turntable angle: {30*position}Deg',fontsize='xx-large')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs,np.unwrap(np.angle(All_KU100FR_Interpolated[position,i,:])),label = f'{labels[0][i][1]} at {30*position} Deg',linestyle=':')
    plt.semilogx(freqs,np.unwrap(np.angle(All_FRO2D_Convolved_Interpolated_Shifted[position,i,:])),label = f'{labels[0][i][0]} at {30*position} Deg',linestyle='--')
    plt.semilogx(freqs,np.unwrap(np.angle(All_FRO3D_newhor_Convolved_Interpolated_Shifted[position,i,:])),label = f'{labels[0][i][2]} at {30*position} Deg',linestyle='-.')
    plt.xlim(20, 20000)
    plt.ylim(-1000, 50)
    plt.xlabel('Frequency (Hz)',fontsize='xx-large')
    plt.ylabel('Phase (Rad)',fontsize='xx-large')
    plt.legend(fontsize='xx-large') 
    plt.grid()
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()

#%%
CalculateComplexErrorAndPlot(All_KU100FR_Interpolated,All_FRO2D_Convolved_Interpolated_Shifted,ylower=-50,yupper=50,title='Complex error of KU100 measurements & Koln 2D')
#%%
CalculateMagnitudeErrorAndPlot(All_KU100FR_Interpolated,All_FRO2D_Convolved_Interpolated_Shifted,ylower=-50,yupper=50,title='Magnitude error of KU100 measurements & Koln 2D')
#%%
CalculatePhaseErrorAndPlot(All_KU100FR_Interpolated,All_FRO2D_Convolved_Interpolated_Shifted,ylower=-100,yupper=50,title='Phase error of KU100 measurements & Koln 2D')
#%%
CalculateComplexErrorAndPlot(All_KU100FR_Interpolated,All_FRO3D_newhor_Convolved_Interpolated_Shifted,ylower=-50,yupper=50,title='Complex error of KU100 measurements & Koln 3D')
#%%
CalculateMagnitudeErrorAndPlot(All_KU100FR_Interpolated,All_FRO3D_newhor_Convolved_Interpolated_Shifted,ylower=-50,yupper=50,title='Magnitude error of KU100 measurements & Koln 3D')
#%%
CalculatePhaseErrorAndPlot(All_KU100FR_Interpolated,All_FRO3D_newhor_Convolved_Interpolated_Shifted,ylower=-100,yupper=50,title='Phase error of KU100 measurements & Koln 3D')

# %%
ITD_KU100=CalculateITD(All_KU100HRIR,48000,20,1000,4)
ITD_HRIRO3D=CalculateITD(All_HRIRO3D_newhor_Convolved,48000,20,1000,4)
ITD_HRIRO2D=CalculateITD(All_HRIRO2D_Convolved,48000,20,1000,4)

Azimuth_deg = -Azimuth/np.pi*180
indices = np.arange(len(ITD_KU100 ))
width = 0.2
plt.figure(figsize=(10, 6))

plt.bar(indices - width, np.abs(ITD_KU100), width=width, label='KU100')
plt.bar(indices, np.abs(ITD_HRIRO2D), width=width, label='Koln 2D')
plt.bar(indices + width, np.abs(ITD_HRIRO3D), width=width, label='Koln 3D')

plt.xlabel('Turntable angle (Deg)',fontsize='x-large')
plt.ylabel('ITD (s)',fontsize='x-large')
plt.ylim(-0.00005,0.0008)
plt.title('ITD of KU100 measurements & Koln 2D & Interpolated Koln 3D',fontsize='x-large')
plt.xticks(indices, [f'{deg}' for deg in Azimuth_deg])
plt.legend(fontsize='x-large')
plt.grid()
plt.show()

# %%
fc1,ILD_KU100=CalculateILD(All_KU100FR,48000)
fc2,ILD_HRIRO3D=CalculateILD(All_FRO3D_newhor_Convolved,48000)
fc3,ILD_HRIRO2D=CalculateILD(All_FRO2D_Convolved,48000)

# plot ILD
position=3
plt.figure(figsize=(20, 10))
plt.semilogx(fc1,ILD_KU100[position,:],label='My KU100 binaural signal',marker = 'o',markerfacecolor='none',markersize=3)
plt.semilogx(fc1,ILD_HRIRO2D[position,:],label='Koln 2D',marker = 'o',markerfacecolor='none',markersize=3)
plt.semilogx(fc1,ILD_HRIRO3D[position,:],label='Koln 3D',marker = 'o',markerfacecolor='none',markersize=3)

plt.title(f'ILD of KU100 measurements & Koln 2D & Interpolated Koln 3D: Turntable angle: {30*position}Deg', fontsize='xx-large',pad=20)
plt.legend(fontsize='x-large', loc='upper left', framealpha=0.5, handlelength=1, handletextpad=0.3)

plt.ylim(-10,20)
plt.xlim(15,20200)
plt.xlabel('Frequency (Hz)', ha='center', fontsize='xx-large')
plt.ylabel('ILD (dB)', va='center', rotation='vertical', fontsize='xx-large')
plt.grid()
plt.tight_layout(rect=[0, 0, 1, 0.98])  
plt.show()
# %%
