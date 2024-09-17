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

# %matplotlib widget
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

def plotIR(pt_analytical, pt_measured=None, fs=48000, xlower=0, xupper=0, ylower=-1, yupper=1, labels=None, title='Impulse Response'):
    IRtime = np.arange(pt_analytical.shape[2]) / fs
    plt.figure(figsize=(12, 20))

    for i in range(pt_analytical.shape[0]):
        for ii in range(pt_analytical.shape[1]):
            ax = plt.subplot(pt_analytical.shape[0], 2, i * 2 + ii + 1)
            angle_label = f'at {30 * i} Deg'
            if ii == 0:
                label_analytical = f'IR(L)_1 {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.plot(IRtime, pt_analytical[i, ii, :], label=label_analytical)
                if pt_measured is not None:
                    label_measured = f'IR(L)_2 {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.plot(IRtime, pt_measured[i, ii, :], label=label_measured)
            else:
                label_analytical = f'IR(R)_1 {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.plot(IRtime, pt_analytical[i, ii, :], label=label_analytical)
                if pt_measured is not None:
                    label_measured = f'IR(R)_2 {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.plot(IRtime, pt_measured[i, ii, :], label=label_measured)

            plt.xlim(xlower, xupper)
            plt.ylim(ylower, yupper)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend(fontsize='small', loc='upper right')
    
    plt.suptitle(title, fontsize='large')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plotFR(pf_analytical, pf_measured=None, fs=48000, ylower=-60, yupper=20, labels=None, title='Frequency Response'):
    freqs = np.fft.rfftfreq((len(pf_analytical[0, 0, :]) - 1) * 2, 1 / fs)
    plt.figure(figsize=(12, 20))

    for i in range(pf_analytical.shape[0]):
        for ii in range(pf_analytical.shape[1]):
            ax = plt.subplot(pf_analytical.shape[0], 2, i * 2 + ii + 1)
            angle_label = f'at {30 * i} Deg'
            if ii == 0:
                label_analytical = f'FR_1(R) {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.semilogx(freqs, 20 * np.log10(np.abs(pf_analytical[i, ii, :])), label=label_analytical)
                if pf_measured is not None:
                    label_measured = f'FR_2(R) {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.semilogx(freqs, 20 * np.log10(np.abs(pf_measured[i, ii, :])), label=label_measured)
            else:
                label_analytical = f'FR_1(L) {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.semilogx(freqs, 20 * np.log10(np.abs(pf_analytical[i, ii, :])), label=label_analytical)
                if pf_measured is not None:
                    label_measured = f'FR_2(L) {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.semilogx(freqs, 20 * np.log10(np.abs(pf_measured[i, ii, :])), label=label_measured)

            plt.xlim(20, 20000)
            plt.ylim(ylower, yupper)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.grid(True)
            plt.legend(fontsize='small', loc='lower left')

    plt.suptitle(title, fontsize='large')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plotPR(pf_analytical, pf_measured=None, fs=48000, ylower=-1000, yupper=50, labels=None, title='Phase Response'):
    freqs = np.fft.rfftfreq((len(pf_analytical[0, 0, :]) - 1) * 2, 1 / fs)
    plt.figure(figsize=(12, 20))

    for i in range(pf_analytical.shape[0]):
        for ii in range(pf_analytical.shape[1]):
            ax = plt.subplot(pf_analytical.shape[0], 2, i * 2 + ii + 1)
            angle_label = f'at {30 * i} Deg'
            if ii == 0:
                label_analytical = f'PR_1(R) {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.semilogx(freqs, np.unwrap(np.angle(pf_analytical[i, ii, :])), label=label_analytical)
                if pf_measured is not None:
                    label_measured = f'PR_2(R) {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.semilogx(freqs, np.unwrap(np.angle(pf_measured[i, ii, :])), label=label_measured)
            else:
                label_analytical = f'PR_1(L) {angle_label}' if labels is None else f'{labels[i][ii][0]} {angle_label}'
                plt.semilogx(freqs, np.unwrap(np.angle(pf_analytical[i, ii, :])), label=label_analytical)
                if pf_measured is not None:
                    label_measured = f'PR_2(L) {angle_label}' if labels is None else f'{labels[i][ii][1]} {angle_label}'
                    plt.semilogx(freqs, np.unwrap(np.angle(pf_measured[i, ii, :])), label=label_measured)

            plt.xlim(20, 20000)
            plt.ylim(ylower, yupper)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (rad)')
            plt.grid(True)
            plt.legend(fontsize='small', loc='lower left')

    plt.suptitle(title, fontsize='large')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def calculate_time_aligned_hrtf(HRTF, SourcePositions, r_H=0.0875, c=343, f_c=2000,fs=48000):

    # Calculate time offsets
    # SourcePositions: (2702*3) 0:azimuth(DEG) 1:Elevation(DEG) 2:Distance(m)
    # HRTF
    azimuth = np.radians(SourcePositions[:, 0])
    elevation = np.radians(SourcePositions[:, 1])

    tau_r= np.zeros(SourcePositions.shape[0])
    tau_l = np.zeros_like(tau_r)
    tau_r = np.cos(elevation) * np.sin(azimuth) * (r_H / c)
    tau_l = -tau_r

    # print("tau_r:", tau_r)
    # print("tau_l:", tau_l)

    # Define frequency range 
    freqs = np.fft.rfftfreq((len(HRTF[0,0,:])-1)*2,1/fs)
    print("Frequency range:", freqs)

    # Compute all-pass filter for all positions
    A_l = np.ones((tau_l.shape[0], freqs.shape[0]), dtype='complex')
    A_r = np.ones((tau_r.shape[0], freqs.shape[0]), dtype='complex')
    
    for i, f in enumerate(freqs):
        if f >= f_c:
            A_l[:, i] = np.exp(-1j * (f - f_c) * tau_l)
            A_r[:, i] = np.exp(-1j * (f - f_c) * tau_r)
    # print("A_l:", A_l)
    # print("A_r:", A_r)

    # Compute the time-aligned HRTF
    HRTF_aligned = np.zeros_like(HRTF, dtype=complex)
    HRTF_aligned[:, 0, :] = HRTF[:, 0, :] * A_l  # leftt ear
    HRTF_aligned[:, 1, :] = HRTF[:, 1, :] * A_r # right ear
    # print("HRTF_aligned:", HRTF_aligned)
    return HRTF_aligned

def CalculateComplexErrorAndPlot(pf_analytical, pf_measured, fs=48000,ylower=-40,yupper=40,title='Complex Error'):
    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    ComplexError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')

    for i in range(ComplexError.shape[0]):
        for ii in range(ComplexError.shape[1]):
            ComplexError[i,ii,:]= (np.abs(pf_analytical[i,ii,:]-pf_measured[i,ii,:])**2 / np.abs(pf_measured[i,ii,:])**2)

    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(title,fontsize='large')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(ComplexError.shape[0]):
            axs[ii].semilogx(freqs, 10*np.log10(ComplexError[i, ii, :]), label=f'Complex error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)')
        axs[ii].set_ylabel('Error (dB)')
        axs[ii].legend(fontsize='small')
        axs[ii].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def CalculateMagnitudeErrorAndPlot(pf_analytical, pf_measured, fs=48000,ylower=-20,yupper=20):
    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    MagnitudeError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')

    for i in range(MagnitudeError .shape[0]):
        for ii in range(MagnitudeError .shape[1]):
            MagnitudeError [i,ii,:]= 10*np.log10((np.abs(pf_analytical[i, ii, :]) )** 2 / (np.abs(pf_measured[i, ii, :]) )** 2)
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Magnitude error ')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(MagnitudeError.shape[0]):
            axs[ii].semilogx(freqs, MagnitudeError[i, ii, :], label=f'Magnitude error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)')
        axs[ii].set_ylabel('Error(dB)')
        axs[ii].legend(fontsize='small')
        axs[ii].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def CalculatePhaseErrorAndPlot(pf_analytical, pf_measured, fs=48000,ylower=-180,yupper=40):
    freqs = np.fft.rfftfreq((len(pf_analytical[0,0,:])-1)*2,1/fs)
    PhaseError = np.zeros((pf_analytical.shape[0],pf_analytical.shape[1],pf_analytical.shape[2]), dtype = 'float64')
    for i in range(PhaseError .shape[0]):
        for ii in range(PhaseError .shape[1]):
            PhaseError[i,ii,:]= np.angle(pf_analytical[i,ii,:] * pf_measured[i,ii,:].conjugate())
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Phase error ')
    labels = ['L', 'R']
    for ii in range(2):
        for i in range(PhaseError.shape[0]):
            axs[ii].semilogx(freqs, np.unwrap(PhaseError[i, ii, :]), label=f'Phase error({labels[ii]}) at {30*i} Deg')
        axs[ii].set_xlim(20, 20000)
        axs[ii].set_ylim(ylower, yupper)
        axs[ii].set_xlabel('Frequency (Hz)')
        axs[ii].set_ylabel('Error(rad)')
        axs[ii].legend(fontsize='small')
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

def CalculateIACC (Pt,fs):
    IACC_values = []
    max_lag_ms = 1
    max_lag = int(max_lag_ms * fs / 1000)
    
    for i in range(Pt.shape[0]):
        Pt_left = Pt[i, 1, :]
        Pt_right = Pt[i, 0, :]

        correlation = signal.correlate(Pt_left, Pt_right, mode='full')#?????
        mid = len(correlation) // 2
        corr= correlation[mid - max_lag:mid + max_lag + 1]
        norm_factor = np.sqrt(np.sum(Pt_left**2) * np.sum(Pt_right**2))#????
        IACF = corr / norm_factor
        IACC = np.max(np.abs(IACF))
        IACC_values.append(IACC)
    
    return np.array(IACC_values)
#%%
"""
------Read 3D HRIR dataset online------
original: Right ear: 1
          Left ear: 0
          elevation:-90 - 90
"""
#Far field 3.25m 
HRIRO3D_path = "/Users/katakuri/Desktop/Msc project/Experiment/HRIR/HRIR_Farfield/3928297/HRIR_L2702.sofa"

# Near field 1m data: Gain factor:0.16 
# HRIRO3D_path = "/Users/katakuri/Desktop/Msc project/Experiment/HRIR/HRIR_Nearfield/4297951/NFHRIR_L2702_SOFA/HRIR_L2702_NF100.sofa"

HRIRO3D = sofa.Database.open(HRIRO3D_path)

#HRIRO3D_source_positions : (azimuth,elevation,r)
HRIRO3D_source_positions_original = HRIRO3D.Source.Position.get_values(system = 'spherical')
HRIRO3D_receiver_positions =HRIRO3D.Receiver.Position.get_values()

# Degree to Radians
HRIRO3D_source_positions = HRIRO3D_source_positions_original.astype(np.float64)

for i in range(HRIRO3D_source_positions.shape[0]):
    HRIRO3D_source_positions[i,0] = (HRIRO3D_source_positions[i,0]/180) * np.pi
    HRIRO3D_source_positions[i,1] = ((90-HRIRO3D_source_positions[i,1])/180) * np.pi

    
# %%
"""
------ Calculate Y matrix ------

"""
order = 4
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
nfft_interpolated = 96000
rfft_interpolated = nfft_interpolated//2 + 1 

HRIRO3D_data_RightEar=np.zeros((HRIRO3D_source_positions.shape[0], HRIRO3D.Dimensions.N),dtype='float64')
HRIRO3D_data_LeftEar=np.zeros((HRIRO3D_source_positions.shape[0], HRIRO3D.Dimensions.N),dtype='float64')

for i in range(HRIRO3D_source_positions.shape[0]):
    HRIRO3D_data_RightEar[i,:]=HRIRO3D.Data.IR.get_values(indices={"M":i, "R":1, "E":0}) 
    HRIRO3D_data_LeftEar[i,:]=HRIRO3D.Data.IR.get_values(indices={"M":i, "R":0, "E":0}) 
 

# HRTFO3D_R = np.zeros((HRIRO3D_source_positions.shape[0],rfftn),dtype = 'complex')
# HRTFO3D_L = np.zeros((HRIRO3D_source_positions.shape[0],rfftn),dtype = 'complex')

HRTFO3D_R_upsample = np.zeros((HRIRO3D_source_positions.shape[0],rfft_interpolated ),dtype = 'complex')
HRTFO3D_L_upsample = np.zeros((HRIRO3D_source_positions.shape[0],rfft_interpolated ),dtype = 'complex')
HRTFO3D_upsample = np.zeros((HRIRO3D_source_positions.shape[0],2,rfft_interpolated ),dtype = 'complex')

for i in range(HRIRO3D_data_RightEar.shape[0]):
    # HRTFO3D_R[i,:] = np.fft.rfft(HRIRO3D_data_RightEar[i,:])
    # HRTFO3D_L[i,:] = np.fft.rfft(HRIRO3D_data_LeftEar[i,:])

    HRTFO3D_R_upsample[i,:] = np.fft.rfft(HRIRO3D_data_RightEar[i,:], n=nfft_interpolated)
    HRTFO3D_L_upsample[i,:] = np.fft.rfft(HRIRO3D_data_LeftEar[i,:], n=nfft_interpolated)
    
    HRTFO3D_upsample[i,0,:] = HRTFO3D_L_upsample[i,:]
    HRTFO3D_upsample[i,1,:] = HRTFO3D_R_upsample[i,:]

freqs = np.fft.rfftfreq(nfft,1/fs)

#%%
HRTFO3D_interpolated_aligned=calculate_time_aligned_hrtf(HRTFO3D_upsample,HRIRO3D_source_positions_original)
HRTFO3D_R_upsample_aligned = np.zeros_like(HRTFO3D_R_upsample)
HRTFO3D_L_interpolated_aligned =np.zeros_like(HRTFO3D_L_upsample)

for i in range(HRTFO3D_interpolated_aligned.shape[0]):
    HRTFO3D_R_upsample_aligned[i,:] = HRTFO3D_interpolated_aligned[i,1,:]
    HRTFO3D_L_interpolated_aligned[i,:] = HRTFO3D_interpolated_aligned[i,0,:]
#%%
"""
------ Calculate the Hnm(f) matix
Hnm(f) : ((order+1)**2 , Z)

"""

Y_pinv = np.linalg.pinv(Y)

Hnm_R = np.dot(Y_pinv, HRTFO3D_R_upsample)
Hnm_L = np.dot(Y_pinv, HRTFO3D_L_upsample)

Hnm_R_aligned = np.dot(Y_pinv, HRTFO3D_R_upsample_aligned)
Hnm_L_aligned = np.dot(Y_pinv, HRTFO3D_L_interpolated_aligned)
#%%
"""

-----Calculate Farfield 3.25m New a_nm(k) = Y_nm(theta_k,phi_k,k)-------

"""
Azimuth_T = np.arange(0,210,30)/180 *np.pi
Azimuth = -Azimuth_T   

Elevation_hor = 90/180 * np.pi
new_position_hor = np.zeros((Azimuth.shape[0],2))

for i in range(Azimuth.shape[0]):
    new_position_hor[i, :] = [Azimuth[i], Elevation_hor]

a_nm_hor_analytical = np.zeros((new_position_hor.shape[0], (order + 1)**2), dtype='float64')

for i in range(new_position_hor.shape[0]):
    index = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            a_nm_hor_analytical[i, index] = sphHarmReal(n, m, new_position_hor[i, 0], new_position_hor[i, 1], normalisation='N3D')
            index += 1

#%%
"""
------ Synthesis analytical binaural signal for different position :Farfield 3.25m ------

"""
Pf_L_analytical = np.dot(a_nm_hor_analytical,Hnm_L)
Pf_R_analytical = np.dot(a_nm_hor_analytical,Hnm_R)

# Pf_L_analytical_aligned = np.dot(a_nm_hor_analytical,Hnm_L_aligned)
# Pf_R_analytical_aligned = np.dot(a_nm_hor_analytical,Hnm_R_aligned)

Pt_analytical = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,nfft_interpolated),dtype = 'float64')

freqs_analytical = np.fft.rfftfreq(len(Pt_analytical[0,0,:]),1/48000)
Pf_analytical = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,rfft_interpolated),dtype = 'complex')

# Pt_analytical_aligned = np.zeros_like(Pt_analytical)
# Pf_analytical_aligned = np.zeros_like(Pf_analytical)

for i in range(Pt_analytical.shape[0]):

    Pt_analytical[i,0,:] = np.fft.irfft(Pf_L_analytical[i,:])
    Pt_analytical[i,1,:] = np.fft.irfft(Pf_R_analytical[i,:])

    Pf_analytical[i,0,:] = Pf_L_analytical[i,:]
    Pf_analytical[i,1,:] = Pf_R_analytical[i,:]

    # Pt_analytical_aligned[i,0,:] = np.fft.irfft(Pf_R_analytical_aligned[i,:])
    # Pt_analytical_aligned[i,1,:] = np.fft.irfft(Pf_L_analytical_aligned[i,:])

    # Pf_analytical_aligned[i,0,:] = Pf_R_analytical_aligned[i,:]
    # Pf_analytical_aligned[i,1,:] = Pf_L_analytical_aligned[i,:]
#%%
plotIR(Pt_analytical,xlower=0,xupper=0.002,ylower=-0.5,yupper=1.0)    
#%%
"""
------ Read 2D Koln's dataset to compare ------
"""
HRIRO2D_path = "/Users/katakuri/Desktop/Msc project/Experiment/HRIR/HRIR_Farfield/3928297/HRIR_CIRC360.sofa"
HRIRO2D = sofa.Database.open(HRIRO2D_path)
HRIRO2D_source_positions = HRIRO2D.Source.Position.get_values(system = 'spherical')
HRIRO2D_receiver_positions = HRIRO2D.Receiver.Position.get_values(system = 'spherical')

DEG = [0,330,300,270,240,210,180]
measurement_index = np.zeros(len(DEG),dtype = 'int')
for i  in range (len(DEG)):
    measurement_index[i] = np.where(HRIRO2D_source_positions[:,0] == DEG[i])[0]


# Read 2D dataset
HRIRO2D_data = np.zeros((len(measurement_index),2,128))

for i,measurement in enumerate(measurement_index):
    print(measurement)
    for ii in np.arange(HRIRO2D.Dimensions.R):  
        HRIRO2D_data[i,ii,:]=HRIRO2D.Data.IR.get_values(indices={"M":measurement, "R":ii, "E":0})


HRTFO2D_interpolated = np.zeros((HRIRO2D_data.shape[0],2,rfft_interpolated ),dtype = 'complex')
HRIRO2D_interpolated = np.zeros((HRIRO2D_data.shape[0],2,nfft_interpolated ),dtype = 'float64')

for i in range(HRTFO2D_interpolated.shape[0]):
    for ii in range(HRTFO2D_interpolated.shape[1]):
        HRTFO2D_interpolated[i,ii,:]=np.fft.rfft(HRIRO2D_data[i,ii,:], n=nfft_interpolated)

        HRIRO2D_interpolated[i,ii,:] = np.fft.irfft(HRTFO2D_interpolated[i,ii,:])
#%%        
plotIR(HRIRO2D_data,xlower=0,xupper=0.002,ylower=-0.5,yupper=1.0)       
#%%
plotIR(Pt_analytical,HRIRO2D_interpolated,xlower=0,xupper=0.001,ylower=-0.5,yupper=0.8)
plotFR(Pf_analytical,HRTFO2D_interpolated)
plotPR(Pf_analytical,HRTFO2D_interpolated,ylower = -200,yupper=50)

# plotIR(Pt_analytical_aligned,HRIRO2D_interpolated,xlower=0,xupper=0.001,ylower=-0.5,yupper=0.8)
# plotFR(Pf_analytical_aligned,HRTFO2D_interpolated)
# plotPR(Pf_analytical_aligned,HRTFO2D_interpolated,ylower = -200,yupper=50)
#%%
CalculateComplexErrorAndPlot(Pf_analytical,HRTFO2D_interpolated)
CalculateMagnitudeErrorAndPlot(Pf_analytical,HRTFO2D_interpolated)
CalculatePhaseErrorAndPlot(Pf_analytical,HRTFO2D_interpolated)

# CalculateComplexErrorAndPlot(Pf_analytical_aligned,HRTFO2D_interpolated)
# CalculateMagnitudeErrorAndPlot(Pf_analytical_aligned,HRTFO2D_interpolated)
# CalculatePhaseErrorAndPlot(Pf_analytical_aligned,HRTFO2D_interpolated)
#%%
"""
------ check the my calculation of spherical harmonics is same as the sparta ------
------ max gain = 5dB
"""
# a_nm_path_0 = '/Users/katakuri/Documents/REAPER Media/EM32_Farfield_0DEG_3.wav'
# a_nm_path_0 ='/Users/katakuri/Documents/REAPER Media/EM32_Farfield_0DEG_Gain5.wav'
a_nm_path_0 = '/Users/katakuri/Documents/REAPER Media/EM32_Farfield_3.25m_0DEG_Gain5_5.wav'
a_nm_hor_measured_0,fs = sf.read(a_nm_path_0,dtype = 'float64')
a_nm_hor_measured_0_T= a_nm_hor_measured_0.T
a_nm_hor_measured_0_T_fft = np.zeros((a_nm_hor_measured_0_T.shape[0],rfft_interpolated),dtype = 'complex')
for ii in range(a_nm_hor_measured_0_T.shape[0]):
    a_nm_hor_measured_0_T_fft[ii,:] = np.fft.rfft(a_nm_hor_measured_0_T[ii,:])

a_ratio_measured_0 = np.zeros_like(a_nm_hor_measured_0_T_fft,dtype = 'complex')

for i in range(a_nm_hor_measured_0_T_fft.shape[0]):
    a_ratio_measured_0[i,:]=a_nm_hor_measured_0_T_fft[i,:]/a_nm_hor_measured_0_T_fft[0,:]

# a_ratio_analytical_0 = a_nm_hor_analytical[0,:]/a_nm_hor_analytical[0,0]

ynm_0 = np.zeros((order + 1) ** 2)
index = 0
for n in range(order + 1):
    for m in range(-n, n + 1):
        ynm_0[index] = sphHarmReal(n, m,0, (np.pi/2), normalisation='N3D')
        index += 1
a_ratio_analytical_0 = ynm_0/ynm_0[0] 
freqs_0 = np.fft.rfftfreq(96000,1/48000)

plt.figure(figsize=(12, 36))
x = np.arange(4,9)
plt.suptitle('Magnitude and Phase for anm(k)/a00(k) vs Ynm/Y00 at 0 azimuth of MaxGain30dB(N=3, M=(-3,3))')
for i ,n in enumerate(x):
    plt.subplot(len(x),2,i*2+1)
    plt.semilogx(freqs_0,20*np.log10(np.abs(a_ratio_measured_0[n,:])),label = 'anm(f)/a00(f)')
    plt.hlines(20*np.log10(np.abs(a_ratio_analytical_0[n])),20,24000,colors='r',label = 'Ynm/Y00')
    plt.xlim(20,20000)
    plt.ylim(-30,20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Ratio (dB)')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()

    plt.subplot(len(x),2,i*2+2)
    plt.semilogx(freqs_0,180/np.pi*np.angle(a_ratio_measured_0[n,:]),label = 'anm(f)/a00(f)')
    plt.hlines(180/np.pi*np.angle(a_ratio_analytical_0[n]),20,24000,colors='r',label = 'Ynm/Y00')
    plt.xlim(20,20000)
    plt.ylim(-190,190)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Degree ')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
#%%
file_path_1 = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output/JuneListRoom_Omni/Farfield_horizontal_3.25m_Gain0/IR_Farfield_3.25m/IRs/IR_Farfield_3.25m_1s1r_S1.wav'
OmniIR_data, fs = sf.read(file_path_1, dtype = 'float64')
Pf_omni = np.fft.rfft(OmniIR_data)
fac = np.abs(ynm_0[0])/np.mean(np.abs(a_nm_hor_measured_0_T_fft[0,200:6000]))
# fac = np.abs(ynm_0[0])/np.abs(a_nm_hor_measured_0_T_fft[0,3000])
# fac = np.abs(ynm_0[0])/np.abs(Pf_omni[3000])
# %%
"""

------ Synthesis binaural signal from my EM32 measurements order:4-------

"""
Azimuth_deg = -Azimuth/np.pi *180
Pt_measured = np.zeros_like(Pt_analytical)
Pf_measured = np.zeros_like(Pf_analytical)

Pt_measured_aligned = np.zeros_like(Pt_measured)
Pf_measured_aligned = np.zeros_like(Pf_measured)

for i in range(Azimuth_deg.shape[0]):
    a_nm_path = f'/Users/katakuri/Documents/REAPER Media/EM32_Farfield_3.25m_{int(Azimuth_deg[i])}DEG_Gain5_5.wav'
    print(a_nm_path)

    # read the a_nm data in time domain
    a_nm_hor_measured,fs = sf.read(a_nm_path,dtype = 'float64')
    # Calibration by multiplying fac
    a_nm_hor_measured = a_nm_hor_measured*fac
    
    a_nm_hor_measured_T= a_nm_hor_measured.T
    
    # a_nm time domain to the frequency domain
    a_nm_hor_measured_T_fft = np.zeros((a_nm_hor_measured_T.shape[0],rfft_interpolated),dtype = 'complex')
    for ii in range(a_nm_hor_measured_T.shape[0]):
        a_nm_hor_measured_T_fft[ii,:] = np.fft.rfft(a_nm_hor_measured_T[ii,:])

    print(np.mean(np.abs(a_nm_hor_measured_T_fft[0,200:6000])))
    # Synthesis binaural signal in the frequency domain
    Pf = np.zeros((2,rfft_interpolated), dtype='complex')
    Pf_aligned = np.zeros_like(Pf)

    for iii in range(Hnm_R.shape[1]):
        for j in range(Hnm_R.shape[0]):
            Pf[1,iii] += a_nm_hor_measured_T_fft[j,iii] * Hnm_R[j,iii] 
            Pf[0,iii] += a_nm_hor_measured_T_fft[j,iii] * Hnm_L[j,iii] 

            Pf_aligned[1,iii] += a_nm_hor_measured_T_fft[j,iii] * Hnm_R_aligned[j,iii]
            Pf_aligned[0,iii] += a_nm_hor_measured_T_fft[j,iii] * Hnm_L_aligned[j,iii] 

    # Synthesis binaural signal in the time domain
    Pt = np.zeros((2,nfft_interpolated),dtype = 'float64')
    Pt_aligned = np.zeros_like(Pt)

    Pt[0,:] = np.fft.irfft(Pf[0,:])
    Pt[1,:] = np.fft.irfft(Pf[1,:])

    Pt_aligned[0,:] = np.fft.irfft(Pf_aligned[0,:])
    Pt_aligned[1,:] = np.fft.irfft(Pf_aligned[1,:])

    # Store Pt and Pf for all positions
    Pt_measured[i,:,:] = Pt
    Pf_measured[i,:,:] = Pf

    Pt_measured_aligned[i,:,:]=Pt_aligned
    Pf_measured_aligned[i,:,:]=Pf_aligned
#%%
"""
------ check the my calculation of spherical harmonics is same as the sparta ------
------ max gain = 5dB
"""
# a_nm_path_0 = '/Users/katakuri/Documents/REAPER Media/EM32_Farfield_0DEG_3.wav'
# a_nm_path_0 ='/Users/katakuri/Documents/REAPER Media/EM32_Farfield_0DEG_Gain5.wav'
a_nm_path_0 = '/Users/katakuri/Documents/REAPER Media/EM32_Farfield_3.25m_0DEG_Gain5_4.wav'
a_nm_hor_measured_0,fs = sf.read(a_nm_path_0,dtype = 'float64')
a_nm_hor_measured_0_T= a_nm_hor_measured_0.T
a_nm_hor_measured_0_T_fft = np.zeros((a_nm_hor_measured_0_T.shape[0],rfft_interpolated),dtype = 'complex')
for ii in range(a_nm_hor_measured_0_T.shape[0]):
    a_nm_hor_measured_0_T_fft[ii,:] = np.fft.rfft(a_nm_hor_measured_0_T[ii,:])

a_ratio_measured_0 = np.zeros_like(a_nm_hor_measured_T_fft,dtype = 'complex')
for i in range(a_nm_hor_measured_0_T_fft.shape[0]):
    a_ratio_measured_0[i,:]=a_nm_hor_measured_0_T_fft[i,:]/a_nm_hor_measured_0_T_fft[0,:]

# a_ratio_analytical_0 = a_nm_hor_analytical[0,:]/a_nm_hor_analytical[0,0]

ynm_0 = np.zeros((order + 1) ** 2)
index = 0
for n in range(order + 1):
    for m in range(-n, n + 1):
        ynm_0[index] = sphHarmReal(n, m,0, (np.pi/2), normalisation='N3D')
        index += 1
a_ratio_analytical_0 = ynm_0/ynm_0[0] 
freqs_0 = np.fft.rfftfreq(96000,1/48000)

plt.figure(figsize=(12, 36))
x = np.arange(4,9)
plt.suptitle('Magnitude and Phase for anm(k)/a00(k) vs Ynm/Y00 at 0 azimuth of MaxGain30dB(N=3, M=(-3,3))')
for i ,n in enumerate(x):
    plt.subplot(len(x),2,i*2+1)
    plt.semilogx(freqs_0,20*np.log10(np.abs(a_ratio_measured_0[n,:])),label = 'anm(f)/a00(f)')
    plt.hlines(20*np.log10(np.abs(a_ratio_analytical_0[n])),20,24000,colors='r',label = 'Ynm/Y00')
    plt.xlim(20,20000)
    plt.ylim(-30,20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Ratio (dB)')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()

    plt.subplot(len(x),2,i*2+2)
    plt.semilogx(freqs_0,180/np.pi*np.angle(a_ratio_measured_0[n,:]),label = 'anm(f)/a00(f)')
    plt.hlines(180/np.pi*np.angle(a_ratio_analytical_0[n]),20,24000,colors='r',label = 'Ynm/Y00')
    plt.xlim(20,20000)
    plt.ylim(-190,190)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Degree ')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()


#%%
plotIR(Pt_measured,Pt_measured_aligned,xlower=0.006,xupper=0.015,ylower=-0.5,yupper=1.0)
plotFR(Pf_measured,Pf_measured_aligned)

# %%
"""

------ Synthesize binaural signals from Eigenmike (order 4)  VS  
Interpolated binaural signals from Koln 3D database (order 35) ------

"""

horder = 35
Y_horder = np.zeros((HRIRO3D_source_positions.shape[0],(horder+1)**2),dtype = 'float64')
index = 0
for i in range (HRIRO3D_source_positions.shape[0]):
    index = 0
    for n in range(horder + 1):
        for m in range(-n,n+1):
            Y_horder[i, index] = sphHarmReal(n, m, HRIRO3D_source_positions[i, 0], HRIRO3D_source_positions[i, 1], normalisation='N3D')
            index += 1
con_num_h = np.linalg.cond(Y_horder)
print(f'Y_horder matrix: order: {horder} con_num: {con_num_h}')

Y_horder_pinv = np.linalg.pinv(Y_horder)

Hnm_R_horder = np.dot(Y_horder_pinv, HRTFO3D_R_upsample)
Hnm_L_horder = np.dot(Y_horder_pinv, HRTFO3D_L_upsample)

# calculate the new Y matrix 

Y_new_interpolated = np.zeros((new_position_hor.shape[0], (horder + 1)**2), dtype='float64')
for i in range(new_position_hor.shape[0]):
    index = 0
    for n in range(horder + 1):
        for m in range(-n, n + 1):
            Y_new_interpolated[i, index] = sphHarmReal(n, m, new_position_hor[i, 0], new_position_hor[i, 1], normalisation='N3D')
            index += 1

Pf_L_horder_interpolated = np.dot(Y_new_interpolated,Hnm_L_horder)
Pf_R_horder_interpolated = np.dot(Y_new_interpolated,Hnm_R_horder)

Pt_horder_interpolated = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,nfft_interpolated),dtype = 'float64')
Pf_horder_interpolated = np.zeros((new_position_hor.shape[0],HRIRO3D.Dimensions.R,rfft_interpolated),dtype = 'complex')

for i in range(Pt_horder_interpolated .shape[0]):

    Pt_horder_interpolated [i,0,:] = np.fft.irfft(Pf_L_horder_interpolated[i,:])
    Pt_horder_interpolated [i,1,:] = np.fft.irfft(Pf_R_horder_interpolated[i,:])

    Pf_horder_interpolated[i,0,:] = Pf_L_horder_interpolated[i,:]
    Pf_horder_interpolated[i,1,:] = Pf_R_horder_interpolated[i,:]
#%%
plotIR(Pt_horder_interpolated,xlower=0.0,xupper=0.002,ylower=-0.5,yupper=1.0)

# %%
# Read omni data Far field and convolve

file_path_1 = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output/JuneListRoom_Omni/Farfield_horizontal_3.25m_Gain0/IR_Farfield_3.25m/IRs/IR_Farfield_3.25m_1s1r_S1.wav'
OmniIR_data, fs = sf.read(file_path_1, dtype = 'float64')
Pf_omni = np.fft.rfft(OmniIR_data)
#%%
fac2 = np.abs(ynm_0[0])/np.mean(np.abs(Pf_omni[200:6000]))
OmniIR_data = OmniIR_data *fac2
#%%
Pt_horder_interpolated_convolved = np.zeros_like(Pt_horder_interpolated)
# Pt_horder_interpolated_convolved = np.zeros((Pt_horder_interpolated.shape[0],Pt_horder_interpolated.shape[1],2*Pt_horder_interpolated.shape[2]-1), dtype = 'float')
Pf_horder_interpolated_convolved = np.zeros_like(Pf_horder_interpolated)

for i in np.arange(Pt_horder_interpolated.shape[0]):
    for ii in np.arange(Pt_horder_interpolated.shape[1]):
        
        convolved_signal= np.convolve(Pt_horder_interpolated[i,ii,:],OmniIR_data, mode = 'full')
        Pt_horder_interpolated_convolved[i,ii,:] = convolved_signal[0:nfft_interpolated,]

        Pf_horder_interpolated_convolved[i,ii,:] = np.fft.rfft(Pt_horder_interpolated_convolved[i,ii,:])
        # Pt_horder_interpolated_convolved[i,ii,:] = np.fft.irfft(Pf_horder_interpolated_convolved[i,ii,:])
#%%
Pf_omni = np.fft.rfft(OmniIR_data)
plt.figure()
plt.semilogx(freqs_analytical,20*np.log10(np.abs(Pf_omni)))
plt.semilogx(freqs_analytical,20*np.log10(np.abs(a_nm_hor_measured_T_fft[0,:])))
plt.grid()
plt.show()
#%%
shift_amount_Koln = np.argmax(np.abs(Pt_measured[0,1,:])) - \
                np.argmax(np.abs(Pt_horder_interpolated_convolved[0,1,:]))

Pt_measured_Kolnshifted = np.zeros_like(Pt_measured)
Pf_measured_Kolnshifted = np.zeros_like(Pf_measured)

for i in range(Pt_measured.shape[0]):
    for ii in range(Pt_measured.shape[1]):
        # Shift amount

        Pt_measured_Kolnshifted [i,ii,:len(Pt_measured_Kolnshifted [i,ii,:])-shift_amount_Koln] = \
        Pt_measured[i,ii,shift_amount_Koln:]

        Pf_measured_Kolnshifted [i,ii,:] = np.fft.rfft(Pt_measured_Kolnshifted[i,ii,:])
#%%
label1= [[['Synthesized(L)', 'Koln(L)'], ['Synthesized(R)', 'Koln(R)']] for _ in range(7)]
title1 = 'Far-field IR of binaural signal at different turntable position' 
plotIR(Pt_measured_Kolnshifted,Pt_horder_interpolated_convolved ,xlower=0.008,xupper=0.0125,ylower=-0.5,yupper=0.5,labels=label1,title=title1)
#%%
time = np.arange(0,len(Pt_measured[0,0,:]))/fs
plt.figure(figsize=(24, 12))
plt.suptitle('IR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.plot(time, Pt_measured_Kolnshifted[0, i, :], label=label1[0][i][0])
    plt.plot(time, Pt_horder_interpolated_convolved[0, i, :], label=label1[0][i][1])
    plt.xlim(0, 0.02)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
plt.show()
#%%
label2= [[['Synthesized(L)', 'Koln(L)'], ['Synthesized(R)', 'Koln(R)']] for _ in range(7)]
title2 = 'Far-field FR of binaural signal at different turntable position' 
plotFR(Pf_measured_Kolnshifted,Pf_horder_interpolated_convolved,labels=label2,title=title2) 

#%%
plt.figure(figsize=(20, 12))
plt.suptitle('FR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured[0, i, :])), label=label1[0][i][0])
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_horder_interpolated_convolved[0, i, :])), label=label1[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-60, 25)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.legend()
    plt.grid()

plt.show()
#%%
label3= [[['Synthesized(L)', 'Koln(L)'], ['Synthesized(R)', 'Koln(R)']] for _ in range(7)]
title3 = 'Far-field PR of binaural signal at different turntable position' 
plotPR(Pf_measured_Kolnshifted,Pf_horder_interpolated_convolved,labels=label3,title=title3) 
#%%
plt.figure(figsize=(20, 12))
plt.suptitle('PR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured[0, i, :])), label=label1[0][i][0])
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_horder_interpolated_convolved[0, i, :])), label=label1[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-180, 40)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Phase(rad)')
    plt.legend()
    plt.grid()

plt.show()
#%%
CalculateComplexErrorAndPlot(Pf_measured_Kolnshifted,Pf_horder_interpolated_convolved,ylower=-50,yupper=50) 
CalculateMagnitudeErrorAndPlot(Pf_measured_Kolnshifted,Pf_horder_interpolated_convolved,ylower=-50,yupper=50)
CalculatePhaseErrorAndPlot(Pf_measured_Kolnshifted,Pf_horder_interpolated_convolved,ylower=-50,yupper=100)        
# %%
"""

------ Synthesize binaural signals from Eigenmike (order 4)  VS   
Binaural signals from Bingcheng measurement of KU-100 ------

"""

degree = np.arange(30,210,30)
degree = np.insert(degree, 0, 360)
# print(degree)

distance = 3.25
ele = 0
session_name_prefix = 'Farfield_{}m'.format(distance) 

HRIR_KU100 = np.zeros((len(degree),2,nfft_interpolated),dtype='float64')
HRTF_KU100 = np.zeros((len(degree),2,rfft_interpolated),dtype='complex')
for i, angle in enumerate(degree):
   
    session_name = '{}-{:03}deg'.format(session_name_prefix, angle)
    KU100_file_path = '/Users/katakuri/Desktop/Msc project/vaae-audio-toolbox/output/JuneListRoom_KU100/Farfield_horizontal_3.25m_Gain0/IR_' + session_name + '/IRs/IR_' + session_name +'_1s2r_S1.wav'

    data_KU100, fs = sf.read(KU100_file_path,dtype = 'float64')
    
    HRIR_KU100 [i,0,:] = data_KU100[:,0]
    HRIR_KU100 [i,1,:] = data_KU100[:,1]

    HRTF_KU100[i,0,:] = np.fft.rfft(HRIR_KU100 [i,0,:])
    HRTF_KU100[i,1,:] = np.fft.rfft(HRIR_KU100 [i,1,:])

shift_amount_KU100 = np.argmax(np.abs(Pt_measured[0,1,:])) - \
                np.argmax(np.abs(HRIR_KU100[0,1,:]))

Pt_measured_KU100shifted = np.zeros_like(Pt_measured)
Pf_measured_KU100shifted = np.zeros_like(Pf_measured)

for i in range(Pt_measured.shape[0]):
    for ii in range(Pt_measured.shape[1]):
        # Shift amount

        Pt_measured_KU100shifted [i,ii,:len(Pt_measured_KU100shifted [i,ii,:])-shift_amount_KU100] = \
        Pt_measured[i,ii,shift_amount_KU100:]

        Pf_measured_KU100shifted [i,ii,:] = np.fft.rfft(Pt_measured_KU100shifted[i,ii,:])
#%%
label7= [[['Synthesized(L)', 'KU100(L)'], ['Synthesized(R)', 'KU100(R)']] for _ in range(7)]
title7 = 'Far-field IR of binaural signal at different turntable position' 
plotIR(Pt_measured_KU100shifted,HRIR_KU100,xlower=0.008,xupper=0.0125,ylower=-0.1,yupper=0.3,labels=label7,title=title7)
#%%
time = np.arange(0,len(Pt_measured[0,0,:]))/fs
plt.figure(figsize=(24, 12))
plt.suptitle('IR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.plot(time, Pt_measured_KU100shifted[0, i, :], label=label7[0][i][0])
    plt.plot(time, HRIR_KU100[0, i, :], label=label7[0][i][1])
    plt.xlim(0, 0.015)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
plt.show()
#%%
label8= [[['Synthesized(L)', 'KU100(L)'], ['Synthesized(R)', 'KU100(R)']] for _ in range(7)]
title8 = 'Far-field FR of binaural signal at different turntable position' 
plotFR(Pf_measured_KU100shifted,HRTF_KU100,labels=label8,title=title8) 
#%%
plt.figure(figsize=(20, 12))
plt.suptitle('FR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured_KU100shifted[0, i, :])), label=label8[0][i][0])
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(HRTF_KU100[0, i, :])), label=label8[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-60, 25)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.legend()
    plt.grid()

plt.show()
#%%
label9= [[['Synthesized(L)', 'KU100(L)'], ['Synthesized(R)', 'KU100(R)']] for _ in range(7)]
title9 = 'Far-field FR of binaural signal at different turntable position' 
plotPR(Pf_measured_KU100shifted,HRTF_KU100,labels=label9,title=title9) 
#%%
plt.figure(figsize=(20, 12))
plt.suptitle('PR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured_KU100shifted[0, i, :])), label=label9[0][i][0])
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(HRTF_KU100[0, i, :])), label=label9[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-180, 60)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Phase(rad)')
    plt.legend()
    plt.grid()

plt.show()
#%%
CalculateComplexErrorAndPlot(Pf_measured_KU100shifted,HRTF_KU100,ylower=-50,yupper=50) 
CalculateMagnitudeErrorAndPlot(Pf_measured_KU100shifted,HRTF_KU100,ylower=-50,yupper=50)
CalculatePhaseErrorAndPlot(Pf_measured_KU100shifted,HRTF_KU100,ylower=-50,yupper=100)  

#%%
"""

------ Synthesize binaural signals from Eigenmike (order 4)  VS   
Synthesize and aligned binaural signals from Eigenmike (order 4) vs 
Analytical binaural signal------

"""
Pt_analytical_convolved = np.zeros_like(Pt_analytical)
Pf_analytical_convolved = np.zeros_like(Pf_analytical)
for i in np.arange(Pt_analytical_convolved .shape[0]):
    for ii in np.arange(Pt_analytical_convolved.shape[1]):
        
        convolved_signal= np.convolve(Pt_analytical[i,ii,:],OmniIR_data, mode = 'full')
        Pt_analytical_convolved[i,ii,:] = convolved_signal[0:nfft_interpolated,]

        Pf_analytical_convolved[i,ii,:] = np.fft.rfft(Pt_analytical_convolved[i,ii,:])

shift_amount_synthesized = np.argmax(np.abs(Pt_analytical_convolved[0,1,:])) - \
                np.argmax(np.abs(Pt_measured[0,1,:]))

Pt_analytical_convolved_SynthsisShifted = np.zeros_like(Pt_measured)
Pf_analytical_convolved_SynthsisShifted  = np.zeros_like(Pf_measured)

for i in range(Pt_measured.shape[0]):
    for ii in range(Pt_measured.shape[1]):
        # Shift amount
        if shift_amount_synthesized >= 0:
            Pt_analytical_convolved_SynthsisShifted[i, ii, :len(Pt_analytical_convolved_SynthsisShifted[i, ii, :])-shift_amount_synthesized] = \
                Pt_analytical_convolved[i, ii, shift_amount_synthesized:]
        else:
            Pt_analytical_convolved_SynthsisShifted[i, ii, -shift_amount_synthesized:] = \
                Pt_analytical_convolved[i, ii, :shift_amount_synthesized]

        Pf_analytical_convolved_SynthsisShifted [i,ii,:] = np.fft.rfft(Pt_analytical_convolved_SynthsisShifted[i,ii,:])

label4= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title4 = 'Far-field IR of binaural signal at different turntable position' 
plotIR(Pt_measured,Pt_measured_aligned,xlower=0.008,xupper=0.0125,ylower=-0.5,yupper=0.5,labels=label4,title=title4)
#%%
time = np.arange(0,len(Pt_measured[0,0,:]))/fs
plt.figure(figsize=(24, 12))
plt.suptitle('IR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.plot(time,Pt_analytical_convolved_SynthsisShifted[0,i,:],label = 'Analytical binaural signal')
    plt.plot(time, Pt_measured[0, i, :], label=label4[0][i][0])
    plt.plot(time, Pt_measured_aligned[0, i, :], label=label4[0][i][1])
    plt.xlim(0, 0.02)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
plt.show()
#%%
label5= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title5 = 'Far-field FR of binaural signal at different turntable position' 
plotFR(Pf_measured,Pf_measured_aligned,labels=label5,title=title5) 

#%%
plt.figure(figsize=(20, 12))
plt.suptitle('FR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_analytical_convolved_SynthsisShifted[0, i, :])), label='Analytical Binaural signal')
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured[0, i, :])), label=label5[0][i][0])
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured_aligned[0, i, :])), label=label5[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-60, 25)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.legend()
    plt.grid()

plt.show()
#%%
label6= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title6 = 'Far-field PR of binaural signal at different turntable position' 
plotPR(Pf_measured,Pf_measured_aligned,labels=label6,title=title6) 
#%%
plt.figure(figsize=(20, 12))
plt.suptitle('PR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured[0, i, :])), label=label6[0][i][0])
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured_aligned[0, i, :])), label=label6[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-180, 40)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Phase(rad)')
    plt.legend()
    plt.grid()

plt.show()
#%%
CalculateComplexErrorAndPlot(Pf_measured,Pf_analytical_convolved_SynthsisShifted,ylower=-50,yupper=50) 
CalculateMagnitudeErrorAndPlot(Pf_measured,Pf_analytical_convolved_SynthsisShifted,ylower=-50,yupper=50)
CalculatePhaseErrorAndPlot(Pf_measured,Pf_analytical_convolved_SynthsisShifted,ylower=-50,yupper=100)
#%%
"""

------ Synthesize binaural signals from Eigenmike (order 4)  VS   
Synthesize and aligned binaural signals from Eigenmike (order 4) vs 
high-order binaural signal(35)------

"""

shift_amount_synthesized = np.argmax(np.abs(Pt_horder_interpolated_convolved[0,1,:])) - \
                np.argmax(np.abs(Pt_measured[0,1,:]))

Pt_horder_interpolated_convolved_SynthsisShifted = np.zeros_like(Pt_measured)
Pf_horder_interpolated_convolved_SynthsisShifted = np.zeros_like(Pf_measured)

for i in range(Pt_measured.shape[0]):
    for ii in range(Pt_measured.shape[1]):
        # Shift amount
        if shift_amount_synthesized >= 0:
            Pt_horder_interpolated_convolved_SynthsisShifted[i, ii, :len(Pt_horder_interpolated_convolved_SynthsisShifted[i, ii, :])-shift_amount_synthesized] = \
                Pt_horder_interpolated_convolved[i, ii, shift_amount_synthesized:]
        else:
            Pt_horder_interpolated_convolved_SynthsisShifted[i, ii, -shift_amount_synthesized:] = \
                Pt_horder_interpolated_convolved[i, ii, :shift_amount_synthesized]

        Pf_horder_interpolated_convolved_SynthsisShifted [i,ii,:] = np.fft.rfft(Pt_horder_interpolated_convolved_SynthsisShifted[i,ii,:])

#%%
label10= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title10 = 'Far-field IR of binaural signal at different turntable position' 
plotIR(Pt_measured,Pt_measured_aligned,xlower=0.008,xupper=0.0125,ylower=-0.5,yupper=0.5,labels=label10,title=title10)
#%%
time = np.arange(0,len(Pt_measured[0,0,:]))/fs
plt.figure(figsize=(24, 12))
plt.suptitle('IR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.plot(time,Pt_horder_interpolated_convolved_SynthsisShifted[0,i,:],label = 'Koln binaural signal(order35)')
    plt.plot(time, Pt_measured[0, i, :], label=label10[0][i][0])
    plt.plot(time, Pt_measured_aligned[0, i, :], label=label10[0][i][1])
    plt.xlim(0, 0.02)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
plt.show()

#%%
label11= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title11 = 'Far-field FR of binaural signal at different turntable position' 
plotFR(Pf_measured,Pf_measured_aligned,labels=label11,title=title11) 

#%%
plt.figure(figsize=(20, 12))
plt.suptitle('FR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_horder_interpolated_convolved_SynthsisShifted[0, i, :])), label='Koln binaural signal(order35)')
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured[0, i, :])), label=label11[0][i][0])
    plt.semilogx(freqs_analytical, 20*np.log10(np.abs(Pf_measured_aligned[0, i, :])), label=label11[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-60, 25)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.legend()
    plt.grid()
plt.show()

#%%
label12= [[['Synthesized(L)', 'Synthesized&aligned(L)'], ['Synthesized(R)', 'Synthesized&aligned(R)']] for _ in range(7)]
title12 = 'Far-field PR of binaural signal at different turntable position' 
plotPR(Pf_measured,Pf_measured_aligned,labels=label12,title=title12) 
#%%
plt.figure(figsize=(20, 12))
plt.suptitle('PR of Binaural signal at 0Deg')
for i in range(2): 
    plt.subplot(1, 2, i + 1)
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured[0, i, :])), label=label12[0][i][0])
    plt.semilogx(freqs_analytical, np.unwrap(np.angle(Pf_measured_aligned[0, i, :])), label=label12[0][i][1])
    plt.xlim(20, 20000)
    plt.ylim(-180, 40)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Phase(rad)')
    plt.legend()
    plt.grid()

plt.show()
#%%
CalculateComplexErrorAndPlot(Pf_measured,Pf,ylower=-50,yupper=50) 
CalculateMagnitudeErrorAndPlot(Pf_measured,Pf_measured_aligned,ylower=-50,yupper=50)
CalculatePhaseErrorAndPlot(Pf_measured,Pf_measured_aligned,ylower=-50,yupper=100)


# %%
"""
ITD compare: 1.synthesized binaural signal (order=4)
             2.koln's measurement interpolated convolved(order=35)
             3.my KU100 measurement
             4.synthesized and aligned binaural signal (order=4) 
             5.analytical binaural signal convolved(order=4)
"""
ITD_synthesized = CalculateITD(Pt_measured,48000,20,1000,4)
ITD_koln = CalculateITD(Pt_horder_interpolated_convolved,48000,20,1000,4)
ITD_KU100 = CalculateITD(HRIR_KU100,48000,20,1000,4)
ITD_synthesized_aligned = CalculateITD(Pt_measured_aligned,48000,20,1000,4)
ITD_analytical = CalculateITD(Pt_analytical_convolved,48000,20,1000,4)
# Plot ITD
indices = np.arange(len(ITD_synthesized ))
width = 0.15
plt.figure(figsize=(10, 6))

# plt.bar(indices - width, np.abs(ITD_synthesized) , width=width, label='Synthesized binaural signal(order:4)')
# plt.bar(indices, np.abs(ITD_koln) , width=width, label='Koln binaural signal(order:35)')
# plt.bar(indices + width, np.abs(ITD_KU100), width=width, label='My KU100 binaural signal')

plt.bar(indices - 2 * width, np.abs(ITD_synthesized), width=width, label='Synthesized binaural signals (order:4)')
plt.bar(indices - width, np.abs(ITD_synthesized_aligned), width=width, label='Synthesized and aligned binaural signals (order:4)')
plt.bar(indices, np.abs(ITD_KU100), width=width, label='My KU100 binaural signals')
plt.bar(indices + width, np.abs(ITD_analytical), width=width, label='Analytical binaural signals (order:4)')
plt.bar(indices + 2 * width, np.abs(ITD_koln), width=width, label='Koln binaural signals (order:35)')

# plt.bar(indices - 2 * width, np.abs(ITD_synthesized), width=width, label='Synthesized binaural signal (order:4)')
# plt.bar(indices - width, np.abs(ITD_koln), width=width, label='Koln binaural signal (order:35)')
# plt.bar(indices, np.abs(ITD_KU100), width=width, label='My KU100 binaural signal')
# plt.bar(indices + width, np.abs(ITD_synthesized_aligned), width=width, label='Synthesized and aligned binaural signal (order:4)')
# plt.bar(indices + 2 * width, np.abs(ITD_analytical), width=width, label='Analytical binaural signal (order:4)')

plt.xlabel('Turntable angle (Deg)')
plt.ylabel('ITD (s)')
plt.title('ITD of far-field binaural signals at different turntable position',fontweight='bold')
plt.xticks(indices, [f'{deg}' for deg in Azimuth_deg])
plt.legend(fontsize='small')
plt.grid()
plt.show()
# %%
"""
ILD compare: 1.synthesized binaural signal (order=4)
             2.koln's measurement interpolated convolved(order=35)
             3.my KU100 measurement   
"""
fc1,ILD_synthesized = CalculateILD(Pf_measured,48000)
fc2,ILD_koln = CalculateILD(Pf_horder_interpolated_convolved,48000)
fc3,ILD_KU100 = CalculateILD(HRTF_KU100,48000)
fc4,ILD_synthesized_aligned = CalculateILD(Pf_measured_aligned,48000)
fc5,ILD_analytical = CalculateILD(Pf_analytical_convolved,48000)

# plot ILD
fig = plt.figure(figsize=(10, 40))
fig.suptitle('ITD (1/3 octave band) of far-field binaural signal at different turntable position', y=0.92, fontweight='bold', fontsize='large')

gs = gridspec.GridSpec(7, 1, height_ratios=[1.5]*7)  
for i in range(ILD_synthesized.shape[0]):
    ax = plt.subplot(gs[i])

    ax.semilogx(fc1,ILD_synthesized[i,:],label='Synthesized binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)
    ax.semilogx(fc1,ILD_koln[i,:],label='Koln binaural signal(order:35)',marker = 'o',markerfacecolor='none',markersize=3)
    ax.semilogx(fc1,ILD_KU100[i,:],label='My KU100 binaural signal',marker = 'o',markerfacecolor='none',markersize=3)
    ax.semilogx(fc1,ILD_synthesized_aligned[i,:],label='Synthesized and aligned binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)
    ax.semilogx(fc1,ILD_analytical[i,:],label='Analytical binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)

    ax.set_title(f'ILD at Turntable angle: {Azimuth_deg[i]}Deg', fontsize='x-small',fontweight='bold')
    ax.legend(fontsize='xx-small', loc='upper left', framealpha=0.5, handlelength=1, handletextpad=0.3)
    ax.grid()
    ax.set_ylim(-10,30)
    ax.set_xlim(15,20200)

plt.subplots_adjust(hspace=1, top=0.85)
fig.text(0.5, 0.06, 'Frequency (Hz)', ha='center', fontsize='medium')
fig.text(0.07, 0.5, 'ILD (dB)', va='center', rotation='vertical', fontsize='medium')
plt.show()
# %%
position = 3
plt.figure()

plt.semilogx(fc1,ILD_synthesized[position,:],label='Synthesized binaural signal(order:4)',marker = 's',markerfacecolor='none',markersize=3)
# plt.semilogx(fc1,ILD_synthesized_aligned[position,:],label='Synthesized and aligned binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)
plt.semilogx(fc1,ILD_KU100[position,:],label='My KU100 binaural signal',marker = '+',markerfacecolor='none',markersize=3)
plt.semilogx(fc1,ILD_analytical[position,:],label='Analytical binaural signal(order:4)',marker = 'x',markerfacecolor='none',markersize=3)
plt.semilogx(fc1,ILD_koln[position,:],label='Koln binaural signal(order:35)',marker = '^',markerfacecolor='none',markersize=3)

# plt.semilogx(fc1,ILD_synthesized[position,:],label='Synthesized binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)
# plt.semilogx(fc1,ILD_koln[position,:],label='Koln binaural signal(order:35)',marker = 'o',markerfacecolor='none',markersize=3)
# plt.semilogx(fc1,ILD_KU100[position,:],label='My KU100 binaural signal',marker = 'o',markerfacecolor='none',markersize=3)
# plt.semilogx(fc1,ILD_synthesized_aligned[position,:],label='Synthesized and aligned binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)
# plt.semilogx(fc1,ILD_analytical[position,:],label='Analytical binaural signal(order:4)',marker = 'o',markerfacecolor='none',markersize=3)

plt.title(f'ILD at Turntable angle: {position*30}Deg', fontsize='x-small',fontweight='bold')
plt.legend(fontsize='large', loc='upper left', framealpha=0.5, handlelength=1, handletextpad=0.3)

plt.ylim(-5,20)
plt.xlim(15,20200)
plt.xlabel('Frequency (Hz)', ha='center', fontsize='x-large')
plt.ylabel('ILD (dB)', va='center', rotation='vertical', fontsize='x-large')
plt.grid()
plt.tight_layout()
plt.show()

# %%
