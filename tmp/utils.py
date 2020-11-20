from __future__ import division
import numpy as np
# import scipy.interpolate 
# import tensorflow as tf
import math
import os

mu = 2
K = 256
CP = 32


def print_something():
    print('utils.py has been loaded perfectly')


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x) ** 2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB


def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return 0.7071 * (2 * bit_r[:, 0] - 1) + 0.7071j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation


def deModulation(Q):
    Qr=np.real(Q)
    Qi=np.imag(Q)
    bits=np.zeros([64,2])
    bits[:,0]=Qr>0
    bits[:,1]=Qi>0
    return bits.reshape([-1])  # This is just for QAM modulation

def Modulation1(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (bit_r[:, 0]) + 1j * (bit_r[:, 1])


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples ...
    # cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):

    convolved = np.convolve(signal, channelResponse)

    sigma2 = 0.35 * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))



def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):

    # --- training inputs ----

    CR=1
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, 2 * K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,  K)
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword, mu)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP,  K)
    OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword)
    AA = np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))


    CC=OFDM_RX_noCP/np.max(AA)
    BB = np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))

    return np.concatenate((AA, BB)), CC  # sparse_mask


def MIMO(X, HMIMO, SNRdb,flag,P):
    P = P * 2
    Pilot_file_name = 'Pilot_' + str(P)
    if os.path.isfile(Pilot_file_name):
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')
    pilotValue = Modulation(bits, mu)


    if flag==1:
        cpflag, CR = 0, 0
    elif flag==2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]



    bits0=X[0]
    bits1=X[1]
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]
    signal_output00, para = ofdm_simulate(bits0, HMIMO[0,:], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                                    pilotCarriers1, dataCarriers, CR)
    signal_output01, para = ofdm_simulate(bits0, HMIMO[1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output10, para = ofdm_simulate(bits1, HMIMO[2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output11, para = ofdm_simulate(bits1, HMIMO[3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)

    signal_output0=signal_output00+signal_output10
    signal_output1=signal_output01+signal_output11
    output=np.concatenate((signal_output0, signal_output1))
    output=np.transpose(np.reshape(output,[8,-1]),[1,0])

    #print(np.shape(signal_output00))
    return np.reshape(output,[-1])

# def ofdm_simulateMIMO22(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers1,pilotCarriers2,
#                   Clipping_Flag):
#     # --- training inputs ----
#
#
#     OFDM_data = np.zeros([2,K], dtype=complex)
#     OFDM_data[0,pilotCarriers1] = pilotValue[0:2:P]  # allocate the pilot subcarriers
#     OFDM_data[1, pilotCarriers2] = pilotValue[1:2:P]
#
#     OFDM_time = IDFT(OFDM_data)
#     OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, 2 * K)
#     # OFDM_withCP = addCP(OFDM_time)
#     OFDM_TX = OFDM_withCP
#     if Clipping_Flag:
#         OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
#     OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
#     OFDM_RX_noCP = removeCP(OFDM_RX, CP,  K)
#     OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)
#     # OFDM_RX_noCP = removeCP(OFDM_RX)
#     # ----- target inputs ---
#     symbol = np.zeros(K, dtype=complex)
#     codeword_qam = Modulation(codeword, mu)
#     if len(codeword_qam) != K:
#         print('length of code word is not equal to K, error !!')
#     symbol = codeword_qam
#     OFDM_data_codeword = symbol
#     OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
#     OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
#     # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
#     if Clipping_Flag:
#         OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
#     OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
#     OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP,  K)
#     OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword)
#     AA = np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))
#
#     #AA = AA / np.max(AA)
#     CC=OFDM_RX_noCP/np.max(AA)
#     BB = np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))
#     #BB = BB / np.max(BB)
#     # OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
#     return np.concatenate((AA, BB)), CC  # sparse_mask


'''

def ofdm_simulate(codeword, channelResponse,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):       
    payloadBits_per_OFDM = mu*len(dataCarriers)

    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))  
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 
'''

'''


def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,Clipping_Flag):       
    OFDM_data = np.zeros(K, dtype=complex)
    allCarriers = np.arange(K)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP_flag, mu, K)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX,CP,K)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP_flag, mu, K)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 


'''
