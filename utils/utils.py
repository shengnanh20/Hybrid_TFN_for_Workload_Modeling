import numpy as np
import numpy.matlib
import glob
from pandas import read_csv
from scipy import signal


def my_dir(pathName, filter='*', i=-1):
    dirs = glob.glob(pathName + '/' + filter)
    dirs.sort(key=lambda x: x)
    if i == -1:
        outName = dirs
    elif i < dirs.__len__():
        outName = dirs[i]
    else:
        outName = None
    return outName


def readCsvFiles(files=None):
    values = []
    for fI in np.arange(0, len(files)).reshape(-1):
        f = open(files[fI], encoding='UTF-8')
        valuesTmp = read_csv(f)
        f.close()

        if fI == 0:
            values = valuesTmp
        else:
            values = values.append(valuesTmp)

    return values


def procStd(dataIn=None, wdSize=None):
    dataOut = []
    dataInTmp = np.concatenate((np.matlib.repmat(dataIn[0, :], wdSize // 2, 1), dataIn, np.matlib.repmat(
        dataIn[-1, :], wdSize // 2, 1)), 0)
    for dI in np.arange(0, dataInTmp.shape[0] - wdSize).reshape(-1):
        dataStd = np.matlib.std(dataInTmp[dI:dI+wdSize, :], 0, keepdims=True, where=~np.isnan(dataInTmp[dI:dI+wdSize, :]))
        if dI == 0:
            dataOut = dataStd
        else:
            dataOut = np.concatenate((dataOut, dataStd), 0)

    return dataOut


def procMean(dataIn=None, wdSize=None):
    dataOut = []
    dataInTmp = np.concatenate((np.matlib.repmat(dataIn[0, :], wdSize // 2, 1), dataIn, np.matlib.repmat(
        dataIn[-1, :], wdSize // 2, 1)), 0)
    for dI in np.arange(0, dataInTmp.shape[0] - wdSize).reshape(-1):
        dataStd = np.mean(dataInTmp[dI:dI+wdSize, :], 0, keepdims=True, where=~np.isnan(dataInTmp[dI:dI+wdSize, :]))
        if dI == 0:
            dataOut = dataStd
        else:
            dataOut = np.concatenate((dataOut, dataStd), 0)

    return dataOut


def procFilter(dataIn=None, fSize=None):
    dataInTmp = np.concatenate((np.matlib.repmat(dataIn[0, :], fSize // 2, 1), dataIn, np.matlib.repmat(
        dataIn[-1, :], fSize // 2, 1)), 0)
    dataOut = np.zeros(dataIn.shape)
    for dI in np.arange(0, dataOut.shape[1]).reshape(-1):
        dataOut[:, dI] = np.matlib.convolve(dataInTmp[:, dI], np.ones((fSize + 1)) / (fSize + 1), 'valid')

    return dataOut


def getEyeFeatures(data1Cach=None, blinkFlagCache=None, freq1=None):
    #######################
    #            Blink Features
    #######################

    blinkMaxDur = 0
    blinkCurDur = 0
    blinkTotalDur = 0
    blinkNum = 0
    for dI in np.arange(1, blinkFlagCache.shape[0]).reshape(-1):
        tmp = blinkFlagCache[dI - 1] + blinkFlagCache[dI] * 10
        if tmp == 0:
            continue
        else:
            if tmp == 1:
                blinkNum = blinkNum + 1
            else:
                if tmp == 10:
                    blinkCurDur = 0
                else:
                    blinkCurDur = blinkCurDur + 1
                    blinkTotalDur = blinkTotalDur + 1
                    blinkMaxDur = np.amax([blinkMaxDur, blinkCurDur])

    blinkTotalDur = blinkTotalDur / freq1
    blinkMaxDur = blinkMaxDur / freq1
    blinkMeanDur = blinkTotalDur / (blinkNum + 1e-06)
    blinkRate = blinkNum / (blinkFlagCache.shape[0] / freq1)

    #######################
    #           Pupil Diameter
    #######################

    diaCach = np.nanmean(data1Cach[:, (0, 3)], 1)
    diaMean = np.nanmean(diaCach, 0)
    diaStd = np.nanstd(diaCach, 0)

    #######################
    #           Saccade Features
    #######################

    fSize = 10

    data1Filter = procFilter(data1Cach, fSize)
    tmp = data1Filter[1:, (1, 2, 4, 5)] - data1Filter[0:-1, (1, 2, 4, 5)]
    tmp = np.concatenate((tmp, tmp[-1, :].reshape(1, tmp.shape[1])), 0)
    speed = np.abs(tmp[:, (0, 2)])

    sacThr = 0.03

    sacFlag = np.amax(speed, 1) > sacThr
    sacSpeed = speed[sacFlag, :]
    if len(sacSpeed) == 0:
        sacSpeed = np.zeros((1, speed.shape[1]))

    sacPathLength = sum(sacFlag) / freq1
    sacNum = np.round(sum((np.asarray(sacFlag[1:], 'float') - np.asarray(sacFlag[:-1], 'float')) != 0) / 2)
    sacDurationMean = sacPathLength / (sacNum + 1e-06)
    sacRate = sacNum / (blinkFlagCache.shape[1 - 1] / freq1)

    #######################
    #           Speed Features
    #######################

    speedTmp = np.nanmean(sacSpeed, 1)

    speedMean = np.nanmean(speedTmp, 0)
    speedStd = np.nanstd(speedTmp, 0)
    speedMax = np.nanmax(speedTmp, 0)

    #######################
    #           Return Eye Features
    #######################

    features1 = np.array([])
    features1 = np.concatenate((features1, [diaMean, diaStd]), 0)
    features1 = np.concatenate((features1, [blinkRate, blinkMaxDur, blinkMeanDur]), 0)
    features1 = np.concatenate((features1, [sacPathLength, sacDurationMean, sacRate]), 0)
    features1 = np.concatenate((features1, [speedMean, speedStd, speedMax]), 0)

    return features1, diaCach, sacFlag, speed


def getPulseFeatures(data2Cach=None, freq2=None):
    #######################
    #           Identify peaks
    #######################

    data2Peaks = np.array(data2Cach)
    fSize = 700
    peakTmp = procFilter(data2Peaks, fSize)
    peakTmpIdx = data2Peaks * 0.8 < peakTmp
    data2Peaks[peakTmpIdx] = np.amin(data2Peaks)
    peakIdx = signal.find_peaks(data2Peaks.squeeze())
    peakIdx = peakIdx[0]
    peakValue = data2Peaks[peakIdx]

    #######################
    #            HR and PRV
    #######################

    IBI = peakIdx[1:] - peakIdx[:-1]
    IBIavg = np.mean(IBI)

    SDNN = np.std(IBI)
    SDSD = np.std(IBI[1:] - IBI[:-1])
    RMSSD = np.mean((IBI[1:] - IBI[:-1])**2)**.5

    srate = freq2
    FFT = np.fft.fft(data2Cach.squeeze())
    signalspec = np.abs(FFT)
    fftpts = len(data2Cach)
    hpts = fftpts / 2
    signalspec = signalspec / hpts
    binwidth = srate / fftpts
    f = np.arange(0, srate, binwidth)
    PSDavg = np.array(
        [np.mean(signalspec[(f > 0) & (f < 0.2), ]), np.mean(signalspec[(f > 0.2) & (f < 0.4), ]),
         np.mean(signalspec[(f > 0.4) & (f < 0.6), ]), np.mean(signalspec[(f > 0.6) & (f < 0.8), ]),
         np.mean(signalspec[(f > 1.) & (f < 2.), ])])

    #######################
    #           Return Pulse Features
    #######################

    features2 = np.array([])
    features2 = np.concatenate((features2, [IBIavg, SDNN]), 0)
    features2 = np.concatenate((features2, [SDSD, RMSSD]), 0)
    features2 = np.concatenate((features2, PSDavg), 0)

    return features2, peakValue, peakIdx

