import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

from utils import my_dir, readCsvFiles, procStd, procMean, procFilter, getEyeFeatures, getPulseFeatures

rootPath = 'data/HPO-CLD'
fileNames = np.array(['*-tobii*.csv', '*-bitalino*.csv', '*-labels*.csv'])
dirs = my_dir(rootPath)
freq1 = 120
freq2 = 1000
dataSet = np.array([])

testFlag = False

dirIs = np.arange(0, len(dirs)) if not testFlag else [7, 54, 55]
for dirI in dirIs:
    curDir = dirs[dirI]

    #######################
    #           Read files
    #######################

    files = my_dir(curDir, fileNames[0])
    values1 = readCsvFiles(files)
    values1 = np.asarray(values1.values[:, np.array([1, 11, 13, 14, 12, 16, 17])-1], 'float')
    files = my_dir(curDir, fileNames[1])
    values2 = readCsvFiles(files)
    values2 = np.asarray(values2.values[:, np.array([1, 2])-1], 'float')
    files = my_dir(curDir, fileNames[2])
    ts = readCsvFiles(files)
    ts = np.asarray(ts.values[:, np.array([2, 3, 7])-1], 'float')
    ts = np.concatenate((ts, np.matlib.repmat(np.transpose(np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]])), ts.shape[0] // 9, 1)), 1)

    if len(values1) == 0 or len(values2) == 0 or len(ts) == 0:
        continue

    tIs = np.arange(0, ts.shape[0]) if not testFlag else np.arange(0, 1)
    for tI in tIs:

        #######################
        #           Get data
        #######################

        idx1 = np.where(np.abs(values1[:, 0] - ts[tI, 0]) == np.amin(np.abs(values1[:, 0] - ts[tI, 0])))[0]
        idx2 = np.where(np.abs(values1[:, 0] - ts[tI, 1]) == np.amin(np.abs(values1[:, 0] - ts[tI, 1])))[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        data1 = values1[idx1[0]:idx2[0]+1, 1:]

        idx1 = np.where(np.abs(values2[:, 0] - ts[tI, 0]) == np.amin(np.abs(values2[:, 0] - ts[tI, 0])))[0]
        idx2 = np.where(np.abs(values2[:, 0] - ts[tI, 1]) == np.amin(np.abs(values2[:, 0] - ts[tI, 1])))[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        data2 = values2[idx1[0]:idx2[0]+1, 1:]

        if data1.shape[0] < 10 or data2.shape[0] < 10:
            continue
        
        #######################
        #           Eye Signal Normalization
        #######################

        wdSize = 1500
        stdD = procStd(data1, wdSize)
        meanD = procMean(data1, wdSize)
        data1Norm = (data1 - meanD) / stdD

        #######################
        #           Blink Detection
        #######################

        wdSize = 30
        blinkThr = 1
        stdD = procStd(data1Norm, wdSize)
        blinkFlag = np.amax(stdD[:, (0, 3)], 1) > blinkThr

        #######################
        #           Blink Removal
        #######################

        nanFlag = np.any(np.isnan(data1Norm), 1)
        xTmp1 = np.where(np.any(np.concatenate(([blinkFlag], [nanFlag]), 0), 0) == 0)[0]
        xTmp2 = np.arange(0, data1Norm.shape[0])
        data1Fixed = data1Norm
        for dI in np.arange(0, data1Norm.shape[1]).reshape(-1):
            data1Fixed[:, dI] = np.interp(np.transpose(xTmp2), xTmp1, data1Norm[xTmp1, dI])

        #######################
        #           Pupil Signal Quality
        #######################

        dilaDiff = data1[:, 0] - data1[:, 3]
        dilaDiff = np.reshape(dilaDiff, [len(dilaDiff), 1])
        fSize = 120
        dilaDiffFilter = procFilter(dilaDiff, fSize)
        wdSize = 120
        dilaDiffStd = procStd(dilaDiff, wdSize)

        #######################
        #           Filter Pulse Data
        #######################

        fs = freq2
        fl = 0.5
        fh = 5
        N = 128
        wp = np.array([fl / (fs / 2), fh / (fs / 2)])
        b2 = signal.firwin(N+1, wp, pass_zero='bandpass')
        data2Filter = signal.filtfilt(b2, 1, np.squeeze(data2))
        data2Filter = np.reshape(data2Filter, [len(data2Filter), 1])

        #######################
        #           Generate dataset
        #######################

        cachT = 12.5
        maxCach = int(np.floor(np.amin([data1.shape[0] / freq1, data2.shape[0] / freq2]))) - 2

        sIs = np.arange(1, int(np.floor(maxCach - cachT))) if not testFlag else np.arange(1, 2)
        for sI in sIs:

            idx1 = np.asarray(sI * freq1 + (np.arange(0, freq1 * cachT)), 'int')
            idx2 = np.asarray(sI * freq2 + (np.arange(0, freq2 * cachT)), 'int')

            #######################
            #           Noisy detection
            #######################

            dilaDiffThr = 0.3
            dilaDiffRate = 0.3
            dilaDiffStdCach = dilaDiffStd[idx1, :]
            if sum(dilaDiffStdCach > dilaDiffThr) / len(dilaDiffStdCach) > dilaDiffRate:
                continue
            data1Cach = data1Fixed[idx1, :]
            data2Cach = data2Filter[idx2, :]
            blinkFlagCache = blinkFlag[idx1]
            if np.any(np.isnan(data1Cach)) or np.any(np.isnan(data2Cach)):
                continue

            #######################
            #           Get Eye Features
            #######################

            # diaMean, diaStd; blinkRate, blinkMaxDur, blinkMeanDur;
            # sacPathLength, sacDurationMean, sacRate; speedMean, speedStd, speedMax
            featureEye, diaCach, sacFlag, speed = getEyeFeatures(data1Cach, blinkFlagCache, freq1)

            #######################
            #           Get Pulse Features
            #######################

            # IBIavg, SDNN; SDSD, RMSSD, PSDavg(5)
            featurePulse, peakValue, peakIdx = getPulseFeatures(data2Cach, freq2)

            #######################
            #           Get Label
            #######################

            label = ts[tI, (3, 2)]
            dataSetTmp = np.concatenate((np.array([dirI+1, tI+1, sI]), featureEye, featurePulse, label), 0)
            if len(dataSet) == 0:
                dataSet = np.array(dataSetTmp.reshape([1, len(dataSetTmp)]))
            else:
                dataSet = np.concatenate((dataSet, dataSetTmp.reshape([1, len(dataSetTmp)])), 0)

            #######################
            #           show details
            #######################

            if testFlag:
                t1 = (np.arange(0, data1Cach.shape[0])) / freq1
                t2 = (np.arange(0, data2Cach.shape[0])) / freq2

                plt.figure(figsize=(9, 7))
                plt.subplot(3, 1, 1)
                plt.xlim([0, cachT])
                plt.plot(t1, data1Cach[:, 0] / np.amax(np.abs(data1Cach[:, 0]), 0) + 1, '.')
                plt.plot(t1, data1Cach[:, 3] / np.amax(np.abs(data1Cach[:, 3]), 0) + 1, '.')
                plt.plot(t1, np.asarray(blinkFlagCache, 'float') + 0.5)
                plt.legend(['Eye Dia (R)', 'Eye Dia (L)', 'Blink'])
                plt.suptitle('Eye diameter, Blink')

                plt.subplot(3, 1, 2)
                plt.xlim([0, cachT])
                plt.plot(t1, data1Cach[:, 1] / np.amax(np.abs(data1Cach[:, 1]), 0) + 1, '.')
                plt.plot(t1, speed[:, 0] / np.amax(np.abs(speed[:, 0]), 0) + 0.8)
                plt.plot(t1, np.asarray(sacFlag, 'float') + 0.2)
                plt.legend(['X Pos', 'X Speed', 'Moving'])
                plt.title('X Pos, X Speed, Saccade')

                plt.subplot(3, 1, 3)
                plt.xlim([0, cachT])
                plt.plot(t2, data2Cach)
                plt.plot(peakIdx / freq2, peakValue, 'ro')
                plt.legend(np.array(['Pulse', 'Peaks']))
                plt.title(str('Heart Rate %.1f Hz' % (60000 / featurePulse[0])))

                plt.draw(), plt.pause(3)

    print('Processing %d / %d' % (dirI+1, len(dirs)))


columns = ['Person', 'Trial', 'Time',
           'diaMean', 'diaStd', 'blinkRate', 'blinkMaxDur', 'blinkMeanDur',
           'sacPathLength', 'sacDurationMean', 'sacRate', 'speedMean', 'speedStd', 'speedMax',
           'IBIavg', 'SDNN', 'SDSD', 'RMSSD', 'PSDavg1', 'PSDavg2', 'PSDavg3', 'PSDavg4', 'PSDavg5',
           'LabelCls', 'LabelReg']
dt = pd.DataFrame(dataSet, columns=columns)
dt.to_csv("dataSet.csv", index=False)
