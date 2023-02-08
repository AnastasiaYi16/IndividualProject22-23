import numpy as np
import os
import cv2
import cmath
from skimage.color import rgb2hsv
import colorsys

TrainOginRoot = './TrackingNet-devkit/data'
OutputRoot = './data'
totalSeq = 30
intensityGain = np.random.rand(0,totalSeq) * 0.1 + 0.3
intensityOffset = np.random.rand(0,totalSeq) * 0.03 + 0.035
gammaGain = np.random.rand(0,totalSeq) * 1.5 + 1.8
saturationGain = np.random.rand(0,totalSeq) * 0.2 + 0.6
for subroot in np.arange(0,totalSeq).reshape(-1):
    subRootName = TrainOginRoot + '/' + subroot
    subOutRood = os.path.join(OutputRoot, subroot)
    os.mkdir(subOutRood)
    imgList = os.listdir(subRootName + '/' + '*.png')
    # read clean image
    for f in np.arange(0,len(imgList)).reshape(-1):
        img = img = cv2.imread(imgList(f).folder + '/' + imgList(f).name)
        imgHR = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # resize to lower resolution
        imgHR = cv2.resize(imgHR,0.5)
        synDark = cmath.real(imgHR ** gammaGain(subroot + 1))
        # adding noise: poisson and guassian
        a = 1
        b = 0.05
        noisy = cmath.real(synDark + (np.multiply(np.sqrt(np.multiply(a,synDark)),np.random.normal(0,0.01,synDark.shape))) + np.random.normal(0,b,synDark.shape))
        if f == 0:
            minnoise = np.amin(noisy)
            rangenoise = range(noisy)
        # normalise with the same parameter for the whole sequence
        noisy = (noisy - minnoise) / rangenoise
        noisy = noisy + intensityOffset(subroot + 1)
        # dim brightness
        synDark = np.multiply(intensityGain(subroot + 1),noisy)
        # dim saturation
        hsvimg = rgb2hsv(synDark)
        hsvimg[:,:,1] = saturationGain(subroot + 1) * hsvimg(:,:,2) # How to convert to python?
        synDark = colorsys.hsv_to_rgb(hsvimg)
        cv2.imwrite(synDark,subOutRood + '/' + imgList(f).name)
