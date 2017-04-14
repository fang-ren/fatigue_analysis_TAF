# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:14:05 2016

@author: fangren

Qsum function modified by Ronald Pandolf
"""

# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
import smtplib
from email.mime.text import MIMEText
import os.path
import time


def transRev(xdet, ydet, x0, y0):
    """
    translation from detector coordinates (detector corner as the origin)
    into detector plane coordinates (beam center as the new origin)
    x0, y0 are coordiantes of the beam center in  the old coordinate system
    xdet, ydet are pixel coordinates on the detector coordinate system
    return new coordiantes x, y in the new coordiante system
    """
    x = xdet - x0
    y = ydet - y0
    return x, y


def rotRev(Rot, x, y):
    """
    Rotation according to the beam center (0, 0), in the detector plane
    """
    xbeam = x*np.cos(-Rot) - y*np.sin(-Rot)
    ybeam = x*np.sin(-Rot) + y*np.cos(-Rot)
    return xbeam, ybeam


def calTheta_chi(d, tilt, xbeam, ybeam):
    """
    calculate theta angle from the detector distance d (along beam travel
    direction), and tilting angle of the detector
    return theta and chi angles
    """
    if ybeam>0 and xbeam>0: 
        p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = d*np.tan(gama)
        p2 = d/np.cos(gama)
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = -np.arctan(x/y)
        
    elif ybeam>0 and xbeam<=0: 
        p1 = np.sqrt(d**2+xbeam**2+2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = -np.arctan(x/y)

    elif ybeam<0 and xbeam<=0:
        p1 = np.sqrt(d**2+xbeam**2+2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = np.pi - np.arctan(x/y)
        
    elif ybeam<0 and xbeam>0:
        p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = - np.pi + np.arctan(x/y)
        
    elif ybeam == 0 and xbeam>0:
        p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = -np.pi/2

    elif ybeam == 0 and xbeam<=0:
        p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = np.pi/2        
        
        
    return twoTheta/2, chi


vcalTheta_chi = vectorize(calTheta_chi)

def calQ(lamda, theta):
    """
    calculate Q from theta angle and beam energy lamda
    """
    return 4*np.pi*np.sin(theta)/lamda
    

def Qchi(xdet, ydet, x0, y0, d, Rot, tilt, lamda):
    """
    Integrate the four functions: transRev, rotRev, calTheta_chi, calQ
    return Q and chi
    """
    x, y = transRev(xdet, ydet, x0, y0)
    xbeam, ybeam = rotRev(Rot, x, y)
    theta, chi = vcalTheta_chi(d, tilt, xbeam, ybeam)
    Q = calQ(lamda, theta)
    return Q, chi


def polarCorr(intensity, Q, chi, lamda, PP):
    """
    polarization correction
    """
    Qx = Q*np.sin(chi)
    Qy = Q*np.cos(chi)
    thetax = np.arcsin(Qx * lamda / (4 * np.pi ))
    thetay = np.arcsin(Qy * lamda / (4 * np.pi ))
    PCorr = 0.5 * ((1-PP)*np.cos(thetay)+PP*np.cos(thetax)+1) 
    return intensity / PCorr
        
def chiSum(Intensity, chi, step):
    """
    sum up Intensity values for each Q value
    return, a list of Q (Qlist), a list of the summation of intensities for 
    each Q bin (IntSum), a list of the number of intensity value added to 
    that bin (count)
    """
    chiMin = np.min(chi)    
    chi = chi + np.pi
    resol = 1/step    
    keep = np.where(Intensity != 0)
    IntSum = np.bincount((chi[keep].ravel()*resol).astype(int), Intensity[keep].ravel().astype(int))
    count = np.bincount((chi[keep].ravel()*resol).astype(int), np.ones((1800,1800))[keep].ravel().astype(int))
    chiLen = len(IntSum)
    chiList = [i*step+chiMin for i in range(chiLen)]
    return chiList, IntSum, count


def chiAverage(Intensity, chi, step):
    """
    the summation of intensities for each Q bin (IntSum) averaged by count
    """
    chiList, IntSum, count = chiSum(Intensity, chi, step)
    IntAve = list(np.array(IntSum)/np.array(count))
    return chiList, IntAve

def binNorm(Intensity, chi, step, binSize):
    chiList, IntAve = chiAverage(Intensity, chi, step) 
    sigma = np.nanstd(IntAve)
    mean = np.nanmean(IntAve)
    criterion = sigma/mean
    binList, binAve = chiAverage(Intensity, chi, binSize) 
    normIntAve = []
    chiMin = np.min(chiList)
    #print chiList
    for i in range(len(IntAve)):
        #print i, chiList[i], chiMin
        normIntAve.append(IntAve[i]/(binAve[int((chiList[i]-chiMin)/binSize)]))
    return normIntAve, chiList, criterion

def Qmask(Qleft, Qright, Q, Qmask_status):
    if Qmask_status == 'off':
        return np.ones((1800,1800))
    elif Qmask_status == 'on':
        QMask = np.ones((1800,1800))
        QMask = (Q>Qleft)*(Q<Qright)*QMask
        return QMask
    


###########################################
############ code starts here #############
###########################################

file_path ='P:\\bl11-3\\May2016\\NiFe_scans\\test\\test_0004.tif'

# fitting parameters
x0 = 1537.30216637
y0 = 1547.90094844
d = 1529.14534219
Rot = 0
tilt = -0.00279522267447
lamda = 0.974399
PP = 0.95   # beam polarization, decided by beamline setup

# define Q-range
Qleft = 3.13
Qright =3.17

# mask_status
Qmask_status = 'on'

# set up 1800X1800 X, Y grids for cropped image array
X = [i+601 for i in range(1800)]
Y = [i+601 for i in range(1800)]
X, Y = np.meshgrid(X, Y)


im = Image.open(file_path)
imArray = np.array(im)
imArray = imArray[600:2400,600:2400]   # crop image to speed up
im.close()

# Calculate Q, chi, polarized corrected intensity(Ipol) arrays
Q = np.zeros((1800,1800))
chi = np.zeros((1800,1800))
Ipol = np.zeros((1800,1800))
Q[:], chi[:] = Qchi(X, Y, x0, y0, d, Rot, tilt, lamda)
Ipol = polarCorr(imArray, Q, chi, lamda, PP)

beamStop = np.ones((1800,1800))
beamStop = (chi<-0.095)*beamStop + (chi>0.037)*beamStop
QMask1 = Qmask(Qleft, Qright, Q, Qmask_status)

Ipol = Ipol * QMask1 *beamStop


# generate a tiff image
plt.figure(1)
plt.title('Marccd.tif')
plt.pcolormesh(X, Y, np.ma.masked_where(Ipol == 0, Ipol))
plt.show()
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.clim(0, np.max(imArray))

# generate a Q-chi plot with polarization correction
plt.figure(2)
plt.title('Q-chi_polarization corrected.tif')  
plt.pcolormesh(Q, chi, np.ma.masked_where(Ipol == 0, Ipol)) # not plotting zero intensities
plt.show()
plt.colorbar()
plt.xlabel('Q')
plt.ylabel('chi')
if Qmask_status == 'on':
    plt.xlim((Qleft, Qright))
else:
    plt.xlim(np.nanmin(Q), np.nanmax(Q))
plt.clim(0, np.nanmax(Ipol))
plt.ylim((-3.142, 3.142))
plt.grid()

    
# generate column average image
plt.figure(3)
plt.title('Normalized Average')
normIntAve, chiList, criterion = binNorm(Ipol, chi, 0.0005, 0.06)
plt.plot(normIntAve, chiList)
plt.xlabel('Normalized Intensity by Local Average')
plt.ylabel('chi')


    

