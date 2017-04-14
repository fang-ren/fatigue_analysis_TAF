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
        
    if ybeam>0 and xbeam<=0: 
        p1 = np.sqrt(d**2+xbeam**2+2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = -np.arctan(x/y)

    if ybeam<=0 and xbeam<=0:
        p1 = np.sqrt(d**2+xbeam**2+2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = np.pi - np.arctan(x/y)
        
    if ybeam<=0 and xbeam>0:
        p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
        gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
        x = -d*np.tan(gama)
        p2 = d/np.cos(gama)
        #print 'p1=',p1,'p2=', p2
        y = ybeam*p2/p1
        rSqr = x**2 + y**2
        twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
        chi = - np.pi + np.arctan(x/y)
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
    criterion = sigma/mean*5
    binList, binAve = chiAverage(Intensity, chi, binSize) 
    normIntAve = []
    chiMin = np.min(chiList)
    #print chiList
    for i in range(len(IntAve)):
        #print i, chiList[i], chiMin
        normIntAve.append(IntAve[i]/(binAve[int((chiList[i]-chiMin)/binSize)]))
    return normIntAve, chiList, criterion

def Qmask(Q1, hw1, Q):
    QMask = np.ones((1800,1800))
    QMask = (Q>(Q1-hw1))*(Q<(Q1+hw1))*QMask
    return QMask
    


###########################################
############ code starts here #############
###########################################
folder_path = 'P:\\bl11-3\\May2016\\Tim_Allen_data\\'
base_filename = 'after2_'
basefile_path = folder_path + base_filename
file_index = 0.0001

save_path = folder_path + 'Processed\\'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# fitting parameters
x0 = 1730.65*3000/3450
y0 = 1730.66*3000/3450
d = 1622.75*3000/3450
Rot = 3.23-np.pi
tilt = 0.0
lamda = 0.974399
PP = 0.95   # beam polarization, decided by beamline setup

# parameters of mask
Q1 = 3.07
hw1 = 0.01


# set up 1800X1800 X, Y grids for cropped image array
X = [i+601 for i in range(1800)]
Y = [i+601 for i in range(1800)]
X, Y = np.meshgrid(X, Y)

while (1):
    print 'waiting for image', basefile_path + str(file_index)[2:], '...'
    while not os.path.exists(basefile_path + str(file_index)[2:] + '.tif'):
        time.sleep(1)
    print 'processing image', basefile_path + str(file_index)[2:], '...'
    im = Image.open(basefile_path + str(file_index)[2:] + '.tif')
    imArray = np.array(im)
    imArray = imArray[600:2400,600:2400,0]   # crop image to speed up
    im.close()

    # Calculate Q, chi, polarized corrected intensity(Ipol) arrays
    Q = np.zeros((1800,1800))
    chi = np.zeros((1800,1800))
    Ipol = np.zeros((1800,1800))
    Q[:], chi[:] = Qchi(X, Y, x0, y0, d, Rot, tilt, lamda)
    Ipol = polarCorr(imArray, Q, chi, lamda, PP)

    beamStop = np.ones((1800,1800))
    beamStop = (chi<-0.091)*beamStop + (chi>-0.037)*beamStop
    QMask = Qmask(Q1, hw1, Q)
    Ipol = Ipol * QMask *beamStop

#
## generate a tiff image
#plt.figure(1)
#plt.title('Mar.tif')
#plt.pcolormesh(X, Y, np.ma.masked_where(Ipol == 0, Ipol))
#plt.show()
#plt.colorbar()
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.grid()
#plt.clim(0, np.max(imArray))

    # generate a Q-chi plot with polarization correction
    plt.figure(2)
    plt.title('Q-chi_polarization corrected.tif')
    plt.pcolormesh(Q, chi, np.ma.masked_where(Ipol == 0, Ipol)) # not plotting zero intensities
    plt.show()
    plt.colorbar()
    plt.xlabel('Q')
    plt.ylabel('chi')
    plt.xlim((Q1-hw1*2, Q1+hw1*2))
    plt.ylim((-3.142, 3.142))
    plt.grid()
    plt.savefig(save_path + base_filename + str(file_index)[2:] + '_Qchi')
    
    # generate column average image
    plt.figure(6)
    plt.title('Normalized Average')
    normIntAve, chiList, criterion = binNorm(Ipol, chi, 0.01, 0.06)
    plt.plot(normIntAve, chiList)
    plt.xlabel('Normalized Intensity by Local Average')
    plt.ylabel('chi')
    plt.savefig(save_path + base_filename + str(file_index)[2:] + '_Normalized_Average')
    
    plt.close("all")

# https://github.com/CrakeNotSnowman/Python_Message/blob/master/sendMessage.py
    if np.nanmax(normIntAve) > (1+5*criterion):
        print 'sending alert...'
        fromaddr = 'hitpxrd@gmail.com'
        toaddrs1  = '2036688741@messaging.sprintpcs.com'
        toaddrs2  = '5052214328@vzwpix.com'
        toaddrs3  = 'tafurni@sandia.gov'
        msg = "\r\n".join([
        "From: hitpxrd@gmail.com"
        "To: 2036688741@messaging.sprintpcs.com, 5052214328@vzwpix.com, tafurni@sandia.gov",
        "Subject: 5-sigma ALERT",
        "",
        "The measurement is done."
        ])
        username = 'hitpxrd@gmail.com'
        password = 'HiTpXRD2016'
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(username,password)
        server.sendmail(fromaddr, toaddrs2, msg)
        server.sendmail(fromaddr, toaddrs3, msg)
        #server.sendmail(fromaddr, toaddrs1, msg)
        server.quit()
        
    elif np.nanmax(normIntAve) > (1+3*criterion) and np.nanmax(normIntAve) <= (1+5*criterion):
        print 'sending alert...'
        fromaddr = 'hitpxrd@gmail.com'
        toaddrs1  = '2036688741@messaging.sprintpcs.com'
        toaddrs2  = '5052214328@vzwpix.com'
        toaddrs3  = 'tafurni@sandia.gov'
        msg = "\r\n".join([
        "From: hitpxrd@gmail.com"
        "To: 2036688741@messaging.sprintpcs.com, 5052214328@vzwpix.com, tafurni@sandia.gov",
        "Subject: 3-sigma ALERT",
        "",
        "The measurement is almost done."
        ])
        username = 'hitpxrd@gmail.com'
        password = 'HiTpXRD2016'
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(username,password)
        server.sendmail(fromaddr, toaddrs2, msg)
        server.sendmail(fromaddr, toaddrs3, msg)
        #server.sendmail(fromaddr, toaddrs1, msg)
        server.quit()
    
    file_index += 0.0001