# -*- coding: utf-8 -*-
"""
Created on Wed May 18 2016

@author: fangren

Qsum function modified by Ronald Pandolf
"""
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

def Qmask(Q1, hw1, Q, Qmask_status):
    if Qmask_status == 'off':
        return np.ones((1800,1800))
    if Qmask_status == 'on':
        QMask = np.ones((1800,1800))
        QMask = (Q>(Q1-hw1))*(Q<(Q1+hw1))*QMask
        return QMask
    

def file_index(index):
    if len(str(index)) == 1:
        return '000' + str(index)
    elif len(str(index)) == 2:
        return '00' + str(index)
    elif len(str(index)) == 3:
        return '0' + str(index)


###########################################
############ code starts here #############
###########################################
folder_path = 'P:\\bl11-3\\May2016\\NiFe_scans\\NiFe-W3-db37\\F2after28k\\'  
base_filename = 'F2after28k_05181933_'
basefile_path = folder_path + base_filename
index = 114
save_path = folder_path + 'Processed\\'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# fitting parameters
x0 = 1537.30216637
y0 = 1547.90094844
d = 1529.14534219
Rot = 0
tilt = -0.00279522267447
lamda = 0.974399
PP = 0.95   # beam polarization, decided by beamline setup

# parameters of Qrange
Qmask_status = 'on'

# to change the Qrange only in the plot, copy the lines below into the console after the program finished running
#plt.pcolormesh(Q, chi, np.ma.masked_where(Ipol_total == 0, Ipol_total)) # not plotting zero intensities
#plt.colorbar()
#plt.xlim((3.12, 3.16))

Qleft = 3.11
Qright = 3.15

#########################
hw = (Qright - Qleft)/8

Q1 = Qleft + hw
Q2 = Qleft + hw * 3
Q3 = Qleft + hw * 5
Q4 = Qleft + hw * 7


# set up 1800X1800 X, Y grids for cropped image array
X = [i+601 for i in range(1800)]
Y = [i+601 for i in range(1800)]
X, Y = np.meshgrid(X, Y)

while (1):
    print("\r")    
    print 'waiting for image', basefile_path + file_index(index) + '.tif...'
    print("\r")    
    while not os.path.exists(basefile_path + file_index(index) + '.tif'):
        time.sleep(10)
    print 'processing image '+ basefile_path + file_index(index) + '.tif'+' and '\
     + basefile_path + file_index(index-15) + '.tif...'
    print("\r")
    print("\r")
    im_A = Image.open(basefile_path + file_index(index) + '.tif')
    imArray_A = np.array(im_A)[600:2400,600:2400]  # crop image to speed up
    #im_B = Image.open(basefile_path + file_index(index-15) + '.tif')
    #imArray_B = np.array(im_B)[600:2400,600:2400]  # crop image to speed up
    #imArray = imArray_A + imArray_B - np.abs(imArray_A-imArray_B)
    imArray = imArray_A    
    im_A.close()
    #im_B.close()

    # Calculate Q, chi, polarized corrected intensity(Ipol) arrays
    Q = np.zeros((1800,1800))
    chi = np.zeros((1800,1800))
    Ipol = np.zeros((1800,1800))
    Q[:], chi[:] = Qchi(X, Y, x0, y0, d, Rot, tilt, lamda)
    Ipol = polarCorr(imArray, Q, chi, lamda, PP)

    beamStop = np.ones((1800,1800))
    beamStop = (chi<-0.095)*beamStop + (chi>0.037)*beamStop
    QMask1 = Qmask(Q1, hw, Q, Qmask_status)
    QMask2 = Qmask(Q2, hw, Q, Qmask_status)
    QMask3 = Qmask(Q3, hw, Q, Qmask_status)
    QMask4 = Qmask(Q4, hw, Q, Qmask_status)
    Ipol1 = Ipol * QMask1 *beamStop
    Ipol2 = Ipol * QMask2 *beamStop
    Ipol3 = Ipol * QMask3 *beamStop
    Ipol4 = Ipol * QMask4 *beamStop
    Ipol_total = Ipol * (QMask1 + QMask2 + QMask3 + QMask4) * beamStop

    #
    ## generate a tiff image
    #plt.figure(1)
    #plt.title('Marccd.tif')
    #plt.pcolormesh(X, Y, np.ma.masked_where(Ipol == 0, Ipol))
    #plt.show()
    #plt.colorbar()
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.grid()
    #plt.clim(0, np.max(imArray))

    # generate a Q-chi plot with polarization correction
    plt.figure(2, figsize=(20,20))
    plt.title('Q-chi_polarization corrected.tif')  
    plt.pcolormesh(Q, chi, np.ma.masked_where(Ipol_total == 0, Ipol_total)) # not plotting zero intensities
    plt.show()
    plt.colorbar()
    plt.xlabel('Q')
    plt.ylabel('chi')
    if Qmask_status == 'on':
        plt.xlim((Qleft, Qright))
    elif Qmask_status == 'off':
        plt.xlim(np.nanmin(Q), np.nanmax(Q))
    plt.clim(0, np.nanmax(Ipol_total))
    plt.ylim((-3.142, 3.142))
    plt.grid()
    plt.savefig(save_path + base_filename + file_index(index) + '_Qchi.tif', format = 'tif')
    
    # generate column average image
    normIntAve1, chiList, criterion1 = binNorm(Ipol1, chi, 0.0011, 0.06)
    normIntAve2, chiList, criterion2 = binNorm(Ipol2, chi, 0.0011, 0.06)
    normIntAve3, chiList, criterion3 = binNorm(Ipol3, chi, 0.0011, 0.06)
    normIntAve4, chiList, criterion4 = binNorm(Ipol4, chi, 0.0011, 0.06)
    
    IntAve_total = np.array(normIntAve1) + np.array(normIntAve2) + np.array(normIntAve3)\
    + np.array(normIntAve4)
        
    
    plt.figure(3)
    plt.title('Normalized Average_added by four channels')
    plt.plot(IntAve_total, chiList)
    plt.xlabel('Normalized Intensity by Local Average')
    plt.ylabel('chi')
    plt.savefig(save_path + base_filename + file_index(index) + '_Normalized_Average4')

    plt.close("all")
    

    criterion = np.average([criterion1, criterion2, criterion3, criterion4])
    
    zinger = 0    
    for i in range(len(chiList)):
        if IntAve_total[i] > (2*criterion +4):
            zinger += 1
    
    # https://github.com/CrakeNotSnowman/Python_Message/blob/master/sendMessage.py
    if zinger > 1:
        print 'Multiple 3-sigma anomalous spot was found in file ' + base_filename + file_index(index) + ', sending level-1 alert...'
        print("\r\n")
        fromaddr = 'hitpxrd@gmail.com'
        toaddrs1 = '5053663332@vzwpix.com'
        toaddrs2  = '5052214328@vzwpix.com'
        toaddrs3  = 'tfurnish22@gmail.com'
        msg = "\r\n".join([
        "From: hitpxrd@gmail.com"
        "To: 5053663332@vzwpix.com,5052214328@vzwpix.com, tfurnish22@gmail.com",
        "Subject: Multiple 3-sigma ALERT",
        "",
        "Multiple 3-sigma anomalous spots were found in file " + base_filename + file_index(index)
        ])
        username = 'hitpxrd@gmail.com'
        password = 'HiTpXRD2016'
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(username,password)
        server.sendmail(fromaddr, toaddrs1, msg)
        server.sendmail(fromaddr, toaddrs2, msg)
        server.sendmail(fromaddr, toaddrs3, msg)
        server.quit()
        
    elif zinger == 1:
        print '1 3-sigma anomalous spot was found in file ' + base_filename + file_index(index) + ', sending level-1 alert...'
        print("\r")
        fromaddr = 'hitpxrd@gmail.com'
        toaddrs1 = '5053663332@vzwpix.com'
        toaddrs2  = '5052214328@vzwpix.com'
        toaddrs3  = 'tfurnish22@gmail.com'
        msg = "\r\n".join([
        "From: hitpxrd@gmail.com"
        "To: 5053663332@vzwpix.com, 5052214328@vzwpix.com, tfurnish22@gmail.com",
        "Subject: Single 3-sigma ALERT",
        "",
        "A 3-sigma anomalous spot was found in file " + base_filename + file_index(index)
        ])
        username = 'hitpxrd@gmail.com'
        password = 'HiTpXRD2016'
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(username,password)
        server.sendmail(fromaddr, toaddrs1, msg)
        server.sendmail(fromaddr, toaddrs2, msg)
        server.sendmail(fromaddr, toaddrs3, msg)
        server.quit()
    
    index += 1