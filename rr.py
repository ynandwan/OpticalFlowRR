from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys,os
import cv2
import numpy as np
import scipy as sp
import sys
from IPython.core.debugger import Pdb

sys.path.append('/home/yatin/phd/vision/code/project')
import UtilityClasses as uc
from numpy import linalg as LA

BASE_PATH = '/home/yatin/phd/vision'
os.chdir(BASE_PATH)
#100 cols, 80 rows
SIZE = (100,80)
FLOW_MEMORY = 0.95
PFF_MEMORY = 0.95

n = 0
prevGray = None
favg = uc.EWMA(FLOW_MEMORY)
pff = uc.EWMA(PFF_MEMORY)
rrsignal = []
cap = cv2.VideoCapture('data/project/rr.mov')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        n += 1
        #if n%20 != 0:
        #    continue
        #
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, SIZE) 
        gray = gray.astype(float)
        if(n > 1):
            imx,imy = np.gradient(gray)
            
            dn = gray - prevGray
            #Pdb().set_trace()
            gradNorm2 = np.multiply(imy,imy) + np.multiply(imx,imx) 
            gradNorm2[gradNorm2==0] = 1.0
            flowx = -1*np.multiply(imx,dn)/gradNorm2
            flowy = -1*np.multiply(imy,dn)/gradNorm2
            fn = np.hstack((flowx.flatten(),flowy.flatten()))
            favg.update(fn)
            if pff.count == 0:
                pff.update(fn)
            else:
                pff.update(np.sign(pff.avg.dot(fn))*fn)
            #
            pffNorm = LA.norm(pff.avg)
            rn = np.dot(favg.avg,pff.avg)/pffNorm
            rrsignal.append(rn)
        #
        if n%100 == 0:
        #if n > 890:
            print("n: {0}, this rr: {1}".format(n,rn))
        #
        prevGray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("n: {0}".format(n))
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

%matplotlib inline
plt.plot(np.array(rrsignal[100:]))

