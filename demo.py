import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from PIL import Image

#Facial recognition 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#Eye recognition
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def findFace(img):
    face_rect = face_cascade.detectMultiScale(img,
        scaleFactor = 1.2, minNeighbors = 5)
    for (x,y, w, h) in face_rect:
        return (x, y, w, h)

def findEyes(img):
    eyes = eye_cascade.detectMultiScale(img,
    scaleFactor = 1.2, minNeighbors = 5)
    if len(eyes) != 2: return None
    leftEye = eyes[0]
    rightEye = eyes[1]
    return (leftEye, rightEye)


cap = cv2.VideoCapture("test.mp4")

def zoom(img, rect):
    if rect is None: return img
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    # print(x+,y,w,h)
    return img[y:y+h, x:x+w]

def transform(img, rect):
    if rect is None: return img
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    bottom_left  =    np.float32([x, y + h])
    top_left     =    np.float32([x, y])
    top_right    =    np.float32([x + w, y])
    bottom_right =    np.float32([x + w, y + h])
    
    height = img.shape[0]
    width = img.shape[1]
    
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[0,0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(img, matrix, (width, height))
    return res

eyesFound = False
#For contours
thresh = 20 


#Blink method 1
#Compute distane between highest and lowest white pixels in middle
def blink1(img):
    img = np.asarray(img)
    width = img.shape[0]
    height = img.shape[1]
    mid = width//2
    high = 0
    low = height-1
    j = 0
    for i in range(height):
        if img[i,mid] == 255:
            high = i
            break
    while low >= 0:
        if img[low, mid] == 255 or low == high:
            break
        else:
            low -=1

    distance = low-high
    return distance

def blink2(img):
    width = img.shape[0]
    height = img.shape[1]
    high = 0
    low = height-1
    esc = False
    q = False
    j = 0
    i = 0
    while i < height and not q:
        for j in range(width):
            if img[i,j] == 255:
                high = i
                q = True
                break
            else:
                i+=1
    k = width - 1
    while low >= 0 and not esc:
        while k >= 0 :
            if img[low, k] == 255 or low == high:
                esc = True
                break
            else:
                low -=1

    distance = k-high
    return distance

def genCurve(data):
    frames = len(data)
    t = np.linspace(0, 4 * np.pi, frames)
    mean = np.mean(data)
    std =3 * np.std(data)/(2 ** 0.5)/(2 ** 0.5)
    phase = 0
    freq = 1
    amp = 1
    guess = std * np.sin(t + phase) + mean
    optimize = lambda x: x[0] + np.sin(x[1] * t + x[2]) + x[3] - data
    eAmp, eFreq, ePhase, eMean = scipy.optimize.leastsq(optimize, [amp, freq, phase, mean])[0]

    fit = eAmp * np.sin(eFreq * t + ePhase) + eMean
    fine_t = np.arange(0, max(t), 0.1)
    fit = eAmp * np.sin(eFreq * fine_t + ePhase) + eMean

    plt.plot(data, '.')
    plt.plot(fine_t, label='Fitted')
    plt.plot(guess, label="Guess")
    plt.legend()
    plt.show()


def getSin(data):
    frames = len(data)
    t = np.linspace(0, np.pi/2, frames)

    gFreq = 1
    gAmp = 1
    gPhase = 0
    gOffset = np.mean(data)

    p0 = [gFreq, gAmp, gPhase, gOffset]

    def sin(x, f, a, p, o): return np.sin(x * f + p) * a + o

    fit = scipy.optimize.curve_fit(sin, t, data, p0=p0)

    guess = sin(t, *p0)
    fitted = sin(t, *fit[0])

    plt.plot(data, '.')
    plt.plot(fitted, label='Fitted')
    plt.plot(guess, label="Guess")
    plt.legend()
    plt.show()

#Sin fitting
def fit_sin(data, tt,yy):
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    # guess_amp  = np.std(yy) * 2.0 ** 0.5
    guess_amp = np.max(data)/2
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sin(t, A, w, p, c): return A * np.sin(w * t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sin, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    
    # fit = A * np.sin(w * t + p) + c

    plt.plot(data, '+')
    plt.plot(fitfunc(tt), label="Fitted")
    plt.legend()
    plt.savefig("img.png")
    plt.cla()


    return Image.open("img.png")
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov)} #, "rawres": (guess,popt,pcov)}



eyes = None
left = []
right = []
while True:
   try :
    ret, frame = cap.read()
    # cv2.imshow("Face", frame)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    justFace = transform(frame, rect = findFace(frame))
    # justFace = frame
    justFace = cv2.bilateralFilter(frame, 2, 75, 75)
   
    # cv2.imshow("Face", justFace)
    if eyes is None:
        eyes = findEyes(justFace)

    if eyes is not None:    
        leftEye = zoom(justFace, eyes[0])
        leftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Left eye", leftEye)

        rightEye = zoom(justFace, eyes[1])
        rightEye = cv2.cvtColor(rightEye, cv2.COLOR_BGR2GRAY)

        
        #Contours
        leftRet, leftThresh = cv2.threshold(leftEye, thresh, 255, cv2.THRESH_BINARY)
        leftContours, leftHierarchy = cv2.findContours(leftThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        leftBlank = np.zeros(leftThresh.shape)
        cv2.drawContours(leftBlank, leftContours, -1, (255,0,0), 3)

        rightRet, rightThresh = cv2.threshold(rightEye, thresh, 255, cv2.THRESH_BINARY)
        rightContours, rightHierarchy = cv2.findContours(rightThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rightBlank = np.zeros(rightThresh.shape)
        cv2.drawContours(rightBlank, rightContours, -1, (255, 0, 0), 3)

        # #Canny
        leftFilter = cv2.bilateralFilter(leftEye, 40 , 60, 60) #Smoothing
        rightFilter = cv2.bilateralFilter(rightEye, 40 , 60, 60) #Smoothing

        leftFilter2 = cv2.bilateralFilter(leftEye, 15, 50, 50)
        edges_f = cv2.Canny(leftFilter, 19 , 120)
        edges_f2 = cv2.Canny(rightFilter, 19, 120)
        # cv2.imshow(edges_f)
        left.append(blink2(edges_f))
        right.append(blink2(edges_f2))

   except:
    #    genCurve(left)
    #    genCurve(right)
       lN = len(left)
       rN = len(right)
       ttL = np.linspace(0, 10, lN)
       ttR = np.linspace(0,10, rN)
       limg = fit_sin(left, ttL, left)
       rimg = fit_sin(right, ttR, right)

       width, height = limg.size
       outwidth, outheight = width * 2, height

       image = Image.new('RGB', (outwidth, outheight))

       image.paste(limg, (0,0))
       image.paste(rimg, (width, 0))

       image.save("side-by-side.png")
       break
   