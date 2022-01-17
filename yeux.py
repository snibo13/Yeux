import cv2
import sys
import numpy as np
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
cap = cv2.VideoCapture("TV.mp4")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
facemarks = cv2.face.createFacemarkLBF()
facemarks.loadModel("lbf.yaml")

def sample(lst, step):
  return lst[::step]

def span(lst):
  vert = [y for x,y in lst]
  return max(vert) - min(vert)

  
  

def fit_sin(data, t, y, title):
  # Type conversion for numpy
  t = np.array(t)
  y = np.array(y)
  # Fast fourier transform sinusoidal regression
  # Used for initial guess
  # Determine frequency of samples
  f = np.fft.fftfreq(len(t), (t[1] - t[0]))
  # Single dimensional fourier transform
  Fy = abs(np.fft.fft(y))
  guess_freq = abs(f[np.argmax(Fy[1:]) + 1])
  guess_amp = np.max(data) / 2.0 ** 0.5
  guess_offset = np.mean(y)
  guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

  def sin(t,A, w, p, c): return A * np.sin(w * t + p) + c
  # Curve fitting based on fft approximation
  popt, pcov = scipy.optimize.curve_fit(sin, t, y, p0 = guess)

  A, w, p, c = popt
  f = w / (2.0 * np.pi)
  fit_func = lambda t: A * np.sin(w * t + p)
  


  plt.cla()
  plt.plot(data - c, '+')
  # plt.plot(sin(t, guess_amp, 2.0 * np.pi * guess_freq, 0.0, 0), label="Original")
  plt.plot(fit_func(t), label="Fitted")
  plt.legend()
  plt.savefig(title)
  plt.cla()
  return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f}

def computeSineWaves(cap):
  # Span measurements for both eyes
  left = []
  right = []
  ret = True
  while ret:
    ret, frame = cap.read()
    res = face_cascade.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5)
    if len(res) == 0:
      continue
    
    x = res[0][0]
    y = res[0][1]
    w = res[0][2]
    h = res[0][3]
    face = frame[y: y+h, x: x + w]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    success, landmarks = facemarks.fit(gray, res)
    landmarks = landmarks[0][0]
    
    if success:
      left_eye = landmarks[36:42]
      right_eye = landmarks[42:48]
      left.append(span(left_eye))
      right.append(span(right_eye))

  # print(right[0:10])
  ls = sample(left, 1)
  rs = sample(right, 1)


  # Normalizing for pupil height Results in percentages
  leftSin = fit_sin(ls, np.linspace(0, 1, len(ls)), ls, "L.png")
  rightSin = fit_sin(rs, np.linspace(0, 1, len(rs)), rs, "R.png")

  leftAmp = leftSin["amp"]
  rightAmp = rightSin["amp"]

  fullyOpen = max(max(leftAmp), max(rightAmp))
  scalingFactor = 11.4 / float(fullyOpen)
  ls = [x * scalingFactor for x in ls]
  rs = [x * scalingFactor for x in rs]

  leftSin = fit_sin(ls, np.linspace(0, 1, len(ls)), ls, "L.png")
rightSin = fit_sin(rs, np.linspace(0, 1, len(rs)), rs, "R.png")


if __name__ == '__main__':
  if len(sys.argv) is not 1:
    print("Usage: yeux.py <Path to Video>")
  else:
    cap = cv2.VideoCapture(sys.argv[1])
    computeSineWaves(cap)
  
