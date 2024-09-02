import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

def get_max_streak(boolvector: np.ndarray, what_to_find : bool = True):
    max_streak = 0
    current_streak = 0
    for current_value in boolvector:
        if current_value == what_to_find:
            current_streak +=1
        else:
            max_streak = max([max_streak, current_streak])
            current_streak = 0
    return max_streak

def z_scores(y: np.ndarray) -> np.ndarray:
    median_y = np.median(y)
    MAD_y = np.median(np.abs(y-median_y))
    z_y = 0.6745*(y - median_y)/MAD_y
    return z_y

def WhitakerHayes(y: np.ndarray, neighbours: int = 2, threshold: float = 6) -> np.ndarray:

    dy = np.diff(y)
    z_dy = z_scores(dy)

    detected_as_outlier = abs(z_dy) >= threshold
    den_y = np.copy(y)
    detected_as_outlier[0:neighbours-1] = False
    detected_as_outlier[-neighbours:] = False
    
    max_streak_outliers = get_max_streak(detected_as_outlier, True)
    if max_streak_outliers >= 2*neighbours:  # TODO: Check this recursive condition
        den_y = WhitakerHayes(y, neighbours+2, 1.25*threshold)
        return den_y

    restart = False
    for i in np.arange(len(detected_as_outlier)):
        if detected_as_outlier[i]:
            w = np.arange(i-neighbours, i+1+neighbours)
            w_cor = w[detected_as_outlier[w] == 0]
            if w_cor.size == 0:
                restart = True
                raise Exception
                break
            den_y[i] = np.mean(y[w_cor])
            
    if any(np.isnan(den_y)) or restart: # TODO: Check this recursive condition
        #den_y = WhitakerHayes(y, neighbours, 1.1*threshold)
        den_y = WhitakerHayes(y, neighbours+2, 1.25*threshold)

    return den_y

if __name__ == "__main__":
    K = 10
    theta = np.arange(0, 2*np.pi, 0.01)
    y = np.zeros_like(theta)
    for idx in np.arange(50):
        y += np.sin(2*(idx+1)*theta)/((idx+1)**2)

    r_idx = random.sample(range(len(y)), K)
    y[r_idx] += np.random.normal(0, 1, K)

    neighbours = 3
    threshold = 10
    y_despiked = WhitakerHayes(y, neighbours, threshold)

    line_y, = plt.plot(y)
    line_y_despiked, = plt.plot(y_despiked, '--')
    line_y.set_label('spiky spectrum')
    line_y_despiked.set_label('despiked spectrum')
    plt.legend()
    plt.show()

    '''
    dy = np.diff(y)
    z_dy = z_scores(dy)
    line_y, = plt.plot(y)
    line_z, = plt.plot(10 + (z_dy > threshold))
    line_y.set_label('spiky spectrum')
    line_z.set_label('spikes')
    plt.legend()
    plt.show()
   
    plt.plot(z_dy)
    plt.show()
    '''