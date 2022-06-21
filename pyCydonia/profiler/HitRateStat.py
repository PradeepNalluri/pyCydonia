import numpy as np 

DEFAULT_HIT_RATE_TRACKED = [0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

class HitRateStat:
    def __init__(self, 
            hit_rate_file,
            hit_rate_tracked = DEFAULT_HIT_RATE_TRACKED):
        
        self.data = np.genfromtxt(hit_rate_file, delimiter=',')
        self.hit_rate_tracked = hit_rate_tracked

        print("In HitRateStat")