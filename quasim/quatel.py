import numpy as np
from scipy import constants
import pandas as pd

'''
-Given location of the two telescope in Earth coordinate
-Width of time bin for correlation: 0.15ns
-Instrumental path length difference = 0
'''

class QuaTel:


    def __init__(self, N=2, A=1.0, tau=0.15, DL = 0.0):

        self.N = N                            # Only works for N=2 for now.
        self.A = A                            # collecting Area in meter-squared
        self.tau = tau                        # time bin for correlation in ns
        self.DL = DL                          # instrumental path length diff
        self.BW = 1.0/(2.0*np.pi*self.tau)    # Detector Bandwith in GHz
        self.Omega_E = 7.292*10.0**(-5.0)     # Earth Rotation speed (+z) [rad/sec]
                            
        
    def get_num_photon(self, pos_s, pos_t, lam, T, type_xy):
        # pos_s (source)  enter in 2-D list [[RA1,DEC1,S1],[RA2,DEC2,S2]] in rad and JY
        # or use output of BSC_fil as input for pos_s
        # pos_t (tele) enter in 2-D list [[RA1,DEC1,R1],[RA2,DEC2,R2]] in rad and meter
        # lambda in meter
        # T: period of observation
        # type_xy: enter 'pos' for cg,dh and 'neg' for ch,dg

        if isinstance(pos_s,list):
            pos_s = np.array(pos_s)
            pos_s = pos_s.reshape(pos_s.size/6,2,3)     # convert list to 3D array (M,2,3)
        elif isinstance(pos_s, np.ndarray):
            try:
                pos_s = np.delete(pos_s,0,axis=2)       # delete axis that contains star# (M,2,3)
            except:
                pos_s = np.delete(pos_s.reshape((1,2,4)),0,axis=2)     #When one pair of star is entered as array
                
        pos_t = np.array(pos_t).reshape(1,2,3)          # convert to (M,2,3) same as pos_s
        s1 = pos_s[:,0,2]*10.0**(-26)                   # convert Jy to mks unit , M-array
        s2 = pos_s[:,1,2]*10.0**(-26)

        
        if (np.absolute(pos_s[:,:,1] - np.tile(pos_t[:,:,1],(len(pos_s),1))) >= (np.pi/2.)).any():

            raise ValueError("At least 1 source out of reach")

        else:
            
            #change position of sources from declination to theta
            pos_s[:,:,1] = (np.pi/2.0)-pos_s[:,:,1]
            pos_t[:,:,1] = (np.pi/2.0)-pos_t[:,:,1]

        def pos_carte(posi,ti):
            # enter time (1D Array) in sec and posi in 2-D array, [[PHI1,THETA1],[PHI2,THETA2]]
            # find source position as function of time due to Earth rotation in cartesian

            N = ti.size
            M = len(posi)  
            ti = np.tile(ti,(M,2,1))                                #(M,2,N)
            PHI =  np.tile(posi[:,:,0].reshape(M,2,1), (1,1,N))     #(M,2,N)
            THETA = np.tile(posi[:,:,1].reshape(M,2,1), (1,1,N)) 

            new_posi = np.zeros((M,2,N,3))
            #change from spherical to cartesian

            x = np.sin(THETA)*np.cos(PHI - self.Omega_E*ti)   #[[[x1t1,x1t2,x1t3],[x2t1,x2t2,x2t3]],...] (M,2,N)
            y = np.sin(THETA)*np.sin(PHI - self.Omega_E*ti)
            z = np.cos(THETA)     

            new_posi[:,:,:,0] = x
            new_posi[:,:,:,1] = y
            new_posi[:,:,:,2] = z

            return new_posi
        

        #Find position vector of the two telescope in [x,y,z]
        new_pos_t = pos_carte(pos_t,np.array([0.])).reshape(2,3)                     #(1,2,1,3) -> (2,3)
        new_pos_t[0] *= pos_t[0,0,2]
        new_pos_t[1] *= pos_t[0,1,2]

        #Find the baseline vector in [x,y,z]:
        B_v = new_pos_t[1] - new_pos_t[0]                                       #(3) 
        B = np.linalg.norm(B_v)

        
        # Approximate the fringe rate in order to approximate dt
        w_f_ref= 2.0*np.pi*B*self.Omega_E/lam                                               

        #approximate time interval then break up period into small time intervals
        #assume dt << 1/w_f

        dt = 10.0/w_f_ref

        L = int(T/dt)
        t = np.linspace(0.0, T, L)

        new_pos_s = pos_carte(pos_s,t)                     # position vector of two sources in cartesian  (M,2,N,3)
        D_source = new_pos_s[:,0,:,:]-new_pos_s[:,1,:,:]   # difference between the two unit vector pointing to sources (M,N,3)

        #Dot product between baseline vector and D_source unit vector
        dot = B_v[0]*D_source[:,:,0] + B_v[1]*D_source[:,:,1] + B_v[2]*D_source[:,:,2]     #(M,N)
        
        k_const = self.tau*10**(-9)*(self.A*self.BW*10**(9)*lam/constants.h/constants.c)**2

        s1 = np.tile(s1,(L,1)).T           #(M,N)
        s2 = np.tile(s2,(L,1)).T
        
        vis = (2.0*s1*s2)/((s1+s2)**2)     #(M,N)
        
        N_xy = 1.0/8.0*k_const*(s1+s2)**2
        
        #phase term due to instrumental length diff
        ph = 2.0*np.pi*self.DL/lam


        if (type_xy == 'pos'):
            res_pos = N_xy*(1+vis*np.cos(2*np.pi/lam*dot+ph))        #(M,N), finds coincidence rate, rather than # of concidence
          #  excess = -N_xy*vis*np.cos(np.pi/2 -(2*np.pi/lam*dot+ph)) #term used for finding w(t) for func: freq_func
            phase = 2*np.pi/lam*dot+ph

            return res_pos, t, B_v, phase, D_source
        
        elif (type_xy == 'neg'):             
            res_neg = N_xy*(1-vis*np.cos(2*np.pi/lam*dot+ph))
          #  excess =  N_xy*vis*np.cos(np.pi/2 -(2*np.pi/lam*dot+ph))
            phase = 2*np.pi/lam*dot+ph

            return res_neg, t, B_v, phase

        else:
            raise ValueError("Invalid type")



    def get_rates(self, res_rate, time):
    # Calculates the max oscillation frequency, average coincidence rates, and fft
    # Input needs results from def above.
        
    # Find average concidence rates:
        max_res_rate = np.amax(res_rate, axis=1)
        min_res_rate = np.amin(res_rate, axis=1)
        avg_res_rate = (max_res_rate + min_res_rate)/2   #(M)

    # Find frequencies using FFT
        N = time.size
        M = len(res_rate)
    
        mean = np.tile(avg_res_rate, (N,1)).T                 #(M,N)                 
        sample_period = time[1]-time[0]

        fft = np.fft.fft(res_rate - mean)                     #(M,N),
        fft_amp = np.absolute(fft)                            #amplitude spectrum.
        fft_freq = np.fft.fftfreq(N, d = sample_period)       #(N)

        fft_freq_tile = np.tile(fft_freq, (M,1))
        ma = np.tile(np.amax(fft_amp, axis=1), (N,1)).T
        cond = np.where(fft_amp == ma)
        freq = pd.unique(np.absolute(fft_freq_tile[cond]))
    
    
        return avg_res_rate, fft, fft_freq, freq  #avg_res_rate for M pairs, fft,fft_freq, and peak_freq
        

    def freq_func(self, phase, time):
        # find frequency as a function of time by differentiating phase:
        # phase has shaoe (M,N), for M stars and N time steps
        M = len(phase)
        dy = np.diff(phase, axis=1)
        dx = np.tile(np.diff(time),(M,1))

        f_t = (dy/dx)/(2*np.pi)
        new_t = (time[:-1] + time[1:])/2

        return f_t, new_t

    
