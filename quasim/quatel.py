import numpy as np
from scipy import constants
import pandas as pd

'''
-Given location of the two telescope in Earth coordinate
-Width of time bin for correlation: 0.15ns
-Instrumental path length difference = 0
'''

class QuaTel:


    def __init__(self, N=2, A=1.0, tau=0.15, ph = 0.0):

        self.N = N                            # Only works for N=2 for now.
        self.A = A                            # collecting Area in meter-squared
        self.tau = tau                        # time bin for correlation in ns
        self.ph = ph                          # offset phase due to instrumental diff
        self.BW = 1.0/(2.0*np.pi*self.tau)    # Detector Bandwith in GHz
        self.Omega_E = 7.292e-5               # Earth Rotation speed (+z) [rad/sec]
                          
        
    def get_num_photon(self, pos_s, pos_t, lam, T, type_xy):
        # pos_s (source)  enter in 2-D list [[RA1,DEC1,S1],[RA2,DEC2,S2]] in rad and JY
        # or use output of BSC_fil as input for pos_s
        # pos_t (tele) enter in 2-D list [d_EW,d_NS,L=latitude] in rad and meter
        # lambda in meter
        # T: period of observation [T_begin,T_end]
        # type_xy: enter 'pos' for cg,dh and 'neg' for ch,dg

        if isinstance(pos_s,list):
            pos_s = np.array(pos_s)
            pos_s = pos_s.reshape(pos_s.size/6,2,3)     # convert list to 3D array (M,2,3)
        elif isinstance(pos_s, np.ndarray):
            try:
                pos_s = np.delete(pos_s,0,axis=2)       # delete axis that contains star# (M,2,3)
            except:
                pos_s = np.delete(pos_s.reshape((1,2,4)),0,axis=2)     #When one pair of star is entered as array
                
        baseline = np.array(pos_t)          
        s1 = pos_s[:,0,2]*10.0**(-26)                   # convert Jy to mks unit , M-array
        s2 = pos_s[:,1,2]*10.0**(-26)
        
        def source_pos(posi,ti):
            # enter time (1D Array) in sec and posi in 2-D array, [[PHI1,THETA1],[PHI2,THETA2]]
            # find source position as function of time due to Earth rotation in cartesian

            N = ti.size
            M = len(posi)  
            ti = np.tile(ti,(M,1))
            
            PHI = posi[:,:,0]
            d_PHI = PHI[:,0]-PHI[:,1]
            PHI_mid = np.tile((PHI[:,0]+PHI[:,1])/2,(1,N))-self.Omega_E*ti
            DEC = posi[:,:,1]
            d_DEC = DEC[:,0]-DEC[:,1]  #M
            DEC_mid = np.tile((DEC[:,0]+DEC[:,1])/2,(1,N))  #M
            
            new_posi = np.zeros((M,N,3))
            dx = -np.sin(DEC_mid)*np.cos(PHI_mid)*d_DEC-d_PHI*np.cos(DEC_mid)*np.sin(PHI_mid)
            dy = -np.sin(DEC_mid)*np.sin(PHI_mid)*d_DEC+d_PHI*np.cos(DEC_mid)*np.cos(PHI_mid)
            dz = d_DEC * np.cos(DEC_mid)
           
            
            new_posi[:,:,0] = dx
            new_posi[:,:,1] = dy
            new_posi[:,:,2] = dz

            return new_posi
        

        #Find norm of baseline
        B = np.sqrt(baseline[0]**2+baseline[1]**2)
        
        # Approximate the fringe rate in order to approximate dt
        w_f_ref= 2.0*np.pi*B*self.Omega_E/lam                                               

        #approximate time interval then break up period into small time intervals
        #assume dt << 1/w_f

        dt = 10.0/w_f_ref

        L = int((abs(T[0])+abs(T[1]))/dt)
        t = np.linspace(T[0], T[1], L)
        
        #new_pos_s = pos_carte(pos_s,t)                     # position vector of two sources in cartesian  (M,2,N,3)
        #D_source = new_pos_s[:,0,:,:]-new_pos_s[:,1,:,:]   # difference between the two unit vector pointing to sources (M,N,3)
        D_source = source_pos(pos_s,t)
        #Dot product between baseline vector and D_source unit vector
        dot = -baseline[1]*np.sin(baseline[2])*D_source[:,:,0] + baseline[0]*D_source[:,:,1] \
              +baseline[1]*np.cos(baseline[2])*D_source[:,:,2]     #(M,N)
        
        k_const = self.tau*10**(-9)*(self.A*self.BW*10**(9)*lam/constants.h/constants.c)**2

        s1 = np.tile(s1,(L,1)).T           #(M,N)
        s2 = np.tile(s2,(L,1)).T
        
        vis = (2.0*s1*s2)/((s1+s2)**2)     #(M,N)
        self.vis = vis
        
        N_xy = 1.0/8.0*k_const*(s1+s2)**2
        

        if (type_xy == 'pos'):
            res_pos = N_xy*(1+vis*np.cos(2*np.pi/lam*dot+self.ph))        #(M,N), finds coincidence rate, rather than # of concidence
          #  excess = -N_xy*vis*np.cos(np.pi/2 -(2*np.pi/lam*dot+ph)) #term used for finding w(t) for func: freq_func
            phase = 2*np.pi/lam*dot+self.ph

            return res_pos, t, B, phase, D_source
        
        elif (type_xy == 'neg'):             
            res_neg = N_xy*(1-vis*np.cos(2*np.pi/lam*dot+self.ph))
          #  excess =  N_xy*vis*np.cos(np.pi/2 -(2*np.pi/lam*dot+ph))
            phase = 2*np.pi/lam*dot-self.ph

            return res_neg, t, B, phase, D_source

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

    
