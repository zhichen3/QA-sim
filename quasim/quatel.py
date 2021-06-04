import numpy as np
from scipy import constants

'''
-Two telescopes separated by 200m with collecting area 1m^2, straight E-W
-Width of time bin for correlation: 0.15ns
-Instrumental path length difference = 0
'''

class QuaTel:
    
    def __init__(self, N=2, B=200, A=1.0, tau=0.15, DL = 0.0, RA=1.3, DEC=0.71):

        self.N = N
        self.B = B                            # baseline in meter
        self.A = A                            # collecting Area in meter-squared
        self.tau = tau                        # time bin for correlation in ns
        self.DL = DL                          # instrumental path length diff
        self.BW = 1.0/(2.0*np.pi*self.tau)    # Detector Bandwith in GHz
        self.RA = RA                          # RA of tele in rad
        self.DEC = DEC                        # DEC of tele in rad
        self.THETA = (np.pi/2.0)-self.DEC     # convert DEC to THETA
        self.Omega_E = 7.292*10.0**(-5.0)     # Earth Rotation speed (+z) [rad/sec]
        
    def get_w_f(self, pos, lam):
        # pos enter in 2-D array [[RA1,DEC1],[RA2,DEC2]] in rad
        # lambda in meter
  
        pos = np.array(pos)

        if (np.absolute(pos[:,1]-self.DEC) >= (np.pi/2.)).all():

            raise ValueError("Sources out of reach")

        else:
            
            #change position of sources from declination to theta
            pos[:,1] = (np.pi/2.0)-pos[:,1]

        def source_pos(pos,ti):
            # enter time in sec and pos in 2-D array

            #change from spherical to cartesian for easier rotation:
            x = np.sin(pos[:,1])*np.cos(pos[:,0]+self.Omega_E*ti)
            y = np.sin(pos[:,1])*np.sin(pos[:,0]+self.Omega_E*ti)
            z = np.cos(pos[:,1])

            # rotate sources to telescope coord around z then y:
        
            x_new1 = np.cos(-self.RA)*x - np.sin(-self.RA)*y
            y_new = np.sin(-self.RA)*x + np.cos(-self.RA)*y

            x_new = np.cos(-self.THETA)*x_new1 + np.sin(-self.THETA)*z
            z_new = -np.sin(-self.THETA)*x_new1 + np.cos(-self.THETA)*z
            
            # THETA in tele coord: theta1 = pos[0,1], theta2 = pos[1,1]
            
            pos[:,1] = np.arccos(z_new)                # Give THETA
            pos[:,0] = np.arctan2(y_new,x_new)         # Give PHI


            return pos


        # Find Change in THETA in new coord in 1sec:

        Omega_E_new = abs(source_pos(pos,1)[0,1] - source_pos(pos,2)[0,1])
        

        # Find delta_THETA
        D_THETA = source_pos(pos,0)[0,1] - source_pos(pos,0)[1,1]

        #Let the source with smaller THETA be THETA_0
        if (D_THETA >= 0.0):
            THETA_0 = source_pos(pos,0)[1,1]
        else:
            THETA_0 = source_pos(pos,0)[0,1]


        D_THETA = abs(D_THETA)

        #Find w_f, fringe rate
        if (D_THETA < 0.1):
            w_f = (2.0*np.pi*self.B*Omega_E_new*np.sin(THETA_0)*D_THETA)/lam
        else:
            w_f = (2.0*np.pi*self.B*Omega_E_new/lam)*(np.sin(THETA_0)*
                    np.sin(D_THETA)+np.cos(THETA_0)*(1-np.cos(D_THETA)))

        return w_f

    def get_num_photon(self, w_f, lam, s1, s2, t, type_xy):
        # find w_f use func above
        # assume length of integration is small compared to fringe period
        # dt << 1/w_f , let dt = 0.01(1/w_f)
        # s1 and s2 are the spectral flux density of two sources
        # type_xy: enter 'pos' for cg,dh and 'neg' for ch,dg

        dt = 0.01/w_f
        
        k_const = self.tau*10**(-9)*dt*(self.A*self.BW*10**(9)*lam/constants.h/constants.c)**2

        vis = (2.0*s1*s2)/(s1+s2)**2
        
        N_xy = 1.0/8.0*k_const*(s1+s2)**2

        #phase term due to instrumental length diff
        ph = 2.0*np.pi*self.DL/lam


        if (type_xy == 'pos'):

            return N_xy*(1+vis*np.cos(w_f*t+ph))
        
        elif (type_xy == 'neg'):             

            return N_xy*(1-vis*np.cos(w_f*t+ph))

        
  #  def precision(self)
        
