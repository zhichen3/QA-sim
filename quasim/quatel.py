import numpy as np
from scipy import constants

'''
-Two telescopes separated by 200m with collecting area 1m^2, straight E-W
-Width of time bin for correlation: 0.15ns
-Instrumental path length difference = 0
'''

class QuaTel:
    
    def __init__(self, N=2, B=200, A=1.0, tau=0.15, DL = 0.0, RA=-1.3, DEC=0.71):

        self.N = N                            # Only works for N=2 for now.
        self.B = B                            # baseline in meter
        self.A = A                            # collecting Area in meter-squared
        self.tau = tau                        # time bin for correlation in ns
        self.DL = DL                          # instrumental path length diff
        self.BW = 1.0/(2.0*np.pi*self.tau)    # Detector Bandwith in GHz
        self.RA = RA                          # RA of tele in rad
        self.DEC = DEC                        # DEC of tele in rad
        self.THETA = (np.pi/2.0)-self.DEC     # convert DEC to THETA
        self.Omega_E = 7.292*10.0**(-5.0)     # Earth Rotation speed (+z) [rad/sec]
        
    def get_num_photon(self, pos, lam, s1, s2, T, type_xy):
        # pos enter in 2-D list [[RA1,DEC1],[RA2,DEC2]] in rad
        # lambda in meter
        # s1 and s2 are the spectral flux density of two sources
        # type_xy: enter 'pos' for cg,dh and 'neg' for ch,dg

        pos = np.array(pos)

        if (np.absolute(pos[:,1]-self.DEC) >= (np.pi/2.)).any():

            raise ValueError("At least 1 source out of reach")

        else:
            
            #change position of sources from declination to theta
            pos[:,1] = (np.pi/2.0)-pos[:,1]

        def source_pos(posi,ti):
            # enter time (1D Array) in sec and posi in 2-D array
            # find source position in tele coord

            N = ti.size

            #change from spherical to cartesian for easier rotation:
            x1 = np.sin(posi[0,1])*np.cos(posi[0,0]-self.Omega_E*ti)  #x,y for source 1
            y1 = np.sin(posi[0,1])*np.sin(posi[0,0]-self.Omega_E*ti)

            x2 = np.sin(posi[1,1])*np.cos(posi[1,0]-self.Omega_E*ti)  #x,y for source 2
            y2 = np.sin(posi[1,1])*np.sin(posi[1,0]-self.Omega_E*ti)

            x = np.vstack((x1,x2))   #[[x1t1,x1t2,...],[x2t1,x2t2,...]]
            y = np.vstack((y1,y2))
            
            z = np.transpose(np.tile(np.cos(posi[0,1]),(N,1)))   #[[z1,z1,...],[z2,z2,...]]

            # rotate sources to telescope coord around z then y:
        
            x_new1 = np.cos(-self.RA)*x - np.sin(-self.RA)*y
            y_new = np.sin(-self.RA)*x + np.cos(-self.RA)*y

            x_new = np.cos(-self.THETA)*x_new1 + np.sin(-self.THETA)*z
            z_new = -np.sin(-self.THETA)*x_new1 + np.cos(-self.THETA)*z
            
            # THETA in tele coord: theta1 = pos[0,1], theta2 = pos[1,1]
            new_posi = np.zeros((2,N,2))
            new_posi[:,:,1] = np.arccos(z_new)                # Give THETA
            new_posi[:,:,0] = np.arctan2(y_new,x_new)         # Give PHI


            return new_posi


        # Find delta_THETA: total angle diff (THETA PHI) or just angle diff in THETA? 
        D_THETA = source_pos(pos,np.array([0.]))[0,0,1] - source_pos(pos,np.array([0.]))[1,0,1]

        #Let the source with smaller THETA be THETA_0
        if (D_THETA >= 0.0):
            THETA_0 = source_pos(pos,np.array([0.]))[1,0,1]
        else:
            THETA_0 = source_pos(pos,np.array([0.]))[0,0,1]

        D_THETA = abs(D_THETA)

        # Omega_E_sam = np.absolute(source_pos(pos,np.array([1.]))[0,0,1] - source_pos(pos,np.array([2.]))[0,0,1])

        # Use self.Omega_E for w_f_ref since Omega_E is the largest possible value for change in THETA 
        #Find w_f_ref, reference fringe rate used to approx time interval
        if (D_THETA < 0.1):
            w_f_ref = (2.0*np.pi*self.B*self.Omega_E*np.sin(THETA_0)*D_THETA)/lam
        else:
            w_f_ref= (2.0*np.pi*self.B*self.Omega_E/lam)*(np.sin(THETA_0)*
                    np.sin(D_THETA)+np.cos(THETA_0)*(1-np.cos(D_THETA)))


        #approximate time interval then break up period into small time intervals
        #assume dt << 1/w_f
        dt = 0.08/w_f_ref

        t = np.linspace(0.0, T, int(T/dt)) 


        # Calculate actual Omega_E and w_f  more accurately since it depends on time:
        pos_t1 = source_pos(pos,t)
        pos_t2 = source_pos(pos,t+dt)
        Omega_E_new =(np.absolute(pos_t1[0,:,1] - pos_t2[0,:,1]))/dt

       
        
        if (D_THETA < 0.1): 
            w_f = (2.0*np.pi*self.B*Omega_E_new*np.sin(THETA_0)*D_THETA)/lam   #w_f: 1-D with N-ele
        else:
            w_f = (2.0*np.pi*self.B*Omega_E_new/lam)*(np.sin(THETA_0)*
                    np.sin(D_THETA)+np.cos(THETA_0)*(1-np.cos(D_THETA)))

        
       # dt[cond] = 0.0
        
        k_const = self.tau*10**(-9)*dt*(self.A*self.BW*10**(9)*lam/constants.h/constants.c)**2

        vis = (2.0*s1*s2)/((s1+s2)**2)
        
        N_xy = 1.0/8.0*k_const*(s1+s2)**2
        
        #phase term due to instrumental length diff
        ph = 2.0*np.pi*self.DL/lam

        res_pos = N_xy*(1+vis*np.cos(w_f*t+ph))
        res_neg = N_xy*(1-vis*np.cos(w_f*t+ph))

        # let res=0 when sources are out of sight due to rotation.

        cond = np.where(  (np.absolute(pos_t1[0,:,1])>(np.pi/2.)) | (np.absolute(pos_t1[1,:,1]) > (np.pi/2.)) )
        res_pos[cond] = 0.0
        res_neg[cond] = 0.0
        
        if (type_xy == 'pos'):

            return res_pos, t, w_f
        
        elif (type_xy == 'neg'):             

            return res_neg, t, w_f

        
  #  def precision(self)
        
