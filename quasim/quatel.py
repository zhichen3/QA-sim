import numpy as np
from scipy import constants

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
                            
        
    def get_num_photon(self, pos_s, pos_t, lam, s1, s2, T, type_xy):
        # pos_s (source)  enter in 2-D list [[RA1,DEC1],[RA2,DEC2]] in rad
        # pos_t (tele) enter in 2-D list [[RA1,DEC1,R1],[RA2,DEC2,R2]] in rad and meter
        # lambda in meter
        # s1 and s2 are the spectral flux density of two sources, enter in [Jy]
        # T: period of observation
        # type_xy: enter 'pos' for cg,dh and 'neg' for ch,dg

        pos_s = np.array(pos_s)
        pos_t = np.array(pos_t)
        s1 *= 10.0**(-26)
        s2 *= 10.0**(-26)
        
        if (np.absolute(pos_s[:,1]- pos_t[:,1]) >= (np.pi/2.)).any():

            raise ValueError("At least 1 source out of reach")

        else:
            
            #change position of sources from declination to theta
            pos_s[:,1] = (np.pi/2.0)-pos_s[:,1]
            pos_t[:,1] = (np.pi/2.0)-pos_t[:,1]

        def pos_carte(posi,ti):
            # enter time (1D Array) in sec and posi in 2-D array, [[RA1,DEC1],[RA2,DEC2]]
            # find source position as function of time due to Earth rotation in cartesian

            N = ti.size

            #change from spherical to cartesian for easier rotation:
            x1 = np.sin(posi[0,1])*np.cos(posi[0,0]-self.Omega_E*ti)  #x,y for source 1
            y1 = np.sin(posi[0,1])*np.sin(posi[0,0]-self.Omega_E*ti)

            x2 = np.sin(posi[1,1])*np.cos(posi[1,0]-self.Omega_E*ti)  #x,y for source 2
            y2 = np.sin(posi[1,1])*np.sin(posi[1,0]-self.Omega_E*ti)

            x = np.vstack((x1,x2))   #[[x1t1,x1t2,...],[x2t1,x2t2,...]]
            y = np.vstack((y1,y2))   #[[y1t1,y1t2,...].[y2t1,y2t2,...]]
            
            z1 = np.tile(np.cos(posi[0,1]),N)
            z2 = np.tile(np.cos(posi[1,1]),N)
            z = np.vstack((z1,z2))

            new_posi = np.zeros((2,N,3))
            new_posi[:,:,0] = x
            new_posi[:,:,1] = y
            new_posi[:,:,2] = z
            
            return new_posi

        def coord_rot_theta(position_s,position_t):
            # rotate sources to telescope coord around z then y
            # for determining theta in tele coord so res=0 when sources are out of position of tele-plane
            # enter position_s after using pos_carte func above
            # enter position_t in  [[PHI1,THETA1,R1],[PHI2,THETA2,R2]]
            
            x_new1 = np.cos(-position_t[0,0])*position_s[:,:,0] - np.sin(-position_t[0,0])*position_s[:,:,1]
          # y_new = np.sin(-position_t[0,0])*position_s[:,:,1] + np.cos(-position_t[0,0])*position_s[:,:,0]

          # x_new = np.cos(-position_t[0,1])*x_new1 + np.sin(-position_t[0,1])*position_s[:,:,2]
            z_new = -np.sin(-position_t[0,1])*x_new1 + np.cos(-position_t[0,1])*position_s[:,:,2]
            
           # posi_rot = np.zeros((2,position_s[0,:,0].size))
            posi_rot_theta = np.arccos(z_new)                        # Give THETA in new coord
           # posi_rot[:,:,0] = np.arctan2(y_new,x_new)         # Give PHI in new coord

            return posi_rot_theta

        
        #Find position vector of the two telescope in [x,y,z]
        new_pos_t = pos_carte(pos_t,np.array([0.]))
        new_pos_t[0] *= pos_t[0,2]
        new_pos_t[1] *= pos_t[1,2]

        #Find the baseline vector in [x,y,z]:
        B_v = new_pos_t[1] - new_pos_t[0]
        B = np.linalg.norm(B_v)

        
        # Approximate the fringe rate in order to approximate dt
        w_f_ref= 2.0*np.pi*B*self.Omega_E/lam

        #approximate time interval then break up period into small time intervals
        #assume dt << 1/w_f

        dt = 1.0/w_f_ref
        
        t = np.linspace(0.0, T, int(T/dt))

        new_pos_s = pos_carte(pos_s,t)               # position vector of two sources in PHI and THETA
        D_source = new_pos_s[0]-new_pos_s[1]         # difference between the two unit vector pointing to the two sources

        #Dot product between baseline vector and D_source unit vector
        dot = B_v[0,0]*D_source[:,0]+B_v[0,1]*D_source[:,1]+B_v[0,2]*D_source[:,2]
        
        k_const = self.tau*10**(-9)*dt*(self.A*self.BW*10**(9)*lam/constants.h/constants.c)**2

        vis = (2.0*s1*s2)/((s1+s2)**2)
        
        N_xy = 1.0/8.0*k_const*(s1+s2)**2
        
        #phase term due to instrumental length diff
        ph = 2.0*np.pi*self.DL/lam

        # let res=0 when sources are out of sight due to rotation.
        source_theta_rot = coord_rot_theta(new_pos_s,pos_t)
        cond = np.where(  (np.absolute(source_theta_rot[0,:])>(np.pi/2.)) | (np.absolute(source_theta_rot[1,:]) > (np.pi/2.)) )
        
       
        if (type_xy == 'pos'):
            res_pos = N_xy*(1+vis*np.cos(2*np.pi/lam*dot+ph))
            res_pos[cond] = 0.0

            return res_pos, t, B_v
        
        elif (type_xy == 'neg'):             
            res_neg = N_xy*(1-vis*np.cos(2*np.pi/lam*dot+ph))
            res_neg[cond] = 0.0

            return res_neg, t, B_v

        else:
            raise ValueError("Invalid type")
        


        
