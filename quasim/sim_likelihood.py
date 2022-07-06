
import sys
sys.path=["../external/april/py"]+sys.path
from Parameter import Parameter
from BaseLikelihood import BaseLikelihood
import MCMCAnalyzer
import numpy as np

"""
This is the simulation likelihood for MCMC.

Parameters
-----------
seed: a list of initial guess visibility, d_ra and d_dec. e.g. [V_init, d_ra_init, d_dec_init]
pos_t: position of telescope. A list of the baseline. e.g. [B_ew, B_ns, Latitude]
pos_s: position of the source in list, takes result from the either BSC.BSC_filter() or a list of [[ra1,dec1,spectra_flux_density1],[ra2,dec2,spectral_flux_density2]
lam: wavelength of observation
t_pos: time stamp of positive mode data
t_neg: time stamp of negative mode data
err: list of errors for the four parameter: [err_vis, err_ew, err_ns, err_phase] Adjust error in order to get nice fits.
"""

class sim_like(BaseLikelihood):    
    def __init__(self, seed, pos_t, pos_s, lam, t_pos = None, t_neg = None, err=[0.04, 5e-10, 15e-10, np.pi/10.0]):  # seed = [V_init,d_ra_init, d_dec_init]
        BaseLikelihood.__init__(self,"sim_data")  
        
        # free par
        self.seed = seed
        self.V = seed[0]
        self.d_ew = seed[1]
        self.d_ns = seed[2]
        self.offset = seed[3]

        self.err = err
        self.t_pos = t_pos             # timestamp for the sim data
        self.t_neg = t_neg
        
        self.baseline = np.array(pos_t)

        if len(pos_s[0]) == 4:
            self.pos_s = np.delete(pos_s, 0, axis=1)  # delete star # part.
        else:
            self.pos_s = pos_s     # position of sources to determine midpoint
        
        self.lam = lam         # lambda for observation
        self.Omega_E = 7.292e-5

    def freeParameters(self):
    # Adjust err or bounds accordingly for constrained triangle plots
        return [
                Parameter("V", self.seed[0], err=self.err[0],bounds=(0.0,1.0)),    #0.1 for 1arcsec ,bounds=(-0.1,0.7)
                Parameter("d_ew",self.seed[1], err=self.err[1],),   #5e-10
                Parameter("d_ns",self.seed[2], err=self.err[2],),   #12e-10 for 1arcsec, 5e-10 for 15arcsec
                Parameter("offset",self.seed[3], err=self.err[3],bounds=(-np.pi,np.pi)),   #bounds=(-np.pi,np.pi)
                ]
    
    def updateParams(self,params):    #params is also a class, updates param value.
        for p in params:
            if p.name=="V":
                self.V=p.value
            if p.name=="d_ew":
                self.d_ew=p.value
            if p.name=="d_ns":
                self.d_ns=p.value
            if p.name=="offset":
                self.offset=p.value
        
    def source_pos(self,ti):
        N = ti.size
        M = len(self.pos_s)  
            
        PHI = self.pos_s[:,0]
        PHI_mid = np.tile((PHI[0]+PHI[1])/2,(N))-self.Omega_E*ti
        DEC = self.pos_s[:,1]
        DEC_mid = np.tile((DEC[0]+DEC[1])/2,(N))  
            
        dx = -np.sin(DEC_mid)*np.cos(PHI_mid)*self.d_ns - self.d_ew*np.sin(PHI_mid)
        dy = -np.sin(DEC_mid)*np.sin(PHI_mid)*self.d_ns + self.d_ew*np.cos(PHI_mid)
        dz = self.d_ns * np.cos(DEC_mid)
        
        new_posi = np.column_stack((dx,dy,dz))
        
        return new_posi
    
    def get_phase(self):
        if self.t_pos is not None:
            new_pos_s = self.source_pos(self.t_pos)
            dot = -self.baseline[1]*np.sin(self.baseline[2])*new_pos_s[:,0] \
               + self.baseline[0] *new_pos_s[:,1] \
               +self.baseline[1]*np.cos(self.baseline[2])*new_pos_s[:,2]
       
            phase_pos = 2*np.pi/self.lam*dot + self.offset   # Total phase with the offset for plus mode

        if self.t_neg is not None:
            new_pos_s = self.source_pos(self.t_neg)
            
            dot = -self.baseline[1]*np.sin(self.baseline[2])*new_pos_s[:,0] \
               + self.baseline[0] *new_pos_s[:,1] \
               +self.baseline[1]*np.cos(self.baseline[2])*new_pos_s[:,2]
       
            phase_neg = 2*np.pi/self.lam*dot + self.offset   # Total phase with the offset for plus mode
                        
        if (self.t_neg is None and self.t_pos is not None):
            return phase_pos
        
        elif (self.t_neg is not None and self.t_pos is None):
            return phase_neg
        
        else:            
            return [phase_pos,phase_neg]

    def loglike_wprior(self):

        phase = self.get_phase()
        if (self.t_neg is None and self.t_pos is not None):
            res = np.sum(np.log(1+self.V*np.cos(phase)) , axis=None)
            
        elif (self.t_neg is not None and self.t_pos is None):
            res = np.sum(np.log(1-self.V*np.cos(phase)), axis=None)
        
        else:
            res = np.sum(np.log(1+self.V*np.cos(phase[0])), axis=None) \
                    + np.sum(np.log(1-self.V*np.cos(phase[1])), axis=None)
                   
        return res
