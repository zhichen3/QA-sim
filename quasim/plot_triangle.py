import corner
import numpy as np
import copy

"""
Object for plotting
Parameters
-------------------
fname: string of input file
theo_par: theoretical parameters, get from QuaTel.get_theo_par()
obs_duration: observation duration in secs
avg_rate: average pair rate
n_cycle: number of cycles, enter only number of cycles for one mode, no matter used both positive and negative mode.
output_file: string of the output file
single_mode: Whether the simulated data used a single positive or negative mode, or two modes combined.  
"""
class plot_triangle:
    def __init__(self, fname, theo_par, obs_duration, avg_rate, n_cycle, output_file="output_data.jpg", single_mode=False):

        self.fname = fname
        self.theo_par = theo_par
        self.obs_duration = obs_duration
        self.avg_rate = avg_rate
        self.n_cycle = n_cycle
        self.output_file = output_file
        self.single_mode = single_mode

    def plot(self):
        
        res = np.loadtxt(self.fname)
        mcmc_data = np.delete(np.delete(res, [0,1], axis=1),np.s_[:int(res[:,0].size/2)], axis=0) 

        # Find Sigma using formula in the paper:
        k = (1.0-np.sqrt(1-self.theo_par[0]**2))/self.theo_par[0]**2
        
        # Using the General formula:
        self.theo_sigma_ew = np.absolute(np.sqrt(6/(k*np.pi**2))/self.theo_par[0]/np.sqrt(self.obs_duration*self.avg_rate) \
                                    *self.theo_par[1]/self.n_cycle)/4.8481368e-9
        self.theo_sigma_ns = np.absolute(np.sqrt(6/(k*np.pi**2))/self.theo_par[0]/np.sqrt(self.obs_duration*self.avg_rate) \
                                    *self.theo_par[2]/self.n_cycle)/4.8481368e-9 

        if not self.single_mode:
            self.theo_sigma_ew /= np.sqrt(2.0)
            self.theo_sigma_ns /= np.sqrt(2.0)
        print(f"Sigma ew from general formula = {self.theo_sigma_ew}")
        print(f"Sigma ns from general formula = {self.theo_sigma_ns}")
        
        # Find center point of d_ew and d_ns
        d_ew_mid = corner.quantile(mcmc_data[:,1],0.5)
        d_ns_mid = corner.quantile(mcmc_data[:,2],0.5)
        
        mcmc_data[:,1] = (mcmc_data[:,1] - d_ew_mid)/4.8481368e-9
        mcmc_data[:,2] = (mcmc_data[:,2] - d_ns_mid)/4.8481368e-9

        self.theo_par_mod = copy.deepcopy(self.theo_par[:])
        self.theo_par_mod[1] = (self.theo_par[1] - d_ew_mid)/4.8481368e-9
        self.theo_par_mod[2] = (self.theo_par[2] - d_ns_mid)/4.8481368e-9

        
        figure = corner.corner(mcmc_data,labels=['Visbility','Δd_E [mas]','Δd_N [mas]','offset_phase [rad]'],quantiles=(0.023, 0.16, 0.5, 0.84,0.977),levels=(0.68,0.952,0.994))
        corner.overplot_points(figure, np.array(list(self.theo_par_mod))[None], marker="s", color="C1",label="Theo")
        figure.savefig(self.output_file,facecolor='white')
        
        # Find sigma from the MCMC by calculating the square root of covariance matrix
        self.sigma_mcmc = np.sqrt(np.cov(mcmc_data,rowvar=False).diagonal())
        print("Sigma from MCMC =",self.sigma_mcmc)
