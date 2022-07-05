import numpy as np
from scipy.integrate import quad

class sim_data:

    def __init__(self, num, t, theo_phase, type_xy):
        '''
        This is a data point sampler given the pair rate as a function of time, theoretical phase and type of correlation.
            ________PARAMETERS___________
        num = pair rate for one star pair, 1D array  
        t = time
        theo_phase = theoretical phase array
        type_xy = positive correlated channels or anticorrelated channels, enter as 'pos' or 'neg'
        '''
        
        self.avg_rate = (np.amax(num) + np.amin(num))/2
        self.V = (np.amax(num/self.avg_rate -1 ))
        self.num = num
        self.theo_phase = np.mod((theo_phase + np.pi),2*np.pi) - np.pi


        # Define cdf and pdf
        if type_xy == 'pos':
            def pdf(x, V):
                return (1+V*np.cos(x))/(2*np.pi)

            def cdf(x, V):
                return (x+np.pi+V*np.sin(x))/(2*np.pi)

            #Inverse cdf: and find corresponding phase using bisection method:
            def inverse_cdf(r,a,b,V):
                def inverse_cdf_func(x,r,V):
                    return (x+V*np.sin(x)+np.pi)/(2*np.pi) - r
    
                if (inverse_cdf_func(a,r,V)*inverse_cdf_func(b,r,V) >= 0).any():
                    raise ValueError("bisection method failed, need to reset boundary a and b")
    
                a_n = a
                b_n = b
                m_n = (a_n + b_n)/2
                f_m_n = inverse_cdf_func(m_n, r, V)    #where we have len(r) equations, or len(r) 1-D array
    
                while (np.absolute(f_m_n) > 0.00001).any():
                    f_a_n = inverse_cdf_func(a_n, r, V)
                    f_b_n = inverse_cdf_func(b_n, r, V)
                    cond1 = np.where(f_a_n*f_m_n < 0) 
                    cond2 = np.where(f_b_n*f_m_n < 0)
                    b_n[cond1] = m_n[cond1]
                    a_n[cond2] = m_n[cond2]
                    m_n = (a_n+b_n)/2
                    f_m_n = inverse_cdf_func(m_n, r, V)
        
                return m_n




        elif type_xy == 'neg':
            def pdf(x, V):
                return (1 - V*np.cos(x))/(2*np.pi)

            def cdf(x, V):
                return (x+np.pi - V*np.sin(x))/(2*np.pi)


            #Inverse cdf: and find corresponding phase using bisection method:
            def inverse_cdf(r,a,b,V):
                def inverse_cdf_func(x,r,V):
                    return (x-V*np.sin(x)+np.pi)/(2*np.pi) - r
    
                if (inverse_cdf_func(a,r,V)*inverse_cdf_func(b,r,V) >= 0).any():
                    raise ValueError("bisection method failed, need to reset boundary a and b")
                    
    
                a_n = a
                b_n = b
                m_n = (a_n + b_n)/2
                f_m_n = inverse_cdf_func(m_n, r, V)    #where we have len(r) equations, or len(r) 1-D array
    
                while (np.absolute(f_m_n) > 0.00001).any():
                    f_a_n = inverse_cdf_func(a_n, r, V)
                    f_b_n = inverse_cdf_func(b_n, r, V)
                    cond1 = np.where(f_a_n*f_m_n < 0) 
                    cond2 = np.where(f_b_n*f_m_n < 0)
                    b_n[cond1] = m_n[cond1]
                    a_n[cond2] = m_n[cond2]
                    m_n = (a_n+b_n)/2
                    f_m_n = inverse_cdf_func(m_n, r, V)
                    
                return m_n

        #condition to find end point of cycle
        cond = np.asarray(np.where((num-np.amin(num)) < 0.00001*self.avg_rate))

        #condition to eliminate extra points
        cond1 = np.asarray(np.where(np.diff(cond) == 1))

        # indices for theo_phase ~ pi
        cond = np.delete(cond, cond1)

        # Split array at each pi, we have N arrays, each element corresponds to one cycle
        phase_split = np.array(np.split(self.theo_phase, cond),dtype=object)
        t_split = np.array(np.split(t,cond),dtype=object)

        #timestamp corresponding to randomly generated phase
        self.N = len(phase_split)  # Total number of cycles
        self.cycle_period= np.zeros(self.N)

        for i in range(self.N):
            self.cycle_period[i] = abs(t_split[i][0]-t_split[i][-1])
        
        avg_num = self.avg_rate * self.cycle_period   # average number of counts
        self.p = np.random.poisson(avg_num)
        M = np.sum(self.p)     # number of occurences

        self.r = np.random.random(M)
        a = np.full(shape= M, fill_value = np.pi+0.5)
        b = np.full(shape= M, fill_value = -np.pi-0.5)
        phase = inverse_cdf(self.r,a,b,self.V)

        timestamp = np.zeros(M)

        # find timestamp for all phase
        i = 0
        index = np.where(self.p > 0)
        for j in range(index[0].size):    # j-th cycle
            for k in range(self.p[index][j]):
                indices = np.abs(phase[i] - phase_split[index][j]).argmin()
                timestamp[i] = t_split[index][j][indices]
                i += 1
        
        self.timestamp = timestamp[np.argsort(timestamp)]
        self.phase = phase[np.argsort(timestamp)]
        
        self.sim_num = pdf(self.phase,self.V)*2*np.pi*self.avg_rate


        # Outputs theoretical pdf and cdf for visual effects / Irrelevant
        self.x = np.linspace(-np.pi, np.pi, 200)
        self.pdf_value = pdf(self.x,self.V)
        self.cdf_value = cdf(self.x,self.V)

