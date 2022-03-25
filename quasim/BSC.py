import numpy as np

'''
It processes the Bright star catalogue file to output
as an array [[star#1,RA1,DEC1,s1],[star#2,RA2,DEC2,s2]....]
'''


class BSC_process:
    def __init__(self, BSC_file):
        lis = []
        for line in open(BSC_file,"rt"):
            if (len(line[20:35].strip()) < 5 ):                     #Ignore stars that have no info
                continue
        
            star_num = int(line[0:4])
            try:
                RA_hours = float(line[75:77])          
                RA_min = float(line[77:79])
                RA_sec = float(line[79:83])
                RA = (RA_hours + RA_min/60 + RA_sec/3600)/24*360*np.pi/180    # RA for J2000
    
                DEC_sign = (line[83] == '-')
                DEC_deg = float(line[84:86])
                DEC_arcmin = float(line[86:88])
                DEC_arcsec = float(line[88:90])
                DEC = (DEC_deg + DEC_arcmin/60 + DEC_arcsec/3600)*np.pi/180   # DEC for J2000
    
                if DEC_sign:
                    DEC *= -1
                # lam ~ 0.55um, F_v,0 = 3640Jy
                V_mag = float(line[102:107])               # Convert to flux density
                s_V = 3640.*10**(V_mag/-2.5)                # flux density in Jy
                #s_V = 2000.*10**(V_mag/-2.5)
            except ValueError:
                continue


            lis.append(star_num)
            lis.append(RA)
            lis.append(DEC)
            lis.append(s_V)
            
        row_num = int(len(lis)/4)
        pos_s = np.array(lis).reshape((row_num,4))    #[[star_num1,RA1,DEC1,s_V1],[star_num2,RA2,DEC2,s_V2],...]
        self.pos_s = np.delete(pos_s, 883, axis=0)       # Deletes a duplicate star in the file*


    def BSC_filter(self, obs_t = None, limit = None):
        # position of tele (pos_t) in [[RA1,DEC1,R1],[RA2,DEC2,R2]]
        # obs_t: at what month of the year for observation, set to 21th day. If obs_t = 3, then 3/21
        # All conditions can be adjusted accordingly.
        
        #pos_t = np.array(pos_t)

        
        #Select out stars that are never in the plane of tele, diff in DEC less than some deg 90deg
        #cond1 = np.where((np.absolute(self.pos_s[:,2] - pos_t[0,1]) < np.pi/6.) &
        #                 (np.absolute(self.pos_s[:,2] - pos_t[1,1]) < np.pi/6.))
        
        cond1 = np.where(self.pos_s[:,2] > 0 )   # consider stars only in northern hemisphere
        
        pos_s = self.pos_s[cond1]

        #flux density condition: eliminate stars whose flux density is less than 50Jy
        cond2 = np.where(pos_s[:,3] > 30.0)
        pos_s = pos_s[cond2]


        # Select out stars that can be seen during the night depending on time of year.
        # Want stars who lags behind the sun at [pi, 5pi/4] during observation periods.
        # So between 12 hours and 16 hours behind the sun.
        # RA defined at 3/21. Just as approxiamtion:
        if obs_t is not None:
            delay = ((obs_t - 3.0)*30*np.pi/180)
            cond3 = np.where((np.mod(pos_s[:,1]-delay,2*np.pi)> np.pi) & (np.mod(pos_s[:,1]-delay,2*np.pi) < 5.7*np.pi/4))
            pos_s = pos_s[cond3]
        
        
        #Create star pairs NxN matrix of all pairs and select out onces in the lower triangle:
        N = len(pos_s)
        row_del = []

        # might want to optimize for big N. *Nested for-loop too time consuming.
        for i in range(N):
            for j in range(i+1):
                k = i*N+j
                row_del.append(k)

        pos_s_rep = np.tile(pos_s,(1,N)).reshape((N**2,4)) 
        pos_s_rep = np.delete(pos_s_rep, row_del, axis=0)  #[11111...22222....3333..]
        
        pos_s_seq = np.tile(pos_s,(N,1))
        pos_s_seq = np.delete(pos_s_seq, row_del, axis=0)  #[12345..23456..34567..]

        #[ [[num1,RA1,DEC1,S1],[num2,RA2,DEC2,S2]],.....]
        ndim = int(N*(N-1)/2)
        pos_s_mat = np.hstack((pos_s_seq,pos_s_rep)).reshape((ndim,2,4))

        # select out pairs of stars that are far away from each other
        # convert from spherical to carte? and find difference.

        def dis_diff(posi):
            # posi in (N,2,4) array and convert posi from RA,DEC to carte
            new_posi = np.zeros((len(posi),2,4))
            new_posi[:,:,2] = np.pi/2 - posi[:,:,2]        #DEC to THETA
            
            #change from spherical to cartesian
            x = np.sin(new_posi[:,:,2])*np.cos(posi[:,:,1])     #x,y,z for all pairs (N,2)
            y = np.sin(new_posi[:,:,2])*np.sin(posi[:,:,1])
            z = np.cos(new_posi[:,:,2])

            dx = np.absolute(x[:,1] - x[:,0])
            dy = np.absolute(y[:,1] - y[:,0])
            dz = np.absolute(z[:,1] - z[:,0])

            dis_diff = np.sqrt(dx*dx + dy*dy + dz*dz)      # 1-D N pair stars array
            
            return dis_diff
        
        dis_diff = dis_diff(pos_s_mat)
        
        if limit is None: 
            cond4 = np.where(dis_diff < 0.01)           #  Check for star pairs less than 0.01rad separation
        else:
            cond4 = np.where(np.logical_and(dis_diff > limit[0],dis_diff < limit[1]))
        
        
        pos_s_mat = pos_s_mat[cond4]
        
        if obs_t is not None:
            cond5 = np.where(pos_s_mat[:,0,3]+pos_s_mat[:,1,3] == np.amax(np.sum(pos_s_mat[:,:,3],axis=1)))
            pos_s_mat = pos_s_mat[cond5]
            
        return pos_s_mat
