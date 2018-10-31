import numpy as np



def generate_k_table_pioneer_6(number_of_actions, k_step,k_min,k_max,k_size):

    k1_step =k_step[0] 
    k1_min  =k_min[0] 
    k1_max  =k_max[0] 
    k1_size =k_size 

    k2_step =k_step[1] 
    k2_min = k_min[1] 
    k2_max = k_max[1] 
    k2_size =k_size 

    k3_step =k_step[2] 
    k3_min  =k_min[2] 
    k3_max  =k_max[2] 
    k3_size =k_size 

    k4_step =k_step[3] 
    k4_min = k_min[3] 
    k4_max = k_max[3] 
    k4_size =k_size 

    k5_step =k_step[4] 
    k5_min  =k_min[4] 
    k5_max  =k_max[4] 
    k5_size =k_size 

    k6_step =k_step[5] 
    k6_min = k_min[5] 
    k6_max = k_max[5] 
    k6_size =k_size 

    k_table = np.zeros((number_of_actions,6))
    count = 0

    for p in [k6_min,  k6_min + k6_step, k6_max]: 
        for n in [k5_min,  k5_min + k5_step, k5_max]:
            for m in [k4_min,  k4_min + k4_step, k4_max]:
                 for k in [k3_min,  k3_min + k3_step, k3_max]:
                    for j in [k2_min,  k2_min + k2_step, k2_max]:
                        for i in [k1_min,  k1_min + k1_step, k1_max]:

                            k1=i
                            k2=j
                            k3=k
                            k4=m
                            k5=n
                            k6=p
                            
                            if k1<0.:
                                k1 = 0.
                            
                            if k2<=0.:
                                k2 = 0. + k2_step
                            
                            if k3<0.:
                                k3 = 0.

                            if k4<0.:
                                k4 = 0.
                            
                            if k5<0.:
                                k5 = 0. + k5_step
                            
                            if k6<0.:
                                k6 = 0.
                            

                            k_table[count]=np.array([k1, k2, k3, k4, k5, k6])
                            count = count + 1


    return k_table




def generate_k_table_pioneer_4(number_of_actions, k_step,k_min,k_max,k_size):

    k1_step =k_step[0] 
    k1_min  =k_min[0] 
    k1_max  =k_max[0] 

    k2_step =k_step[1] 
    k2_min = k_min[1] 
    k2_max = k_max[1] 

    k3_step =k_step[2] 
    k3_min  =k_min[2] 
    k3_max  =k_max[2] 

    k4_step =k_step[3] 
    k4_min = k_min[3] 
    k4_max = k_max[3] 

    k_table = np.zeros((number_of_actions,4))
    count = 0

    
    for m in [k4_min,  k4_min + k4_step, k4_max]:
         for k in [k3_min,  k3_min + k3_step, k3_max]:
            for j in [k2_min,  k2_min + k2_step, k2_max]:
                for i in [k1_min,  k1_min + k1_step, k1_max]:

                    k1=i
                    k2=j
                    k3=k
                    k4=m
                    
                    if k1<0.:
                        k1 = 0.
                    
                    if k2<=0.:
                        k2 = 0. + k2_step
                    
                    if k3<0.:
                        k3 = 0.

                    if k4<0.:
                        k4 = 0.
                    
                    k_table[count]=np.array([k1, k2, k3, k4])
                    count = count + 1


    return k_table


def generate_k_table_pioneer_3(number_of_actions, k_step,k_min,k_max,k_size):

    k1_step = k_step[0] 
    k1_min  = k_min[0] 
    k1_max  = k_max[0] 

    k2_step = k_step[1] 
    k2_min  = k_min[1] 
    k2_max  = k_max[1] 

    k3_step = k_step[2] 
    k3_min  = k_min[2] 
    k3_max  = k_max[2] 


    k_table = np.zeros((number_of_actions,3))
    count = 0

    
    
    for k in [k3_min,  k3_min + k3_step, k3_max]:
        for j in [k2_min,  k2_min + k2_step, k2_max]:
            for i in [k1_min,  k1_min + k1_step, k1_max]:

                k1=i
                k2=j
                k3=k
                
                if k1<0.:
                    k1 = 0.
                
                if k2<=0.:
                    k2 = 0. + k2_step
                
                if k3<0.:
                    k3 = 0.

                
                k_table[count]=np.array([k1, k2, k3])
                count = count + 1


    return k_table


def generate_k_table_pioneer_2(number_of_actions, k_step,k_min,k_max,k_size):

    k1_step = k_step[0] 
    k1_min  = k_min[0] 
    k1_max  = k_max[0] 

    k2_step = k_step[1] 
    k2_min  = k_min[1] 
    k2_max  = k_max[1] 

    
    k_table = np.zeros((number_of_actions,2))
    count = 0

    
    
    
    for j in [k2_min,  k2_min + k2_step, k2_max]:
        for i in [k1_min,  k1_min + k1_step, k1_max]:

            k1=i
            k2=j
            
            if k1<0.:
                k1 = 0.
            
            if k2<=0.:
                k2 = 0. + k2_step
            
            
            k_table[count]=np.array([k1, k2])
            count = count + 1


    return k_table



