import numpy as np

def Nelder_Mead(S, f, N, tau = 0.0000001, mu_r = -1, mu_e = -2, mu_oc = -0.5, mu_ic = 0.5, sgm = 0.5):
    n = len(S[0]) #dimension
    k = 0 #numero de iteraciones

    # ordenamiento
    f_S = np.array([f(x) for x in S])
    s_idx = np.argsort(f_S)  #sorted indices    
    x = np.array([S[idx] for idx in s_idx])

    while k < N and f(x[n]) - f(x[0]) >= tau :
        k += 1
        x_bar = np.mean(x[:n], axis = 0)
        x_mu_r = x_bar + mu_r*(x[n] - x_bar)
        f_r = f(x_mu_r)
        f_n = f(x[n-1])
        f_1 = f(x[0])

        if f_1 <= f_r and f_r < f_n: #Reflexion
            x[n] = x_mu_r

        elif f_r < f_1:
            x_mu_e = x_bar + mu_e*(x[n] - x_bar)
            f_e = f(x_mu_e)
            if f_e < f_r: #expandir
                x[n] = x_mu_e
            else:
                x[n] = x_mu_r #reflejar

        else:
            f_n1 = f(x[n])
            x_mu_ic = x_bar + mu_ic*(x[n] - x_bar)
            f_ic = f(x_mu_ic)

            x_mu_oc = x_bar + mu_oc*(x[n] - x_bar)
            f_oc = f(x_mu_oc)

            flag = True #bandera para saber si es necesario un encojimiento
            if f_r < f_n1:
                if f_oc <= f_r: #contraccion exterior
                    flag = False
                    x[n] = x_mu_oc
            
            elif f_ic < f_n1: #contracción interior
                flag = False
                x[n] = x_mu_ic

            
            if flag: # encojimiento
                for j in range(1, n+1):
                    x[j] = x[0] + sgm*(x[j]-x[0])

        ## Ordenamiento de vertices para la siguiente iteracion
        f_S = np.array([f(vec) for vec in x])
        s_idx = np.argsort(f_S)
        x_aux = np.copy(x)    
        x = np.array([x_aux[idx] for idx in s_idx])

    return x, k