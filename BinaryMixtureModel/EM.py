import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import util

file = scipy.io.loadmat('/Users/Angela/Desktop/master/inference/data_lab_1/oec_data.mat')

X_oec= file['X_oec']
countries= file['countries']
products= file['products']


#plt.plot(X_oec,'*')
#plt.show()

N = np.shape(X_oec)[0]
D = np.shape(X_oec)[1]

#K=2 #probar con diferentes Ks a ver como se desacoplan los clustering de paises

x = X_oec

Ks_list = [2,3,4,5,6]

log_likes_ks = []
log_likes_ks_MAP = []

# Instanciate the variables

for K in Ks_list:

    rik_ini = np.zeros((N,K))

    # Initializate the variables

    # http: // citeseerx.ist.psu.edu / viewdoc / download?doi = 10.1.1.104.4104 & rep = rep1 & type = pdf

    alpha = 50/K #Elegimos un alpha= 50/K porque asi lo sugieren Griffiths and Steyvers (2004)
    pi_ini = np.random.dirichlet(alpha*np.ones((1,K))[0])
    mu_ini = np.random.rand(x.shape[1],K)

    # Parameters for the priors of the parameters
    a=1
    b=1
    alpha=np.ones((1,K))

    # Realizamos EM
    iterations = 100  # Numero de iteraciones del EM

    rik, best_log_like,log_likes_iters = util.EM_algorithm(K, N, pi_ini, mu_ini, rik_ini, x, iterations)
    log_likes_ks.append(best_log_like)

    rik_MAP, best_log_like_MAP, log_likes_iters_MAP = util.EM_algorithm_MAP(K, N, pi_ini, mu_ini, rik_ini, x, iterations, a, b, alpha)
    log_likes_ks_MAP.append(best_log_like_MAP)

   # print(log_likes_iters)


    plt.plot(log_likes_iters)
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.plot(log_likes_iters_MAP)
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.savefig('Plot_with_MAP_'+str(K))
    plt.show()


print(log_likes_ks)
# No aplicamos model selection
k_index = np.argmax(log_likes_ks)
k_optim = Ks_list[k_index]
print('k optima sin seleccion de modelo: '+ str(k_optim))

# BIC criteria for model selection
k_bic = util.model_selection(log_likes_ks,Ks_list, D, N, 'bic')
print('k optima con BIC: '+ str(k_bic))

# AIC criteria for model selection
k_aic = util.model_selection(log_likes_ks,Ks_list,D, N, 'aic')
print('k optima con AIC: '+ str(k_aic))





