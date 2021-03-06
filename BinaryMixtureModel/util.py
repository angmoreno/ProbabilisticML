
import numpy as np
import math

def E_step(K , N, pi, mu, rik, x):

    # Defino el rik

    for k in range(K):
        for i in range(N):

            rik[i, k] = pi[k] * np.prod((mu[:,k] ** x[i, :]) * ((1 - mu[:,k]) ** (1 - x[i, :])))
            #rik[i, k] = np.prod((mu[k, :] ** x[i, :]) * ((1 - mu[k, :]) ** (1 - x[i, :])))

    # Normalizo el rik
    for i in range(N):
        rik[i,:] = rik[i,:] / rik[i,:].sum()

    log_likelihood = 0
    for i in range(N):
        for k in range(K):
            '''
            pos=0
            # Para evitar que la log salga NaN
            for elem in mu[:,k]:
                if int(elem)==1:
                    mu[pos,k] = 1-1e-10
                    pos+=1
                elif int(elem)==0:
                    mu[pos, k] = 1e-10
                    pos+=1
            '''
            log_likelihood += rik[i, k]*np.log(pi[k]) + \
                              rik[i, k]*np.sum(x[i, :]*np.log(mu[:,k])) + \
                              rik[i, k]*np.sum((1 - x[i, :]) * np.log(1 - mu[:,k])) # esta línea YA NO da guerra

    return (rik,log_likelihood)



def M_step(K, N, rik, pi ,mu, x):

    for k in range(K):
        for j in range(x.shape[1]):
            pi = rik.sum(axis=0) / N
            #mu[:,k] = np.sum(rik[:,k]* x[:,j]) / np.sum(rik[:,k])  aca el problema
            mu[:, k] = np.sum(rik[:, k] * x[:, j]) / np.sum(rik)
    return (pi, mu)



def M_step_MAP(K, N, rik, pi ,mu, x,a,b,alpha):


    for k in range(K):
        for j in range(x.shape[1]):
            alpha_0 = np.sum(alpha)

            pi = rik.sum(axis=0) + alpha[:,k] - 1 / (N + K - alpha_0)
            mu[:,k] = (a - 1 - np.sum(rik[:,k]* x[:,j])) / (a + b - 2 - np.sum(rik))

    return (pi, mu)


def EM_algorithm(K, N, pi, mu, rik_ini, x, iterations):

    log_likes = []
    log_likes.append(-np.infty)
    epsilon=1e-4

    for iter in range(1,iterations):

        # E - step
        rik, likelihood = E_step(K, N, pi, mu,rik_ini, x)
        log_likes.append(likelihood)
        print('Iteration %d: log-likelihood is %f'%(iter, likelihood))

        if(abs(likelihood-log_likes[iter-1])<epsilon):
            print('Converged at iteration %d: loglikelihood is %f' % (iter, likelihood))
            return (rik, likelihood,log_likes)
        else:
             # M - step
            pi, mu = M_step(K, N, rik, pi, mu, x)


    return (rik, log_likes)

def EM_algorithm_MAP(K, N, pi, mu, rik_ini, x, iterations,a,b,alpha):
    log_likes = []
    log_likes.append(-np.infty)
    epsilon = 1e-4

    for iter in range(1, iterations):

        # E - step
        rik, likelihood = E_step(K, N, pi, mu, rik_ini, x)
        log_likes.append(likelihood)
        print('Iteration %d: log-likelihood is %f' % (iter, likelihood))

        if (abs(likelihood - log_likes[iter - 1]) < epsilon):
            print('Converged at iteration %d: loglikelihood is %f' % (iter, likelihood))
            return (rik, likelihood, log_likes)
        else:
            # M - step
            pi, mu = M_step_MAP(K, N, rik, pi, mu, x,a,b,alpha)



def model_selection(list_log_likelihood,k_list, D, N, criterion):

    model_list = []
    if criterion == 'bic':
        #print(list_log_likelihood)
        #print(k_list)
        no_cost = np.empty((len(k_list), 1))
        bic_cost = np.empty((len(k_list), 1))
        i = 0
        for k in k_list:
            dof = k * (D+1)
            bic_cost[i] = list_log_likelihood[k-k_list[0]] - 0.5*dof*np.log(N)
            no_cost[i] = list_log_likelihood[k-k_list[0]]

            i += 1


        best_k = np.argmax(bic_cost)+k_list[0]

        #print('bic:')
        #print(bic_cost)
        #print('no_cost:')
        #print(no_cost)

        return best_k

    elif criterion == 'aic':
        no_cost = np.empty((len(k_list), 1))
        aic_cost = np.empty((len(k_list), 1))
        i = 0
        for k in k_list:
            dof = k * (D+1)
            aic_cost[i] =  list_log_likelihood[k-k_list[0]] - dof
            no_cost[i] =  list_log_likelihood[k-k_list[0]]
            i += 1

        best_k = np.argmax(aic_cost)+k_list[0]
        #print(aic_cost)
        #print(no_cost)


        return  best_k

