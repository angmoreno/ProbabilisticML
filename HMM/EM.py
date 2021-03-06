
import numpy as np
import expectation
import maximization
import aux_functions
import matplotlib.pyplot as plt

def EM_algorithm(X,pi,A,B,N,max_it,ini):


    iter = 0
    converged = False
    Q = []
    K = np.shape(pi)[0]
    T = np.shape(X[0][0])[1]

    epsilon = 1e-2

    alpha = np.zeros((N,K,T)) # one alpha for each sequence
    beta = np.zeros((N,K,T)) # one beta for each sequence
    gamma = np.zeros((N,K,T)) # one gamma for each sequence
    chi = np.zeros((N,T,K,K)) # one chi for each sequence

    pb = np.zeros((N,K,T))

    while iter!=max_it :

        if iter == max_it:
            print('---------Max iteration reached--------')
            break

        #print('--------- ITERATION ---> '+str(iter))
        # bucle por secuencia
        for seq in range(N):
            pb[seq,:,:] = aux_functions.calculate_pb(X[seq][0],B,K)
            [alpha[seq,:,:], beta[seq,:,:] ,gamma[seq,:,:], chi[seq,:,:,:] ] = expectation.expectation(X[seq][0],pi,A,pb[seq])


        [pi_est,A_est,B_est] = maximization.maximization(X , gamma, chi)

        pi = pi_est
        A = A_est
        B = B_est

        Q.append(aux_functions.calculate_cost(X,pi_est,A_est,B_est,gamma,chi))

        print('Cost at iteration ' +str(iter)+': '+ str(Q[iter]) )

        if iter==0:
            iter += 1
        else:
            if (np.abs(Q[iter] - Q[iter-1]) < epsilon) or (iter==max_it-1 ):
                print('Convergence reached at iteration ' + str(iter))

                plt.plot(Q)
                plt.xlabel('Iterations')
                plt.ylabel('Log-likelihood')
                plt.savefig('Plot_K_' + str(K)+'_ini_'+str(ini))
                plt.show()



                return [pi_est, A_est, B_est, gamma, Q ,Q[iter], pb]
            else:
                iter += 1


















