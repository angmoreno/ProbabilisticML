import numpy as np

def maximization(X,gamma, chi):

    # Update model parameters: pi, A, B
    K = np.shape(gamma[0])[0]
    N = len(gamma)
    T = np.shape(gamma[0])[1]
    D = np.shape(X[0][0])[0]

    # -------------- pi update ---------------
    pi = np.zeros((1, K))
    for n in range(N):
        pi = pi + gamma[n][:,0]
    pi = pi/N

    # -------------- A update ---------------


    num = 0
    for n in range(N):
        for t in range(1,T):
            num += chi[n,t,:,:]

    den = 0

    for n in range(N):
        for t in range(1,T):
            for k in range(K):
                den+= chi[n,t,:,k]

    A = num/den
    A = A / A.sum(axis=0)


    # -------------- B update ---------------

    B = np.zeros((K, D))

    for k in range(K):
        den = 0
        for n in range(N):
            prod = np.sum(X[n][0]*np.tile(gamma[n][k,:,np.newaxis],D).T, axis=1)
            B[k,:] = B[k,:] + prod
            den += np.sum(gamma[n][k,:])
        B[k,:]=B[k, :]/den

        #print(gamma[n].shape)
        #print(np.transpose(X[0][n]).shape)
        #prod = np.dot(gamma[n],np.transpose(X[0][n]))
        #B = B + prod

   # B = B / B.sum(axis = 0)

    pi_update = pi
    A_update = A
    B_update = B


    return [pi_update, A_update, B_update]