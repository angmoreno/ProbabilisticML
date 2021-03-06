
import numpy as np

def expectation(sequence,pi,A,pb):

    [alpha_n, beta_n] = forward_backward(sequence,pi,A,pb)
    gamma_n = compute_gamma(alpha_n,beta_n)
    chi_n = compute_chi(sequence,alpha_n,beta_n,A,pb)

    return [alpha_n, beta_n ,gamma_n, chi_n]


def forward_backward(sequence, pi, A, pb):

    K = np.shape(A)[0] #num hidden states
    T = np.shape(sequence)[1] #num observations

    # Forward (calculate alpha)

    alpha_n = np.zeros((K, T))
    alpha_aux = np.zeros((K, T))


    alpha_n[:,0] =  pi * pb[:,0]  #alpha_0
    alpha_aux[:, 0] = alpha_n[:,0]

    for t in range(1, T):
        for k in range(K):
            for k_ in range(K):
                alpha_aux[k,t] += alpha_aux[k_,t-1] * A[k,k_]
            alpha_n[k,t] = alpha_aux[k,t]* pb[k,t]

    #-----------------------

    # Backward (Calculate beta)
    beta_n = np.zeros((K,T))
    beta_n[:, T-1] = np.ones(K)  #beta_T


    for t in range(T - 2, -1, -1):
        for k in range(K):
            for k_ in range(K):
                beta_n[k,t] += A[k,k_] * pb[k_,t+1] * beta_n[k_,t+1]


    return [alpha_n, beta_n]


def compute_gamma(alpha_n, beta_n):

    K = np.shape(alpha_n)[0]
    T = np.shape(alpha_n)[1]

    gamma = np.zeros((K, T))
    for j in range(K):
        gamma[j,:] = beta_n[j,:]*alpha_n[j,:]


    gamma =  gamma/gamma.sum(axis=0)

    return gamma

def compute_chi(sequence,alpha,beta,A,pb):

    K = np.shape(A)[0]
    T = np.shape(sequence)[1]

    chi = np.zeros((T,K,K))
    chi_t = np.zeros((K,K))

    for t in range(T-1):
        chi_it = alpha[:,t]*A * pb[:,t+1]*beta[:,t+1]
        chi_t = chi_t + chi_it/chi_it.sum()
        chi[t,:,:] = chi_t

    return chi


