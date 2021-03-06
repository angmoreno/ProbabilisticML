import numpy as np

def initialize(K,D):

    # Probabilities of the components of the mixture
    #pi = np.random.uniform(size=(1, K))[0]
    #pi = pi / pi.sum()
    pi = 1/K * np.ones(K)


    # Transition probabilities (from one state to another)
    A = np.random.rand(K, K)
    A = A / A.sum(axis=1,keepdims=True)

    # Emission probabilities or Observation likelikoods (from latent space to observations)
    B = np.random.rand(K, D)

    B = B / B.sum(axis=1,keepdims=True)

    return [pi,A,B]


def calculate_pb(sequence,B,K):

    T = np.shape(sequence)[1]
    D = np.shape(sequence)[0]

    pb = np.zeros((K,T))
    for t in range(T):
        prob = 0
        for j in range(D):
            prob+= sequence[j,t]*np.log(B[:,j] + 1e-7) + (1-sequence[j,t])*np.log(1-B[:,j] + 1e-7)
        pb[:,t] = np.exp(prob)


    return pb

def calculate_cost(X,pi,A,B,gamma,chi):

    N = len(gamma)
    K = np.shape(gamma[0])[0]
    T = np.shape(gamma[0])[1]
    D = np.shape(B)[1]

    prior_term = 0
    state_transition_term = 0
    log_like = 0


    for n in range(N):
        prior_term +=  np.sum(gamma[n][:, 0]*np.log(pi))

    for n in range(N):
        for k in range(K):
            for t in range(1,T):
                for k_ in range(K):
                    state_transition_term += chi[n][t,k,k_] * np.log(A[k, k_])

    for n in range(N):
        seq = X[n][0]
        for k in range(K):
            for t in range(T):
                for j in range(D):

                    log_like += gamma[n][k, t] * (seq[j,t]*np.log(B[k,j]) + (1-seq[j,t])*np.log(1-B[k,j]))

    Q = prior_term + state_transition_term + log_like




    return Q



def Viterbi_decoder(pi_est, A_est, pb):

    K = np.shape(pb)[1]
    N = len(pb)
    T = np.shape(pb)[2]

    states = np.zeros((N,T))
    delta = np.zeros((K,T,N))
    fi = np.zeros((K,T,N))


    # Forward
    for n in range(N):
        delta[:,0,n] = pi_est[0] * pb[n,:,0]

    for t in range(1,T):
        for n in range(N):
            #print(A_est*(delta[:,t-1,n].T))
            max = np.amax(A_est*delta[:,t-1,n],axis=1)
            #print(max)
            index_max = np.argmax(A_est*delta[:,t-1,n],axis=1)
            #print(index_max)
            delta[:,t,n] = pb[n,:,t]*max
            fi[:,t,n] = index_max #mirar si transponer o no

    # Backward

    states[:,T-1] = np.argmax(delta[:,T-1,:],axis=0)

    for n in range(N):
        for t in range(T-2,-1,-1):
            states[n,t] = fi[int(states[n,t+1]),t+1,n]

    return states



def MAP_decoder(gamma_est):

    N = len(gamma_est)
    T = np.shape(gamma_est)[2]

    states = np.zeros((N,T))
    for i in range(N):
        states[i,:] = np.argmax(gamma_est[i],axis = 0)  # max wrt k

    return states
