import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import aux_functions
import EM

file = scipy.io.loadmat('./observed.mat')

X = file['observed']


N = np.shape(X)[0]  # number of sequences 10
D = np.shape(X[0][0])[0]  # dimension of sequences 6
T = np.shape(X[0][0])[1]  # length of each sequence 100

max_it = 100
Ks_list = [2,3,4,5]  #[2,3,4,5]
Num_inis = 3

max_it= 100
converged = False

num_k = len(Ks_list)

ll_ks_opt = []



A_ks_opt = []
B_ks_opt = []
pi_ks_opt = []
gamma_ks_opt = []
pb_ks_opt = []

for K in Ks_list:
    print('--------------------------')
    print('      K = '+str(K))
    print('--------------------------')



    ll_inis = np.zeros((Num_inis,1))
    pi_est_inis = np.zeros((Num_inis,1,K))
    #pi_est_inis = np.zeros((Num_inis))
    A_est_inis = np.zeros((Num_inis,K,K))
    B_est_inis = np.zeros((Num_inis,K,D))
    gamma_inis = np.zeros((Num_inis,N,K,T))
    pb_inis = np.zeros((Num_inis, N, K, T))

    Q_total = []

    for ini in range(Num_inis):

        pi_est = np.zeros((1, K))
        A_est = np.zeros((K, K))
        B_est = np.zeros((K, D))
        gamma = np.zeros((N,K,T))


        pb = np.zeros((N, K, T))
        print('--------- Initialization: '+str(ini)+' ---------\n\n')
        # Parameter initialization
        [pi_ini,A_ini,B_ini] = aux_functions.initialize(K, D)
        [pi_est, A_est, B_est, gamma, Q_tot, Q,  pb] = EM.EM_algorithm(X,pi_ini,A_ini,B_ini, N, max_it ,ini)

        pi_est_inis[ini,:,:] = pi_est
        A_est_inis[ini,:,:] = A_est
        B_est_inis[ini,:,:] = B_est
        gamma_inis[ini,:,:,:] = gamma
        ll_inis[ini] = Q
        pb_inis[ini,:,:,:] = pb

        Q_total.append(Q_tot)



    # Best model for all iterations and one particular inicialization
    ini_opt = np.argmax(ll_inis)
    ll_ks_opt.append(ll_inis[ini_opt][0])
    A_ks_opt.append(A_est_inis[ini_opt])
    B_ks_opt.append(B_est_inis[ini_opt])
    pi_ks_opt.append(pi_est_inis[ini_opt])
    gamma_ks_opt.append(gamma_inis[ini_opt])
    pb_ks_opt.append(pb_inis[ini_opt])


# Best model (best K)

k_opt = np.argmax(ll_ks_opt) + Ks_list[0]
print('----- OPTIMAL K: '+str(k_opt)+'------')
ll_k_opt = ll_ks_opt[k_opt-Ks_list[0]]
A_k_opt = A_ks_opt[k_opt-Ks_list[0]]
B_k_opt = B_ks_opt[k_opt-Ks_list[0]]
pi_k_opt = pi_ks_opt[k_opt-Ks_list[0]]
gamma_opt = gamma_ks_opt[k_opt-Ks_list[0]]
pb_opt =  pb_ks_opt[k_opt-Ks_list[0]]


# FOR DECODING: UNCOMMENT AND TRY WITH JUST ONE K

'''
# MAP decoder (FB algorithm)
states_MAP = aux_functions.MAP_decoder(gamma_opt)

# ML decoder (viterbi)

states_ML = aux_functions.Viterbi_decoder(pi_k_opt, A_k_opt, pb_opt)

print(' States MAP decoder \n')
print(states_MAP)
print(' States Viterbi decoder \n')
print(states_ML)


for n in range(N):

    x = np.linspace(0,T,100)
    plt.plot(x,states_MAP[n],'-',label = 'MAP decoder')
    plt.plot(x, states_ML[n],'--',label = 'Viterbi decoder')
    plt.xlabel('Observations')
    plt.ylabel('Sequence '+str(n))
    plt.title(' MAP and Viterbi sequence decoder for K='+str(K))
    plt.legend(loc='upper right')
    plt.savefig('Decoded_sequence_K' + str(K)+'_N='+str(n))
    plt.show()

'''