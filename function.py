import numpy as np

def shrink(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def lagrangian(D):
    theshold = 1e-3

    m, n = D.shape
    lambda_ = 1 / np.sqrt(max(m, n)) #C
    #synthetic image (lambda_ = 0.3 / np.sqrt(max(m, n)))

    Y = np.zeros((m, n)) #Y1
    A = np.zeros((m, n)) #A1
    E = np.zeros((m, n)) #E1
    p = 1.1
    d_norm = np.linalg.norm(D, 'fro')
    mu = 1 / d_norm
    mu_bar = mu * 1e7 
    iter = 0
    flag_b = 0
    conv_k = False
    while not conv_k:
        iter = iter + 1
        E = shrink(D+(Y/mu)-A, lambda_/mu) #j+1
        U,S,V = np.linalg.svd((Y/mu) + D - E, full_matrices=False) #i
        A = U @ np.diag(shrink(S,1/mu)) @ V
        T = (D - A - E)
        Y = Y + mu * T #k+1
        mu = mu * p
        mu = min(mu * p, mu_bar)
        flag = np.abs(np.linalg.norm(T, 'fro') - flag_b)
        flag_b = np.linalg.norm(T, 'fro')
        if flag < theshold:
            conv_k = True
    return A, E, iter