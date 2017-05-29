import numpy as np

def main():

    data = np.random.rand(5,3)
    for i in range(3):
        mean = 0
        for j in range(5):
            mean+=data[j][i]
        for j in range(5):
            data[j][i] = data[j][i] - mean
    u,s,v = np.linalg.svd(data, full_matrices = 1)
    print u.shape
    print s.shape
    print v.shape
    #for i in range(u.shape[0]):
    #    for j in range(u.shape[1]):
    #        u[i][j] = u[i][j]/s[j]
    print u
    m_u = np.mat(u)
    print m_u*m_u.T
main()
