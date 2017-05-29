import math
import numpy as np
from numpy import linalg  as la
from sklearn import preprocessing
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def gene_rad_ortho_mat_3(a,b,c):
    gene = np.random.rand(a,b,c)
    for i in range(a):
        gene[i] = gs(gene[i])
    return gene

def gs(matrix):
    #print matrix.shape
    tran_matrix = matrix.T
    res = np.zeros(shape = (matrix.shape[0],matrix.shape[1]))
    tran_res = res.T
    tran_res[0] = tran_matrix[0]
    for i in range(1, tran_matrix.shape[0]):
        tmp = tran_matrix[i]
        for j in range(0,i):
            tmp -= ((np.dot(tran_matrix[i],tran_res[j]))/(np.dot(tran_res[j],tran_res[j])))*tran_res[j]
        tran_res[i] = tmp
    norms = la.norm(tran_res, axis = 1)
    #print tran_res
    for i in range(tran_res.shape[0]):
        tran_res[i] = tran_res[i]/norms[i]
    return tran_res.T
class HierarchicalNetwork:

    #def gene_rad_ortho_mat_1(n):
    def sigmoid_inverse(self, x):
        print x
        dim = len(x)
        if (dim != self.output_num):
            print 'dimension of y wrong'
            return 0
        res = [0.0]*dim
        for i in range(dim):
            res[i] = -(math.log(1/x[i] - 1))
        return res
    def __init__ (self):
        self.input_num = 0
        #number of input features, i.e. n
        self.output_num = 0
        #number of output nodes i.e. m
        self.hidden_net_neuron_num = 0
        #number of hidden neworks, i.e. L
        self.hidden_subnet_num = 0
        #for each of subnetwork
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.input_bias = []
        self.output_bias = []
    def setup(self, num_input, num_output, num_subnet, num_subneuron):
        self.input_num = num_input
        self.output_num = num_output
        self.hidden_net_neuron_num = num_subneuron
        self.hidden_subnet_num = num_subnet

        self.input_cells = [1.0]*self.input_num
        self.output_cells = [1.0]*self.output_num

        #self.input_weights = np.random.rand(self.hidden_subnet_num, self.hidden_net_neuron_num, self.input_num)
        # L d n
        #self.output_weights = np.random.rand(self.hidden_subnet_num, self.output_num, self.hidden_net_neuron_num)
        # L m d
        #self.input_bias = np.random.rand(self.hidden_subnet_num, 1)
        #self.hidden_bias = np.random.rand(self.hidden_subnet_num, 1)

        #generate orthogonal matrix
        self.input_weights = gene_rad_ortho_mat_3(self.hidden_subnet_num, self.hidden_net_neuron_num, self.input_num)
        print 'over'
        #self.output_weights = gene_rad_ortho_mat_3(self.hidden_subnet_num, self.output_num, self.hidden_net_neuron_num)
        self.output_weights = np.random.rand(self.hidden_subnet_num, self.output_num, self.hidden_net_neuron_num)
        m = np.mat(np.zeros(shape = (self.output_num, self.hidden_net_neuron_num)))
        for i in range(self.hidden_subnet_num):
            self.output_weights[i] = m
        self.input_bias = np.ones(shape = (self.hidden_subnet_num,self.hidden_net_neuron_num,1))
        self.output_bias = np.ones(shape = (self.hidden_subnet_num,self.hidden_net_neuron_num,1))
    def feature_extraction(self,x, y):
        #batch not implemented
        #y r^d
        #y already normalized
        #print 'normalized y', u_ny.shape
        for i in range(self.hidden_subnet_num):
            print x.shape
            print self.input_weights[i].shape
            h_c = np.zeros(shape = (self.hidden_net_neuron_num,1))
            tmp = np.mat(self.input_weights[i])*np.mat(x) + self.input_bias[i]
            for j in range(self.hidden_net_neuron_num):
                h_c[j] = sigmoid(tmp[j])
            print 'h c', h_c.shape
            tmp = np.mat((h_c*h_c.T+np.eye(self.hidden_net_neuron_num)))
            print tmp.shape
            h_inverse_c = h_c.T*tmp.I
            print 'h inverse c', h_inverse_c.shape
            #print np.mat(self.sigmoid_inverse(y))
            #print np.mat(h_inverse_c)
            tmp = np.mat(self.sigmoid_inverse(y)).T*np.mat(h_inverse_c)
            #print tmp.shape
            #print type(tmp)
            #print type(self.output_weights[i])
            #print self.output_weights.shape
            self.output_weights[i] = tmp
            #calculate output bias

    def train(self, x, y, batch_size = 1):
        #first, subspace feature extraction
        #then, pattern learning
        length = len(y)
        print length,'samples'
        min_max_scaler = preprocessing.MinMaxScaler()
        y_scaled = min_max_scaler.fit_transform(y)
        print y
        #min_max_scaler = preprocessing.MinMaxScaler()
        batches = length/batch_size
        for i in range(length):
            #H = self.feature_extraction(x[batch_size*i: min(length,batch_size*(i+1))],
            #                        y[batch_size*i: min(length, batch_size*(i+1))]
            #                        ) # H = {H_1, ..., H_L}
            H = self.feature_extraction(x[i],y_scaled[i])
            break

def main():
    network = HierarchicalNetwork()
    network.setup(num_input = 3,num_output = 2,num_subnet =10, num_subneuron = 15)
    x = [[1, 3, 5],[3,6,1],[2,2,2]]
    y = [[2,4],[1,5],[3,3]]
    x = np.reshape(x,(len(x), 3,1))
    y = np.reshape(y,(len(y), 2))
    network.train(x,y)

main()
