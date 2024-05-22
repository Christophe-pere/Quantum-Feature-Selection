import pennylane as qml 
from pennylane import numpy as np
from sklearn import datasets 
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class FeatureSelection:
    '''
    Class to perform quantum feature selection using variational approach as a black box like Zoufal et al. 2023, Variational quantum algorithm for unconstrained black box binary optimization Application to feature selection. 
    The class uses the QN-SPSA by default but can also be used with QNG Optimizer.
    
    Args: 
        - ansatz: 
        - ansatz_state: 
        - num_qubits:
        - diff_method:
        - opt: optimizer 
        - max_features: the maximum number of features needed to extract 
        - provider: 

    Return:
        - Feature selection object 
        
    Tuto: 

    >> from varQFS import FeatureSelection
    >> varqfs = FeatureSelection(ansatz_1,ansatz_2,  num_qubits=np.shape(X)[1])
    >> selected_variables, energy = varqfs.fit(X, depth, batch_size, num_batches)
    >> _bitstring_ = varqfs.bitstring_final(X[0], selected_variables)
    
    # ansatz_1 is the ansatz to encode the data + parametrize circuit  
    # ansatz_2 is the parametrized circuit returning the state to extract the bitstring 
    # num_qubits is initialized with the number of features from the dataset 
    # X is the dataset 
    # depth is the number of times the ansatz is repeated
    # batch_size is the number of samples to evaluate the optimizer 
    # num_batches is the number of batches, depending on the batch_size and the size of the dataset

    # selected_variables are the list of feature indexes 
    # energy is the energy computed by the optimizer for a specific bitstring 
    # _bitstring_ is the final bitstring obtained from the dataset 1 are valuable features, and 0 those which are not kept 
    
    '''
    def __init__(self, 
                 ansatz, 
                 ansatz_state,
                 num_qubits, 
                  
                 diff_method="parameter-shift",  # parameter-shift
                 opt=qml.QNSPSAOptimizer(stepsize=1e-2),
                 max_features=7,
                 provider=None):
        
        self.ansatz = ansatz 
        self.ansatz_state= ansatz_state
        self.opt = opt
        self.num_qubits = num_qubits
        self.provider = provider
        self.diff_method = diff_method
        self.max_features = max_features

        self.dev = qml.device('lightning.qubit', wires = self.num_qubits)

    def select_variables(self, X, depth: int, batch_size: int, num_batches: int): 
        '''
        Function to select the features through variational approach as a black box using QN-SPSA or QNG

        Args
            - X: dataset 
            - depth: number of times the ansatz is repeated 
            - batch_size: Number of samples in the dataset passed to evaluate the gradient 
            - num_batches: Number of batches depending of the size of the whole dataset and the size of the batch 
        Returns 
            - bit_save: final bitstring of the selected features  
            - cost_save: energy associated to the bitstring (bit_save) obtained by the optimizer 
        '''
        # test to define the new number of qubits 
        #dev = qml.device('default.qubit', wires = num_qubits)
        #dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider)
        
        qnode = qml.QNode(self.ansatz, self.dev, diff_method=self.diff_method)
        init_params = np.random.random(self.num_qubits*2*depth) 
    
        # init parameters 
        cost_save = 10 
        params = init_params
    
        # starting the process 
        print(
            f"Starting learning process with {num_batches} repetitions on {self.num_qubits} qubits."
        )
        
        # loop on the number of batches determined by the number of rows divided by the size of the batch 
        for n in tqdm(range(num_batches)):
        
            # select indices to build the batch 
            shuffled_idices = np.random.randint(0, len(X), (batch_size,))#np.random.permutation(len(X_))
            X_batch = np.array(X[shuffled_idices])
            
            # train the optimizer on the batch
            for i in range(len(X_batch)):
                params, cost = self.opt.step_and_cost(qnode, params, X=X_batch[i], depth=depth, num_qubits=self.num_qubits)
                params = np.array(params, requires_grad=True) # needed ???
    
            # determine the cost
            if cost <= cost_save:
                bit_save = self.bitstring(params, depth).zfill(self.num_qubits)
                # to delete when it will work
                print(
                    f"{'Iteration = ' + str(n)+',':<12}  {'Cost = '+ str(round(float(cost), 8)):<12} for bitstring {bit_save}"
                )
                cost_save = cost
    
            # delete elements where the gradient was obtained 
            X = np.delete(X, shuffled_idices, axis=0)
            
        return bit_save, cost_save
    
    def fit(self, X, depth, batch_size, num_batches):
        '''


        Args
            - X: dataset 
            - depth: number of times the ansatz is repeated 
            - batch_size: Number of samples in the dataset passed to evaluate the gradient 
            - num_batches: Number of batches depending of the size of the whole dataset and the size of the batch 
        Returns 
            - index: new list of indices for the feature selection 
            - cost_save: energy of the Hamiltonian associated to the combination of features (bitstring) 
        '''
        features = np.shape(X)[1]
        index = np.array(list(range(features))) 
        
        while features > self.max_features:
            
            _bitstring_, cost_save = self.select_variables(X, depth, batch_size, num_batches)
            features = _bitstring_.count('1') # count the number of 1 so selected features 
            columns_index = [i[0] for i in enumerate(list(_bitstring_)) if i[1]=='1'] # extract the index of each 1 in the bitstring
            del_col_index = [i[0] for i in enumerate(list(_bitstring_)) if i[1]=='0']
            self.num_qubits = features
            if not del_col_index:
                return index , cost_save
            
            index = np.delete(index, np.array(del_col_index))
            #print(index)
            X = X[:, columns_index]
            features = np.shape(X)[1]
        return index , cost_save
        
    def bitstring(self, params, depth):
    
        '''


        Args 
            - params:
            - depth:
        Returns
            bitstring 
        '''
        #dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider)
        qnode = qml.QNode(self.ansatz_state, self.dev, diff_method=self.diff_method)
        states = qnode(params=params, depth=depth, num_qubits=self.num_qubits)
        return "{0:b}".format(np.argmax(qnode(params=params, depth=depth, num_qubits=self.num_qubits)))
    

    def bitstring_final(self, col, index):
        '''


        Args 
            - col:
            - index:
        Returns
            bitstring 
        '''
        _bitstring_ = np.zeros(len(col))
        for id in index:
            _bitstring_[id] = 1
        return ''.join([str(int(i)) for i in _bitstring_]) 
    
    
        