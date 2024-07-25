import pennylane as qml 
from pennylane import numpy as np
from sklearn import datasets 
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import random 

def ansatz_4(params, depth, num_qubits):
    '''
    
    '''
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits
        
    return qml.state()# qml.sample(range(num_qubits))#qml.state()

def ansatz_2(params, X,depth, num_qubits):
    '''
    
    '''
    zz_feature_map(data=X, num_qubits=num_qubits, depth=depth)
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits
        
    return qml.state()

def bitstring(ansatz, params, depth, num_qubits):
    '''

    '''
    
    dev = qml.device('lightning.qubit', wires=num_qubits)
    qnode = qml.QNode(ansatz, dev)#, diff_method="parameter-shift")
    states = qnode(params=params, depth=depth, num_qubits=num_qubits)
    return "{0:b}".format(np.argmax(qnode(params=params, depth=depth, num_qubits=num_qubits)))

def bitstring_2(ansatz,  params, X, depth, num_qubits):
    '''

    '''
    #dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider, use_probs=False)
    #dev = qml.device('qiskit.aer', backend = 'aer_simulator_matrix_product_state', wires=num_qubits, shots=1000) 
    
    dev = qml.device('lightning.qubit', wires=num_qubits)
    qnode = qml.QNode(ansatz, dev)#, diff_method="parameter-shift")
    #states = qnode(params=params,X=X, depth=depth, num_qubits=num_qubits)
    return "{0:b}".format(np.argmax(qnode(params=params, X=X, depth=depth, num_qubits=num_qubits)))


#@qml.qnode(dev)
def ansatz_3(params, X, depth=2, num_qubits=5):
    '''
    
    '''
    H = hamiltonian(num_qubits)
    
    #amplitudes(X_batch, num_qubits=num_qubits)
    qml.AngleEmbedding(features=X, wires=range(num_qubits))#,pad_with=0.,normalize=True)
    #qml.StatePrep(np.linalg.norm(X).reshape(-1), wires=range(num_qubits))
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits
    return qml.expval(H)

def ansatz_1(params, X, depth=2, num_qubits=5):
    '''
    
    '''
    H = hamiltonian(num_qubits)
    
    #amplitudes(X_batch, num_qubits=num_qubits)
    #qml.AngleEmbedding(features=X, wires=range(num_qubits))#,pad_with=0.,normalize=True)
    #qml.StatePrep(np.linalg.norm(X).reshape(-1), wires=range(num_qubits))
    zz_feature_map(data=X, num_qubits=num_qubits, depth=depth)
    
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits
    return qml.expval(H)


def hamiltonian(num_qubits):
    
    coeffs = [1 for i in range(num_qubits)]
    obs = [qml.PauliZ(i) for i in range(num_qubits)]
    H = qml.Hamiltonian(coeffs, obs)
    return H

def zz_feature_map(num_qubits , data, depth):
    """
    Function that generate the circuit of the ZZ feature map depending on the data row receive
    :param data: the data row that we want to encode in the circuit
    :return:
    """
    size: int = min(num_qubits, len(data))
    for i in range(size):
        # Hadamard gate apply to every qubits that we will be using
        qml.Hadamard(wires=i)
    for _ in range(depth):
        for i in range(size):
            qml.RZ(2 * data[i], wires=i)

        for qubit_pair in list(combinations(range(size), 2)):
            # Integers that represent the qubits linked for the RZ gate
            qubit_0, qubit_1 = qubit_pair[0], qubit_pair[1]

            # Combinaison of quantum gates to apply to our circuit
            qml.CNOT(wires=[qubit_0, qubit_1])
            qml.RZ(2 * (np.pi - data[qubit_0]) * (np.pi - data[qubit_1]), wires=qubit_1)
            qml.CNOT(wires=[qubit_0, qubit_1])

def ansatz_new(params, X, depth=2, num_qubits=5):
    '''
    
    '''
    zz_feature_map(data=X, num_qubits=num_qubits, depth=depth)
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits

    return qml.state()

def ansatz_new_test(params, depth=2, num_qubits=5):
    '''
    
    '''
    #H = hamiltonian(num_qubits)
    
    #amplitudes(X_batch, num_qubits=num_qubits)
    #qml.AngleEmbedding(features=X, wires=range(num_qubits))#,pad_with=0.,normalize=True)
    #qml.StatePrep(np.linalg.norm(X).reshape(-1), wires=range(num_qubits))
    #zz_feature_map(data=X, num_qubits=num_qubits, depth=depth)
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(params[i+step], wires=i)
        step += num_qubits

def ansatz_final(params, X, depth=2, num_qubits=5, out='state'):
    H = hamiltonian(num_qubits)
    num_params = int(len(params)/2) 
    ansatz_new(params[:num_params], X, depth=depth, num_qubits=num_qubits)
    qml.adjoint(ansatz_new)(params[num_params:], X, depth=depth, num_qubits=num_qubits)

    return qml.expval(H) #if out!= 'state' else qml.state() #[qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def ansatz_final_2(params, X, depth=2, num_qubits=5):
    H = hamiltonian(num_qubits)
    num_params = int(len(params)/2) 
    ansatz_new(params[:num_params], X, depth=depth, num_qubits=num_qubits)
    qml.adjoint(ansatz_new)(params[num_params:], X, depth=depth, num_qubits=num_qubits)
    return qml.state()

def ansatz_final_3(params, X, depth=2, num_qubits=5):
    #H = hamiltonian(num_qubits)
    num_params = int(len(params)/2) 
    ansatz_new_test(params[:num_params], depth=depth, num_qubits=num_qubits)
    qml.adjoint(ansatz_new_test)(params[num_params:], depth=depth, num_qubits=num_qubits)
    return qml.state()

#@qml.qnode(dev)#, interface='tf')#, interface="autograd")
#def cost_fn(params, X):
#    ansatz_3(params, X)
#    return qml.expval(H)

#def ansatz_5(params, X, depth=2, num_qubits=5):
#    '''
#    
#    '''
#    H = hamiltonian(num_qubits)
    #print(params)
    #amplitudes(X_batch, num_qubits=num_qubits)
    #qml.AngleEmbedding(features=X, wires=range(num_qubits))#,pad_with=0.,normalize=True)
    #qml.StatePrep(np.linalg.norm(X).reshape(-1), wires=range(num_qubits))
#    step = 0
#    for _ in range(depth):
#        for i in range(num_qubits):
#            qml.RY(params[i+step], wires=i)
#        for i in range(num_qubits-1):
#            qml.CNOT([i,i+1])
#        for i in range(num_qubits):
#            qml.RY(params[i+step], wires=i)
#        step += num_qubits
#    return qml.expval(qml.AngleEmbedding(features=X, wires=range(num_qubits)))


def select_variables(X, ansatz, depth: int, batch_size: int, num_batches: int, opt: str, num_qubits: int): 

    # test to define the new number of qubits 
    #dev = qml.device('default.qubit', wires = num_qubits)
    #dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider, use_probs=False)
    dev = qml.device('lightning.qubit', wires=num_qubits)
    #dev = qml.device('qiskit.aer', backend = 'aer_simulator_matrix_product_state', wires=num_qubits, shots=1000) 
    
    qnode = qml.QNode(ansatz, dev)#, diff_method="parameter-shift")
    init_params = np.random.random(num_qubits*2*depth*2) 
    # Compute a single metric tensor with new strategy, as indicated with `use_probs=False`.
    print(f'Number of parameters: {len(init_params)}')
    #print(init_params)
    # define the optimizer 
    if opt=='QNG':
        opt = qml.QNGOptimizer(step_size=5e-2, lam=0.001, approx="block-diag")
    else:
        opt = qml.QNSPSAOptimizer(stepsize=1e-3, resamplings=10)

    # init parameters 
    cost_save = 1000
    params = init_params

    # starting the process 
    print(
        f"Starting learning process with {num_batches} repetitions on {num_qubits} qubits."
    )
    #print(np.shape(X))

    mt_fn = qml.metric_tensor(qnode, argnum=0, use_probs=False, approx="block-diag")
    #mt = mt_fn(init_params)
    # loop on the number of batches determined by the number of rows divided by the size of the batch 
    for n in tqdm(range(num_batches)):
    
        # select indices to build the batch 
        shuffled_idices = np.random.randint(0, len(X), (batch_size,))#np.random.permutation(len(X_))
        X_batch = np.array(X[shuffled_idices])
        
        # train the optimizer on the batch
        for i in range(len(X_batch)):
            params, cost = opt.step_and_cost(qnode,  params, X=X_batch[i], depth=depth, num_qubits=num_qubits)#, metric_tensor_fn=mt_fn)
            params = np.array(params, requires_grad=True) # needed ???

        # determine the cost
        if cost <= cost_save:
            #test = bitstring(params, depth, num_qubits)
            bit_save = bitstring_2(ansatz_2, params,X_batch[i],  depth, num_qubits).zfill(num_qubits)
            #print(len(bit_save), len(test), num_qubits)
            # to delete when it will work
            print(
                f"{'Iteration = ' + str(n)+',':<12}  {'Cost = '+ str(round(float(cost), 8)):<12} for bitstring {bit_save}"
            )
            
            cost_save = cost

        # delete elements where the gradient was obtained 
        X = np. delete(X, shuffled_idices, axis=0)
        
    return bit_save, cost_save

def fit(X, ansatz, depth, batch_size, num_batches, opt, max_features):
    
    features = np.shape(X)[1]
    index = np.array(list(range(features))) 
    #while features > 0: #max_features:
        
    _bitstring_, cost_save = select_variables(X, ansatz, depth, batch_size, num_batches,  opt, num_qubits=features)
    features = _bitstring_.count('1') # count the number of 1 so selected features 
    columns_index = [i[0] for i in enumerate(list(_bitstring_)) if i[1]=='1'] # extract the index of each 1 in the bitstring
    del_col_index = [i[0] for i in enumerate(list(_bitstring_)) if i[1]=='0']

    if not del_col_index:
        return index , cost_save
        
    index = np.delete(index, np.array(del_col_index))
        #print(index)
    X = X[:, columns_index]
    features = np.shape(X)[1]
    return index , cost_save

def bitstring_final(col, index):

    _bitstring_ = np.zeros(len(col))
    for id in index:
        _bitstring_[id] = 1
    return ''.join([str(int(i)) for i in _bitstring_]) 

def preprocess(X):
    '''

    ''' 
    enc = OneHotEncoder(handle_unknown='ignore')
    X_cat = X.select_dtypes(include=['object'])
    enc.fit(X_cat)
    names = list()
    #for i in enc.categories_:
    #    names.extend(i)
    X_transform = pd.DataFrame(enc.transform(X_cat).toarray())
    data_result = pd.concat([X.select_dtypes(exclude=['object']), X_transform], axis=1)
    return data_result




def id_features_selection(max_n_feature, sample_size=20, sampling_generation=100 ): 

    population = list(range(0, max_n_feature))
    
    sample_store =  []
    for i in list(range(sampling_generation)):
        #perform simple random sampling by using the random.sample() function
        sample = random.sample(population, sample_size) 
        sample_store.append(sample)
    
    return sample_store

def store_res(bits, features, data):
    idx = list(bits)
    
    for i in enumerate(features):
        if i[1] in data.keys() and int(idx[i[0]])==1:
            data[i[1]] += 1 
        elif i[1] not in data.keys() and int(idx[i[0]])==1:
            data[i[1]] = 1
        else:
            pass 
    return data

if __name__ == "__main__":

    # dataset 
    data = pd.read_csv('../Pennylane/data/german.data', sep = ' ')
    data = preprocess(data).to_numpy() # just use the first 60 columns 

    #X_ = data[0]
    #Y_ = data[1]
    X_ = data[:,:-1]
    Y_ = data[:,-1]
    Y_ = Y_ * 2 - 1 
    X_new = deepcopy(X_)
    max_n_feature = np.shape(X_new)[1]
    # probabilistic sampling 
    num_qubits = 10 #20 # number of qubits 

    # grab the indices of the features 
    idx = id_features_selection(max_n_feature, sample_size=num_qubits, sampling_generation=100 )

    # device 
    dev = qml.device('lightning.qubit', wires=num_qubits)
    depth = 2 # depth of the circuit 
    batch_size = 5*depth #5
    opt =  None # 'QNG' # None #
    max_features = 3
    
    # set up variables 
    num_batches = 10#len(X_new) // batch_size
    bitstring_save = []
    energy_save = []
    values_final = {}
    
    for id_features in tqdm(idx): 
        
        sample = X_new[:, id_features]
        # run feature selection 
        selected_variables, energy = fit(sample, ansatz_1, depth, batch_size, num_batches, opt, max_features)
        
        #_bitstring_ = varqfs.bitstring_final(X_new[0], selected_variables)
        _bitstring_ = bitstring_final(sample[0], selected_variables)
        bitstring_save.append(_bitstring_)
        energy_save.append(energy)
        values_final = store_res(_bitstring_, id_features, values_final)
        print(values_final)

    


