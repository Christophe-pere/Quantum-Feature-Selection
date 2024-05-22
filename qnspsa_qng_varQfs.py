import pennylane as qml 
from pennylane import numpy as np
from sklearn import datasets 
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


#@qml.qnode(dev)
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

def bitstring(ansatz, params, depth, num_qubits):

    '''

    '''
    dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider, use_probs=False)
    #dev = qml.device('lightning.qubit', wires=num_qubits)
    qnode = qml.QNode(ansatz, dev, diff_method="parameter-shift")
    states = qnode(params=params, depth=depth, num_qubits=num_qubits)
    return "{0:b}".format(np.argmax(qnode(params=params, depth=depth, num_qubits=num_qubits)))



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

def hamiltonian(num_qubits):
    
    coeffs = [1 for i in range(num_qubits)]
    obs = [qml.PauliZ(i) for i in range(num_qubits)]
    H = qml.Hamiltonian(coeffs, obs)
    return H

#@qml.qnode(dev)#, interface='tf')#, interface="autograd")
def cost_fn(params, X):
    ansatz_3(params, X)
    return qml.expval(H)

def ansatz_5(params, X, depth=2, num_qubits=5):
    '''
    
    '''
    H = hamiltonian(num_qubits)
    #print(params)
    #amplitudes(X_batch, num_qubits=num_qubits)
    #qml.AngleEmbedding(features=X, wires=range(num_qubits))#,pad_with=0.,normalize=True)
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
    return qml.expval(qml.AngleEmbedding(features=X, wires=range(num_qubits)))


def select_variables(X, ansatz, depth: int, batch_size: int, num_batches: int, opt: str, num_qubits: int): 

    # test to define the new number of qubits 
    #dev = qml.device('default.qubit', wires = num_qubits)
    dev = qml.device('qiskit.ibmq.circuit_runner', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider, use_probs=False)
    #dev = qml.device('lightning.qubit', wires=num_qubits)
    qnode = qml.QNode(ansatz, dev, diff_method="parameter-shift")
    init_params = np.random.random(num_qubits*2*depth) 

    # define the optimizer 
    if opt=='QNG':
        opt = qml.QNGOptimizer(step_size=5e-2, lam=0.001, approx="block-diag")
    else:
        opt = qml.QNSPSAOptimizer(stepsize=1e-3, resamplings=10)

    # init parameters 
    cost_save = 10 
    params = init_params

    # starting the process 
    print(
        f"Starting learning process with {num_batches} repetitions on {num_qubits} qubits."
    )
    #print(np.shape(X))
    # loop on the number of batches determined by the number of rows divided by the size of the batch 
    for n in tqdm(range(num_batches)):
    
        # select indices to build the batch 
        shuffled_idices = np.random.randint(0, len(X), (batch_size,))#np.random.permutation(len(X_))
        X_batch = np.array(X[shuffled_idices])
        
        # train the optimizer on the batch
        for i in range(len(X_batch)):
            params, cost = opt.step_and_cost(qnode,  params, X=X_batch[i], depth=depth, num_qubits=num_qubits)
            params = np.array(params, requires_grad=True) # needed ???

        # determine the cost
        if cost <= cost_save:
            #test = bitstring(params, depth, num_qubits)
            bit_save = bitstring(ansatz_4, params, depth, num_qubits).zfill(num_qubits)
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
    while features > max_features:
        
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


if __name__ == "__main__":

    #data = datasets.make_classification(n_samples=10000, n_features=num_qubits, n_classes=2) 
    from varQFS import FeatureSelection

    
    # Here we import the data 
    data = pd.read_csv('data/german.data', sep = ' ')
    data = preprocess(data).to_numpy()[:,:60] # just use the first 60 columns 
    num_qubits = np.shape(data)[1] # set the number of qubits with the data dimension 

    # Real hardware config 
    from qiskit_ibm_runtime import QiskitRuntimeService
    provider = QiskitRuntimeService(channel='ibm_quantum', token='my_token')
    # use sampler 
    dev = qml.device('qiskit.ibmq.sampler', wires=num_qubits,  backend='ibm_quebec', shots=1000, provider=provider, use_probs=False)

    # simulator 
    #dev = qml.device('lightning.qubit', wires=num_qubits)
    
    print(
        'Dataset generated!'
    )

    #varqfs = FeatureSelection(ansatz_3,  num_qubits=np.shape(data)[1])

    
    depth = 2 # depth of the circuit 
    batch_size = 5*depth #5
    opt =  None # 'QNG' # None #
    
    #max_iterations = 1000
    #conv_tol = 1e-06
    max_features = 6
    
    X_ = data[:,:-1]
    Y_ = data[:,-1]
    #X_ = data[0]#[:, :num_qubits]#np.concatenate([X_new, X_transform.to_numpy()], axis=1) #data_res.to_numpy() X_new
    X_new = deepcopy(X_)
    
    #Y_ = data[1] #y
    Y_ = Y_ * 2 - 1
    num_batches = len(X_new) // batch_size
    bitstring_save = []
    energy_save = []
    #selected_variables, energy = select_variables(X_new, ansatz_3, depth, batch_size, num_batches, init_params, opt)
    for _ in range(1):
        #selected_variables, energy = varqfs.fit(X_new, depth, batch_size, num_batches)
        selected_variables, energy = fit(X_new, ansatz_3, depth, batch_size, num_batches, opt, max_features)
        
        #_bitstring_ = varqfs.bitstring_final(X_new[0], selected_variables)
        _bitstring_ = bitstring_final(X_new[0], selected_variables)
        bitstring_save.append(_bitstring_)
        energy_save.append(energy)
    print(
        f'The final bitstring is: {_bitstring_} with an energy of {energy}'
    )
    
    print(bitstring_save)