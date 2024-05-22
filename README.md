# VarQFS
Implementation of the variation Quantum Feature Selection algorithm provided by Zoufal et al. 2023 

--- 

## Library 
This implementation is done with `PennyLane 0.37.0-dev`

## Content 

### varQFS.py

Contains the class for the feature selection algorithm. Work in progress.

How to use the class: 

```python

from varQFS import FeatureSelection
varqfs = FeatureSelection(ansatz_1,ansatz_2,  num_qubits=np.shape(X)[1])
selected_variables, energy = varqfs.fit(X, depth, batch_size, num_batches)
bitstring_ = varqfs.bitstring_final(X[0], selected_variables)
    
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

```

### qnspsa_qng_varQFS.py 

Source code to run the feature selection algorithms. All functions are working. I'm currently migrating the approach into varQFS.py in the class `FeatureSelection`. 

## Tutorial 

- Import your data 
- Preprocess the data 
- Build the device 
- build the ansatzes 
- Set the `depth`, `batch_size`, optimizer (`opt`) and `max_feature`
- Transform the label (`Y`) if needed 
- Then: 

```python

selected_variables, energy = fit(X_new, ansatz_3, depth, batch_size, num_batches, opt, max_features)
        
_bitstring_ = bitstring_final(X_new[0], selected_variables)

```


