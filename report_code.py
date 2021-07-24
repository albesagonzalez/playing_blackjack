import os
import pickle
from copy import deepcopy
import numpy as np
from numpy import pi
from scipy.optimize import minimize
import random
import itertools
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.jupyter import *
from qiskit import *






###################################################################################################################
#################################### MARKOV CLASS #################################################################
###################################################################################################################
def infinite_next_state(state, key):
    count = state[0]
    ace = state[1]
    count_ = count + key
    ace_ = ace
    if key == 1:
        if count_ + 10 <= 21:
            count_ += 10
            ace_ = True
    if ace:
        if count_ > 21:
            count_ -= 10
            ace_ = False
    if count_ > 21:
        count_ = 0
    if count_ == 21:
        ace_ = False
    if count_ == 0:
        ace_ = False
    return (count_, ace_)
def finite_next_state(state, key):
    new_hand = False
    count = state[0]
    ace = state[1]
    total_count = state[2]
    count_ = count + key
    ace_ = ace
    if key == 1:
        if count_ + 10 <= 21:
            count_ += 10
            ace_ = True
    if ace:
        if count_ > 21:
            count_ -= 10
            ace_ = False
    if count_ > 21:
        count_ = 0
        ace_ = False
        new_hand = True
    if count_ == 21:
        ace_ = False
        count_ = 0
        total_count += 1
        new_hand = True
    return (count_, ace_, total_count), new_hand
def finite_hands_state(state, key):
    new_hand = False
    count = state[0]
    ace = state[1]
    total_count = 0
    count_ = count + key
    ace_ = ace
    if key == 1:
        if count_ + 10 <= 21:
            count_ += 10
            ace_ = True
    if ace:
        if count_ > 21:
            count_ -= 10
            ace_ = False
    if count_ > 21:
        count_ = 0
        ace_ = False
        new_hand = True
    if count_ == 21:
        ace_ = False
        count_ = 0
        new_hand = True
    return (count_, ace_), new_hand
def infinite_state_dict(probs, state):
    line_dict = {}
    for key, value in probs.items():
        if value != 0:
            state_ = infinite_next_state(state, key)
            key = state_
            try:
                line_dict[key] += value
            except:
                line_dict[key] = value
    return line_dict
def finite_pp_dict(deck, state):
    line_dict = {}
    probs = dynamic_probs(deck)
    for key, value in probs.items():
        if value != 0:
            state_ = infinite_next_state(state, key)
            key = state_
            try:
                line_dict[key] += value
            except:
                line_dict[key] = value
    return line_dict
def finite_state_dict(observation):
    line_dict = {}
    state = (observation[0], observation[1], observation[2])
    next_deck = list(observation[3:])
    probs = dynamic_probs(next_deck)
    for key, value in probs.items():
        if not np.isclose(value, 0):
            state_, new_hand = finite_next_state(state, key)
            aux_deck = deepcopy(next_deck)
            aux_deck[key - 1] -= 1
            final_key = state_ + tuple(aux_deck)
            if new_hand:
                new_deck = deepcopy(aux_deck)
                new_probs = dynamic_probs(aux_deck)
                for new_key, new_value in new_probs.items():
                    if not np.isclose(new_value, 0):
                        new_state, new_hand = finite_next_state(state_, new_key)
                        new_aux_deck = deepcopy(aux_deck)
                        new_aux_deck[new_key - 1] -= 1
                        final_key = state_ + tuple(new_aux_deck)
                        try:
                            line_dict[final_key] += new_value*value
                        except:
                            line_dict[final_key] = new_value*value
            else:
                try:
                    line_dict[final_key] += value
                except:
                    line_dict[final_key] = value     
    return line_dict
def static_probs():
    probs = {}
    for i in range(1, 10):
        probs[i] = 1/13
    probs[10] = 4/13
    return probs
def dynamic_probs(current_deck):
    if np.sum(current_deck) == 0:
        return {}
    probs = {}
    probs = {i + 1 : current_deck[i]/np.sum(current_deck) for i in range(10)}
    return probs
class Markov():
    def __init__(self, environment):
        self.environment = environment
        if self.environment == 'infinite_deck':
            self.probs = static_probs()     
    def dictionary(self, observation):
        if self.environment == 'infinite_deck':
            probs = self.probs
            return infinite_state_dict(probs, observation)
        if self.environment == 'finite_deck':
            return finite_state_dict(observation)
        if self.environment == 'finite_pp':
            state = (observation[0], observation[1])
            deck = list(observation[2:])
            return finite_pp_dict(deck, state)
        
        
        
###################################################################################################################
#################################### QUANTUM MODEL CLASS ##########################################################
###################################################################################################################

backend = BasicAer.get_backend('qasm_simulator')
def quantum_circuit(qbit_params_, g_params_):  
    num_bits = len(qbit_params_)
    qreg_q = QuantumRegister(num_bits, 'q')
    creg_c = ClassicalRegister(num_bits, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    for bit in qbit_params_:
        bit_id = bit["id"]
        encoding_angle = bit["theta"]
        circuit.rx(encoding_angle, qreg_q[bit_id])
        circuit.rz(encoding_angle, qreg_q[bit_id])   
    num_layers = len(g_params_)
    for i in range(num_layers): 
        for j in range(num_bits - 1):
            circuit.cx(qreg_q[j], qreg_q[j + 1])
        for channel in g_params_[i]:
            channel_id = channel["id"]
            circuit.rx(channel["x"], qreg_q[channel_id])
            circuit.ry(channel["y"], qreg_q[channel_id])
            circuit.rz(channel["z"], qreg_q[channel_id])     
    return circuit
def measure_action(circuit, action):
    if action == 0:
        channel = 0
    else:
        channel = 1
    n = circuit.num_qubits
    qreg_q = QuantumRegister(n, 'q')
    creg_c = ClassicalRegister(n, 'c')
    circuit.measure(qreg_q[channel], creg_c[channel])
    result = execute(circuit,backend,shots=1000).result()
    counts = result.get_counts(circuit)
    ones_assigned = False
    zeros_assigned = False
    for key in list(counts.keys()):
        if '1' in key:
            num_of_ones = counts[key]
            ones_assigned = True
        else:
            num_of_zeros = counts[key]
            zeros_assigned = True
    if not ones_assigned:
        num_of_ones = 0
    if not zeros_assigned:
        num_of_zeros = 0
    return num_of_ones/(num_of_zeros + num_of_ones)
def gate_params(architecture):
    n = architecture[0]
    num_layers = architecture[1]
    g_params_ = []
    for i in range(num_layers):
        layer = []
        for j in range(n):
            channel = {}
            channel["id"] = j
            channel["x"] = 2*pi*np.random.rand()
            channel["y"] = 2*pi*np.random.rand()
            channel["z"] = 2*pi*np.random.rand()
            #channel["x"] = 0.1
            #channel["y"] = 0.1
            #channel["z"] = 0.1
            layer.append(channel)
        g_params_.append(layer)
    return g_params_
def q_loss(output, label):
    #print("m: ", m)
    #print("Y: ", Y)
    #print("AL: ", AL)
    cost = np.abs(output - label)
    return cost
def compute_gradient(qbit_params_, action, g_params_, param_):
    layer = param_["layer"]
    channel = param_["channel"]
    gate = param_["gate"]
    theta =  g_params_[layer][channel][gate]
    r = 1/2
    gate_params_1 = deepcopy(g_params_)
    gate_params_2 = deepcopy(g_params_)
    if r == 0:
        return 0
    gate_params_1[layer][channel][gate] = theta - pi/(4*r)
    gate_params_2[layer][channel][gate] = theta + pi/(4*r)
    circuit1 = quantum_circuit(qbit_params_, gate_params_1)
    circuit2 = quantum_circuit(qbit_params_, gate_params_2)
    num_bits = circuit1.num_qubits
    measurement_1 = measure_action(circuit1, action)
    measurement_2 = measure_action(circuit2, action)
    gradient = r*(measurement_2 - measurement_1)
    return gradient
def update_parameters(initial_state, initial_q, action, final_q, layer_params_, learning_rate):
    param_ = {}
    diff = final_q - initial_q
    num_layers = len(layer_params_) 
    new_params_ = deepcopy(layer_params_)
    delta_params_ = deepcopy(layer_params_)
    for i in range(num_layers):
        for channel in layer_params_[i]:
            param_["layer"] = i
            channel_id = channel["id"]
            param_["channel"] = channel_id
            if i + 1 < num_layers or channel_id == action:
                param_["gate"] = 'x'
                grad_x = compute_gradient(initial_state, action, layer_params_, param_)
                #print(grad_x)
                param_["gate"] = 'y'
                grad_y = compute_gradient(initial_state, action, layer_params_, param_)
            #print(grad_y)
                param_["gate"] = 'z'
                grad_z = compute_gradient(initial_state, action, layer_params_, param_)
            else:
                grad_x = 0
                grad_y = 0
                grad_z = 0
            delta_params_[i][channel_id]["x"] = learning_rate*diff*grad_x
            delta_params_[i][channel_id]["y"] = learning_rate*diff*grad_y
            delta_params_[i][channel_id]["z"] = learning_rate*diff*grad_z          
    for i in range(num_layers):
        for channel in layer_params_[i]:
            channel_id = channel["id"]
            new_params_[i][channel_id]["x"] += delta_params_[i][channel_id]["x"]
            new_params_[i][channel_id]["y"] += delta_params_[i][channel_id]["y"]
            new_params_[i][channel_id]["z"] += delta_params_[i][channel_id]["z"]
    return new_params_
class Quantum_Model():   
    def __init__(self, architecture): 
        self.architecture = architecture
        self.parameters = gate_params(architecture)  
    def train(self, exp_sample, learning_rate = 0.5, num_iterations = 15, print_cost = False):
        for X, Y in exp_sample:
            action = Y[0]
            label = Y[1]
            output = self.forward(X, action)
            for i in range(0, num_iterations):
                output = self.forward(X, action)
                cost = q_loss(output, label)
                self.parameters = update_parameters(X, output, action, label, self.parameters, learning_rate)
        return
    def forward(self, state, action):
        circuit = quantum_circuit(state, self.parameters)
        return measure_action(circuit, action)
    
    
    
###################################################################################################################
#################################### DEEPQ MODEL CLASS ############################################################
###################################################################################################################

def relu_function(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache
def Neural_Network_backward(diff_A, cache):
    Z = cache
    diff_Z = np.array(diff_A, copy=True) 
    diff_Z[Z <= 0] = 0
    return diff_Z
def initialize_params(layer_dims):
    params = {}
    L = len(layer_dims)           
    np.random.seed(2)
    for l in range(1, L):
        params['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1])*0.05
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))    
    return params
def l_forward(A, W, b):
    Z = W.dot(A) + b  
    cache_value = (A, W, b) 
    return Z, cache_value
def la_forward(A_prev, W, b):
    Z, linear_cache_value = l_forward(A_prev, W, b)
    A, a_cache_value = relu_function(Z)
    cache_value = (linear_cache_value, a_cache_value)
    return A, cache_value
def Forward(X, params):
    caches_value = []
    A = X
    L = len(params) // 2                 
    for l in range(L):
        A_prev = A 
        A, cache_value = la_forward(A_prev, params['W' + str(l + 1)], params['b' + str(l + 1)])
        caches_value.append(cache_value)       
    return A, caches_value
def loss(AL, Y):
    m = Y.shape[1]
    cost_matrix = Y - AL  
    cost = 0
    for row in cost_matrix:
        for element in row:
            cost += element**2
    cost = (1/m)*cost
    return cost
def l_backward(diff_Z, cache_value):
    A_prev, W, b = cache_value
    m = A_prev.shape[1]
    dW = 1./m * np.dot(diff_Z,A_prev.T)
    db = 1./m * np.sum(diff_Z, axis = 1, keepdims = True)
    diff_A_prev = np.dot(W.T,diff_Z)
    return diff_A_prev, dW, db
def la_backward(diff_A, cache_value):
    linear_cache_value, a_cache_value = cache_value
    diff_Z = Neural_Network_backward(diff_A, a_cache_value)
    diff_A_prev, dW, db = l_backward(diff_Z, linear_cache_value)
    return diff_A_prev, dW, db
def Backward(AL, Y, caches_value):
    grads = {}
    L = len(caches_value)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    diff_AL = 2*(AL - Y)
    current_cache_value = caches_value[L-1]
    grads["diff_A" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = la_backward(diff_AL, current_cache_value)
    for l in reversed(range(L-1)):
        current_cache_value= caches_value[l]
        l_string = str(l + 1)
        diff_A_prev_temp, dW_temp, db_temp = la_backward(grads["diff_A" + str(l + 2)], current_cache_value)
        grads["diff_A" + l_string] = diff_A_prev_temp
        grads["dW" + l_string] = dW_temp
        grads["db" + l_string] = db_temp
    return grads
def update_params_regular(params, grads, learning_rate):
    L = len(params) // 2 
    for l in range(L):
        l_string = str(l + 1)
        params["W" + l_string] = params["W" + l_string] - learning_rate * grads["dW" + l_string]
        params["b" + l_string] = params["b" + l_string] - learning_rate * grads["db" + l_string]    
    return params
def update_params_adam(params, grads, learning_rate, m_dict, v_dict, t):
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 1e-8
    m_hat = {}
    v_hat = {}
    L = len(params) // 2
    for l in range(L):
        W = "W" + str(l+1)
        dW = 'd' + W
        b = "b" + str(l+1)
        db = 'd' + b
        m_dict[W] = beta_1*m_dict[W] + (1 - beta_2)*grads[dW]
        m_dict[b] = beta_1*m_dict[b] + (1 - beta_2)*grads[db]
        v_dict[W] = beta_2*v_dict[W] + (1 - beta_1)*np.multiply(grads[dW], grads[dW])
        v_dict[b] = beta_2*v_dict[b] + (1 - beta_1)*np.multiply(grads[db], grads[db])
        m_hat[W] = m_dict[W]/(1 - beta_1**t)
        m_hat[b] = m_dict[b]/(1 - beta_1**t)
        v_hat[W] = v_dict[W]/(1 - beta_2**t)
        v_hat[b] = v_dict[b]/(1 - beta_2**t)
        params[W] = params[W] - learning_rate * np.divide(m_hat[W], (v_hat[W] + epsilon))
        params[b] = params[b] - learning_rate * np.divide(m_hat[b], (v_hat[b] + epsilon))
    return params, m_dict, v_dict
def update_params_stochastic(params, V_params, grads, learning_rate):
    beta = 0.5
    L = len(params) // 2 
    for l in range(L):
        W = "W" + str(l+1)
        dW = 'd' + W
        b = "b" + str(l+1)
        db = 'd' + b
        V_params[W] = beta*V_params[W] + learning_rate*grads[dW]
        V_params[b] = beta*V_params[b] + learning_rate*grads[db]
        params[W] -= V_params[W]
        params[b] -= V_params[b]
    return params, V_params
class DeepQ_Model():
    def __init__(self, architecture): 
        self.architecture = architecture
        self.params = initialize_params(architecture)
    def train(self, X, Y, learning_rate = 0.001, max_iterations = 3000, print_cost = False, optimizer = 'regular'):
        AL, caches_value = Forward(X, self.params)
        cost = loss(AL, Y)
        if optimizer == 'regular':
            costs = []
            i = 1
            while cost > 0.005 and i < max_iterations:
                AL, caches_value = Forward(X, self.params)
                cost = loss(AL, Y)
                grads = Backward(AL, Y, caches_value)
                self.params = update_params_regular(self.params, grads, learning_rate)
                i += 1
                if print_cost and i % 100 == 0:
                    print("\n\n\n\n\ Cost after iteration %i: %f" %(i, cost))
                if print_cost and i % 100 == 0:
                    costs.append(cost)               
        if optimizer == 'adam':
            m_dict = {}
            v_dict = {}
            L = len(self.params)//2
            for l in range(L):
                m_dict["W" + str(l+1)] = 0
                v_dict["W" + str(l+1)] = 0
                m_dict["b" + str(l+1)] = 0
                v_dict["b" + str(l+1)] = 0
            i = 1
            while cost > 0.05 and i < max_iterations:
                AL, caches_value = Forward(X, self.params)
                cost = loss(AL, Y)
                grads = Backward(AL, Y, caches_value)
                self.params, m_dict, v_dict = update_params_adam(self.params, grads, learning_rate, m_dict, v_dict, i)
                i += 1
                # Print the cot every 100 training example
                if print_cost and i % 1 == 0:
                    print("\n\n\n\n\ Cost after iteration %i: %f" %(i, cost))
            #print(i)         
        if optimizer == 'stochastic':
            i = 1
            V_params = self.params
            while cost > 0.05 and i < max_iterations:
                AL, caches_value = Forward(X, self.params)
                cost = loss(AL, Y)
                grads = Backward(AL, Y, caches_value)
                self.params, V_params = update_params_stochastic(self.params, V_params, grads, learning_rate)
                i += 1
                # Print the cot every 100 training example
                if print_cost and i % 1 == 0:
                    print("\n\n\n\n\ Cost after iteration %i: %f" %(i, cost))
            print(i)
        return         
    def forward(self, X):
        AL, caches_value = Forward(X, self.params)
        return AL  
    def load_params(params):
        self.params = params
        
    
    
###################################################################################################################
############################## POLICY GRADIENT NN CLASS ###########################################################
################################################################################################################### 

class Policy_Gradient_NN:
    def __init__(self,architecture,activation, learning_rate):
        if activation=="sigmoid":
            self.architecture=architecture+[1]
        if activation=="softmax":
            self.architecture=architecture+[2]            
        self.activation=["relu" for i in range(len(architecture)-1)]+[activation] #activation functions ["relu","relu", "sigmoid"/"softmax"]
        #print("archi",self.architecture)
        #print("activation",self.activation)
        self.W=0.01*np.array([np.random.randn(self.architecture[i],self.architecture[i-1]) for i in range(1,len(self.architecture))])
        self.b=np.array([np.zeros((i,1)) for i in self.architecture[1:]])
        self.X=np.array([[0.5] for i in range(self.architecture[0])]) #input #self.Y=np.array([[0.5] for i in range(self.architecture[-1])]) #output
        self.l=learning_rate #learning rate, must be taken between [0,1]                        
        
        
    def Sigmoid(self,M):
        return 1/(1+np.exp(-M))
    def Relu(self,M):
        return np.maximum(0,M)
    def Softmax(self,M):
        c=max(M) # for stability  mulitply by log(max(z))
        exps = np.exp(M-c)
        return exps / np.sum(exps)
    def Relu_deriv(self,M):
        R=np.array([1 if i>0 else 0 for i in M])
        return np.reshape(R,np.shape(M))
    def Relu_back(self,dA,Z):
        R=np.array([1 if i>0 else 0 for i in Z])
        return np.reshape(R,np.shape(dA))*dA       
    def Sigmoid_deriv(self,M):
        One=np.ones((len(M),1))
        return np.multiply(Policy_Gradient_NN.Sigmoid(self,M),One-Policy_Gradient_NN.Sigmoid(self,M))
    def Sigmoid_back(self,dA,Z):
        One=np.ones((len(Z),1))
        A=Policy_Gradient_NN.Sigmoid(self,Z)
        return dA*A*(One-A)   
    def Softmax_back(self, dA, Z):
        m, n = Z.shape
        p = Policy_Gradient_NN.Softmax(self,Z)
        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
        dSoftmax = tensor2 - tensor1
        dZ = np.einsum('ijk,ik->ij', dSoftmax, dA)  # (m, n)
        return dZ        
        
    def Linear_forward(self,A,W,b,activation):
        Z=np.dot(W,A)+b #An+1 
        if activation=="relu":
            A_new=Policy_Gradient_NN.Relu(self,Z)
        if activation=="sigmoid":
            A_new=Policy_Gradient_NN.Sigmoid(self,Z)
        if activation=="softmax":
            A_new=Policy_Gradient_NN.Softmax(self,Z)        
        cache=A,W,b,Z        
        return np.array(A_new) ,cache
    
    def Forward(self):
        caches=[]
        A=self.X
        for i in range(len(self.architecture)-1):
            A_prev=A
            A,cache=Policy_Gradient_NN.Linear_forward(self,A_prev,self.W[i],self.b[i],self.activation[i])
            caches.append(cache)
        return A,caches
    
    def Linear_backward(self,dAlp1,cache,activation):
        A_prev,W,b,Z=cache
        if activation=="sigmoid":
            dAl=Policy_Gradient_NN.Sigmoid_back(self,dAlp1,Z)
        if activation=="relu":
            dAl=Policy_Gradient_NN.Relu_back(self,dAlp1,Z)
        if activation=="softmax":
            dAl=Policy_Gradient_NN.Softmax_back(self,dAlp1,Z)
        dW=np.dot(dAl,A_prev.T)
        db=dAl
        dA=np.dot(W.T,dAl)
        return dA,dW,db
    
    def Backward(self,loss,caches): #gradients calculation
        grads_W,grads_b=[],[]
        dAlp1=loss  #loss=Y-AL      
        for i in range(1,len(caches)+1):
            (A,W,b,Z)=caches[-i] #cache=A,W,b,Z            
            dA,dW,db=Policy_Gradient_NN.Linear_backward(self,dAlp1,(A,W,b,Z),self.activation[-i])
            grads_W.append(dW)
            grads_b.append(db)
            dAlp1=dA
        return np.flip(grads_W,axis=0),np.flip(grads_b,axis=0)
        
    def Update(self,grads_W,grads_b):# Gradient descent method
        for i in range(len(self.W)):
            self.W[i]-=self.l*grads_W[i]
            self.b[i]-=self.l*grads_b[i]
        return self.W,self.b          
    
    
    
        
###################################################################################################################
############################## RANDOM AGENT CLASS #################################################################
###################################################################################################################

class Random_Agent():
    def __init__(self, actionSpace):
        self.exp_size = 100000000000000
        self.actionSpace = actionSpace
    def choose_action(self, state, observation):
        action = np.random.choice(self.actionSpace)
        return action
    def choose_optimal_action(self, state, observation):
        action = np.random.choice(self.actionSpace)
        return action
    def get_parameters(self):
        return
    def process_observation(self, observation):
        return observation
    def process_episode(self, episode_memory):
        return
    def update_epsilon(self):
        return
    
    
    
###################################################################################################################
#################################### PP AGENT CLASS ###############################################################
###################################################################################################################

class PP_Agent(Markov): 
    def __init__(self, stateSpace, model):
        self.environment = model['environment']
        self.stateSpace = stateSpace
        self.Q = {}
        if self.environment == 'infinite_deck':
            self.policy = self.generate_policy()
        self.exp_size = 100000000000000
    def get_parameters(self):
        return self.Q
    def process_observation(self, observation):
        return observation
    def choose_action(self, state, observation):
        action = self.policy[state]
        return action
    def choose_optimal_action(self, state, observation):
        if self.environment == 'infinite_deck':
            pass
        else:
            state = (observation[0], observation[1])
            deck = observation[3:]
            self.policy = self.generate_policy(deck)
        action = self.policy[state]
        return action
    def process_episode(self, episode_memory):
        return
    def update_epsilon(self):
        return
    def generate_policy(self, deck = {}):
        rewards = {}
        policy = {}
        num_states = len(self.stateSpace)
        rewards[(21, False)] = 21**2
        rewards[(0, False)] = 0
        finished = False
        leftStates = deepcopy(self.stateSpace)
        while not finished:
            for state in leftStates:
                reward = 0
                if self.environment == 'infinite_deck':
                    markov_dict = Markov(self.environment).dictionary(state)
                if self.environment == 'finite_pp':
                    obs = state + deck
                    markov_dict = Markov(self.environment).dictionary(obs)
                    if len(markov_dict) == 0:
                        leftStates.remove(state)
                try:
                    for key, value in markov_dict.items():
                        reward = reward + value*rewards[key]
                    sticking_reward = state[0]**2
                    if reward > sticking_reward:
                        rewards[state] = reward
                        policy[state] = 0
                        self.Q[(state, 1)] = sticking_reward
                    else:
                        rewards[state] = sticking_reward
                        policy[state] = 1
                        self.Q[(state, 0)] = reward                       
                        leftStates.remove(state)
                except:
                    pass
            if len(policy) == num_states:
                finished = True
        return policy




###################################################################################################################
############################## MONTE CARLO AGENT CLASS ############################################################
###################################################################################################################

class Monte_Carlo_Agent(Markov):
    def __init__(self, stateSpace, actionSpace, model, gamma, initial_epsilon, epsilon_step):
        self.model_free = model["free"]
        if not self.model_free:
            self.environment = model["environment"]
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        self.EPSILON = initial_epsilon
        self.EPSILON_STEP = epsilon_step
        self.exp_size = 100000000000000
        self.GAMMA = gamma
        self.Q = {}
        self.times_visited = {}
        self.initialise_q_times()     
    def initialise_q_times(self):
        for state in self.stateSpace:
            for action in self.actionSpace:
                self.Q[(state, action)] = 0
                self.times_visited[(state, action)] = 0
    def get_parameters(self):
        return self.Q
    def load_parameters(params):
        self.Q = params
    def choose_action(self, state, observation):
        r = np.random.random()
        if r < 1 - self.EPSILON:
            values = np.array([self.Q[(state, a)] for a in self.actionSpace ])
            best = np.random.choice(np.where(values==values.max())[0])
            action = self.actionSpace[best]  
        else:
            action = np.random.choice(self.actionSpace)   
        return action
    def choose_optimal_action(self, state, observation):
        values = self.values(state, observation)
        best = np.random.choice(np.where(values==values.max())[0])
        action = self.actionSpace[best]     
        return action
    def process_observation(self, observation):
        return observation   
    def process_episode(self, episode_memory):
        last_visited = False
        G = 0
        for state, action, reward, state_, observation_ in reversed(episode_memory):
            tv = self.times_visited[(state, action)]
            G = self.GAMMA*G + reward
            self.Q[(state, action)] = ((self.Q[(state, action)])*tv + G)/(tv + 1)
            self.times_visited[(state, action)] += 1    
    def update_epsilon(self):
        if self.EPSILON - 1e-3 > 0:
            self.EPSILON -= 1e-3
        else:
            self.EPSILON = 0       
    def values(self, state, observation):
        values = np.array([self.Q[(state, a)] for a in self.actionSpace])
        if self.model_free:
            pass
        else:
            values = np.array([self.Q[(state, a)] for a in self.actionSpace])
            markov_dict = Markov(self.environment).dictionary(observation)
            hit_value = 0
            for key, value in markov_dict.items():
                state_values = np.array([self.Q[(self.process_observation(key), a)] for a in self.actionSpace])
                state_max = np.max(state_values)
                hit_value += value*state_max
            values[0] = hit_value
        return values

    
    
    
###################################################################################################################
############################## Q AGENT CLASS ######################################################################
###################################################################################################################

class Q_Agent(Markov):
    def __init__(self, stateSpace, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate = 0.05):
        self.model_free = model["free"]
        if not self.model_free:
            self.environment = model["environment"]
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        self.EPSILON = initial_epsilon
        self.EPSILON_STEP = epsilon_step
        self.exp_size = 100000000000000
        self.GAMMA = gamma
        self.Q = {}
        self.initialise_q()
        self.learning_rate = learning_rate
    def initialise_q(self):
        for state in self.stateSpace:
            for action in self.actionSpace:
                self.Q[(state, action)] = 0
    def get_parameters(self):
        return self.Q
    def load_parameters(params):
        self.Q = params
        
    def choose_action(self, state, observation):
        r = np.random.random()
        if r < 1 - self.EPSILON:
            values = self.values(state, observation)
            best = np.random.choice(np.where(values==values.max())[0])
            action = self.actionSpace[best]  
        else:
            action = np.random.choice(self.actionSpace)
        return action
    def choose_optimal_action(self, state, observation):
        values = self.values(state, observation)
        best = np.random.choice(np.where(values==values.max())[0])
        action = self.actionSpace[best]     
        return action
    def process_observation(self, observation):
        return observation
    def process_episode(self, episode_memory):
        last_visited = False
        for state, action, reward, state_, observation_ in reversed(episode_memory):
            if last_visited:
                self.Q[(state, action)] = (1 - self.learning_rate)*self.Q[(state, action)] + self.learning_rate*(self.GAMMA*np.max([self.Q[(state_, a)] for a in self.actionSpace]) + reward)
            else:
                self.Q[(state, action)] = reward
                last_visited = True
    def update_epsilon(self):
        if self.EPSILON - self.EPSILON_STEP > 0:
            self.EPSILON -= self.EPSILON_STEP
        else:
            self.EPSILON = 0
    def values(self, state, observation):
        values = np.array([self.Q[(state, a)] for a in self.actionSpace])
        if self.model_free:
            pass
        else:
            values = np.array([self.Q[(state, a)] for a in self.actionSpace])
            markov_dict = Markov(self.environment).dictionary(observation)
            hit_value = 0
            for key, value in markov_dict.items():
                state_values = np.array([self.Q[(self.process_observation(key), a)] for a in self.actionSpace])
                state_max = np.max(state_values)
                hit_value += value*state_max
            values[0] = hit_value
        return values
    

    
    
###################################################################################################################
############################## QUANTUM AGENT CLASS ################################################################
################################################################################################################### 

def process_state_range(state_range):
    min_max = []
    for state in state_range:
        min_ = np.min(state)
        max_ = np.max(state)
        min_max.append((min_, max_))
    return min_max
class Memory():
    def __init__(self, capacity):
        self.c = capacity
        self.storage = []
        self.placement = 0        
    def push(self, move):
        if len(self.storage) < self.c:
            self.storage.append(move)
        else:
            self.storage[self.placement] = move
            self.placement = (self.placement + 1)%self.c            
    def sample(self, batchsize):
        return random.sample(self.storage, batchsize)   
    def __len__(self):
        return len(self.storage)   
class Quantum_Agent(Memory, Quantum_Model):  
    def __init__(self, state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate):
        self.model_free = model["free"]
        if not self.model_free:
            self.environment = model["environment"]
        self.actionSpace = actionSpace
        self.EPSILON = initial_epsilon
        self.EPSILON_STEP = epsilon_step
        self.GAMMA = gamma
        self.learning_rate = learning_rate
        self.BATCH_SIZE = 3
        self.EXPERIENCE_CAPACITY = 10
        self.experience = Memory(self.EXPERIENCE_CAPACITY)
        self.exp_size = 1
        self.MIN_MAX = process_state_range(state_range)
        self.NUM_QUBITS = len(self.MIN_MAX)
        self.NUM_LAYERS = 2
        self.architecture = [self.NUM_QUBITS, self.NUM_LAYERS]
        self.model = Quantum_Model(self.architecture)  
    def get_parameters(self):
        return self.model.parameters
    def load_parameters(params):
        self.model.load_parameters(params)      
    def choose_action(self, state, observation_):
        r = np.random.random()
        if r < 1 - self.EPSILON:
            values = np.array([self.model.forward(state, action) for action in self.actionSpace])
            best = np.random.choice(np.where(values==values.max())[0])
            action = self.actionSpace[best]  
        else:
            action = np.random.choice(self.actionSpace)   
        return action    
    def choose_optimal_action(self, state, observation_):
        values = np.array([self.model.forward(state, action) for action in self.actionSpace])
        best = np.random.choice(np.where(values==values.max())[0])
        action = self.actionSpace[best]  
        return action     
    def process_observation(self, observation):
        qbit_params_ = []
        qbit = {}
        for i in range(len(self.MIN_MAX)):
            qbit["id"] = i
            qbit["theta"] = ((int(observation[i]) - self.MIN_MAX[i][0])/self.MIN_MAX[i][1])*pi
            qbit_params_.append(qbit)
            qbit = {}
        return qbit_params_          
    def process_episode(self, episode_memory):
        last_visited = False
        for state, action, reward, state_, observation_ in reversed(episode_memory):
            if last_visited:
                label_qsa = self.GAMMA*np.max([self.model.forward(state_, action) for action in self.actionSpace]) + reward
            else:
                label_qsa = reward
                last_visited = True
            y = (action, label_qsa)
            self.experience.push((state, y))     
    def learn(self):
        if len(self.experience) < self.BATCH_SIZE:
            return
        sample = self.experience.sample(self.BATCH_SIZE)
        self.model.train(sample, learning_rate = self.learning_rate)        
    def update_epsilon(self):
        if self.EPSILON - self.EPSILON_STEP > 0:
            self.EPSILON -= self.EPSILON_STEP
        else:
            self.EPSILON = 0         
    def print_parameters(self):
        print(self.model.parameters)
 


        
###################################################################################################################
############################## DEEPQ AGENT CLASS ##################################################################
###################################################################################################################    
class Memory():
    def __init__(self, capacity):
        self.c = capacity
        self.storage = []
        self.placement = 0    
    def push(self, move):
        if len(self.storage) < self.c:
            self.storage.append(move)
        else:
            self.storage[self.placement] = move
            self.placement = (self.placement + 1)%self.c         
    def sample(self, batchsize):
        return random.sample(self.storage, batchsize)  
    def reset(self):
        self.storage = []
        self.placement = 0  
    def __len__(self):
        return len(self.storage)
def process_sample(sample):
    X = sample[0][0]
    Y = sample[0][1]
    first = True
    for state, label in sample:
        if not first:
            X = np.concatenate((X, state), axis = 1)
            Y = np.concatenate((Y, label), axis = 1)
        else:
            first = False
    return X, Y
class DeepQ_Agent(Memory, Markov, DeepQ_Model):
    def __init__(self, state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate,
                           optimizer):
        self.model_free = model["free"]
        if not self.model_free:
            self.environment = model["environment"]

        self.actionSpace = actionSpace
        self.output_size = len(actionSpace)
        self.input_size = len(state_range)
        self.EPSILON = initial_epsilon
        self.EPSILON_STEP = epsilon_step
        #self.BATCH_SIZE = 10
        #self.EXPERIENCE_CAPACITY = 50
        #para finite complicado 
        self.BATCH_SIZE = 500
        self.EXPERIENCE_CAPACITY = 5000
        self.experience = Memory(self.EXPERIENCE_CAPACITY)
        self.final_experience = Memory(self.BATCH_SIZE)
        self.exp_size = 4
        self.architecture = [self.input_size, 20, 10, 5, self.output_size]
        self.model = DeepQ_Model(self.architecture)
        self.GAMMA = gamma  
        self.learning_rate = learning_rate
        self.optimizer = optimizer
    def get_parameters(self):
        return self.model.params
    def load_parameters(params):
        self.model.load_parameters(params)      
    def choose_action(self, state, observation):
        r = np.random.random()
        if r < 1 - self.EPSILON:
            values = self.values(state, observation)
            best = np.random.choice(np.where(values==values.max())[0])
            action = self.actionSpace[best]  
        else:
            action = np.random.choice(self.actionSpace)   
        return action    
    def choose_optimal_action(self, state, observation):
        values = self.values(state, observation)
        best = np.random.choice(np.where(values==values.max())[0])
        action = self.actionSpace[best]    
        return action    
    def process_observation(self, observation):
        state = []
        for element in observation:
            state.append(np.double(element))
        state = [state]
        return np.transpose(np.array(state))       
    def process_episode(self, episode_memory):
        last_visited = False
        for state, action, reward, state_, observation in reversed(episode_memory):
            label_qsa = self.model.forward(state)
            if last_visited:
                label_qsa[action] = self.GAMMA*np.max(self.values(state_, observation)) + reward
            else:
                label_qsa[action] = reward
                last_visited = True 
                self.final_experience.push((state, label_qsa))
            self.experience.push((state, label_qsa))    
    def learn(self):
        if len(self.experience) < self.BATCH_SIZE:
            return
        #if len(self.final_experience) == self.BATCH_SIZE:
            #sample = self.final_experience.sample(self.BATCH_SIZE)
            #self.final_experience.reset()
        else:
            sample = self.experience.sample(self.BATCH_SIZE)
        X, Y = process_sample(sample)
        self.model.train(X, Y, learning_rate = self.learning_rate, max_iterations = 300, print_cost = False, optimizer = self.optimizer)
        #self.model.train(X, Y, learning_rate = 0.1, max_iterations = 300, print_cost = True, optimizer = 'stochastic')       
    def update_epsilon(self):
        if self.EPSILON - self.EPSILON_STEP > 0:
            self.EPSILON -= self.EPSILON_STEP
        else:
            self.EPSILON = 0    
    def values(self, state, observation):
        values = self.model.forward(state)
        if self.model_free:
            pass
        else:
            markov_dict = Markov(self.environment).dictionary(observation)
            hit_value = 0
            for key, value in markov_dict.items():
                state_values = self.model.forward(self.process_observation(key))
                state_max = np.max(state_values)
                if key[0] == 0:
                    state_max = 0
                hit_value += value*state_max
            values[0] = hit_value
        return values
    
    
    
###################################################################################################################
############################## POLICY GRADIENT CLASS ##############################################################
###################################################################################################################
        
class Policy_Gradient_Agent(Policy_Gradient_NN):
    def __init__(self, state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate, activation="softmax", reward_method="reinforce"):
        self.reward_method,self.activation=reward_method,activation
        self.model_free = model["free"]
        if not self.model_free:
            self.environment = model["environment"]
        self.input_size = len(state_range)        
        architecture = [self.input_size, 40, 40]
        self.learning_rate = learning_rate
        self.model = Policy_Gradient_NN(architecture,self.activation,learning_rate) #architecture,activation,learning_rate
        self.GAMMA = gamma
        self.EPSILON = initial_epsilon
        self.EPSILON_STEP = epsilon_step
        
    def choose_action(self, state, observation): 
        self.model.X=state # NN input update
        AL, cache=self.model.Forward()
        if self.activation == "sigmoid":
            probs=[float(AL), 1-float(AL)]
            action = int(np.random.choice([0, 1], 1, p = probs))
            delta_log = (1-action)*(1-probs[0])- action*probs[0]
        if self.activation == "softmax":
            probs=[float(AL[0]),float(AL[1])]            
            action = int(np.random.choice([0, 1], 1, p = probs))
            delta_log = np.array([[1 - probs[0]], [-probs[1]]])*(action == 0) + np.array([[-probs[0]], [1 - probs[1]]])*(action == 1)
        return (action, delta_log, cache)
    
    def choose_optimal_action(self, state, observation): # why both?
        self.model.X=state # NN input update
        AL, cache=self.model.Forward()
        if self.activation == 'sigmoid':
            probs=[float(AL), 1-float(AL)]
        if self.activation == 'softmax':
            probs=[float(AL[0]),float(AL[1])]            
        action =int(1-round(probs[0]))
        return action   
    
    def process_observation(self, observation):
        state = []
        for element in observation:
            state.append(np.double(element))
        state = [state]
        return np.array(np.transpose(np.array(state)))
    
    def learn(self,state, action, caches, delta_log, reward,G_t):
        self.model.X=state
        loss=-np.array(delta_log)*G_t 
        grads_W, grads_b = self.model.Backward(loss,caches)
        self.model.W,self.model.b=self.model.Update(grads_W,grads_b)           
    
    def discounted_value(self, batch_memory):# episode_memory=[state, action=(action,delta_log,caches), reward, state_, observation_]
        discounted_rewards,delta,R,E = [],[],[],[]
        # discounted value - reinforce        
        for experience in batch_memory:
            rewards=[r for s,_,r,_,_ in experience]
            R+=rewards; E+=experience
            for t in range(len(rewards)):
                Gt,pw  = 0 ,0
                for r in rewards[t:]:
                    Gt = Gt + self.GAMMA**pw * r
                    pw = pw + 1
                discounted_rewards.append(Gt)
        G=(discounted_rewards-np.mean(discounted_rewards))/np.std(discounted_rewards) #reduce variance
        # TD residual
        V_t=0
        for r_t,V_tp1 in zip(R,discounted_rewards): # or take G=discounted_rewards and d=r+self.gamma*V_t-V_tm1
            d=r_t+self.GAMMA*V_tp1-V_t
            V_t=V_tp1
            delta.append(d)
        if self.reward_method=='reinforce':
            return G,E
        if self.reward_method=='TD':
            return (delta-np.mean(delta))/np.std(delta),E #(delta-np.mean(delta))/np.std(delta)
        
    def process_episode(self, batch_memory):
        G,E=Policy_Gradient_Agent.discounted_value(self,batch_memory)
        for (state, (action,delta_log,caches), reward, state_, observation_), G_t in zip(E, G):
            Policy_Gradient_Agent.learn(self,state, action, caches, delta_log, reward,G_t)  
        
    

    
###################################################################################################################
############################## INFINTIE DECK CLASS ################################################################
###################################################################################################################  
def depurate_stateSpace(sS):
    depurated = []
    for state in sS:
        if state[0] < 11 and state[1]:
            pass
        else:
            depurated.append(state)
    return depurated 
def generate_stateSpace(state_range):
    stateSpace = list(itertools.product(*state_range))
    return stateSpace
class BJ_infinite_game():
    #the constructor does not take any agrument as input   
    def __init__(self):
        #this variables of the hand / episode are initialized
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10 ,10, 10]
        self.croupier = []
        self.player = []
        self.actionSpace = [0, 1]
        self.state_range = [21, 1]
        self.usable_ace = False
        self.finished = False
        self.sumSpace = [i for i in range(0, 22)]
        self.aceSpace = [False, True]
        self.state_range = []
        self.state_range.append(self.sumSpace)
        self.state_range.append(self.aceSpace)
        self.stateSpace = depurate_stateSpace(generate_stateSpace(self.state_range))
    #method returns true if there is an ace and it should be used
    def use_ace(self, cards):  # Does this hand have a usable ace?
        return 1 in cards and sum(cards) + 10 <= 21
    #counts the value of the hand
    def count(self, cards):
        if self.use_ace(cards):
            sum_ = sum(cards) + 10
        else:
            sum_ = sum(cards)    
        if sum_ <= 21:
            return sum_
        else:
            return 0
    #changes the object according to the action "Hit", and returns observation, reward and whether the game is finished or not    
    def hit(self):
        self.player.append(np.random.choice(self.cards))
        if self.use_ace(self.player):
            self.usable_ace = True
        count = self.count(self.player)
        if count == 0 or count == 21:
            reward = count*count/441
            count = 0
            self.finished = True
        else:
            reward = 0
        observation = (count, self.usable_ace)    
        return observation, reward, self.finished
    #changes the object according to the action "stick", and returns observation, reward and finished
    def stick(self):
        self.finished = True
        count = self.count(self.player)
        reward = count*count/441
        observation = (count, self.usable_ace)  
        return observation, reward, self.finished
    #this is the main method of interaction. until a game is finished, the agent continuously decides to either hit or stick
    #by calling game.move(0) or game.move(1)
    def move(self, action):
        if action == 0:
            return self.hit()
        if action == 1:
            return self.stick()

        
        
        
        
###################################################################################################################
############################## FULL FINTIE DECK CLASS #############################################################
###################################################################################################################  

def generate_stateSpace(state_range):
    stateSpace = list(itertools.product(*state_range))
    return stateSpace
class BJ_finite_full:  
    #the constructor takes as input the number of decks (D) and the mode of playing:
    #
    #all_knwon: the player has a memory of how many of each card has already been handed in that episode
    #
    #counting_cards: the player has access to the card counting (as we discussed, definition of counting can be found
    #in count_cards methods)
    #
    #all_hidden: the player only has access to the current hand
    def __init__(self, D, mode):
        valid = mode in ['all_known', 'counting_cards', 'all_hidden']
        try:
            if not valid:
                a = int('d')
        except:
            print("that is not a valid mode. Valid modes are \'all_hidden\', \'counting_cards\' and \'all_known\', ")
        self.mode = mode
        #this number of decks and total cards are initialized
        self.num_decks = D
        self.N = 52*self.num_decks
        self.left_number = 52*self.num_decks
        self.total_reward = 0
        self.reward = 0
        #for elegance, the environment provides with the full action and state space. this is informative to
        #the agent so that it does need to be coded in each one
        self.left_number = 52*self.num_decks
        self.state_range = []
        self.sumSpace = [i for i in range(2, 22)]
        self.aceSpace = [False, True]
        self.accumSpace = [i for i in range(self.num_decks*7952)]
        self.state_range.append(self.sumSpace)
        self.state_range.append(self.aceSpace)
        self.state_range.append(self.accumSpace) 
        #We need to include more decks
        if self.mode == 'all_known':
            num_of_cards = 4*self.num_decks
            num_of_cards_10 = 16*self.num_decks
            range_ = [i for i in range(num_of_cards)]
            for figure in range(9):
                self.state_range.append(range_)
            range10_ = [i for i in range(num_of_cards_10)]
            self.state_range.append(range10_)
        elif self.mode == 'counting_cards':
            counting_range = [i for i in range (0, 20*self.num_decks)]
            self.state_range.append(counting_range)
            self.state_range.append(counting_range)  
        #this variables of the episode are initialized
        self.left_cards = np.array([4,4,4,4,4,4,4,4,4,16])*self.num_decks #nb of cards per value 
        self.player = []
        self.left_number = 52*self.num_decks
        self.usable_ace = False
        self.episode_finished = False 
        self.actionSpace = [0, 1]
        if self.mode == 'counting_cards':
            self.high_cards = 0
            self.low_cards = 0
            minmax = 20*self.num_decks
            self.countedSpace = [i for i in range(-minmax, minmax + 1)]
    def get_state_range(self):
        return self.state_range          
    def get_state_space(self):
        return generate_stateSpace(self.state_range)    
    #same thing for action space
    def get_action_space(self):
        return self.actionSpace   
    #this function tells the environment wether it has a usable ace or not
    def use_ace(self, cards):
        return 1 in cards and sum(cards) + 10 <= 21   
    #this function counts the sum of a hand (not the reward)
    def count(self, cards):
        if self.use_ace(cards):
            sum_ = sum(cards) + 10
        else:
            sum_ = sum(cards)
            
        if sum_ <= 21:
            return sum_
        else:
            return 0     
    def count_left(self):
        self.left_number = np.sum(self.left_cards)
    #this method checks wether the episode is finished. this is done after every handed card
    def check_episode(self):
        if np.max(self.left_cards) == 0:
            self.reward = self.total_reward
            self.episode_finished = True
    #this function is called when the environemnt is in counting_cards mode
    def count_cards(self, card):
        if card in [2, 3, 4, 5, 6]:
            self.low_cards += 1
        if card in [1, 10]:
            self.high_cards += 1
    #this function returns the observation value depending on the environment mode
    def observe(self, count, card):
        self.count_left()
        if self.mode == 'counting_cards':
            self.count_cards(card)
            observation = (count, self.usable_ace, self.high_cards, self.low_cards)
        elif self.mode == 'all_known':
            observation = [count, self.usable_ace, self.total_reward]
            for number in self.left_cards:
                observation.append(number)
            observation = tuple(observation)
        else:
            observation = (count, self.usable_ace)       
        return observation
    def draw(self):
        prob = [i/np.sum(self.left_cards) for i in self.left_cards]
        prob_cumsum = np.cumsum(prob)
        rand = random.random()
        card = np.min(np.argwhere(prob_cumsum > rand))
        self.player.append(card + 1)
        self.left_cards[card] -= 1
        self.check_episode()
        if self.use_ace(self.player):
            self.usable_ace = True
        else:
            self.usable_ace = False
        count = self.count(self.player)   
        return count, card
    #this method updates the game according to a hit action, and returns next state, obtained reward, and whether
    #the episode is finished or not
    def hit(self):
        count, card = self.draw()
        if count == 0 or count == 21:
            self.total_reward += count*count/441
            self.hand_finished = True
            self.player = []
            if not self.episode_finished:
                count, card = self.draw()
        if self.episode_finished:
            count = self.count(self.player)
            self.total_reward += count*count/441
            self.reward += count*count/441
            #self.reward += count*count/441     
        #the observation aka state depends on the mode  
        observation = self.observe(count, card + 1) 
        #after every card is dealt, the environment checks whether there are any cards left and updates self.episode_finished
        return observation, self.reward, self.episode_finished
    def stick(self):
        count = self.count(self.player)
        self.total_reward += count*count/441
        self.player = []
        count, card = self.draw()
        observation = self.observe(count, card + 1)
        return observation, self.reward, self.episode_finished
    #this is the main method of interaction. until a game is finished, the agent continuously decides to either hit or stick
    #by calling game.move(0) or game.move(1)
    def move(self, action):
        self.reward = 0
        self.hand_finished = False
        if action == 0:
            return self.hit()
        if action == 1:
            return self.stick()   
    def inform(self):
        print(self.left_cards)
        
        

        
###################################################################################################################
############################## MAIN  ##############################################################################
###################################################################################################################
def get_best_finite():
    print("computing best game for FD")
    def objective(x):
        sum_ = 0
        for i in x:
            sum_ -= i**2
        return sum_
    def constraint_1(x):
        sum_ = 380
        for i in x:
            sum_ -= i
        return sum_
    b = (0.0, 21.0)
    bnds = []
    for i in range(52):
        bnds.append(b)
    cons1 = {'type': 'eq', 'fun': constraint_1}
    cons = [cons1]
    x0 = 21*np.random.rand(52)
    sol = minimize(objective, x0, method = 'SLSQP', bounds = bnds, constraints = cons)
    clean = np.array(np.round(sol['x']))
    unique, counts = np.unique(clean, return_counts=True)
    best_game = dict(zip(unique, counts))
    for key, item in best_game.items():
        print("score in action: ", key, "number of times: ", item)
    return
def new_game(episode):
    environment = episode['environment']
    num_decks = episode['num_decks']
    if environment == 'infinite_deck':
        game = BJ_infinite_game()
    if environment == 'finite_deck' or environment == 'finite_pp':
        game = BJ_finite_full(num_decks, 'all_known')
    return game
def learn_blackjack_pg(agent, episode, num_episodes, batch_size):
    all_rewards = []
    past_rewards = []
    batch_memory = []          
    for i in range(1, num_episodes + 1):
        percentage = i/num_episodes*100
        if percentage%30 == 0:
            print("% episodes observed: ", percentage, '\n')
            print("mean past reward: ", np.mean(past_rewards), '\n\n')
            past_rewards = []   
        game = new_game(episode)
        observation, reward, done = game.move(0)
        state = agent.process_observation(observation)
        R = 0
        episode_memory = []             
        while not done:
            #print("state=",state)
            action = agent.choose_action(state, observation) #action=(action,delta_log,cache)
            observation_, reward, done = game.move(action[0])
            state_ = agent.process_observation(observation_)
            episode_memory.append((state, action, reward, state_, observation_)) 
            state = state_
            observation = observation_
            R += reward
        batch_memory.append(episode_memory)
        if i>batch_size and i%batch_size==0:
            agent.process_episode(batch_memory)
            batch_memory = []                          
        #agent.update_epsilon()
        past_rewards.append(R)
        all_rewards.append(R)
    return agent, all_rewards
def learn_blackjack(agent, episode, num_episodes):
    all_rewards = []
    past_rewards = []
    for i in range(1, num_episodes + 1):
        percentage = i/num_episodes*100
        if percentage%30 == 0:
            print("% episodes observed: ", percentage, '\n')
        if percentage%30 == 0:
            print("mean past reward: ", np.mean(past_rewards), '\n\n')
            #all_rewards.append(np.mean(past_rewards))
            past_rewards = []   
        game = new_game(episode)
        observation, reward, done = game.move(0)
        state = agent.process_observation(observation)
        R = 0
        episode_memory = []  
        while not done:
            action = agent.choose_action(state, observation)
            observation_, reward, done = game.move(action)
            state_ = agent.process_observation(observation_)
            episode_memory.append((state, action, reward, state_, observation_)) 
            state = state_
            observation = observation_
            R += reward     
        agent.process_episode(episode_memory)
        agent.update_epsilon()
        past_rewards.append(R)   
        all_rewards.append(R)
        if i%agent.exp_size == 0:
            agent.learn()
            agent.update_epsilon()
    return agent, all_rewards
def play_blackjack_pg(agent, episode, num_episodes):
    rewards = []
    for i in range(1, num_episodes + 1):
        percentage = i/num_episodes*100
        if percentage%1 == 0:
            print("% episodes played: ", percentage, '\n')         
        game = new_game(episode)
        observation, reward, done = game.move(0)
        state = agent.process_observation(observation)
        R = 0
        episode_memory = []  
        while not done:
            action = agent.choose_optimal_action(state, observation)
            observation_, reward, done = game.move(action[0])
            state_ = agent.process_observation(observation_)
            episode_memory.append((state, action, reward, state_, observation_)) 
            state = state_
            observation = observation_
            R += reward         
        rewards.append(R)             
    print("mean reward: ", np.mean(rewards), '\n\n')
    return rewards
def play_blackjack(agent, episode, num_episodes):
    rewards = []
    for i in range(1, num_episodes + 1):
        percentage = i/num_episodes*100
        if percentage%20 == 0:
            print("% episodes played: ", percentage, '\n')         
        game = new_game(episode)
        observation, reward, done = game.move(0)
        state = agent.process_observation(observation)
        R = 0
        episode_memory = []  
        while not done:
            action = agent.choose_optimal_action(state, observation)
            observation_, reward, done = game.move(action)
            state_ = agent.process_observation(observation_)
            episode_memory.append((state, action, reward, state_, observation_)) 
            state = state_
            observation = observation_
            R += reward         
        rewards.append(R)             
    print("mean reward: ", np.mean(rewards), '\n\n')
    return rewards
def run_experiment(experiment):   
    episode = experiment['episode']
    training_episodes = experiment['training_episodes']
    testing_episodes = experiment['testing_episodes']
    agent_type = experiment['agent']
    gamma = experiment['gamma']
    initial_epsilon = experiment['initial_epsilon']
    epsilon_step = experiment['epsilon_step']
    learning_rate = experiment['learning_rate']
    optimizer = experiment['optimizer']
    final_layer = experiment['final_layer']
    environment = episode['environment']
    game = new_game(episode)
    if environment == 'infinite_deck':
        stateSpace = game.stateSpace
    actionSpace = game.actionSpace
    state_range = game.state_range  
    if agent_type == 'Random_Agent':
        agent = Random_Agent(actionSpace)
    if  agent_type == 'PP_Agent':
        agent = PP_Agent(stateSpace, model)  
    if agent_type == 'Monte_Carlo_Agent':
        agent = Monte_Carlo_Agent(stateSpace, actionSpace, model, gamma, initial_epsilon, epsilon_step)    
    if agent_type == 'Q_Agent':
        agent = Q_Agent(stateSpace, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate)    
    if agent_type == 'Quantum_Agent':
        agent = Quantum_Agent(state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate) 
    if agent_type == 'DeepQ_Agent':
        agent = DeepQ_Agent(state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step, learning_rate, optimizer)
    if agent_type == 'Policy_Gradient_Agent':
        agent = Policy_Gradient_Agent(state_range, actionSpace, model, gamma, initial_epsilon, epsilon_step,
                                      learning_rate, optimizer)
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    playing_rewards = play_blackjack(agent, episode, testing_episodes)
    return trained_agent, learning_rewards, playing_rewards


def save_results(rewards, experiment_name):
    results = {}
    results['rewards'] = rewards
    path = os.path.join(os.getcwd(), 'Graphs', experiment_name)
    with open(path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_results(experiment_name):
    path = os.path.join(os.getcwd(), 'Graphs', experiment_name)
    with open(path, 'rb') as handle:
        results = pickle.load(handle)
    rewards = results['rewards']
    return rewards

def plot_rewards(R,window,title): #R=[[reward,"label_reward"],[],[]]
    plt.figure('reward',figsize=(6,5))
    color_plot=["b","r","g","yellow","orange","lime","m","olive","indigo"]
    for (reward,label),c in zip(R,color_plot):
        iteration=np.arange(0,len(reward),window)
        mean_reward=np.array([np.mean(reward[i:j]) for (i,j) in zip(iteration[:-1],iteration[1:])])
        std_reward=np.array([np.std(reward[i:j]) for (i,j) in zip(iteration[:-1],iteration[1:])])           
        plt.plot(iteration[:-1], mean_reward,linewidth=3,color=c,label=label)
        plt.fill_between(iteration[:-1], mean_reward-std_reward , mean_reward+std_reward,facecolor=c,alpha=0.2)
    plt.legend()
    plt.title(title,fontsize=15)
    plt.xlabel("Episodes",fontsize=14)
    plt.ylabel("Rewards/episode",fontsize=14)
    plt.show()
    

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    
    episode = {}
    episode['environment'] = 'infinite_deck'
    episode['num_decks'] = 1
    model = {}
    model['free'] = False
    model['environment'] = episode['environment']

    game = new_game(episode)
    if model['environment'] == 'infinite_deck':
        stateSpace = game.stateSpace
        
    infinite_episode = {}
    infinite_episode['environment'] = 'infinite_deck'
    infinite_episode['num_decks'] = 1
    infinite_game = new_game(infinite_episode)
    stateSpace = infinite_game.stateSpace
    actionSpace = game.actionSpace
    state_range = game.state_range
        


    testing_episodes = 100000
    agent = PP_Agent(stateSpace, model)
    playing_rewards = play_blackjack(agent, episode, testing_episodes)
    print("mean of pp agent: ", np.mean(playing_rewards))
    print("std of pp agent: ", np.std(playing_rewards))
  

    training_episodes = 3000
    agent = Monte_Carlo_Agent(stateSpace, actionSpace, model, gamma = 0.999999, 
                              initial_epsilon = 0.5, epsilon_step = 3e-3)
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    plot_rewards([[learning_rewards,"MC_Agent"]], window=20, title = 'MC Agent (ID)')
    
    
    agent = Q_Agent(stateSpace, actionSpace, model, gamma = 0.999999, 
                    initial_epsilon = 0.5, epsilon_step = 3e-4, learning_rate = 0.05)
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    plot_rewards([[learning_rewards,"Q_Agent"]], window=20, title = 'Q Agent (ID)')

    agent = DeepQ_Agent(state_range, actionSpace, model, gamma = 0.999999, initial_epsilon = 0.5, 
                         epsilon_step = 3e-4, learning_rate = 0.001, optimizer = 'regular')
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    plot_rewards([[learning_rewards,"DeepQ_Agent"]], window=20, title = 'DeepQ Agent (ID)')
    

    training_episodes = 150
    agent = Quantum_Agent(state_range, actionSpace, model, gamma = 0.999999, initial_epsilon = 0.5, 
                          epsilon_step = 1e-2, learning_rate = 0.3)
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    plot_rewards([[learning_rewards,"Quantum_Agent"]], window=20, title = 'Quantum Agent (ID)')
    
    training_episodes = 10000
    agent = Policy_Gradient_Agent(state_range, actionSpace, model, gamma = 0.999999, 
    initial_epsilon = 0.5, epsilon_step = 2e-5, learning_rate = 0.01, activation="softmax",
    reward_method="reinforce")
    trained_agent, learning_rewards = learn_blackjack_pg(agent, episode, training_episodes, 5)
    plot_rewards([[learning_rewards,"PG_Agent"]], window=20, title = 'PG Agent (ID)')
    



    get_best_finite()
    

    testing_episodes = 100
    
    
    episode = {}
    episode['environment'] = 'finite_pp'
    episode['num_decks'] = 1
    
    model = {}
    model['free'] = False
    model['environment'] = episode['environment']
    
    agent = PP_Agent(stateSpace, model)
    playing_rewards = play_blackjack(agent, episode, testing_episodes)
    print("mean of pp agent (FD): ", np.mean(playing_rewards))
    print("std of pp agent: (FD): ", np.std(playing_rewards))


    
    game_ = new_game(infinite_episode)
    stateSpace = infinite_game.stateSpace
    actionSpace = game.actionSpace
    state_range = game.state_range

    episode = {}
    episode['environment'] = 'finite_deck'
    episode['num_decks'] = 1 
    
    model = {}
    model['free'] = False
    model['environment'] = episode['environment']
    
    game_ = new_game(episode)
    actionSpace = game_.actionSpace
    state_range = game_.state_range
    
    training_episodes = 10000
    
    agent = Policy_Gradient_Agent(state_range, actionSpace, model, gamma = 0.999999, initial_epsilon = 0.5, epsilon_step = 3e-5, learning_rate = 0.01, activation="softmax", reward_method="reinforce")
    trained_agent, learning_rewards = learn_blackjack_pg(agent, episode, training_episodes, 5)
    plot_rewards([[learning_rewards,"PG_Agent"]], window=20, title = 'PG Agent (FD)')
    
    
    agent = DeepQ_Agent(state_range, actionSpace, model, gamma = 0.999999, initial_epsilon = 0.5, epsilon_step = 3e-5, learning_rate = 0.01, optimizer = 'regular')
    trained_agent, learning_rewards = learn_blackjack(agent, episode, training_episodes)
    plot_rewards([[learning_rewards,"DeepQ_Agent"]], window=20, title = 'DeepQ Agent (FD)')