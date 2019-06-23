#!/usr/bin/env python
# coding: utf-8

# # TODO
# 
# * DONE :) Fijate si podes imprimir el espacio de Q a medida que avanza en el training
# 

# In[3]:


# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# for importing notebooks (.ipynb) as regular .py
import import_ipynb
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np
import pprint


# In[4]:


def random_argmax(v):
    """Like np.argmax(), but if there are several "best" actions,
       chooses and returns one randomly"""
    arguments = np.argwhere(v == np.amax(v)).ravel()
    return np.random.choice(arguments)


# In[5]:


# From layout of chars to layout of numbers
# Beware: This will be painful to see
def ascii_state_to_numeric_state(ascii_state):
    str_state = str(ascii_state)
    score_pos = str(str_state).find("Score: ")
    ascii_map = str(str_state)[:score_pos-1]

    numer_map = np.ndarray(len(ascii_map)+1, dtype=np.double)
    for i, c in enumerate(ascii_map):
        if c==' ':
            numer_map[i] = 1
            continue
        if c=='%':
            numer_map[i] = 2
            continue
        if c=='.':
            numer_map[i] = 3
            continue
        if c=='\n':
            numer_map[i] = 4
            continue
        if c=='G':
            numer_map[i] = 5
            continue
        if c=='o':
            numer_map[i] = 6
            continue
        # Pacman dirs
        if c=='<':
            numer_map[i] = 7
            continue
        if c=='>':
            numer_map[i] = 8
            continue
        if c=='^':
            numer_map[i] = 9
            continue
        if c=='v':
            numer_map[i] = 10
            continue
    numer_map /= 15.0
    #last array position will contain the score
    numer_map[-1] = float(str_state[score_pos+7:])/3000
    return numer_map


# ## Tabular Q Learning
# 
# Q(s,a) is a dictionary with each state-action value it visits  

# In[6]:


class QLearningAgent(ReinforcementAgent):
    # Parent class in learningAgents.py
    """ Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        "*** YOUR CODE HERE ***"
        #self.Q = {}
        self.Q = Counter()
        self.q_size_history = np.ndarray(self.numTraining//10)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # max_a(Q[state, all actions])
        legalActions = self.getLegalActions(state)
        if not legalActions:
            value=0.0
        else:
            # TODO: Find a better way
            value=max([self.getQValue(state, a) for a in legalActions])
        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            action=None
        else:
            # TODO: Find a better way
            action=legalActions[random_argmax([self.getQValue(state, a) for a in legalActions])]
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            #action = None
            return None
        "*** YOUR CODE HERE ***"
        # epsilon decay
        epsmin = 0.01
        eps_decay = 0.9999
        self.epsilon = max(self.epsilon*eps_decay, epsmin)
        if util.flipCoin(self.epsilon):
            # Act randomly
            action = random.choice(legalActions)
        else:
            # Act greedly
            action = self.computeActionFromQValues(state)
        print("habia una vez un bu!")
        return action
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        iteration = self.episodesSoFar
        self.alpha = 1/np.power((iteration+1), 1) # alpha decay
        alpha = self.alpha
        gamma = self.discount
        # -----------------------------v revisar si calculo maximo Q
        estimation = reward + gamma*self.computeValueFromQValues(nextState)
        self.Q[(state, action)] += alpha*(estimation - self.Q[(state, action)])
        import pprint
        pprint.pprint("lalala")
        pprint.pprint("Q (table) size: "+str(len(self.Q)))#, end="\r")

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


# In[7]:


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate/step size
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        #print("Q (table) size: "+str(len(self.Q)), end="\r")
        return action
    
    


# ## Linear Aproximation for Q learning
# 
# Sutton and Barto Ch.9.5 - p.232 *"Feature construction for linear methods"*
# 
# We want to generalize Q values better for different state-action pairs.
# 
# Intuition:
# 
# * If a ghost is close to pacman at one state and dies, we want to generalize "danger" to any other position where a ghost is close.
# 
# 
# Observations:
# 
# * Linear approximators **can't** find relationships between features, so we need to combine them ourselves (if we want that)
#  
#  eg:
#  feature "dist_x" represents horizontal distance to ghost,
#  feature "dist_y" represents vertical distance to ghost,
#  
#  A linear approximator cannot learn if a ghost is close on a plane, because it cannot make operations inbetween features to get a combined value.
#  
#  To solve this, we can add a third feature that combines the other two:
#  
#  *feature\[ "dist_xy" \] = dist_x $*$ dist_x*
# 

# $n $ : number of features
# 
# $Q(s,a) = \sum\limits_{i=1}^n f_i(s,a) * w_i$
# 
# **Prediction error:**
# 
# $advantage = (R + \gamma \max\limits_{a} Q(S', a)) - Q(S,A)$
# 
# **Update:**
# 
# $w_i \leftarrow w_i + \alpha \cdot advantage \cdot f_i(S,A)$
# 

# In[8]:


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        #extractor = 'CoordinateExtractor'
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        featureDict = self.featExtractor.getFeatures(state, action)
#         for feat in featureDict.keys():
#             self.weights[feat]*featureDict[feat]
        #print("aprox Q value: ", np.dot(self.weights, featureDict))
        return np.dot(self.weights, featureDict)
    
    def getMaxQValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # max_a(Q[state, all actions])
        legalActions = self.getLegalActions(state)
        if not legalActions:
            value=0.0
        else:
            # TODO: Find a better way
            value=max([self.getQValue(state, a) for a in legalActions])
        return value
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        iteration = self.episodesSoFar
        self.alpha = 1/np.power((iteration+1), 1) # alpha decay
        alpha = self.alpha
        gamma = self.discount
        #state = str(state)
        featureDict = self.featExtractor.getFeatures(state, action)
        #for key,feat in 
        
        pastVal = self.getQValue(state, action)
        advantage = reward + gamma*self.getMaxQValue(nextState) - pastVal
        for feature in featureDict.keys():
            #print("state: ", state, " action: ", action)
            self.weights[feature] += alpha * advantage * featureDict[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            
#             print("Weights:")
#             pprint.pprint(self.weights)
#             print("Features:")
#             for k in self.weights.keys():
#                 state, action = k
#                 pprint.pprint(self.featExtractor.getFeatures(state, action))

#             print(len(self.weights))
            pass


# **TODO:**
# 
# * Linear model with experience replay 1:18 en lecture 5RL de Hado
# * Linear model with LSTD for solving instantly best parameter for that history
# * Non-linear model

# ## LSTD (still empty)

# ![](../img/lstd-pseudocode.png)

# For number of features: 100s-1000s
# 
# For millones of features it becomes unwieldy

# In[9]:


# TODO: 
# make it work
# add forgetting
class LSTDAgent(PacmanQAgent):
    pass


# ## Episodic Semi-gradient Sarsa for Estimating Q*(s,a)

# ![](../img/episodic-semi-gradient-sarsa.png)

# #### TODO / TOREMEMBER
# * Con el aproximador lineal tenia
# 
#  Q(s,a) = $\sum$ feature(s,a)*wi
# 
# 
# * Acá voy a tener una NN que tiene :
# 
# 
# 1. de entrada 2 valores: state-action pair
# 2. hidden layers no se
# 3. output layer cantidad de features
# 
# 
# * Ej: 
# 
# 
# 1. Voy a necesitar predecir valores a partir de valores de entrada
# 2. Voy a necesitar los gradientes para pesar la Advantage Function
# 3. Voy a necesitar actualizar los pesos de mi red neuronal <- tal vez lo pueda definir en la red

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
#dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        # input: 147 chars from state and 1 from action taken
        self.fc1 = nn.Linear(148, 100)  # 
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


# In[2]:


class NNQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        #extractor = 'CoordinateExtractor'
        #self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        #self.weights = util.Counter()
        self.net = self.initNN()
        #self.net = self.net.to('cuda:0')
        # to float; test with double later
        #self.net = self.net.float() 
        
        
    def initNN(self):
        net = Net()
#         torch.nn.init.xavier_uniform(net.fc1.weight.data)
#         torch.nn.init.xavier_uniform(net.fc2.weight.data)
#         torch.nn.init.xavier_uniform(net.fc3.weight.data)
        torch.nn.init.uniform_(net.fc1.weight.data)
        torch.nn.init.uniform_(net.fc2.weight.data)
        torch.nn.init.uniform_(net.fc3.weight.data)
        torch.nn.init.uniform_(net.fc1.bias.data)
        torch.nn.init.uniform_(net.fc2.bias.data)
        torch.nn.init.uniform_(net.fc3.bias.data)
        #torch.nn.init.xavier_uniform(net.weight)
        #net.bias.data.fill_(0.01)
        # Create random Tensors for weights.
        # Setting requires_grad=True indicates that we want to compute gradients with
        # respect to these Tensors during the backward pass.
#         self.w1 = torch.randn(D_in, H1, device=device, dtype=dtype, requires_grad=True)
#         self.w2 = torch.randn(H1,   H2, device=device, dtype=dtype, requires_grad=True)
#         self.w3 = torch.randn(H2, D_out, device=device, dtype=dtype, requires_grad=True)
        return net

#     def random_argmax(self, tens):
#         """Like np.argmax(), but if there are several "best" actions,
#            chooses and returns one randomly.
#            Works with tensors"""
#         arguments = np.argwhere(tens == torch.max(tens)[0]).ravel()
#         return np.random.choice(arguments)
    
    
    def getQValue(self, state, action, compute_grad=False):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #featureDict = self.featExtractor.getFeatures(state, action)
#         for feat in featureDict.keys():
#             self.weights[feat]*featureDict[feat]
        #print("aprox Q value: ", np.dot(self.weights, featureDict))
        #return np.dot(self.weights, featureDict)
        #numer_state = state.deepCopy()
        numer_state = ascii_state_to_numeric_state(state)
        actions = {'North': [1./6],
                   'South': [2./6],
                   'East' : [3./6],
                   'West' : [4./6],
                   'Stop' : [5./6]}
        numer_action = actions[action]
        input_data = torch.Tensor(np.concatenate((numer_state, numer_action)))#.type(torch.DoubleTensor)
        #print("data",input_data, type(input_data))
        if not compute_grad:
            # Do not compute grad
            with torch.no_grad():
                out_q = self.net(input_data)
        else:
            # Leave trace for calculate grad later
            out_q = self.net(input_data)
        
        return out_q

    
    def computeActionFromNN(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            action=None
        else:
            # TODO: Find a better way
            #action=legalActions[random_argmax([self.getQValue(state, a) for a in legalActions])]
            #numer_state = state.deepCopy()
            numer_state = ascii_state_to_numeric_state(state)
            actions = {'North':[1./6],
                       'South':[2./6],
                       'East' :[3./6],
                       'West' :[4./6],
                       'Stop' :[5./6]}
            #print(numer_state)
            #print(actions['East'])
            #input_data = np.concatenate((numer_state, actions['East']))
            #something = torch.from_numpy(input_data.astype(dtype=np.double))
            
            #all_q_s_values = [self.net(torch.from_numpy(np.concatenate((numer_state, actions[a])))) for a in legalActions]
            #all_q_s_values = [self.net(torch.from_numpy(np.concatenate((numer_state, actions[a]))).type(torch.DoubleTensor)) for a in legalActions]
            #input_data = torch.Tensor(np.concatenate((numer_state, actions['East'])))
            #print("input:")
            #print(input_data)
            #print("net:")
            #print(self.net(input_data))
            all_q_s_values = np.ndarray(len(legalActions))
            for i, a in enumerate(legalActions):
                input_data = torch.Tensor(np.concatenate((numer_state, actions[a])))
                with torch.no_grad():
                    output = self.net(input_data)
                all_q_s_values[i] = output
            #all_q_s_values = [sum(self.net(torch.Tensor(np.concatenate((numer_state, actions[a]))))) for a in legalActions]
            print("all_q_s_values", all_q_s_values)
            best_action = random_argmax(all_q_s_values)
            action = legalActions[best_action]
            #print("action returned", str(action), type(str(action)))
        return action

    
    def getAction(self, state):
        """
          eps-greedy policy.
          Note that if there are no legal actions,
          which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        #action = None
        "*** YOUR CODE HERE ***"
        # epsilon decay
        epsmin = 0.01
        eps_decay = 0.9999
        #self.epsilon = max(self.epsilon*eps_decay, epsmin)
        self.epsilon = 0.1
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check
        self.epsilon=0.1
        if util.flipCoin(self.epsilon):
            # Act randomly
            action = random.choice(legalActions)
        else:
            # Act greedly
            #action = self.computeActionFromNN(state)
            action = self.computeActionFromNN(state)
        #print("segunda que devuelve action:", action, type(action))
        # Leave trace for calculating grad on update
        #_ = self.getQValue(state, action, compute_grad=True)
        self.doAction(state, action)
        
        return action
    
    
    def getPolQValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # max_a(Q[state, all actions])
        legalActions = self.getLegalActions(state)
        if not legalActions:
            value=0.0
        else:
            # TODO: Find a better way
            policy_action = self.getAction(state)
            value = self.getQValue(state, policy_action)
            #value=max([self.getQValue(state, a) for a in legalActions])
        return value
    
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        iteration = self.episodesSoFar
        self.alpha = 0.000001# 1/(1000*(iteration+1)) # alpha decay
        alpha = self.alpha
        gamma = 0.9#self.discount
        #state = str(state)
        #featureDict = self.featExtractor.getFeatures(state, action)
        #for key,feat in 
        
        #pastVal = self.getQValue(state, action)
        pastVal = self.getQValue(state, action, compute_grad=True)
        #with torch.no_grad():
        advantage = reward + gamma*self.getPolQValue(nextState) - pastVal
            #print(advantage)
        #to_maximize = pastVal
        
        pastVal.backward(-advantage)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #for feature in featureDict.keys():
#         for name, param in self.net.named_parameters():
#             #print("state: ", state, " action: ", action)
#             #self.weights[feature] += alpha * advantage * featureDict[feature]
#             #print("name:",name)
#             #print("param: ", param)
#             with torch.no_grad():
#                 #print("alpha * advantage * param: ", alpha * advantage * param)
#                 param.data = param.data + alpha * advantage * param.data
        with torch.no_grad():
            print("Layer 1:")
            print(self.net.fc1.weight.data)
            print("update:")
            print(alpha * advantage * self.net.fc1.weight.grad)
            
            self.net.fc1.weight.data += alpha * advantage * self.net.fc1.weight.grad
            print("Layer 1 - Updated")
            print(self.net.fc1.weight.data)
            
            self.net.fc2.weight.data += alpha * advantage * self.net.fc2.weight.grad
            self.net.fc3.weight.data += alpha * advantage * self.net.fc3.weight.grad
            self.net.fc1.bias.data   += alpha * advantage * self.net.fc1.bias.grad
            self.net.fc2.bias.data   += alpha * advantage * self.net.fc2.bias.grad
            self.net.fc3.bias.data   += alpha * advantage * self.net.fc3.bias.grad
            
            #self.net.zero_grad()
#             self.net.fc1.weight.grad.zero_()
#             self.net.fc2.weight.grad.zero_()
#             self.net.fc3.weight.grad.zero_()
#             self.net.fc1.bias.grad.zero_()
#             self.net.fc2.bias.grad.zero_()
#             self.net.fc3.bias.grad.zero_()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        #print("state:",state)
        #print("getScore:",state.getScore())
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            
#             print("Weights:")
#             pprint.pprint(self.weights)
#             print("Features:")
#             for k in self.weights.keys():
#                 state, action = k
#                 pprint.pprint(self.featExtractor.getFeatures(state, action))

#             print(len(self.weights))
            pass


# In[12]:


# from subprocess import call
# command = ('ipython nbconvert --to script qlearningAgents.ipynb')
# call(command, shell=True)

# %autoreload 2


# In[38]:


np.array([np.array(1),2])


# In[ ]:





# Small Classic Layout (7x20):
# 
#     %%%%%%%%%%%%%%%%%%%%
#     %......%G  G%......%
#     %.%%...%%  %%...%%.%
#     %.%o.%........%.o%.%
#     %.%%.%.%%%%%%.%.%%.%
#     %........P.........%
#     %%%%%%%%%%%%%%%%%%%%
# 
# I need it as input of my NN, flatting it out (1x140):
# 
#     %%%%%%%%%%%%%%%%%%%%%......%G  G%......%%.%%...%%  %%...%%.%%.%o.%........%.o%.%%.%%.%.%%%%%%.%.%%.%%........P.........%%%%%%%%%%%%%%%%%%%%%
# 
# Still need numbers/one-hot vectors for each symbol? or will it work without?
# 
# 
# 
# 
# 
# 

# In[21]:


# state:  
# %%%%%%%
# %  G> %
# % %%% %
# % %.  %
# % %%% %
# %.    %
# %%%%%%%
# Score: -8

# action:  East


# In[ ]:




