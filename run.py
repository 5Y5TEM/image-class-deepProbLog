from train import train_model
from data_loader import load
from mnist import MNIST_Net, neural_predicate,test_MNIST
from model import Model
from optimizer import Optimizer
from network import Network
from dataLoader import load_dataset, load_dataloader
import matplotlib.pyplot as plt 
import torch
import random 
import numpy as np 
import re 



queries = load('train_data.txt')
test_queries = load('test_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()

# =============================================================================
# TRAIN MODEL 
# =============================================================================
network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
model = Model(problog_string, [net], caching=False)
#model.load_state("model.zip")
optimizer = Optimizer(model, 2)
train_model(model,queries, 3, optimizer,test_iter=1000,test=test_MNIST,snapshot_iter=1000)


# =============================================================================
# SAVE MODEL
# =============================================================================
#torch.save(network.state_dict(), "mnist001.pwf")
#model.save_state("model.zip")


# =============================================================================
# LOAD MODEL 
# =============================================================================
#model.load_state("model.zip")
#model.load_state("model_iter_8000.mdl")
#model.load_state("/root/Desktop/DigitAdder (copy)/deepproblog/save_states/model_iter_1000.mdl")
#network = MNIST_Net()
#network.load_state_dict(torch.load("mnist001.pwf"))
#network.eval()
#net = Network(network, 'mnist_net', neural_predicate)
#net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
#model = Model(problog_string, [net], caching=False)
#optimizer = Optimizer(model, 2)


# =============================================================================
# TEST MODEL 
## =============================================================================
##
##PATH ="/root/Desktop/Original/deepproblog/data/dataset/" #normal size 
##mnist_train_data, mnist_test_data = load_dataset(PATH)
#
#train_path = "/root/Desktop/Original/deepproblog/data/train"
#test_path = "/root/Desktop/Original/deepproblog/data/test"
#
#mnist_train_data = load_dataset(train_path)
#mnist_test_data = load_dataset(test_path)
#
#test_set, test_loader = load_dataloader(test_path)
#
#test_MNIST(model)

# =============================================================================
# PLOT EXAMPLE RESULTS
# =============================================================================
#example = 666
#
#
#
#print("\n\nExample result: \n")
#query = test_queries[example]
##print(query)
#result = model.solve(query)
#print(result)
#
##for k in result.keys():
###    print("key: {} \t value: {}".format(k, result[k]))
##    z = result[k]
##    y = z[1]
##    
##    for k in y.keys(): 
###        print(y[k])
##        probs = np.array(y[k])
##        predicted = np.argmax(probs)
##        predicted = predicted +1
#temp = re.findall(r'\d+', str(list(result.keys())[0]))
#res = list(map(int, temp)) 
#r = res[0]
#print( " r: ", r)
#d,l = mnist_test_data[r]
##d,l = mnist_train_data[r]
#plt.imshow(d[0,:,:])
#
#add = 0 
##result = 0 
#c = test_set.classes[l]
#digits = [int(d) for d in str(c)]
#for x in digits: 
#    add += x
#
##digits = [int(d) for d in str(p)]
##for x in digits: 
##    result += x
#
#q = 'addition(%s,_).'%str(r)
#tf = open("query.txt","w")
#tf.write(q)
#tf.close() 
#
#qu = load('query.txt')
#q = qu[0]
#resultQ = model.solve(q)
#print("Result of the example query:\n ",resultQ)
#maximum = str(max(resultQ, key=resultQ.get))
##print(maximum)
#temp = re.findall(r'\d+', str(maximum))
##print(int(temp[1]))
#
#print("\nThe predicted sum of the following image is: {}.\nThe actual sum: {}.\nlabel was: {} ".format(temp[1],add,l+10))
