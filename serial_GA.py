import tensorflow as tf
import numpy as np
from cart_pole import linear_dynamico
from network import fc,pop_weight_creation,update_all_vars
from GA import choose_winner,pop_best,fitness
import numpy as np


'''
this consists of two parts 
a serialized test part that does a session test on the samples 

the and update part that gets the best ones and does the GA shit
'''

test_steps=100

input_size=4
output_size=1

num_pop=5
generation=10
mutation_rate=.2



#############################################
# defining the weights and creating firsts population
with tf.variable_scope('network'):
    ws1,bs1=pop_weight_creation(input_size,1,num_pop)

   # ws2,bs2=pop_weight_creation(5,1,num_pop)
###############################################
#defining the network
net=fc(batch_input,ws1,bs1)
#net=fc(net,ws2,bs2)
#**********************************************
for i in range(test_steps):
