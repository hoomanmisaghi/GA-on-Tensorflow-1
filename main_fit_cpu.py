from network import fc,pop_weight_creation,update_all_vars
from GA import choose_winner,pop_best,fitness
import numpy as np
import tensorflow as tf

''' this a main file for genetic algorithm 
in this fiel fitness function is called from GA.py file
fitness is on cpu 

this is more efficient if you could write you fitness function on gpu


might be pointless not sure
'''
num_pop=5
generation=1000
mutation_rate=.2


input=[[1.],[1.],[1.],[1.]]#
input_size=4
output_size=1


# preparing the input for whole population
batch_input = []
for pep in range(num_pop):
    batch_input.append(input)
##############################################
# defining the weights and creating firsts population
with tf.variable_scope('network'):
    ws1,bs1=pop_weight_creation(input_size,1,num_pop)

   # ws2,bs2=pop_weight_creation(5,1,num_pop)
update_list=[ws1,bs1]
###############################################
#defining the network
net=fc(batch_input,ws1,bs1)
#net=fc(net,ws2,bs2)
#**********************************************
fit=fitness(net)

winner=choose_winner(fit,5,num_pop)
best=pop_best(fit)



##########################################33
summary_writer = tf.summary.FileWriter("board\\",tf.get_default_graph())
summary_writer.add_graph(tf.get_default_graph())
merge=tf.summary.merge_all()
###########################################
sess=tf.Session()
sess.run(tf.global_variables_initializer())
##########################################################################################33

#start the loop
for i in range(generation):
    Winner,WS1,BS1,Best=sess.run([winner]+update_list+[best])


    if i==0:
        total_best=Best
    else:
        if Best>total_best:
            total_best=Best
    vals=[WS1,BS1]
    np.array(vals)
    THE_WINNER=Winner[0]
    #best_weights=vals[:,THE_WINNER,:,:]
    update_op=update_all_vars(Winner,update_list,vals,2,num_pop,mutation_rate)
    sess.run(update_op)
    print('iteration:',i)
    print('generation fitness',Best)
    print('global best:',total_best)





#print(np.array(WS1).shape)
#print(np.array(new_pop).shape)