import tensorflow as tf
import numpy as np
import random

def cross_over(winner,num_slice,w,new_pop,mutation_rate):
    '''
        this is a simple cross over
        sadly this part is on CPU :|
        fuck you GA and fuck me that I don't know any better ways to program this shit on GPU :|
    :param winner:
    :param num_slice:
    :param w:this is numpy!!!
    :return:
    '''

    pop_with_cross_over=int((1-mutation_rate)*new_pop)


    def slice_genes(winner,num_slice,w):
        '''

        :param winner:
        :param num_slice:
        :param w:
        :return:
        '''


        w_shape=np.shape(w)
        num_winner=len(winner)

        slice_place=int(w_shape[1]/num_slice)



        # lets get some good genes  #ژن_خوب
        g1=[]
        g2=[]
        for i in winner:
            g1.append(w[i,0:slice_place,:])
            g2.append(w[i,slice_place:,:])
        return g1,g2

    def breed(g1,g2,pop_with_cross_over):
        new_pop=[]
        for i in range(pop_with_cross_over):

            one=random.choice(g1)
            two=random.choice(g2)

            child=np.concatenate((one,two))
            new_pop.append(child)

        return new_pop

    g1,g2=slice_genes(winner, num_slice, w)

    new_pop=breed(g1, g2, pop_with_cross_over)



    return new_pop


def mutation(w,new_pop,mutation_rate):
    '''

    :param mut_rate: percentage of population that should be mutated
    :param w: the variable that should be mutated
    :return: the mutated variable
    '''
    w_shape = np.shape(w)
    pop_with_cross_over = int((1 - mutation_rate) * new_pop)
    pop_with_mutation=new_pop-pop_with_cross_over
    new_pop=[]
    for pop in range(pop_with_mutation):
        w = np.random.rand(w_shape[1],w_shape[2])


        new_pop.append(w)
    return new_pop






def choose_winner(res,num_of_survive,pop_num,condition='max'):
    '''

    **remember res should be scalar!!!


    :param res:
    :param condition:
    :return:
    '''

    res=tf.reshape(res,[pop_num])
    if condition=='max':
        val,index=tf.nn.top_k(res,num_of_survive)
    elif condition=='min':
        val, index = tf.nn.top_k(-res, num_of_survive)

    return index

def pop_best(res):
    best=tf.reduce_max(res)
    #arg_best=tf.arg_max(res,1)
    return best

def update_pop(winner,num_slice,w,new_pop=50,mutation_rate=.2):
    '''

    :param winner:
    :param num_slice:
    :param w: numpy array
    :param new_pop:
    :param mutation_rate:
    :return:
    '''
    new_pop_c=cross_over(winner,num_slice,w,new_pop,mutation_rate)
    new_pop_m=mutation(w,new_pop,mutation_rate)
    return np.concatenate((new_pop_c,new_pop_m))

def fitness(x):
    fit=-tf.pow(x,2)+.5

    return fit