import tensorflow as tf
from GA import update_pop

''' in this file functions for creating a network that is suitable for evaluation techniques'''

def weight_creation(input_size,output_size,if_bias=True):
    '''
    creates weights for the network randomly
    later on this method should be used in a GA alg
    :param input_size:
    :param output_size:
    :return:
    '''
    w=tf.Variable(tf.random_normal([1,output_size,input_size]),dtype=tf.float32,name='w')
    if if_bias is True:
        b = tf.Variable(tf.random_normal([1,output_size,1]),dtype=tf.float32,name='b')

        return w,b
    else:
        return w,None

def layer_populate(ws,bs):
    '''

    :param ws: collection of all weights in a layer
    :param bs: collection of all weights in a layer
    :return:
    '''

    w=tf.Variable(tf.concat(ws,0),name='W')
    b=tf.Variable(tf.concat(bs,0),name='B')
    return w,b




def fc(input,w,b,activation=None):
    '''
    creats a fulley connected layer by passed in w and b
    :param pop:
    :param input_size:
    :param output_size:
    :param bias_size:
    :param activation_fcn:
    :param passed_vlaue:
    :return:
    '''


    if b is None:
        b=tf.zeros([tf.shape(w)[0]],dtype=tf.float32)

    if activation is None:
        out=tf.matmul(w,input)+b
    else:
        out=activation(tf.matmul(w,input)+b)

    return out
def new_fc(input,input_size,output_size):
    '''

    :param input:
    :param input_size:
    :param output_size:
    :return:
    '''

    input=tf.cast(input,dtype=tf.float32)
    w,b=weight_creation(input_size,output_size)

    net=fc(input,w,b)
    return net




def pop_weight_creation(input_size,output_size,num_pop,if_bias=True):
    '''
    creates weights for the network randomly
    later on this method should be used in a GA alg
    :param input_size:
    :param output_size:
    :return:
    '''

    ws = []
    bs = []

    for pep in range(num_pop):
        w, b = weight_creation(input_size=input_size, output_size=output_size)

        ws.append(w)
        bs.append(b)


    ws, bs = layer_populate(ws, bs)
    return ws,bs




def update_tf_weight(tf_w,value):

    return tf.assign(tf_w,value)

def update_all_vars(winner,var_list,val_list,num_slice,new_pop,mutation_rate):
    for i,var in enumerate(var_list):
        val=val_list[i]

        return update_tf_weight(var,update_pop(winner,num_slice,val,new_pop,mutation_rate))
