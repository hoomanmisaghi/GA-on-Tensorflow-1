import tensorflow as tf
'''
this is  gpu implementation of famous cartpole

check this out:
www-robotics.cs.umass.edu/~grupen/503/SLIDES/cart-pole.pdf
'''
#illustrating example
i = tf.constant(0)
#c = lambda i: tf.less(i, 50)
def c(i):
    return tf.less(i, 50)
def b(i):
    return tf.add(i,1)

#r = tf.while_loop(c, b, [i])
#sess=tf.Session()
#print(sess.run(r))
#'''

"""
def linear_dynamic(x,dx,t,dt,u):
    '''
    t theta
    dt theta dot
    x  cart dist
    dx x dot
    g gravitational acceleration
    l pole length
    e=[dx,ddx,dt,ddt]
    :return:
    '''
    g=9.8
    l=1
    M=2
    ##################################################################################3
    u=tf.reshape(tf.constant(u,dtype=tf.float32),[1,1])
    A=tf.constant([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,g/l,0]],dtype=tf.float32)
    B=tf.constant([x,dx,t,dt],dtype=tf.float32)
    B=tf.reshape(B,[4,1])
    C=tf.constant([0,1/M,0,1/(M*l)],dtype=tf.float32)
    C=tf.reshape(C,[4,1])
    O=tf.matmul(A,B)+tf.matmul(C,u)
    return O
    
"""



def linear_dynamico(initial_cond,u,dtime=.1):
    '''
    t theta
    dt theta dot
    x  cart dist
    dx x dot

    these four shape the initial condition
    g gravitational acceleration
    l pole length
    O=[dx,ddx,dt,ddt]
    :return:
    '''
    g = 9.8
    l = 1
    M = 2


    ##################################################################################3
    x,dx,t,dt=0,0,0,0
    u = tf.reshape(tf.constant(u, dtype=tf.float32), [1, 1])
    A = tf.constant([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, g / l, 0]], dtype=tf.float32)
    #B = tf.constant(initial_cond, dtype=tf.float32)
    #B = tf.reshape(initial_cond, [4, 1])
    C = tf.constant([0, 1 / M, 0, 1 / (M * l)], dtype=tf.float32)
    C = tf.reshape(C, [4, 1])
    O = tf.matmul(A, B) + tf.matmul(C, u)

    new_B=B+O*dtime
    teta=new_B[2]
    fell=tf.greater_equal(tf.abs(teta),.2)

    return O,new_B,fell
u=1
B = tf.placeholder(tf.float32,[4,1])
O,new_B,fell=linear_dynamico(B,u=1.)
sess=tf.Session()
new_cond=[[0],[0],[0],[0]]
for i in range(20):
    out,new_cond,FELL=sess.run([O,new_B,fell],feed_dict={B:new_cond})
    print(new_cond)
