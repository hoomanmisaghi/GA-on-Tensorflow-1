# GA-on-Tensorflow-1
the implementation of genetic algorithm on Tensorflow
This project is the siple implementation of Genetic Algorithm on Tensorflow 1.
In this implementation each population is processed on one pass of data on GPU and the choosing process uses numpy and is processed on CPU. One of the main issues of this implementation is that you have to write your cost function with tensorflow functions to be able to process the cost function on GPU. The advantage is because of the GPU processing you can have big population which is processed in one pass of GPU without any "for" loops, this makes calculations of each generation very fast.
