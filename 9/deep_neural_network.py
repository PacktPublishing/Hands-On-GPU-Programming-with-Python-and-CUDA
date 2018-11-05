from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from Queue import Queue
import csv
from time import time

MAX_ENTROPY = 1

def cross_entropy(predictions=None, ground_truth=None):
    
    if predictions is None or ground_truth is None:
        raise Exception("Error!  Both predictions and ground truth must be float32 arrays")
    
    p = np.array(predictions).copy()
    y = np.array(ground_truth).copy()
    
    if p.shape != y.shape:
        raise Exception("Error!  Both predictions and ground_truth must have same shape.")
    
    if len(p.shape) != 2:
        raise Exception("Error!  Both predictions and ground_truth must be 2D arrays.")
    
    total_entropy = 0
    
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            
            if y[i,j] == 1:            
                total_entropy += min( np.abs( np.nan_to_num(  np.log( p[i,j] ) ) ) , MAX_ENTROPY)             
            else:             
                total_entropy += min( np.abs( np.nan_to_num( np.log( 1 - p[i,j] ) ) ), MAX_ENTROPY)
    
    return total_entropy / p.size


DenseEvalCode = '''
#define _RELU(x) ( ((x) > 0.0f) ? (x) : 0.0f )
#define _SIGMOID(x)  ( 1.0f / (1.0f + expf(-(x)) ))


__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, \
                           float * x, float *y, int batch_size, int w_t, int b_t, float delta)
{
     int i = blockDim.x*blockIdx.x + threadIdx.x;

     
        
     if (i < num_outputs)
     {
         for(int k=0; k < batch_size; k++)
         {    
              double temp = 0.0f;
              
              for (int j = 0; j < num_inputs; j++)
              {
                  temp += ((double) w[ (num_inputs) * i + j ] ) * ( (double) x[k * num_inputs + j]);
              }
                  
              temp += (double) b[i];
              
              
              
              
              y[k * num_outputs + i] = (float) temp;                 
         }
    
         
        
        if( w_t >= 0 && i == (w_t / num_inputs))
        {
              int j = w_t % num_inputs;
              
              for(int k=0; k < batch_size; k++)
                  y[k*num_outputs + i] += delta*x[k*num_inputs+j];
                  
              
        }
         
        if( b_t >= 0 && i == b_t )
        {
              //int j = b_t % num_inputs;
              
              for(int k=0; k < batch_size; k++)
                  y[k*num_outputs + i] += delta;
        }
    
    
        if(relu > 0 || sigmoid > 0)
             for(int k=0; k < batch_size; k++)
             {    
                  float temp = y[k * num_outputs + i];
                  
                  if (relu > 0)
                      temp = _RELU(temp);
                      
                  if (sigmoid > 0)
                      temp = _SIGMOID(temp);
                  
                  
                  
                  
                  y[k * num_outputs + i] = temp;                 
             }
            
    
    }
    
    
         
    return;
}
'''

eval_mod = SourceModule(DenseEvalCode)
eval_ker = eval_mod.get_function('dense_eval')

class DenseLayer:
    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None, stream=None, \
    relu=False, sigmoid=False, delta=None):
        
        self.stream = stream
        
        if delta is None:
            self.delta = np.float32(0.001)
        else:
            self.delta = np.float32(delta)
        
        if weights is None:
            weights = (np.random.rand(num_outputs, num_inputs) -.5 ) 
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)            
        
        if type(weights) != pycuda.gpuarray.GPUArray:
            self.weights = gpuarray.to_gpu_async(np.array(weights, dtype=np.float32) , stream = self.stream)
        else:
            self.weights = weights
        
        
        if num_inputs is None or num_outputs is None:
            
            self.num_inputs = np.int32(self.weights.shape[1])
            self.num_outputs = np.int32(self.weights.shape[0])
            
        else:
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)


        if b is None:
            b = gpuarray.zeros((self.num_outputs,),dtype=np.float32)
            
        if type(b) != pycuda.gpuarray.GPUArray:
            self.b = gpuarray.to_gpu_async(np.array(b, dtype=np.float32) , stream = self.stream)
        else:
            self.b = b   
        
        self.relu = np.int32(relu)
        self.sigmoid = np.int32(sigmoid)
        
        self.block = (32,1,1)
        
        self.grid = (int(np.ceil(self.num_outputs / 32)), 1,1)
        
        
        

    def eval_(self, x, y=None, batch_size=None, stream=None, delta=None, w_t = None, b_t = None):
    
        if stream is None:
            stream = self.stream
        
        if type(x) != pycuda.gpuarray.GPUArray:
            x = gpuarray.to_gpu_async(np.array(x,dtype=np.float32) , stream=self.stream)
            
        if batch_size is None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
                
        if delta is None:
            delta = self.delta
            
        delta = np.float32(delta)
            
        if w_t is None:
            w_t = np.int32(-1)
            
        if b_t is None:
            b_t = np.int32(-1)
        
        
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num_outputs,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num_outputs), dtype=np.float32)


        eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid, \
                 self.weights, self.b, x, y, np.int32(batch_size), w_t, b_t, \
                 delta , block=self.block, grid=self.grid , stream=stream)
        
        return y


# threads: at least "num"
SoftmaxExpCode='''
__global__ void softmax_exp( int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num)
    {
        for (int k=0; k < batch_size; k++)
        {
            y[num*k + i] = expf(x[num*k+i]);
        
        }
    }
}
'''

exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = exp_mod.get_function('softmax_exp')

# threads: at least batch size
SoftmaxMeanCode='''
__global__ void softmax_mean( int num, float *x, float *y, int batch_size)
{

    // parallelize over
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (i < batch_size)
    {
        float temp = 0.0f;
        
        for(int k=0; k < num; k++)
            temp += x[i*num + k];
            
        
        for(int k=0; k < num; k++)
            y[i*num+k] = x[i*num+k] / temp;
    
    }
    
    return;
}'''

mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = mean_mod.get_function('softmax_mean')

        
class SoftmaxLayer:
    def __init__(self, num=None, stream=None):
        self.num = np.int32(num)
        self.stream = stream
        
        
    def eval_(self, x, y=None, batch_size=None, stream=None):
        
        if stream is None:
            stream = self.stream
        
        if type(x) != pycuda.gpuarray.GPUArray:
            temp = np.array(x,dtype=np.float32)
            x = gpuarray.to_gpu_async( temp , stream=stream)
            
        if batch_size==None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
        else:
            batch_size = np.int32(batch_size)
        
        
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num), dtype=np.float32)

                
        exp_ker(self.num, x, y, batch_size, block=(32,1,1), grid=(int( np.ceil( self.num / 32) ), 1, 1), stream=stream)
        
        mean_ker(self.num, y, y, batch_size, block=(32,1,1), grid=(int( np.ceil( batch_size / 32)), 1,1), stream=stream)
    
        return y
    

    

class SequentialNetwork:

    def __init__(self, layers=None, delta=None, stream = None, max_batch_size=32, max_streams=10, epochs = 10):
        
        self.network = []
        self.network_summary = []
        self.network_mem = []
        
        if stream is not None:
            self.stream = stream
        else:
            self.stream = drv.Stream()
            
            
        if delta is None:
            delta = 0.0001
            
        self.delta = delta
        self.max_batch_size=max_batch_size
        
        self.max_streams = max_streams
        
        self.epochs = epochs
        
        if layers is not None:
            for layer in layers:
                add_layer(self, layer)
    
    def add_layer(self, layer):
    
        if layer['type'] == 'dense':
            if len(self.network) == 0:
                num_inputs = layer['num_inputs']
            else:
                num_inputs = self.network_summary[-1][2]
            
            num_outputs = layer['num_outputs']
            sigmoid = layer['sigmoid']
            relu = layer['relu']
            
            weights = layer['weights']
            
            b = layer['bias']
            
            self.network.append(DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs, sigmoid=sigmoid, relu=relu, weights=weights, b=b))
            self.network_summary.append( ('dense', num_inputs, num_outputs))
            
            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty( (self.max_batch_size, self.network_summary[-1][1] ), dtype=np.float32 ) )
                self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32  ) ) 
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append( gpuarray.empty( (self.network_summary[-1][1], ), dtype=np.float32 ) )
                self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32  ) ) 
    
        elif layer['type'] == 'softmax':
            
            if len(self.network) == 0:
                raise Exception("Error!  Softmax layer can't be first!")
            
            if self.network_summary[-1][0] != 'dense':
                raise Exception("Error!  Need a dense layer before a softmax layer!")
            
            
            num = self.network_summary[-1][2]
            
            self.network.append(SoftmaxLayer(num=num))
            
            self.network_summary.append(('softmax', num, num))
            
            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32  ) ) 
            else:
                self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32  ) ) 


            
    
    def predict(self, x, stream=None):
        
        if stream is None:
            stream = self.stream
        
        if type(x) != np.ndarray:
            temp = np.array(x, dtype = np.float32)
            x = temp
        
        if(x.size == self.network_mem[0].size):
            self.network_mem[0].set_async(x, stream=stream)
        else:
            
            if x.size > self.network_mem[0].size:
                raise Exception("Error: batch size too large for input.")
            
            x0 = np.zeros((self.network_mem[0].size,), dtype=np.float32)
            x0[0:x.size] = x.ravel()
            self.network_mem[0].set_async(x0.reshape(self.network_mem[0].shape), stream=stream)
        
        if(len(x.shape) == 2):
            batch_size = x.shape[0]
        else:
            batch_size = 1
        
        for i in xrange(len(self.network)):
            self.network[i].eval_(x=self.network_mem[i], y = self.network_mem[i+1], batch_size=batch_size, stream = stream)
            
        y = self.network_mem[-1].get_async(stream=stream)
        
        if len(y.shape) == 2:
            y = y[0:batch_size, :]
        
        return y
    
    
    def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None, delta=None):
        
        self.network[layer_index].eval_(x=self.network_mem[layer_index], y = partial_mem[layer_index+1], batch_size=batch_size, stream = stream, w_t=w_t, b_t=b_t, delta=delta)
        
        for i in xrange(layer_index+1, len(self.network)):
            self.network[i].eval_(x=partial_mem[i], y =partial_mem[i+1], batch_size=batch_size, stream = stream)

            
    
    def bsgd(self, training=None, labels=None, delta=None, max_streams = None, batch_size = None, epochs = 1, training_rate=0.01):
        
        training_rate = np.float32(training_rate)
        
        training = np.float32(training)
        labels = np.float32(labels)
        
        if( training.shape[0] != labels.shape[0] ):
            raise Exception("Number of training data points should be same as labels!")
        

        if max_streams is None:
            max_streams = self.max_streams
            
        if epochs is None:
            epochs = self.epochs
            
        if delta is None:
            delta = self.delta
        
        streams = []
        bgd_mem = []
        
        
        # create the streams needed for training
        
        for _ in xrange(max_streams):
            streams.append(drv.Stream())
            bgd_mem.append([])
            
        
        # allocate memory for each stream
        
        for i in xrange(len(bgd_mem)):
            for mem_bank in self.network_mem:
                bgd_mem[i].append( gpuarray.empty_like(mem_bank) )
        
        
        # begin training!
        
        num_points = training.shape[0]
        
        if batch_size is None:
            batch_size = self.max_batch_size
        
        index = range(training.shape[0])
        
        for k in xrange(epochs):    
            
            print '-----------------------------------------------------------'
            print 'Starting training epoch: %s' % k
            print 'Batch size: %s , Total number of training samples: %s' % (batch_size, num_points)
            print '-----------------------------------------------------------'
            
            all_grad = []
            
            np.random.shuffle(index)
            
            for r in xrange( int(np.floor(training.shape[0] / batch_size)) ):
            
                batch_index = index[r*batch_size:(r+1)*batch_size] 
                
                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]
                
                batch_predictions = self.predict(batch_training)
        
                cur_entropy = cross_entropy(predictions=batch_predictions, ground_truth=batch_labels)
                
                print 'entropy: %s' % cur_entropy
                
                # need to iterate over each weight / bias , check entropy
                
                for i in xrange(len(self.network)):
                    
                    if self.network_summary[i][0] != 'dense':
                        continue
                    
                    all_weights = Queue()
                    
                    grad_w = np.zeros((self.network[i].weights.size,), dtype=np.float32)
                    grad_b = np.zeros((self.network[i].b.size,), dtype=np.float32)
                    
                    for w in xrange( self.network[i].weights.size ):
                        all_weights.put( ('w', np.int32(w) ) )
                        
                    for b in xrange( self.network[i].b.size ):
                        all_weights.put(('b', np.int32(b) ) )
                        
                    while not all_weights.empty():
                        
                        stream_weights = Queue()
                        
                        for j in xrange(max_streams):
                            
                            if all_weights.empty():
                                break
                            
                            wb = all_weights.get()
                            
                            if wb[0] == 'w':
                                w_t = wb[1]
                                b_t = None
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                w_t = None
                            
                            stream_weights.put( wb )
                            
                            self.partial_predict(layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j], stream=streams[j], batch_size=batch_size, delta=delta)
                            
                        for j in xrange(max_streams):
                            
                            if stream_weights.empty():
                                break
                            
                            wb = stream_weights.get()
                            
                            w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])
                            
                            w_entropy = cross_entropy(predictions=w_predictions[:batch_size,:], ground_truth=batch_labels)
                            

                            if wb[0] == 'w':
                                w_t = wb[1]
                                grad_w[w_t] = -(w_entropy - cur_entropy) / delta
                                
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                grad_b[b_t] = -(w_entropy - cur_entropy) / delta
                        
                    all_grad.append([np.reshape(grad_w,self.network[i].weights.shape) , grad_b])
            
            for i in xrange(len(self.network)):
                
                if self.network_summary[i][0] == 'dense':
                
                    new_weights = self.network[i].weights.get()
                    new_weights += training_rate*all_grad[i][0]
                    new_bias = self.network[i].b.get()
                    new_bias += training_rate*all_grad[i][1]
                    self.network[i].weights.set(new_weights)
                    self.network[i].b.set(new_bias)
                    
    
            
            
def condition_data(data, means=None, stds=None):
    
    if means is None:
        means = np.mean(data, axis=0)
        
    if stds is None:
        stds = np.std(data, axis = 0)
        
    conditioned_data = data.copy()
    conditioned_data -= means
    conditioned_data /= stds
    
    return (conditioned_data, means, stds)
                        
                            
    
if __name__ == '__main__':
     to_class = { 'Iris-setosa' : [1,0,0] , 'Iris-versicolor' : [0,1,0], 'Iris-virginica' : [0,0,1]}
     
     iris_data = []
     iris_labels = []
     
     with open('iris.data', 'rb') as csvfile:
         csvreader = csv.reader(csvfile, delimiter=',')
         for row in csvreader:
             newrow = []
             if len(row) != 5:
                 break
             for i in range(4):
                 newrow.append(row[i])
             iris_data.append(newrow)
             iris_labels.append(to_class[row[4]])
             
     iris_len = len(iris_data)
     shuffled_index = list(range(iris_len))
     np.random.shuffle(shuffled_index)
     
     
     iris_data = np.float32(iris_data)
     iris_labels = np.float32(iris_labels)
     iris_data = iris_data[shuffled_index, :]
     iris_labels = iris_labels[shuffled_index,:]
     
     t_len = (2*iris_len) // 3
     
     iris_train = iris_data[:t_len, :]
     label_train = iris_labels[:t_len, :]
     
     iris_test = iris_data[t_len:,:]
     label_test = iris_labels[t_len:, :]
     
     
     sn = SequentialNetwork( max_batch_size=32 )


     sn.add_layer({'type' : 'dense', 'num_inputs' : 4, 'num_outputs' : 10, 'relu': True, 'sigmoid': False, 'weights' : None, 'bias' : None} )      
     sn.add_layer({'type' : 'dense', 'num_inputs' : 10, 'num_outputs' : 15, 'relu': True, 'sigmoid': False, 'weights': None, 'bias' : None} ) 
     sn.add_layer({'type' : 'dense', 'num_inputs' : 15, 'num_outputs' : 20, 'relu': True, 'sigmoid': False, 'weights': None, 'bias' : None} ) 
     sn.add_layer({'type' : 'dense', 'num_inputs' : 20, 'num_outputs' : 3, 'relu': True, 'sigmoid': False, 'weights': None , 'bias': None } )   
     sn.add_layer({'type' : 'softmax'})
     
     ctrain, means, stds = condition_data(iris_train)
     

     t1 = time()
     sn.bsgd(training=ctrain, labels=label_train, batch_size=16, max_streams=10, epochs=100 , delta=0.0001,
     training_rate=1)
     training_time = time() - t1
     
     hits = 0
     ctest, _, _ = condition_data(iris_test, means=means, stds=stds)
     for i in range(ctest.shape[0]):
         if np.argmax(sn.predict(ctest[i,:])) == np.argmax(label_test[i,:]):
             hits += 1
     
     
     print 'Percentage Correct Classifications: %s' % (float(hits ) / ctest.shape[0])
     print 'Total Training Time: %s' % training_time
