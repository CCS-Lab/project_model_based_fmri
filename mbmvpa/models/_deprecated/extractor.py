import numpy as np

class DefaultExtractor():
    # model is assumed to be linear
    
    def __init__(self, input_shape, n_sample=1):
        self.input_shape = input_shape
        self.n_sample = 1
        
    def __call__(self,model,batch_size=256):
        
        outputs_pool = []
        flatten_shape = np.product(self.input_shape)
        for i in range(self.n_sample):
            
            sample = np.eye(flatten_shape)
            n_step = int(np.ceil((flatten_shape+0.0)/batch_size))

            outputs = []
            for i in range(n_step):
                output = model.predict(sample[i*batch_size:(i+1)*batch_size].reshape(-1,*self.input_shape))
                output = list(output.flatten())
                outputs += output

            outputs = np.array(outputs)[:self.input_shape]
            outputs_pool.append(outputs)
        
        mean_outputs = np.array(outputs_pool).mean(0)
        
        return mean_outputs