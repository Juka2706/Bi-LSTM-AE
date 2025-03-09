import wandb
import onnxruntime
import numpy as np
import tensorflow as tf


class ModelLoader():
    def  __init__(self): 
        self._model = 0
        self._model_file = 0

    def get_model_file(self):
        return self._model_file
    
class WandbModelLoader(ModelLoader):
    def  __init__(self, project_name, model_path, model_name, run_id, login_key = None): 
        super().__init__()

        if login_key != None:
            wandb.login(key=login_key, relogin=True)

        wandb.init(project=project_name, id=run_id)
        run_path = project_name+"/"+project_name+"/"+run_id
        self._model_file = wandb.restore(model_path+'/'+model_name, run_path=run_path)
        wandb.finish()


class FileSystemModelLoader(ModelLoader):
    def  __init__(self, model_name, model_path="",): 
        super().__init__()
        if model_path=="":
            model_file = open(model_name)
            self._model_file = model_file 
        else: 
            model_file = open(model_path+"/"+model_name)
            self._model_file = model_file


class Model():
    def  __init__(self, model_loader: ModelLoader, batch_size=1): 
        if not model_loader is None:
            self._model_file = model_loader.get_model_file() 
        self._model = 0
        self._batch_size = batch_size
        self._batch = []
        self._last_batch = 0
        
    def get_batch_size(self):
        return self._batch_size
    
    def get_current_batch_size(self):
        return len(self._batch)

    def get_last_batch(self):
        return self._last_batch


class ONNXModel(Model):
    def  __init__(self, model_loader: ModelLoader, model_input_name : str, batch_size : int = 1): 
        super().__init__(model_loader, batch_size)
        self._model = onnxruntime.InferenceSession(self._model_file.name)
        self._model_input_name = model_input_name
    
    # Function:
    # Collect data and do inference if batch size is reached
    #
    # input_data:    - Data shape: (Batch_size = 1, ... ) data depending on model
    #
    # return:       - Inference result 
    #               - if batch size isn't reached return None
    def inference(self, input_data):

        self._batch += [input_data]

        if len(self._batch) == self._batch_size:
            self._batch = np.array(self._batch, dtype=np.float32)
            result = self._model.run(None, {self._model_input_name: self._batch})
            result = np.array(result)
            result_shape = result.shape
            result = np.reshape(result, (result_shape[0]*result_shape[1],result_shape[2]))
            self._last_batch = self._batch
            self._batch = []
            return result        
        else:
            return None

#ToDo
class TFModel(Model):
    def  __init__(self, path, batch_size=1): 
        super().__init__(None,batch_size)
        self._model = tf.saved_model.load(path)
        self._model = self._model.signatures["serving_default"]
        t=0


    def inference(self, input_data):

        self._batch += [input_data]

        if len(self._batch) >= self._batch_size:

            result = self._model(tf.constant(self._batch,dtype=tf.float32) )
            result = [i for i in result.items()]
            result = np.array(result[0][1].numpy())

            self._last_batch = self._batch
            if len(self._batch) > self._batch_size:
                self._batch = self._batch[self._batch_size:]
            self._batch = []
            return result        
        else:
            return None







