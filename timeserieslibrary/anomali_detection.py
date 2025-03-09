import sys 
sys.path.append('../../')
import os
import numpy as np
import inspect
#Set parent directory to sys path to find the library in parent folder directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from   timeserieslibrary.ModelDeployment import Model
import timeserieslibrary.Preprocessing as pp

import matplotlib.pyplot as plt


class AnomalieDetectionAlgo():
    def  __init__(self, threshold): 
        self._threshold = threshold


#Receive data and perform anoamlie detection
class AnomalieDetection():
    #Parameter
    # preprocessing:            Normalize and resample
    # model:                    model for inference
    # anomalie_detection:       Anomalie detection algorithm 
    # input_window_length:      length of input window. Resample is done in preprocessing.
    # number_of_variables:      number of variables of the multivariate time series
    # used_variables_index:     Variables of the multivariate time series which should be used# Last variable must be ground truth
    # predict_every_x_steps:    Performance of inference after x inputs. 1: Anomalie detection after each input time step
    
    def  __init__(self, preprocessing:pp.TimeSeriesPreprocessingPipeline, model:Model, anomalie_detection: AnomalieDetectionAlgo, input_window_length, number_of_variables,  sliding_window_stride=1, mqtt_funktions=None): 

        self.__input_window_length = input_window_length     
        self.sliding_window_stride = sliding_window_stride
        self.__model = model
        self.__anomalie_detection = anomalie_detection
        self.__data_collection = None
        self.__anomalie_data_storage_x_data = None        
        self.__anomalie_data_storage_y_data = None      
        self.__anomalie_data_storage_prediction = None 
        self.__mse_data_storage = None
        self.__preprocessing = preprocessing
        self.__data_ready=False
        self.__count=0
        self.__mqtt_funktions = mqtt_funktions


        self.__model.inference(np.ones((input_window_length,number_of_variables+1)))



    #######Interface#######    

    # Fucntion:
    # Interface to receive data
    #
    # Parameter:
    # data:               Input type for message_type="InputData": Float list of length self.__number_of_variables. Data for one time step
    # data:               Input type for message_type=tbd: tbd  
    # message_type:       Message type 
    #
    # return:             - Return for message_type="InputData":  
    #                       - Anoamlie detection result 
    #                       - None, if no dedection is done 
    # return:             - Return for message_type=tbd:  
    #                       - tbd
    #                       - tbd
    def write(self, data, message_type = None):

        if message_type == "InputData":
            result = self.__handle_input_data(data)
            print(result)
            return result
        else:
            return None

    def read(self, message_type = None):

        if message_type == "GetData" and self.__data_ready == True:
           
            return_data = [self.__anomalie_data_storage_x_data.tolist(), self.__anomalie_data_storage_y_data.tolist(), self.__anomalie_data_storage_prediction.tolist(), self.__mse_data_storage.tolist()]

            self.__anomalie_data_storage_x_data=None
            self.__anomalie_data_storage_y_data=None
            self.__anomalie_data_storage_prediction=None
            self.__mse_data_storage=None
            self.__data_ready=False

            return return_data

        return None


    #######Functions#######

    # Fucntion:
    # Handles the input data: - Collect input data
    #                         - If enough data is avalibale (depending on input_window_lenght and predict_every_x_steps) perform anomlie detection 
    #
    # Parameter:
    # input_data:            - 2d input list (sequenz, variablen) 
    #
    # return:               - Anoamlie detection result: list of lenght batch_size. Values: True, False
    #                       - None, if no dedection is done, because data has not reached batch size
    def __handle_input_data(self, input_data):

        self.__collect_data(input_data)

        anomalie_detection_result=[]
        #If data reached lenght of input window lenght. Data (1 Batch) is send to model. 
        while self.__is_data_ready_for_inference():

            self.__data_ready=False # Mutex

            data_to_process = self.__data_collection[:self.__input_window_length] # *self.__preprocessing.downsampling_faktor ????
            self.__data_collection = self.__data_collection[self.sliding_window_stride*self.__preprocessing.downsampling_faktor:] # delete elements, which are not needed for next prediction

            preprocessed_data_x,  preprocessed_data_y = self.__preprocessing.preprocess_one_window(data_to_process) # Resample with this methode for one windows not ideal

            #Send data of batch size 1 to model. Model just return inverence results if the batch size is reached. 
            prediction = self.__model.inference(preprocessed_data_x)

            # plt data for test
            plt.clf()
            plt.cla()
            plt.plot(prediction[0], "r--")
            plt.plot(preprocessed_data_y)
            #plt.show()
            self.__count+=1
            plt.savefig("test"+str(self.__count)+".png")

            if not self.__mqtt_funktions is None:
                self.__mqtt_funktions.publish(prediction[0], preprocessed_data_y)

            if prediction is None:  #if data in the model isn't reached batchsize, no inference is done
                return None
                        
            anomalie_detection_batch = []
            mse_list=[]
            if self.__model.get_batch_size() != 1:
                raise Exception("batch > 1 not implemented")
           
            for i in range(self.__model.get_batch_size()):                
                ground_truth = preprocessed_data_y
                anomalie, mse = self.__anomalie_detection(prediction[i],  ground_truth)
                mse_list += [mse]
                anomalie_detection_batch += [anomalie]

            for idx, anomalie in enumerate(anomalie_detection_batch):
                if anomalie:
                    print("Anomalie detected")    


            print(mse_list)

            for idx, anomalie in enumerate(mse_list):  

                if not self.__anomalie_data_storage_x_data is None :
                    self.__anomalie_data_storage_x_data = np.append(self.__anomalie_data_storage_x_data, preprocessed_data_x[np.newaxis,:,:], axis=0)
                else:
                    self.__anomalie_data_storage_x_data = preprocessed_data_x[np.newaxis,:,:]
                
                if not self.__anomalie_data_storage_y_data is None :
                    self.__anomalie_data_storage_y_data = np.append(self.__anomalie_data_storage_y_data, preprocessed_data_y[np.newaxis,:,:], axis=0)
                else:
                    self.__anomalie_data_storage_y_data = preprocessed_data_y[np.newaxis,:,:]

                if not self.__anomalie_data_storage_prediction is None :
                    self.__anomalie_data_storage_prediction = np.append(self.__anomalie_data_storage_prediction, prediction[:,:], axis=0) #[np.newaxis,:,:]????????
                else:
                    self.__anomalie_data_storage_prediction = prediction[:,:]

                if not self.__mse_data_storage is None :
                    self.__mse_data_storage = np.append(self.__mse_data_storage, mse_list, axis=0)
                else:
                    self.__mse_data_storage = np.array(mse_list)


            print("\n")
                             
            anomalie_detection_result += [anomalie_detection_batch]

            self.__data_ready=True #Mutex und merker

        return anomalie_detection_result
        
    # Fucntion:
    # Collects input data
    #
    # Parameter:
    # input_data:          2d input list (sequenz, variablen)
    #
    def __collect_data(self, input_data):

        input_data = np.array(input_data, dtype=np.float32)

        if self.__data_collection is None:
            self.__data_collection = input_data
        else: 
            self.__data_collection = np.append(self.__data_collection, input_data, axis=0)


    def __is_data_ready_for_inference(self): 

        if self.__data_collection.shape[0] >= self.__input_window_length:
            return True
        else: 
            return False

            




class MSEAnomalieDetection(AnomalieDetectionAlgo):

    def  __init__(self, threshold, mse_compare_sequence_lenght=None): 
        super().__init__(threshold)
            
    def __call__(self, prediction, groundruth):

        mse = np.square(prediction - groundruth).mean()

        if mse > self._threshold:
            return True
        
        return False


class MultiVariantMSEAnomalieDetection(AnomalieDetectionAlgo):

    def  __init__(self, threshold, using_the_mean_mse_over_all_var_for_ad=False): 
        #using_the_mean_mse_over_all_var_for_ad: for Multivariat anomalie detection, 
        # False: use the MSE of all Variables with threshold for each var 
        # True: use the mean MSE of all variables and use one threshold
        super().__init__(threshold)
        self.mean_mse_for_each_variable=using_the_mean_mse_over_all_var_for_ad
            
    def __call__(self, prediction, ground_truth):

        mse=[]

        if len(prediction.shape)==2:
            mse=[]
            for i in range(prediction.shape[-1]):
                mse += [np.mean( np.square(prediction[:,i]-ground_truth[:,i]))]
        else:
            raise Exception("Dimension error")
        
        if self.mean_mse_for_each_variable:
            mse += [np.mean(mse)]
            print("MSE:", mse)
            if mse > self._threshold:
                return True, mse
            
        else: 
            for i in range(len(mse)):
                if mse[i] > self._threshold[i]:
                    return True, mse
        
        return False, mse

