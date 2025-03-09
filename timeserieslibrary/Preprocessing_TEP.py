import numpy as np
import pandas as pd
import einops as eo
from sklearn import preprocessing
from scipy.signal import resample
import json
import os




class TimeSeriesPreprocessingPipeline():
    """Processes one long row time series of format: (sequence length, num variable) or (num variable, sequence length) into multiple windows -> (num samples, sequence length of windows, num variables)
    Processing steps:
    -Resampling (Downsampling of faktor downsampling_faktor)
    -Split the one large time series into (overlapping) windows of size split_time_series_into_window_of_len with stride of time_shift_after_each_split
    -Scale data between 0 and 1
    -Select variables we want to use 
    -Add time stamp
   

    Args:
        data (np.ndarray): Input data. Shape should be either (num variables, sequence length) or (sequence length, num variables).
        index_list_of_used_variables (list, optional): List of indices of variables to be used. Indexes of original Data. Defaults to None.
        downsampling_factor (int, optional): Downsampling factor for resampling the time series. Defaults to 1.
        ground_truth_variable_index (int, optional): returns one extra variable as ground_truth (y) and removes this variable from input data (x). Index of original data
        split_time_series_into_window_of_len (int, optional): Split time series into multiple windows.
        stride_of_window (int, optional): Time shift after each split (useful for creating overlapping segments). Defaults to None.
        scale_data (bool, optional): Whether to normalize the data. Each window of each variable is normalized (set between 0-1) by its own. Defaults to True.
        index_list_of_variables_to_scale (list, optional): List of indices of variables to be normalized. Indexes of original data. Defaults to None.
        add_time_stamp (bool, optional): Whether to add a time stamp to the data. Defaults to True.
        mask_input_data (bool, optional): Mask the input data. But just the Variables which will be used as ground truth.
        data_multiplicator_for_masking (int, optional): multiplicate the data by this factor bevor masking, so we have multiple samples with different masks
        data_mask_lenght (float, optional): Relativ Lenght of masking area in timeseries. Value 0-1

    """

    def __init__(self, 
                split_time_series_into_window_of_len:int=None, sliding_window_step:int=None, steps_to_predict_in_future:int=None,
                input_variable_index_list:list=None,ground_truth_variable_index_list:int=None, 
                downsampling_faktor:float=1, index_list_bool_variablen:list=[],                
                scale_data:bool=True, index_list_of_variables_to_scale:list=[],
                add_time_stamp:bool=True, mask_input_data:bool=False, data_multiplicator_for_masking:int=1, data_mask_lenght:int=0.20, past_window:int=None,
                JSON_config_file:str = None,
                ):


        if not JSON_config_file is None:
            self.load_config_from_JSON(JSON_config_file)

        else:

            self.index_list_of_used_variablen=input_variable_index_list
            self.downsampling_faktor=downsampling_faktor
            self.split_time_series_into_window_of_len=split_time_series_into_window_of_len
            self.sliding_window_step=sliding_window_step      
            self.steps_to_predict_in_future=steps_to_predict_in_future
            self.scale_data=scale_data
            self.scaler = preprocessing.StandardScaler()
            self.scaler_transformer= None
            self.index_list_of_variables_to_scale=index_list_of_variables_to_scale
            self.add_time_stamp=add_time_stamp
            self.ground_truth_variable_index=ground_truth_variable_index_list
            self.index_list_bool_variablen=index_list_bool_variablen
            self.mask_input_data=mask_input_data
            self.data_multiplicator_for_masking=data_multiplicator_for_masking
            self.data_mask_lenght=data_mask_lenght
            self.__min_value_for_scaling=[]
            self.__max_value_for_scaling=[]
            self.past_window=past_window

            if not self.steps_to_predict_in_future is None:
                #self.split_time_series_into_window_of_len=split_time_series_into_window_of_len+steps_to_predict_in_future
                if self.sliding_window_step is None:
                    self.sliding_window_step=steps_to_predict_in_future


            if not self.past_window is None:
                self.split_time_series_into_window_of_len+=self.past_window
       
        if not input_variable_index_list is None and not index_list_of_variables_to_scale is None:               
            if not all(x in input_variable_index_list for x in index_list_of_variables_to_scale): 
                raise Exception("index_list_of_variables_to_scale not in index_list_of_used_variablen")


    def __call__(self, data, is_test_data ,stride_of_window=None, one_window_from_the_end=False):
        """Processes the input data for time series.
        Args:
            is_test_data (bool, optional) if True the stride (window shift) is set to window lenght -> No overlapping segments
        Returns:
            np.ndarray: Processed data with shape:(number of samples, length of windows, num variables)
        """

        if stride_of_window is None:
            stride_of_window = self.sliding_window_step
  
        #Check Data
        if isinstance(data, pd.DataFrame):
            data.fillna(0, inplace=True)
            data=np.array(data) 
        elif not isinstance(data, np.ndarray):
            data=np.array(data) 

        if len(data.shape)>2:
            raise ValueError("Function proceeses only data in format shape: (num variables, sequence length) or ( sequence length, num variables)")
        
        print("\nInput data shape:", np.array(data).shape)

        if (data.shape[0] > data.shape[1]):
            data=data.T #->(variables, sequenz)
    
        #first scale than resampling

        # 


        '''
        if self.scale_data:

            if is_test_data:
                if self.scaler_transformer is None:
                    print("Scaler not fitted")
                data = self.scaler_transformer(data)
            else:
                self.scaler_transformer  =  preprocessing.MinMaxScaler().fit(data)
                data = self.scaler_transformer.transform(data)

            print("Shape of scaled data:",np.array(data).shape)     
        '''

        scaled_data = []
        if self.scale_data:
            for var in range(data.shape[0]):
                if not is_test_data:
                    # Für Trainingsdaten: Den Scaler mit den Daten anpassen (fit) und transformieren
                    self.scaler.fit(data)
                    var_data = self.scaler.transform(data)
                else:
                    # Fürasd Testdaten: Nur transformieren, ohne neu zu fitten!
                    var_data = self.scaler.transform(data)

                scaled_data += [var_data]


        '''
        scaled_data = []
        if self.scale_data:
            
            if not is_test_data and len(self.__max_value_for_scaling)==0:

                self.__max_value_for_scaling = []
                self.__min_value_for_scaling = []
                #print("Set new values for scaling (max and min values)")
            for var in range(data.shape[0]):

                if is_test_data:
                    var_data,_,_ = scale_0_to_1(data[var,:],self.__min_value_for_scaling[var], self.__max_value_for_scaling[var])
                else:
                    var_data, min_, max_ = scale_0_to_1(data[var,:])
                    self.__min_value_for_scaling += [min_]
                    self.__max_value_for_scaling += [max_]
                    #print("var", var)
                    #print("min", min_)
                    #print("max", max_)
                scaled_data += [var_data]
            data=np.array(scaled_data)
            #print("Shape of scaled data:",np.array(data).shape)      
        '''

        #Resample time series
        if not self.downsampling_faktor==1:
            resampled_data=[]
            shape=data.shape
            for var in range(shape[0]):
                
                if var in self.index_list_bool_variablen:# If bool variable methode of resample is: Just pik every x'th (downsample_faktor) point
                    resampled_data += [resample_timeseries(data[var], downsample_factor=self.downsampling_faktor, methode="delete")[0:shape[1]//self.downsampling_faktor]]
                else:
                    resampled_data += [resample_timeseries(data[var],downsample_factor=self.downsampling_faktor)[0:shape[1]//self.downsampling_faktor]]
            data=np.array(resampled_data)
            #print("Shape of resampled data:",np.array(data).shape)


        #Split into multiple windows
        if stride_of_window is None: #None or test data -> No overlapping segments
            stride_of_window=self.split_time_series_into_window_of_len


        #print("stride_of_window", stride_of_window)
        data = to_multiple_series(data, self.split_time_series_into_window_of_len, stride_of_window, False, one_window_from_the_end)
        #print("Shape of data sliced into (overlapping) windows:",np.array(data).shape)
        

 
    

    
        #Use just the variables given in index_list_of_used_variablen
        if not self.index_list_of_used_variablen is None: #Else use all
            used_data_x = []
            for var in self.index_list_of_used_variablen:
                used_data_x += [data[:,:,var]]
            used_data_x =np.transpose(used_data_x, (1,2,0))
        else:
            used_data_x=data
        #print("Shape of used data for input",used_data_x.shape )


        #Seperate ground truth data
        if not self.ground_truth_variable_index is None:
            mask = np.zeros(shape=(data.shape[-1]), dtype=bool)
            for var in self.ground_truth_variable_index:
                mask[var] = 1
            used_data_y = np.array(data[:,:,mask], dtype=np.float32)
        else:
            used_data_y = np.array(data[:,:,], dtype=np.float32)


        #Mask time series. Just mask the time series variables of the input of which will be reconstructed
        if self.mask_input_data and not is_test_data:

            multiplicated_data_x=used_data_x
            multiplicated_data_y=used_data_y
            for var in range(self.data_multiplicator_for_masking-1):
                multiplicated_data_x = np.append(multiplicated_data_x,used_data_x, axis=0)
                multiplicated_data_y = np.append(multiplicated_data_y,used_data_y, axis=0)
            used_data_x=multiplicated_data_x
            used_data_y=multiplicated_data_y

            variablen_to_mask_index_list=[]
            if self.ground_truth_variable_index is None:
                ground_truth_var=range(0,used_data_x.shape[-1])
            else:
                ground_truth_var=self.ground_truth_variable_index
            if self.index_list_of_used_variablen is None:
                used_var=range(0,used_data_x.shape[-1])
            else:
                used_var=self.index_list_of_used_variablen
            for var in ground_truth_var:
                variablen_to_mask_index_list += [np.where(np.array(used_var) == var)]
            used_data_x = mask_time_series_sections(used_data_x, relativ_mask_lenght=[self.data_mask_lenght], variablen_to_mask_index_list=variablen_to_mask_index_list)
            #print("Shape of multiplicated and mask data:",used_data_x.shape )
        

        #Add time stamp
        if self.add_time_stamp:
            used_data_x = add_time_stamp(used_data_x, 0)
            #print("Shape of data with time stamp:",used_data_x.shape )

        if not self.steps_to_predict_in_future is None:
            used_data_x = used_data_x[ : , :-self.steps_to_predict_in_future , : ]
            used_data_y = used_data_y[ : , -self.steps_to_predict_in_future: , : ]
            #print("Y-ata is the future")


        if not self.past_window is None:
            used_data_x = used_data_x[ : , : , : ]
            used_data_y = used_data_y[ : , self.past_window : , : ]


        print("Shape of x data",used_data_x.shape )
        print("Shape of y data",used_data_y.shape )
        return np.array(used_data_x, dtype=np.float32),  np.array(used_data_y, dtype=np.float32)






    #input shape (sequenz, variables)
    def preprocess_one_window(self, data):

        if (data.shape[0] > data.shape[1]):
            data=data.T #->(variables, sequenz)

        print("data shape 1:", data.shape)

        if self.scale_data:
            scaled_data=[]
            for var in range(data.shape[0]):
                var_data,_,_ = scale_0_to_1(data[var,:],self.__min_value_for_scaling[var], self.__max_value_for_scaling[var])
                scaled_data += [var_data]
            data=np.array(scaled_data, dtype=np.float32)
        else:
            Exception("scale mode error")
 
        print("data shape 2:", data.shape)
        #Resample time series
        if not self.downsampling_faktor==1:
            resampled_data=[]
            shape=data.shape
            for i in range(shape[0]):                
                if i in self.index_list_bool_variablen:# If bool variable methode of resample is: Just pik every x'th (downsample_faktor) point
                    resampled_data += [resample_timeseries(data[i], downsample_factor=self.downsampling_faktor, methode="delete")[0:shape[1]//self.downsampling_faktor]]
                else:
                    resampled_data += [resample_timeseries(data[i],downsample_factor=self.downsampling_faktor)[0:shape[1]//self.downsampling_faktor]]
            data=np.array(resampled_data)

        print("data shape 3:", data.shape)
        #Use just the variables given in index_list_of_used_variablen
        if not self.index_list_of_used_variablen is None: #Else use all
            used_data_x = []
            for i in self.index_list_of_used_variablen:
                used_data_x += [data[i,:]]
            used_data_x =np.transpose(used_data_x, (1,0))
        else:
            used_data_x=data


        print("Shape of x data 1",used_data_x.shape )

        #Seperate ground truth data
        if not self.ground_truth_variable_index is None:
            data=np.transpose(data, (1,0))
            mask = np.zeros(shape=(data.shape[-1]), dtype=bool)
            for i in self.ground_truth_variable_index:
                mask[i] = 1
            used_data_y = np.array(data[:,mask], dtype=np.float32)
        else:
            used_data_y = np.array(data[:,:], dtype=np.float32)

        #Add time stamp
        if self.add_time_stamp:
            used_data_x = add_time_stamp(used_data_x, 0)

        print("Shape of x data 2",used_data_x.shape )
        print("Shape of y data",used_data_y.shape )


        return np.array(used_data_x, dtype=np.float32),  np.array(used_data_y, dtype=np.float32)



    def get_min_max_normalization_vaues(self):
        return self.__min_value_for_scaling, self.__max_value_for_scaling
    


    def save_config_as_JSON(self, file_name="config.json"):

        x = {
            "index_list_of_used_variablen": dict(zip(range(len(self.index_list_of_used_variablen)), self.index_list_of_used_variablen)) ,
            "downsampling_faktor": self.downsampling_faktor,
            "split_time_series_into_window_of_len": self.split_time_series_into_window_of_len,
            "sliding_window_step": self.sliding_window_step,
            "scale_data": self.scale_data,
            "index_list_of_variables_to_scale": dict(zip(range(len(self.index_list_of_variables_to_scale)), self.index_list_of_variables_to_scale)) ,
            "add_time_stamp": self.add_time_stamp,
            "ground_truth_variable_index": dict(zip(range(len(self.ground_truth_variable_index)), self.ground_truth_variable_index)) ,
            "index_list_bool_variablen": dict(zip(range(len(self.index_list_bool_variablen )), self.index_list_bool_variablen )),
            "mask_input_data": self.mask_input_data,
            "data_multiplicator_for_masking": self.data_multiplicator_for_masking,
            "data_mask_lenght": self.data_mask_lenght,
            "__min_value_for_scaling":  dict(zip( range(len(self.__min_value_for_scaling )), self.__min_value_for_scaling )),
            "__max_value_for_scaling":  dict(zip( range(len(self.__max_value_for_scaling )), self.__max_value_for_scaling )) ,
        }

        
        with open(file_name, "w") as file:  
            json.dump(x, file, indent=2)
            #file.write(json_object)
            #file.close()

        
    def load_config_from_JSON(self, file_name):

        if os.path.exists(file_name) == True:
            with open(file_name, "r") as file:

                data = json.load(file)

                self.index_list_of_used_variablen=list(data['index_list_of_used_variablen'].values())
                self.downsampling_faktor= data['downsampling_faktor']
                self.split_time_series_into_window_of_len=data['split_time_series_into_window_of_len']
                self.sliding_window_step=data['sliding_window_step']        
                self.scale_data=data['scale_data']
                self.index_list_of_variables_to_scale= list(data['index_list_of_variables_to_scale'].values())
                self.add_time_stamp=data['add_time_stamp']
                self.ground_truth_variable_index= list(data['ground_truth_variable_index'].values())
                self.index_list_bool_variablen= list(data['index_list_bool_variablen'].values())
                self.mask_input_data=data['mask_input_data']
                self.data_multiplicator_for_masking=data['data_multiplicator_for_masking']
                self.data_mask_lenght=data['data_mask_lenght']
                self.__min_value_for_scaling= list(data['__min_value_for_scaling'].values())
                self.__max_value_for_scaling= list(data['__max_value_for_scaling'].values())
                file.close()

        else:
            Exception("No file found")
        


      


def scale_0_to_1(data, std=None, max=None):

    if std is None:
        std=np.std(data)
    if max is None:
        max=np.max(data)

    return ((data - min) / (max - min + 1e-12)) , min, max



def time_series_segmentation(data, segment_size, fill_up_value=0, overlapping_segments:int= 0):
    """Splits a time series into segments.
    
    Args:
        data (np.ndarray): Input time series data.
        segment_size (int): Size of each segment.
        fill_up_value (int, optional): Value used to fill up the sequence size if it's not divisible by segment_size. Defaults to 0.
        overlapping_segments (int, optional): Number of overlapping segments. Defaults to 0.

    Returns:
        np.ndarray: Segmented time series data.


    Info:
        2 Dimensions:
        Split a time series int segments input. (batchsize, sequenz size) -> out :(batchsize, sequenz size/segment_size, segment_size)
        E.g. segment_size = 10: in=(batchsize, 200) -> out=(batchsize, 20, 10)
        if sequenz size is not dividable by segment_size. Sequenz size is filed up with fill_up_value
        3 Dimensions:
        Split a time series int segments input. (batchsize, sequenz size, number of variables) -> out :(batchsize, sequenz size/segment_size, number of variables*segment_size)
        E.g. segment_size = 10: in=(batchsize, 200, 5) -> out=(batchsize, 20, 50)
        if sequenz size is not dividable by segment_size. Sequenz size is filed up with fill_up_value

        Overlapping:
        overlapping_segments= 1 -> Output size is in=(batchsize, 200) -> out=(batchsize, (20 * 2)-1, 10) with segment_size = 10: 
        Overlapping : overlapping_segments= 1 
        __00__ __01__ __02__ __03__ __04__  -> __00__ __10__  __01__ __11__ ... __13__ __04__
                ^      ^      ^      ^
            __10__ __11__ __12__ __13__ 

        Overlapping : overlapping_segments= 2
        __00__ __01__ __02__ __03__ __04__  -> __00__ __10__ __20__  __01__ __11__ __21__ ...  __03__ __13__ __23__   __04__
           __10__ __11__ __12__ __13__ 
               __20__ __21__ __22__ __23__ 
    """
    data=np.array(data)

    if len(data.shape) == 2:

        if overlapping_segments != 0:

            new_data = []
            overlapping_segments += 1
            shape = data.shape

            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size)) + fill_up_value
            input_data_filled_up = np.append(data, fillup, axis=-1)

            batch_size = input_data_filled_up.shape[0]
            seq_size = input_data_filled_up.shape[1]
            new_seq_size = seq_size//segment_size

            new_data += [np.array(np.reshape(input_data_filled_up,(batch_size, new_seq_size, segment_size)))]
            print("shape original segmentation seq:",(new_data[0]).shape)
            data=np.zeros((batch_size,(new_seq_size*overlapping_segments)-(overlapping_segments-1),segment_size))

            shift = segment_size//(overlapping_segments)
            rest=0
           
            if segment_size%overlapping_segments:
                rest = 1

            for i in range(overlapping_segments-1):
               
                cropped = input_data_filled_up[:,shift*(i+1):-(segment_size - (shift*(i+1))+rest)]
                new_data += [np.array(np.reshape(cropped,(batch_size, (new_seq_size-1), segment_size)))]

                print("shape of overlapping seq", i, ":",(new_data[i+1]).shape)

            for batch in range(batch_size):
                for seq in range(new_seq_size-1):
                    for overlap in range(overlapping_segments):
                            data[batch,seq+overlap,:] = new_data[overlap][batch,seq,:]

                data[batch,seq+overlap,: ] = new_data[0][batch,-1,:]

            print("shape of result: ", np.array(data).shape)
            return np.array(data)   

        else:
            shape = data.shape
            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size)) + fill_up_value
            data = np.append(data, fillup, axis=-1)
            data = np.array(np.reshape(data,(data.shape[0], data.shape[1]//segment_size, segment_size)))
            return data

    elif len(data.shape) == 3:

        if overlapping_segments != 0:

            raise "Not yet implemented "

        else:
            shape = data.shape
            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size, shape[2])) + fill_up_value
            data = np.append(data, fillup, axis=1)
            data = np.array(np.reshape(data,(data.shape[0], data.shape[1]//segment_size, data.shape[2]*segment_size)))
            return data
    else: 

        print("Input dim error of add_time_stamp")
        result = None


    return result

    


#2D Data: input shape = (number of variablen, sequenz size) Output shape = (number of variablen+1, sequenz size)  
#3D Data: input shape = (batchsize, sequenz size, number of variablen) Output shape = (batchsize, sequenz size, number of variablen+1)  
def add_time_stamp(data, lower_value = -1):
    """Adds a time stamp to the input data.

    Args:
        data (np.ndarray): Input data.
        lower_value (int, optional): Lower bound of the time stamp. Defaults to -1.

    Returns:
        np.ndarray: Data with the time stamp added.

    Info:
        #2D Data: input shape = (number of variablen, sequenz size) Output shape = (number of variablen+1, sequenz size)  
        #3D Data: input shape = (batchsize, sequenz size, number of variablen) Output shape = (batchsize, sequenz size, number of variablen+1) 
    """

    data = np.array(data)

    if len(data.shape) == 2:

        length = data.shape[0]
        
        Position = np.linspace(lower_value,1,length)
        Position = Position[:, np.newaxis]

        result = np.concatenate(( data[:,:],Position[:,:] ), axis=-1)

    elif len(data.shape) == 3:

        length = data.shape[1]
        batch_size= data.shape[0]
        Position = np.linspace(lower_value,1,length)
        Position = Position[np.newaxis,:]
        Position = eo.repeat(Position, '1 d -> batch d', batch = batch_size)

        result = np.concatenate(( data,Position[:,:,np.newaxis] ), axis=-1)
            
    else: 

        print("Input dim error of add_time_stamp")
        result = None

    return result



def resample_timeseries(data, downsample_factor, axis=-1, methode="fft"):
    """Resamples the time series data along the specified axis.

    Args:
        data (np.ndarray): Input time series data.
        downsample_factor (int): Factor by which to downsample the data.
        axis (int, optional): Axis along which to perform resampling. Defaults to 0.
        methode (str, optional): Given methode of resampling, Values: "fft", "delete"

    Returns:
        np.ndarray: Resampled time series data.
    """

    original_shape = data.shape
    if methode == "fft":
        if len(original_shape)==1:
            new_size=int(len(data)/downsample_factor)
            return resample(data, int(new_size))

        elif len(original_shape)==2:
            transposed=False
            if axis==0:
                data = data.T
                transposed=True

            resampled_array=[]
            new_size=int(data.shape[1]/downsample_factor)

            for i in range(data.shape[0]):        
                resampled_array += [resample(np.array(data)[i], int(new_size))] 

            if transposed==True:
                resampled_array=np.array(resampled_array).T

            return np.array(resampled_array)
        
        elif len(original_shape)>2:
            #transpose resample axis at last position
            transpose_index=[]
            for i in range(len(original_shape)):
                if i != axis:
                    transpose_index += [i]
            transpose_index+=[axis]     
            print(transpose_index)

            data=np.transpose(np.array(data),transpose_index)
            transposed_shape = data.shape


            #reduce to 2 dimesions
            new_size=1
            for i in range(len(data.shape)-1):
                new_size*=data.shape[i]
            data = np.reshape(data, newshape=(new_size, data.shape[-1]))
            

            #resample
            new_size=int(len(data[0])/downsample_factor)
            resampled_array=[]
            for i in range(data.shape[0]):        
                resampled_array += [resample(data[i,:], int(new_size))] 
            resampled_array = np.array(resampled_array)
            

            #reverse Reshape
            new_shape=list(transposed_shape)
            new_shape[-1]=new_size
            resampled_array = np.reshape(resampled_array, newshape=new_shape)

            #reverse transpose
            resampled_array=np.transpose(resampled_array, transpose_index)
            
    elif methode=="delete":

        if len(original_shape)==1:
            return data[::downsample_factor]
        else: 
            raise Exception(">2dim not implemented")
    else:
        raise ValueError("unknown methode")

    return resampled_array




def mask_time_series_sections(data, relativ_mask_lenght = [0.20], variablen_to_mask_index_list:list=None):
    """Masks sections of the time series data.
    Args:
        data (np.ndarray): Input time series data. shape(samples, window, variablen)
        relativ_mask_lenght (list, optional): List of relative mask lengths. Defaults to [0.20].

    Returns:
        np.ndarray: Time series data with masked sections.
    """

    _2d_data=False
    data = np.array(data)
    if len(data.shape) == 2:
        data[:,:,np.newaxis]
        _2d_data=True

    if not variablen_to_mask_index_list is None:
        var_to_mask = np.zeros(shape=(data.shape[-1]), dtype=bool)
        for i in variablen_to_mask_index_list:
            var_to_mask[i] = 1


    if len(data.shape) == 3:
        seq_len = len(data[0,:,0])        
        for d in range(len(data)):
            #if d == 0: #Data Vaildation Example
               #data[d,60:60+37,:] = 0
               #continue            
            if np.random.random()>0.15:   #15% no masking
                used_mask=np.random.randint(0,len(relativ_mask_lenght))
                for i in range(used_mask+1): 
                    used_relativ_mask_lenght = relativ_mask_lenght[used_mask] * (1 +  (np.random.random()-0.5)/ 5  )  #+- 10% Mask size
                    mask_start = int( np.random.randint(0, int(seq_len-(seq_len*used_relativ_mask_lenght))))
                    mask_stop = int(mask_start + (seq_len*used_relativ_mask_lenght))

                    if np.random.random()<0.15: #15% Random value
                        data[d,mask_start:mask_stop,var_to_mask]=np.random.random()
                    else:
                        data[d,mask_start:mask_stop,var_to_mask]=0
    
    if _2d_data:
        data = np.squeeze(data)
    return data   






def to_multiple_timeseries_reverse(data, stride_of_window=None, pred_into_future=None, input_window=None, past_window=None):
    """ Reverses of split on large time series into multiple
        Rebuild the splittet time series into one Large

  
    Args:
        data (np.ndarray): Data of shape (Samples, window, ...)

    Returns:
        np.ndarray: One large time series
    """

    if not past_window is None:

        faktor_sliding = (input_window+past_window) // stride_of_window 
        faktor_past = (input_window+past_window)//input_window

        #print("faktor_sliding", faktor_sliding)
        #print("faktor_past", faktor_past)
        test=[]
        for i in range(0,faktor_past):
            
            step=faktor_sliding//faktor_past

            test += [data[(data.shape[0]//faktor_sliding)*i*step:(data.shape[0]//faktor_sliding)*(i*step+1)]]
            #print(test[-1].shape)


        test=np.array(test)        
        if len(test.shape) ==4:
            test=np.reshape(test, (test.shape[0]*test.shape[1],test.shape[2],test.shape[3]) )
        else: 
            test=np.reshape(test, (test.shape[0]*test.shape[1],test.shape[2]) )
        
        result=[]
        for i in range(test.shape[0]//faktor_past):
            for j in range(faktor_past):
                result += [test[i+(test.shape[0]//faktor_past)*j]]

        result=np.array(result)

        if len(test.shape) ==3:
            result=np.reshape(result, (result.shape[0]*result.shape[1],result.shape[2]) )
        else:
            result=np.reshape(result, (result.shape[0]*result.shape[1]) )

        return result

    elif not pred_into_future is None:

        faktor=input_window//pred_into_future
        iteration_lenght = len(data)//faktor
        input_data=data
        result=[]
        for i in range(0, iteration_lenght):
            for j in range(faktor):
                if i+(iteration_lenght*j) >= len(input_data):
                    break
                else:
                    result+=[input_data[i+(iteration_lenght*j)]]
        result = np.array(result)
        shape=result.shape
        if len(shape)==2:
            result = np.reshape(result, (shape[0]*shape[1]) )
        else:
            result = np.reshape(result, (shape[0]*shape[1], shape[2]) )

        return result

    
    else:

        if not stride_of_window is None:

            window_lenght=data.shape[1]
            stride_of_window
            if window_lenght % stride_of_window:
                raise Exception("For rebuild time series window_lenght must be divisible by stride_of_window")
            
            #just use first part
            faktor = window_lenght // stride_of_window 
            #print("original", data.shape)
            data = data[0:data.shape[0]//faktor]
            #print(data.shape)


        shape=list(data.shape)
        shape1 = shape[0]
        shape.pop(0)
        shape[0]=shape[0]*shape1
        
        return np.reshape(data, newshape=shape)







def to_multiple_series(data, split_time_series_into_window_of_len, stride_of_window = None, verbose=False, one_window_from_the_end=False):
    """Split the time series into multiple smaller series.
    Args:
        data (np.ndarray): Input time series data.
        split_time_series_into_window_of_len (int): Target lengths of each new time series.
        stride_of_window (int, optional): Stride. Shift after each split. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of multiple smaller time series.
    """
    #print("\n\n!!!reconstruction not working right know!!!!\n\n")


    if len(np.array(data).shape) == 1:

        splited_data = split_1d_timeSeries_to_multiple(data, split_time_series_into_window_of_len, stride_of_window, verbose, one_window_from_the_end)

    elif len(np.array(data).shape) == 2:

        if data.shape[0]<data.shape[1]:
            data = np.transpose(data, (1,0))

        splited_data = []
        for i in range(data.shape[1]):
            splited_data += [split_1d_timeSeries_to_multiple(data[:,i], split_time_series_into_window_of_len, stride_of_window, verbose, one_window_from_the_end)]

        splited_data=np.array(splited_data)

        splited_data = np.transpose(splited_data, (1,2,0))
    else:
        return 0 

    
    return splited_data








def split_1d_timeSeries_to_multiple(data, window_size, sliding_window_stride = None, verbose=False, one_window_from_the_end=False):
    """Split a 1D time series into multiple smaller time series.

    Args:
        data (np.ndarray): Input 1D time series data.
        target_length_of_splited_timeseries (int): Target length of each split time series.
        time_shift_after_each_split (int, optional): Shift after each split. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of multiple smaller time series.
    """
    
    
    if window_size==sliding_window_stride and one_window_from_the_end:
        one_window_from_the_end = True
    else:
        one_window_from_the_end = False


    data = np.expand_dims(data, axis=1)

    num_shifts = int(window_size/sliding_window_stride)
    num_splits = int(len(data)/window_size)
    
    data_cut = data[0:(num_splits)*window_size]                   #first split
    input_data_filled_up=np.reshape(data_cut, ((num_splits), window_size))

    if one_window_from_the_end:
        last_window=data[-(window_size+1):-1] 
        last_window=np.squeeze(last_window)
        input_data_filled_up=np.append(input_data_filled_up, last_window[np.newaxis, :], axis=0)


    batches = []
    batch_lenght = []
    batches += [input_data_filled_up]
    batch_lenght += [len(batches[-1])]

    indx_shift =  sliding_window_stride-1
    for i in range(num_shifts-1):#second bis x split ## 

        if data.shape[0] >= num_splits*window_size+indx_shift:
            data_cut = data[ indx_shift : num_splits*window_size+indx_shift]
            input_data_filled_up = np.reshape(data_cut, (num_splits, window_size))

        else:#with this value of shift the number of splits must be reduced by one 
            data_cut = data[ indx_shift : num_splits*window_size+indx_shift-window_size]
            input_data_filled_up = np.reshape(data_cut, (num_splits-1, window_size))

        batches += [input_data_filled_up]
        batch_lenght += [len(batches[-1])]
        indx_shift += sliding_window_stride       #Shift
        
    
    max_lenght=max(batch_lenght)
    #max_lenght=min(batch_lenght)# cut the batches of size of smallest, else reconstruction is almost impossible
    #print("\n\n!!!rekonstruction not working right know!!!!\n\n")
    result = np.array(batches[0][0:max_lenght])

    for i in range(1, len(batches)):

        result= np.append(result, np.array(batches[i][0:max_lenght]), axis=0)    

    i = result.shape[0]

    if verbose:

        print("Finished.", "Total splits:", i)
    
    return result



'''



def split_1d_timeSeries_to_multiple(data, target_lenght_of_splited_timeseries, time_shift_after_each_split = None, verbose=False):
    """Split a 1D time series into multiple smaller time series.

    Args:
        data (np.ndarray): Input 1D time series data.
        target_length_of_splited_timeseries (int): Target length of each split time series.
        time_shift_after_each_split (int, optional): Shift after each split. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of multiple smaller time series.
    """
    data = np.expand_dims(data, axis=1)

    num_shifts = int(target_lenght_of_splited_timeseries/time_shift_after_each_split)
    num_splits = int(len(data)/target_lenght_of_splited_timeseries)
    
    data_cut = data[0:num_splits*target_lenght_of_splited_timeseries]                   #first split
    batches = np.reshape(data_cut, (num_splits, target_lenght_of_splited_timeseries))


    indx_shift =  time_shift_after_each_split-1
    for i in range(num_shifts-1):#second bis x split

        if data.shape[0] >= num_splits*target_lenght_of_splited_timeseries+indx_shift:
            data_cut = data[ indx_shift : num_splits*target_lenght_of_splited_timeseries+indx_shift]
            input_data_filled_up = np.reshape(data_cut, (num_splits, target_lenght_of_splited_timeseries))
        else:#with this value of shift the number of splits must be reduced by one 
            data_cut = data[ indx_shift : num_splits*target_lenght_of_splited_timeseries+indx_shift-target_lenght_of_splited_timeseries]

            input_data_filled_up = np.reshape(data_cut, (num_splits-1, target_lenght_of_splited_timeseries))


        batches = np.append(batches, input_data_filled_up, axis=0)
        indx_shift += time_shift_after_each_split       #Shift
        

    i = batches.shape[0]

    if verbose:

        print("Finished.", "Total splits:", i)
    
    return batches


'''

''' bugs
def min_max_scaler_3d_data(self, data, is_test_data, scale_mode="scale_of_each_variable", index_list_of_variables_to_scale=None, ):
        """scale data to 0 to 1

        Input: 
        data: format (samples, windows, variables)

        Args:
        scale_mode: -scale_of_each_window_and_variable: scales each variable of each window separate
                    -scale_of_each_variable: scales each variable across windows
        index_list_of_variables_to_normalize: Index list of which variable should be scaled 

        Returns:
            np.ndarray: scaled data
        """


        data = np.array(data)
        shape=data.shape
        if  index_list_of_variables_to_scale is None:
            index_list_of_variables_to_scale = range(shape[-1])
        
        scaled_data=[]
        if scale_mode == "scale_of_each_window_and_variable":
            for sample in range(shape[0]):
                sample_result = []
                for variable in index_list_of_variables_to_scale:  
                    if is_test_data:
                        sample_result += [self.minmaxscaler.transform([data[sample, : ,variable]])[0]]
                    else:
                        sample_result += [self.minmaxscaler.fit([data[sample, : ,variable]])[0]]

                sample_result = np.array(sample_result)
                scaled_data += [sample_result]
            scaled_data=np.array(scaled_data)
            scaled_data = np.transpose(scaled_data, (0,2,1))
        elif scale_mode == "scale_of_each_variable":
            variable_result=[]
            for variable in index_list_of_variables_to_scale:  
                    if is_test_data:
                        variable_result += [self.minmaxscaler.transform(data[:, : ,variable])]
                    else:
                        variable_result += [self.minmaxscaler.fit(data[:, : ,variable])]
            scaled_data=np.array(variable_result)
            scaled_data =np.transpose(scaled_data, (1,2,0))
        else:
            raise ValueError("normalization_mode parameter value error")
        
        return scaled_data
'''


def reconstruct_sliding_window_for_each_stack(data, sliding_window_step, input_window_lenght):


    print("Current implementation of split_1d_timeSeries_to_multiple don't allowes this ")

    '''
    overlapping_faktor = input_window_lenght//sliding_window_step  #512/4 = 128

    shape=data.shape
    data = np.reshape(data, newshape=(shape[0]*shape[1], shape[-1]))
    #original size 28479

    stacked_scores=[]
    offset=0
    data_window=len(data)//overlapping_faktor
    for i in range(overlapping_faktor):

        stacked_scores+=[np.concatenate( (np.zeros((i*sliding_window_step,38)),data[offset:offset+data_window], np.zeros((input_window_lenght-i*sliding_window_step ,38))))]
        offset += data_window

    return np.array(stacked_scores)
    '''
