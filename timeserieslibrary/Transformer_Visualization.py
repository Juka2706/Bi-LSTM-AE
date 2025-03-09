import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn import preprocessing    



########Attention visualization ###############

def attention_score_to_1D(attention_score, which_layer="all", which_head="all", layer_head_merge_mode = "mean", remove_first_token=False):
    """Converts attention scores to a 1D array.

    Args:
        attention_score (np.ndarray): The attention scores. Dim: (samples, num_layer, num_head, seq_size, seq_size)
        which_layer (str or int): Index or "all" for the layer to consider.
        which_head (str or int): Index or "all" for the head to consider.
        layer_head_merge_mode (str): The method to merge layers and heads ("mean" or "max").
        remove_first_token (bool): Whether to remove the first token if a cls_token is used.

    Returns:
        np.ndarray: The 1D attention scores.
    """
    list_result=[]

    if remove_first_token:
        attention_score = attention_score[:,:,:,1:,1:]

    for j in range(len(attention_score)):#For all Samaples

        if isinstance(which_layer,str) and which_layer=="all":
            if layer_head_merge_mode == "max":
                result = np.max(attention_score[j], axis=0)
            elif layer_head_merge_mode == "mean":
                result = np.mean(attention_score[j], axis=0)
            else:
                print("error")
        else: 
            result =  attention_score[j,which_layer,:,:,:]

        if isinstance(which_head,str) and which_head=="all":
            if layer_head_merge_mode == "max":
                result = np.max(result, axis=0)
            elif layer_head_merge_mode == "mean":
                result = np.mean(result, axis=0)
            else:
                print("error")
        else: 
            result =  result[which_head,:,:]


        line_score = np.mean(result, axis=0) 
        #line_score_1 = np.sum(result, axis=1)#Other axis. Does not make sense!?!?
        #line_score=line_score+line_score_1
        
        list_result += [line_score]   

    list_result = np.array(list_result)
   
    return list_result


def __chose_x_example_from_each_class(data, label, number_of_samples_per_class_to_return):
    """Selects examples from each class.

    Args:
        data: data, input shape: (samples, ....) or list of data. input shape: ( (data1_samples, ....), (data2_samples, ....), ... )
        label: The labels of the data. one hot encoded shape
        number_of_samples_per_class_to_return (int): Number of selected examples per class.

    Returns:
        tuple: Selected data and labels.
        if input is not list: shape (  number_of_samples_to_return*num_classes, ...)
        if input is list:     shape ( (data1_number_of_samples_to_return*num_classes, ...) , (data2_number_of_samples_to_return*num_classes, ...), ... )
    """
    is_input_data_list = isinstance(data, list)
    
    if not is_input_data_list:
        data=[data]    

    num_classes = len(label[0])
    label=np.argmax(label,axis=-1)

    result_label=[]
    result_data=[[] for _ in range(len(data))]
    output_shape_for_each_data = []

    for cls in range(num_classes):

        indexes = np.where(label == cls)
        indexes = indexes[0][0:number_of_samples_per_class_to_return]
        result_label += [label[indexes]]

        for d in range(len(data)):            
            result_data[d] += [data[d][indexes]]

            new_shape = list(np.array(data[d]).shape)
            new_shape[0] = number_of_samples_per_class_to_return * num_classes
            output_shape_for_each_data += [new_shape]
    
    result_label = np.array(result_label)
    result_label = np.reshape(result_label, newshape=(result_label.shape[0]*result_label.shape[1],-1))

    for d in range(len(data)):
        result_data[d] = np.array(result_data[d])
        result_data[d] = np.reshape(result_data[d], newshape=output_shape_for_each_data[d])

    if is_input_data_list:        
        return result_data, np.array(result_label)    
     
    return result_data[0], np.array(result_label)

    
def __overlapping_attn_score_segments_to_attn_score_per_point(attention_scores_1d, time_series_lenght, segment_size, segment_stride, segment_overlapping_merge_mode="mean" ):
    """Converts overlapping attention scores to attention scores per point.
    The attention scores overlaps because of the overlapping input segments

    Args:
        attention_scores_1d (np.ndarray): 1D attention scores. (samples, seq_size_within_the_segment)
        time_series_lenght (int): Length of the time series.
        segment_size (int): Segment size.
        segment_stride (int): Segment strides.
        segment_overlapping_merge_mode (str): The method to merge overlapping ("mean" or "max").
    Info:
        If transformer uses overlapping segments. Multiple attention scores per point in time series exists. This function merges the attention scores for each point.
        segment_overlapping_merge_mode: Merging mode: "max" -> take the maximum attention score for the dqatapoint
                                                      "mean" -> take the mean over all existing attenion scores for the respective point in time series
    Returns:
        np.ndarray: Attention scores per point. (samples, time_series_seq_size)
    """
    attention_scores_per_point=[]
    for sample in range(len(attention_scores_1d)):

        #Example Segment size 9 (---------) stride 4. Gaps (size=3) filled with 0
        #One segment (--------) is represent by one number in attention score
        #Input attention score:
        #----0-----,----1-----,----2-----,----3-----,----4-----,----5-----,-----6----

        attention_score = attention_scores_1d[sample]        
        segment_overlapping_factor =segment_size//segment_stride #aufrunden

        gaps_in_the_layer=0
        if segment_size%segment_stride:
            gaps_in_the_layer  = segment_stride-(segment_size%segment_stride)
            segment_overlapping_factor+=1

        attn_score_overlapping_per_layer=[]
        for i in range(segment_overlapping_factor):            
            attn_score_overlapping_per_layer += [attention_score[i::segment_overlapping_factor]]

        #Example Segment size 9 (---------) stride 4. Gaps (size=3) filled with 0
        #One segment (--------) is represent by one number in attention score
        #attention score after breakdown in layer:
        #----0-----,----3-----      Layer 0
        #----1-----,----4-----      Layer 1
        #----2-----,----5-----      Layer 2
        attn_score_per_layer_point_granular=[]
        for i in range(segment_overlapping_factor):
            attn_score_point_granular=[]
            for j in range(len(attn_score_overlapping_per_layer[i])):
                for _ in range(segment_size):
                    attn_score_point_granular += [attn_score_overlapping_per_layer[i][j]]
                for _ in range(gaps_in_the_layer): # fill up gaps in the layer with zero #just happens if Segment_size%Segment_stride > 0
                    attn_score_point_granular += [0]

            attn_score_per_layer_point_granular += [attn_score_point_granular]


        #Example Segment size 9 (---------) stride 4. Gaps (size=3) filled with 0
        #One segment (--------) is represent by 9 number in attention score
        #attention score after break down to each data point and fill up with zeros:
        #----0-----000----3-----              Layer 0
        #0000---1------000---4------          Layer 1
        #00000000---2------000----5-----      Layer 2

        attn_score_per_layer_padded = []
        for i in range(segment_overlapping_factor):
            attn_score_per_layer_padded += [np.append(np.zeros(segment_stride*i), np.array(attn_score_per_layer_point_granular[i]))]
        

        #scale to time_series_lenght
        for i in range(segment_overlapping_factor):       
            attn_score_per_layer_padded[i] = attn_score_per_layer_padded[i][0:time_series_lenght]
            #diff = time_series_lenght-attn_score_per_layer_padded[i] 
            #if diff > 0:
            #    attn_score_per_layer_padded[i] = np.append(attn_score_per_layer_padded[i], np.zeros(diff) )
        
        attn_score_per_layer_padded=np.array(attn_score_per_layer_padded)


        #Example Segment size 9 (---------) stride 4. Gaps (size=3) filled with 0
        #One segment (--------) is represent by 9 number in attention score
        #attention score:
        #----0-----000----3-----              Layer 0
        #           +
        #0000----1-----000----4-----          Layer 1
        #           +
        #00000000----2-----000----5-----      Layer 2
        # calculate mean over the layer
        #-----------------------
        # 

        if segment_overlapping_merge_mode == "max":
            attn_score_per_point=np.max(attn_score_per_layer_padded, axis=0)      
        elif segment_overlapping_merge_mode == "mean":
            attn_score_per_point=np.sum(attn_score_per_layer_padded, axis=0)/segment_overlapping_factor
        else:
            print("segment_overlapping_merge_mode: mean or max")
 
        attention_scores_per_point+=[attn_score_per_point]

    return np.array(attention_scores_per_point)




def visualize_attention_score_within_timeseries(time_series, attention_scores_1d, segment_size, segment_stride, number_of_samples_to_plot_from_each_class=None, label=None, prediction_to_print_with_plot=None ,segment_overlapping_merge_mode="mean", attn_score_normalization_mode_for_visualization="sample"):
    """Visualizes attention scores into the respective time series.

    Args:
        time_series (np.ndarray): The time series.
        attention_scores_1d (np.ndarray): 1D attention scores.
        segment_size (int): Segment size.
        segment_stride (int): Segment strides.
        number_of_samples_to_plot_from_each_class (int): Number of selected examples per class.
        label (np.ndarray): The labels of the data.
        prediction_to_print_with_plot (np.ndarray): The predicted labels.
        segment_overlapping_merge_mode (str): The method to merge the attention of overlapping segment  ("mean" or "max").
        attn_score_normalization_mode_for_visualization (str): The normalization mode for visualization ("sample" or "batch").
    """

    cmap = cm.get_cmap(plt.rcParams["image.cmap"])

    time_series_len=len(time_series[0])
    attention_scores_1d=__overlapping_attn_score_segments_to_attn_score_per_point(attention_scores_1d,time_series_len, segment_size, segment_stride, segment_overlapping_merge_mode=segment_overlapping_merge_mode )

    if attn_score_normalization_mode_for_visualization == "batch":
        norm = Normalize(vmin=attention_scores_1d.min(), vmax=attention_scores_1d.max())


    if not number_of_samples_to_plot_from_each_class is None and not label is None :
        if not prediction_to_print_with_plot is None :
            data, label = __chose_x_example_from_each_class([time_series,attention_scores_1d,prediction_to_print_with_plot], label, number_of_samples_to_plot_from_each_class)
            prediction_to_print_with_plot = data[2]
        else:
            data, label = __chose_x_example_from_each_class([time_series,attention_scores_1d], label, number_of_samples_to_plot_from_each_class)

        time_series=data[0]
        attention_scores_1d=data[1]

    for sample in range(len(attention_scores_1d)):

        time_serie = np.squeeze(time_series[sample])
        attention_score=attention_scores_1d[sample]

        if attn_score_normalization_mode_for_visualization == "sample":
            norm = Normalize(vmin=attention_score.min(), vmax=attention_score.max())

        time_serie=np.append(time_serie, np.zeros(segment_size))
        
        y= []
        x= []
        rgba_values= []
        for segment in np.arange(1, time_series_len, segment_stride):    
                
                rgba_values += [cmap(norm(attention_score[segment]))]
                y += [np.arange(segment-1,segment+segment_stride)]
                x += [time_serie[segment-1:segment+segment_stride]]


        if not prediction_to_print_with_plot is None:
            print("Prediction class:", np.argmax(prediction_to_print_with_plot[sample]))        
        if not label is None :
            print("Ground truth class:", label[sample][0])
            
        for i in range(len(y)):
            plt.plot(y[i],x[i], color=rgba_values[i])    

        plt.xlabel("Time")
        plt.ylabel("Normalized Amplitude")

        plt.show()



def visualize_attention_with_timeseries_old(time_series, attention_scores, segment_size, segment_stride, number_of_samples_to_plot_from_each_class=None, label=None):
    """Visualizes encoder output feature vectors within a time series with PCA to RGB.


    Args:
        time_series (np.ndarray): The time series.
        encoder_output_feature_vector (np.ndarray): The encoder output feature vectors.
        segment_size (int): Segment size.
        segment_stride (int): Segment strides.
        number_of_samples_to_plot_from_each_class (int): Number of selected examples per class.
        label (np.ndarray): The labels of the data.
        prediction_to_print_with_plot (np.ndarray): The predicted labels.
    """

    if not number_of_samples_to_plot_from_each_class is None and not label is None :
        data, label = __chose_x_example_from_each_class([time_series,attention_scores], label, number_of_samples_to_plot_from_each_class)

    time_series =data[0]
    attention_scores=data[1]


    for sample in range(len(attention_scores)):
        
        time_serie = np.squeeze(time_series[sample])
        attention_score = attention_scores[sample]
        norm = Normalize(vmin=attention_score.min(), vmax=attention_score.max())
        cmap = cm.get_cmap(plt.rcParams["image.cmap"])
        y= []
        x= []
        rgba_values= []
        i=0
        for segment in np.arange(segment_size, len(time_serie), segment_stride):    

                rgba_values += [cmap(norm(attention_score[i]))]
                y += [np.arange(segment-segment_size,segment)]
                x += [time_serie[segment-segment_size:segment]]
                i += 1 

        for i in range(len(y)):
            plt.plot(y[i],x[i], color=rgba_values[i])                    
        plt.show()





######## Encoder Output Feature Vector Visualization###############



def visualize_encoder_output_feature_vector_within_timeseries_with_PCA_to_RRG(time_series, encoder_output_feature_vector, segment_size, segment_stride, number_of_samples_to_plot_from_each_class=None, label=None, prediction_to_print_with_plot=None):
    """Visualizes encoder output feature vectors within a time series with PCA to RGB (reduction to 3 dim and normalization -> RGB values for segments in time series)

    Args:
        time_series (np.ndarray): respective time series(samples, time_series_seq_size)
        encoder_output_feature_vector (np.ndarray): Encoder output feature vector (samples, number of tokens, model dim)
        segment_size (int): Segment size.
        segment_stride (int): Segment strides.
        number_of_samples_to_plot_from_each_class (int): Number of selected examples per class.
        label (np.ndarray): The labels of the data.
        prediction_to_print_with_plot (np.ndarray): The predicted labels.
    """
    components= 3
    pca = PCA(n_components=components)

    shape=encoder_output_feature_vector.shape

    fv = np.reshape(encoder_output_feature_vector,newshape=(shape[0]*shape[1], shape[2]),order='C')
    scaled_fv = preprocessing.scale(fv,axis=0)
    pca_result = pca.fit(scaled_fv)
    pca_result = pca.transform(scaled_fv)
    pca_result = np.reshape(pca_result,newshape=(shape[0],shape[1], components),order='C')

    norm = Normalize(vmin=pca_result.min(), vmax=pca_result.max())
    RGB_values = norm(pca_result)

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    __plot_time_series_within_RGB_value(time_series, RGB_values, segment_size, segment_stride, number_of_samples_to_plot_from_each_class, label, prediction_to_print_with_plot)



def __plot_time_series_within_RGB_value(time_series, RGB_values, segment_size, segment_stride, number_of_samples_to_plot_from_each_class=None, label=None, prediction_to_print_with_plot=None):
    """Plots time series within RGB value.

    Args:
        time_series (np.ndarray): The time series.
        RGB_values (np.ndarray): The RGB values.
        segment_size (int): Segment size.
        segment_stride (int): Segment strides.
        number_of_samples_to_plot_from_each_class (int): Number of selected examples per class.
        label (np.ndarray): The labels of the data.
        prediction_to_print_with_plot (np.ndarray): The predicted labels.
    """
    time_series_len=len(time_series[0])

    if not number_of_samples_to_plot_from_each_class is None and not label is None :
        if not prediction_to_print_with_plot is None :
            data, label = __chose_x_example_from_each_class([time_series,RGB_values,prediction_to_print_with_plot], label, number_of_samples_to_plot_from_each_class)
            prediction_to_print_with_plot = data[2]
        else:
            data, label = __chose_x_example_from_each_class([time_series,RGB_values], label, number_of_samples_to_plot_from_each_class)

        time_series=data[0]
        RGB_values=data[1]


    for sample in range(len(RGB_values)):

        time_serie = np.squeeze(time_series[sample])
        pca_result=RGB_values[sample]
        time_serie=np.append(time_serie, np.zeros(segment_size))
               
        y= []
        x= []
        rgba_values= []
        for segment in np.arange(1, time_series_len, segment_stride):                    
                index=int(np.round(segment/segment_stride,0))
                rgba_values += [pca_result[index]]
               
                y += [np.arange(segment-1,segment+segment_stride)]
                x += [time_serie[segment-1:segment+segment_stride]]


        if not prediction_to_print_with_plot is None:
            print("Prediction class:", np.argmax(prediction_to_print_with_plot[sample]))        
        if not label is None :
            print("Ground truth class:", label[sample][0])
            
        for i in range(len(y)):
            plt.plot(y[i],x[i], color=rgba_values[i])    


        plt.xlabel("Time")
        plt.ylabel("Normalized Amplitude")

        plt.show()







