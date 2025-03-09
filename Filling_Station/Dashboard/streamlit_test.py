#import streamlit as st
#import streamlit as st
import requests
import time
import json
import numpy as np
import streamlit as st
import pandas as pd
import plotly_express as px
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [15, 7]





VARIABLE=0

class Charts():
    def  __init__(self):
            

        #config
        with open("..\config.json", "r") as file:

            data = json.load(file)
            #config       
            self.window_size=data['split_time_series_into_window_of_len']
            self.threshold =list(data['thresholds'].values()) 
            self.number_of_variables = len(list(data['ground_truth_variable_index'].values()))
            self.names_of_var =list(data['ground_truth_name_list'].values())    
            self.sliding_window_stride =data['sliding_window_stride_deployment']     
            file.close()


        #Interne var/buffer
        self.max_window_lenght_reached = False
        self.plot_data_storage = None
        self.max_window_lenght = self.window_size*5
        self.displayed_windows=self.max_window_lenght/self.window_size
        self.input_count = 0

        
        #Charts            
        self.y_and_pred_chart=[]
        self.mse_storage_all_list=[]
        

        st.write("Visualize model prediction")
        for i in range(self.number_of_variables):             
            st.write(self.names_of_var[i])
            self.y_and_pred_chart += [st.empty()]


    def plot_data(self, data):

        #anomalie_data_storage_x_data = np.array(data[0])
        anomalie_data_storage_y_data  = np.array(data[1])
        anomalie_data_storage_prediction  =  np.array(data[2])
        mse_data_storage=  np.array(data[3])

        for i in mse_data_storage:
            self.mse_storage_all_list+=[i]

        self.mse_storage_all=np.array(self.mse_storage_all_list)
        
        new_data_to_plot=False    
        num_mse_in_future=0



        for i in range(len(anomalie_data_storage_y_data)):

            
            if self.plot_data_storage is None:
                #Add prediction to ground truth in new axis
                self.plot_data_storage=np.append(anomalie_data_storage_y_data[i,:,:,np.newaxis],anomalie_data_storage_prediction[i,:,:,np.newaxis], axis=-1)
                new_data_to_plot=True
            else:
                #if stride, don't take every prediction for Visualisation
                if (self.input_count) % (self.window_size // self.sliding_window_stride) == 0:
                    plot_data= np.append(anomalie_data_storage_y_data[i,:,:,np.newaxis],anomalie_data_storage_prediction[i,:,:,np.newaxis], axis=-1)
                    self.plot_data_storage = np.append(self.plot_data_storage, plot_data, axis=0)
                    new_data_to_plot=True
                    
                    num_mse_in_future=0
                else:
                    num_mse_in_future+=1

            self.input_count +=1
            

        if new_data_to_plot:


            for var in range(self.plot_data_storage.shape[1]):


                if len(self.plot_data_storage) > self.max_window_lenght:
                    self.plot_data_storage=self.plot_data_storage[-self.max_window_lenght:]

                plt.close('all')
                fig, (ax, ax1, ax2) = plt.subplots(3, gridspec_kw={'height_ratios': [3, 1, 1]})

                ax.plot(self.plot_data_storage[:,var,:])  
                plot_max=np.max(self.plot_data_storage[:,var,:])
                ax.set_ylim([-0.05, plot_max*1.2])
                ax.set_ylabel('Normalized Amplitude')  
                ax.title.set_text("Model input and Prediction")
                ax.legend([self.names_of_var[var], "Model Prediction", "Anomalie of Window"])
                            

                if not self.max_window_lenght_reached:
                    shift=self.window_size // 2                        
                    num_of_mse_values_in_plot = len(self.mse_storage_all[:,var])        
                else:
                    shift = 0            
                    num_of_mse_values_in_plot = int((len(self.plot_data_storage) / self.sliding_window_stride) - (self.window_size / self.sliding_window_stride) / 2) +1


                if num_mse_in_future == 0:
                    relevant_mse_values = self.mse_storage_all[-(num_of_mse_values_in_plot):,var]
                else:
                    relevant_mse_values = self.mse_storage_all[-(num_of_mse_values_in_plot+num_mse_in_future):-(num_mse_in_future),var]

                if len(self.plot_data_storage) >= self.max_window_lenght:
                    self.max_window_lenght_reached=True

                
                for mse_value in relevant_mse_values:
                    if mse_value > self.threshold[var]:
                        #ax.plot([shift],[1], "r", marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
                        ax1.plot([shift],[mse_value], "r", marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
                    else:
                        ax1.plot([shift],[mse_value], "g", marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                    shift+=self.sliding_window_stride
                                    
                ax1.plot(np.ones(len(self.plot_data_storage))*self.threshold[var], "orange")
                ax1.set_xlabel('Time')  
                ax1.set_ylabel('MSE/Anomalie Score of Windows')  
                ax1.title.set_text("MSE of Window")
                ax1.legend(["MSE/Anomalie Score of Windows"])
                if np.max(relevant_mse_values)>(self.threshold[var]*2)/1.2:
                    max_val = np.max(relevant_mse_values)*1.2
                else:
                    max_val = self.threshold[var]*2
                ax1.set_ylim(0, max_val)     


                ax2.axvline(self.threshold[var], color='r', linestyle='dashed', linewidth=1)
                bins = np.linspace(0, self.threshold[var] * 5, 50)     
                
                mse_normal=[]
                mse_anoamrl=[]
                for mse in self.mse_storage_all[:,var]:
                    if mse > self.threshold[var]:
                        if mse > self.threshold[var] * 5:
                            mse_anoamrl+=[self.threshold[var] * 5]
                        else:
                            mse_anoamrl+=[mse]
                    else:
                        mse_normal+=[mse]

                


                ax2.hist(mse_anoamrl, color="r", bins=bins)##, labels=label)#, range=[min_,max_])        
                ax2.hist(mse_normal,  color="g", bins=bins)##, labels=label)#, range=[min_,max_])        
                ax2.set_xlabel('MSE')  
                ax2.set_ylabel('Absolute Frequency/Anomalie Score')  
                ax2.title.set_text("Histogram of  MSE")
                fig.tight_layout(pad=1.0)
                self.y_and_pred_chart[var].pyplot(fig)


        


    def load_config_from_JSON(self, file_name):
            
        if os.path.exists(file_name) == True:
            with open("config", "r") as file:

                data = json.load(file)                    
                self.number_of_variables= len(list(data['ground_truth_variable_index'].values()))
                self.window_size=data['split_time_series_into_window_of_len']
                self.threshold=len(list(data['thresholds'].values()))

                self.sliding_window_stride=self.window_size//8
                self.threshold = [0.001615705, 0.008319296, 0.008495063, 0.0074707232]   
                self.names_of_var = ('Gesamtstrom', 'Integerwert für den Motorstrom am Transportförderband', 'Integerwert für den Motorstrom am Zuführförderband', 'Integerwert für den Motorstrom am Drehtisch')


# Streamlit-App
def main():

    st.title("Anomalie Detection: Filling Station")
    conection_status = st.empty()


    #while( not connected )
        #konfig = get_konfig
    
    charts = Charts()
    x=None
    

    while True:

        try: 
            x = requests.get('http://127.0.0.1:8000/getdata')      
            conection_status.write(":green[Connected]")         


        except:
            print("Not Connected")
            conection_status.write(":red[Not Connected]")

        
        if x != None and json.loads(x.content) != None:
            charts.plot_data(json.loads(x.content))
            
            
        time.sleep(1)


tab1, tab2, tab3 = st.tabs(["Filling Station", "MPS-PA", "   "])



with tab1:

    var_names=("Gesamtstrom", "Integerwert für den Motorstrom am Transportförderband", "Integerwert für den Motorstrom am Drehtisch", "Integerwert für den Motorstrom am Zuführförderband" )

    sidebar= st.sidebar
    with sidebar:
        sidebar.button("Live anomaly detection")
        #var_names.index(  st.sidebar.selectbox("Choose variable", var_names))
        print(VARIABLE)
        st.write()


    if __name__ == "__main__":
        main()

