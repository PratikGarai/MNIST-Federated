import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# AUxillary methods
#Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
#         hist = r.history
#         print("Fit history : " ,hist)
#         return model.get_weights(), len(x_train), {}
    def fit(self):       #, parameters, config):
        result = funcs.fit(swift_data_path = swift_data_path,
            bank_data_path = bank_data_path,
            model_dir = model_dir,
            preds_format_path = preds_format_path,
            preds_dest_path = preds_dest_path,
            m = 'xgboost')
        return result

#     def evaluate(self, parameters, config):
    def evaluate(self):     #, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
#         print("Eval accuracy : ", accuracy)
        result = funcs.predict(
            swift_data_path =swift_data_path,
            bank_data_path = bank_data_path,
            model_dir = model_dir,
            preds_format_path = preds_format_path,
            preds_dest_path = preds_dest_path,
            m = 'xgboost'
            )  
        #return loss, len(x_test), {"accuracy": accuracy}
        return result
    
    # Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
