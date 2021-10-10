from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from matplotlib.colors import ListedColormap
import logging
import os

logging_str= "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir ="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str)

def main(data,eta,epochs):
    

    df = pd.DataFrame(data)
    logging.info(f"this is actual dataframe {df}")

    X,y = prepare_data(df)
    

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss()

    save_model(model,'and.model')
    save_plot(df,"and.png",model)

if __name__=='__main__':

    AND = {
        'x1':[0,0,1,1],
        'x2':[0,1,0,1],
        'y':[0,0,0,1]    
    }
    ETA = 0.3  #Normaly bet 0 and 1
    EPOCHS = 10
    try:
        logging.info("\n>>>>> training started  >>>>>>")
        main(data=AND,eta=ETA,epochs=EPOCHS)
        logging.info(">>>>> training done successfully  >>>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e