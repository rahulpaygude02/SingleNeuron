from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from matplotlib.colors import ListedColormap

AND = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,0,0,1]    
}

df = pd.DataFrame(AND)
df

X,y = prepare_data(df)
ETA = 0.3  #Normaly bet 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X,y)

_ = model.total_loss()

save_model(model,'and.model')
save_plot(df,"and.png",model)