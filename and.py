"""
  author : Vishal

"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs'
logging.basicConfig(filename=os.path.join(logs_dir,'running_logs_AND.log'),level=logging.INFO,format=logging_str,filemode='a')


def main(data, eta, epochs, modelName, plotName):
    df = pd.DataFrame(data)
    logging.info(f"This is actual DataFrame \n{df}")
    x,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(x,y)
    _ = model.total_loss()

    save_model(model,modelName)

    save_plot(df,plotName,model)



if __name__ == '__main__':

    AND = {'x1':[0,0,1,1],
       'x2':[0,1,0,1],
       'y':[0,0,0,1]
}

    ETA = 0.3
    EPOCHS = 10

    try:
      logging.info(">>>>>>>>>> Starting Training of the model >>>>>>>>>>")
      main(data=AND, eta=ETA, epochs=EPOCHS, modelName='AND.model', plotName='AND.png')
      logging.info("<<<<<<<<<< Training done successfully <<<<<<<<<<")
    except Exception as e:
      logging.exception(e)
      raise e 


