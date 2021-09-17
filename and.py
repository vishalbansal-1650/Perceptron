from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd

def main(data, eta, epochs, modelName, plotName):
    df = pd.DataFrame(data)
    print(df)
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
    
    main(data=AND, eta=ETA, epochs=EPOCHS, modelName='AND.model', plotName='AND.png')

