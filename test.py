import sklearn.datasets as dataset
from sklearn.model_selction import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
boston=dataset.load_boston()
#print("boston dataset{}").format(boston.DESCR))
trainx,testx,trainy,testy=train_test_split(boston.data,boston.target,test_size=0.2,random_state=42)
lr=LinearRegrassion()
lr.fit(trainx,trainy)
print("training complete")
predy=lr.predict(testx)
mse=metrics.mean_squared_error(testy,predy)
print("mse=",mse)
