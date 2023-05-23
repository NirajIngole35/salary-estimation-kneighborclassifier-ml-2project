import pandas as pd
import numpy as np
#data understanding
pa=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\salary estimation kneighborclassifier ml  2project\dataset\salary estimation.csv")
print(pa.head(5))
print(pa.tail(5))
print(pa.shape)
print(pa.info)
print(pa.describe)

#mapping data (in col.income there are >,= so mapping is needed)

pa['income']=pa['income'].apply(lambda x: 1 if x=='>50K' else 0)
print(pa.head(5))

#x and y
x=pa.iloc[:,:-1].values
y=pa.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)



#find best k values
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

error=[]
for i in range(1,40):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    pre=model.predict(x_test)
    error.append(np.mean(pre!=y_test))

    
plt.plot(range(1,40),error,color='red')
plt.figure(figsize=(12,6))
plt.show()




#modeling
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)


#prediction
predic=model.predict(x_test)


# for new costamer
age=int(input("enter age:-"))
edu=int(input("enter educational-num:-"))
capita=int(input("enter capital-gain:-"))
hourswork=int(input('enter hours-per-week 0-40:-'))

newemp=[[age,edu,capita,hourswork]]
result=model.predict(sc.transform(newemp))
print(result) 
if result==1:
    print('employee might got above 50k')
else:
    print('employee might got below 50k')




    #output
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,predic)*100)
print(confusion_matrix(y_test,predic)*100)
print( classification_report(y_test,predic)*100)