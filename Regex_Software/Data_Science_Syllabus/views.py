from django.shortcuts import render,redirect
import pandas as pd

df = pd.read_csv("C:\\Users\\mahes\\OneDrive\\Desktop\\Project\\Regex_Software\\covid_toy.csv")
print(df.head())
# Create your views here.

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['cough'] = lb.fit_transform(df['cough'])

x = df.iloc[:,2:4].values
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn_ans=knn.fit(x_train,y_train)
print(x_train.head())
print(df.head())

def home(request):
    return render(request,"index.html")

def predict(request):
    if request.method == 'POST':
        a = request.POST.get('age')
        age = int(a)
        s = request.POST.get('cough')
        cough = int(s)
        result = knn_ans.predict([[age,cough]])[0]

print(df.head())