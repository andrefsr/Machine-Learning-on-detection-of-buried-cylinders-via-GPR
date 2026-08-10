import h5py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

N = 1000
x = []
dt = 3e-9/637
tf = 1.6e-9
Nf = int(tf/dt)
T = 3e-9

for i in range(1,N+1):
    with h5py.File(f'Bscan{i}__merged.out', 'r') as f:
        rx = f['rxs/rx1']
        ez = rx['Ez'][:]

    ez_corr = np.zeros((637,60))
    for j in range(Nf+1,637):
        for k in range(60):
            ez_corr[j,k] = ez[j,k]
    
    sinal = ez_corr.flatten()

    features = [
        np.mean(sinal),
        np.var(sinal),
        np.max(np.abs(sinal)),
        np.sum(sinal**2),
        skew(sinal),
        kurtosis(sinal)
    ]

    x.append(features)

df_features = pd.DataFrame(x)
df_features = df_features.rename(columns={0:'Média',1:'Variância',2:'Máximo',3:'Energia',4:'Skewness',5:'Kurtosis'})
df_features.to_csv('Features.csv',index=False)

df_labels = pd.read_csv("labels.csv")
y = df_labels['Label'].tolist()

X = np.array(x)
Y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Acurácia:", accuracy_score(y_test,y_pred))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test,y_pred))

print("\nRelatório:")
print(classification_report(y_test,y_pred))

nomes = ['média','variância','máx','energia','skewness','kurtosis']

imp = modelo.feature_importances_

plt.bar(nomes,imp)
plt.ylabel("Importância")
plt.xticks(rotation=45)
plt.show()