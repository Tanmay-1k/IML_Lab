import pandas as pd, math

class NB:
 def fit(s,X,y):
  s.c=set(y);s.m={};s.v={};s.p={}
  for c in s.c:
   Xc=[X[i] for i in range(len(X)) if y[i]==c]
   s.m[c]=[sum(col)/len(col) for col in zip(*Xc)]
   s.v[c]=[sum((x-m)**2 for x in col)/len(col)+1e-9 for col,m in zip(zip(*Xc),s.m[c])]
   s.p[c]=len(Xc)/len(X)
 def g(s,x,m,v): return (1/math.sqrt(2*math.pi*v))*math.exp(-(x-m)**2/(2*v))
 def pred(s,X):
  return [max({c:math.log(s.p[c])+sum(math.log(s.g(x[i],s.m[c][i],s.v[c][i])) for i in range(len(x))) for c in s.c},key=lambda k:{c:math.log(s.p[c])+sum(math.log(s.g(x[i],s.m[c][i],s.v[c][i])) for i in range(len(x))) for c in s.c}[k]) for x in X]

acc=lambda y,p:sum(i==j for i,j in zip(y,p))/len(y)
split=lambda X,y:(X[:int(.8*len(X))],X[int(.8*len(X)):],y[:int(.8*len(y))],y[int(.8*len(y)):])

# Diabetes
df=pd.read_csv(r"Lab-08-1\diabetes_dataset.csv")
X,y=df.iloc[:,:-1].values.tolist(),df.iloc[:,-1].values.tolist()
Xt,Xs,yt,ys=split(X,y);m=NB();m.fit(Xt,yt);print("Diabetes:",acc(ys,m.pred(Xs)))

# Social Ads
df=pd.read_csv(r"Lab-08-1\Social_Network_Ads.csv")
df["Gender"]=df["Gender"].map({"Male":1,"Female":0})
X,y=df.drop(columns=["Purchased"]).values.tolist(),df["Purchased"].values.tolist()
Xt,Xs,yt,ys=split(X,y);m=NB();m.fit(Xt,yt);print("Social:",acc(ys,m.pred(Xs)))

# Titanic
df=pd.read_csv(r"Lab-08-1\titanic.csv")[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]
df["Age"].fillna(df["Age"].mean(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
df["Sex"]=df["Sex"].map({"male":1,"female":0})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})
X,y=df.drop(columns=["Survived"]).values.tolist(),df["Survived"].values.tolist()
Xt,Xs,yt,ys=split(X,y);m=NB();m.fit(Xt,yt);print("Titanic:",acc(ys,m.pred(Xs)))