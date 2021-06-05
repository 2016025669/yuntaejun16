import pandas as pd
import numpy as np
import sys

#parameter 설정
confidence_rate=30
latent_factor_dim=20
latent_factor_alpha=0.01
lamda_rate=40
running= 10

X=None
Y=None
p=None
c=None

def load_and_change(file):

    f = open(file,"r")
    temp=[]
    for ele in f.read().split("\n"):
        if ele != "":
            temp.append(ele.split("\t"))
      
    data = pd.DataFrame(data=temp,columns=["user_id", "item_id", "rating", "time_stamp"])
    f.close()
    data = data.apply(pd.to_numeric).drop(["time_stamp"],axis=1)
    
    Data_Matrix = data.pivot_table("rating", index= "user_id", columns= "item_id",fill_value=0)
    
    return data ,Data_Matrix 

def init(df):
    M = df.to_numpy()
    #preference
    p = np.where(M>0,1,M)   
    
    #confidence
    c = 1+confidence_rate*M
    
    #latent fator
    x,y = M.shape
    
    user_mat = np.random.rand(latent_factor_dim,x)*latent_factor_alpha
    
    item_mat = np.random.rand(latent_factor_dim,y)*latent_factor_alpha
    
    return p,c,user_mat,item_mat

def cost_function():
    err_mat=c * np.square(p-X.T@Y)
    regularization= lamda_rate*(np.sum(np.square(X))+np.sum(np.square(Y)))
    return np.sum(err_mat)+regularization

def cal_X_Y():
    global X,Y
    # X factor 계산 dx
    
    t_x=X.T #행벡터를 열벡터로 만들기 위해
    t_y=Y.T
    lamda_I=lamda_rate*np.identity(latent_factor_dim)
    
    for u in range(len(t_x)):
        cu=np.diag(c[u])
        Y_cu=Y@cu

        t_x[u]=np.linalg.solve(Y_cu@Y.T + lamda_I,Y_cu@p[u])
    # Y factor 계산 dy
    for  i in range(len(t_y)):
        ci=np.diag(c[:,i])
        X_ci=X@ci
        
        t_y[i]=np.linalg.solve(X_ci@X.T+lamda_I,X_ci@p[:,i])
   

d,m=load_and_change(sys.argv[1])
p,c,X,Y = init(m)

for i in range(running):
    cal_X_Y()
    print(cost_function())

w=X.T@Y
wmax,wmin=np.max(w),np.min(w)
w=(w-wmin)/(wmax-wmin)
w=w*5
d2,m2=load_and_change(sys.argv[2])

f=open(sys.argv[1]+"_prediction.txt","w")
for i in range(len(d2)):
    dx=d2.loc[i][0]
    dy=d2.loc[i][1]
    f.write(str(dx)+"\t"+str(dy)+"\t"+str(w[dx-1,dy-1])+'\n')

