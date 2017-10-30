import pandas as pd
import numpy as np
import time
import pp
job_server=pp.Server()
def gradient(x, y, theta):    
    return (x.transpose())*((x*theta)-y)
def batch_generator(X,y,batch_size):
    	for i in np.arange(0, X.shape[0], batch_size):
         yield (X[i:i + batch_size], y[i:i + batch_size])

def batch_g_d(X, y, alpha, iterations,batch_size):    
    theta = np.matrix(np.zeros((X.shape[1],1)))
    batches=[(xi,yi) for (xi,yi) in batch_generator(X,y,batch_size)]    
    for iteration in range(iterations):                    
        s = np.matrix(np.zeros((X.shape[1],1)))
        jobs=list()
        r=list()
        for i in batches:
            j=job_server.submit(gradient,(i[0],i[1],theta),(),("numpy",))
            jobs.append(j)
        for j in jobs:
            r.append(j())
        """
        for i in batches:
            temp=gradient(i[0],i[1],theta)
            s=s+temp
        """
        """
        for i in range(0,X.shape[0],batch_size):
            temp=gradient(X[i:i+batch_size,:],y[i:i+batch_size,:],theta)            
            s=s+temp
        """
        s=sum(r)
        theta = theta - (alpha*s)/X.shape[0]
        print("Iteration: ",iteration+1," Line: ",float(theta[0])," + ",float(theta[1]),"x              ",end="\r")
    print("\n")    
    return theta
data = pd.read_csv('ex1data1.txt', names = ['x', 'y'])
X_df = pd.DataFrame(data.x)
y_df = pd.DataFrame(data.y)
iterations = 1500
alpha = 0.01
X = np.c_[np.ones((X_df.shape[0],1)),np.matrix(X_df)]
y = np.matrix(y_df)
batch_size = 32
time1=time.time()
t=batch_g_d(X,y,alpha,iterations,batch_size)
time3=time.time()-time1
print("Time taken: ",time3*1000," ms")
print(t)