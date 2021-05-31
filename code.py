import time
import numpy as np
runtime=[]
def matrixgen(n,tau,mu,t):

    #obj func
    c=(np.array([list(3**(n-j) for j in range(1,n+1))])).T
    #
    b=np.zeros(n+n)
    b[0]=1
    for i in range(2,n+1):
        b[i-1]=9**(i-1)
    b=np.array([b]).T

    N=np.zeros((n,n))
    N[0][0]=1
    for i in range(2,n+1):
        N[i-1,0:i-1]=[2*(3**(i-j)) for j in range(1,i)]
        N[i-1,i-1]=1
    #coefficients
    A=np.concatenate((N,np.eye(n)*-1),axis=0).T    
    #starting x
    x=[0.25]
    for i in range(n-1):
        x.append(9*x[-1])
    while True:

        #if np.max
        #d=1/(b-ax)
        slack=b.T-np.sum((A.T*x),axis=1)
        d=(1/slack).T

        #grad=c-(1/t)*A*d
        grad=c-(1/t)*np.dot(A,d)


        #hessian=-A.T*diag(d^2)*A

        hessian=-(1/t)*np.dot(np.dot(A,np.diag((d**2).T[0])),A.T)
        #print(hessian)

        differential=np.dot(np.linalg.inv(hessian),grad)
        x_new=x-differential.T
        print(x)
        if np.max(np.abs(x-x_new))<tau:
            break


        else:
            x=x_new
            t=t*mu

    print('Final Variables: ',np.round(x))
    return np.round(x)
    
    
n=8
tic=time.perf_counter()
matrixgen(n=8,tau=0.001,mu=2,t=0.000001)
toc=time.perf_counter()
print('runtime:',toc-tic)
runtime.append((n,toc-tic))
