import math
import random
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
def kDPPMaxGreedy(X,k):
	K= np.dot(X,np.transpose(X))
	n = int(math.sqrt(K.size))
	# print(n)
	S=[]
	if(n<k):
		print('case1')
		print('n<k')
		return 0
	elif(n==k):
		print('case2')
		S=range(n)
	else:
		print('case3')
		# print(S)
		for i in range(k):
			# print('i',i)
			max=-1
			vals=[0]*n
			for j in range(n):
				# print('j',j)
				if(j not in S):
					T=[]
					for y in S:
						T.insert(0,y)
					T.insert(0,j)
					h=K[T,:]
					h=h[:,T]
					vals[j]=la.det(h)
			# print('vals',vals)
			# print('iter',i)
			# print('s',S)
			S.append(np.argmax(vals))
			# print(S)
		return S
def kDPPGreedySample(X,k):
	n=int(X.shape[0])
	print('n',n)
	S=[]
	for i in range(k):
		# print('i',i)
		multinom=[0]*n
		for j in range(n):
			multinom[j]=pow(la.norm(X[j,:]),2)
		multinomSum=sum(multinom)
		if(multinomSum<0.000000001):
			print('badcase')
			break
		multinom=multinom/multinomSum
		ind=nprand.multinomial(1,multinom)
		ind=np.where(ind==1)
		ind=ind[0][0]
		S.append(ind)
		for j in range(n):
			# print('shapes',X[ind,:].shape,'-',X[j,:].shape)
			X[j,:]=X[j,:] - np.dot(X[ind,:],np.transpose(X[j,:]))/pow(la.norm(X[ind,:]),2)
	return S

def kDPPSampleMCMC(X,k,eps):
	K= np.dot(X,np.transpose(X))
	# S=kDPPGreedySample(X,k)
	n = int(math.sqrt(K.size))
	S=random.sample(np.arange(n),k)
	Sbar=list(set(range(0,n))-set(S))
	numIter=int(n*k*math.log(n/eps))
	print('numIter',numIter)
	for t in range(numIter):
		# print(t)
		outIndex=random.randrange(0,k)
		inIndex=random.randrange(0,n-k)
		outElt=S[outIndex]
		inElt=Sbar[inIndex]
		S[outIndex]=inElt
		Sbar[inIndex]=outElt
	print('samplingdone')
	return S




a=np.random.rand(1000,1000)
a=np.asmatrix(a)
k=a*a.T
k=k/30
print(la.det(k))
# print('max',kDPPMaxGreedy(k,200))
h=kDPPGreedySample(a,10)
print('greedysample',h)
print('greedysampledet',la.det(k[h,:][:,h]))
j=kDPPSampleMCMC(k,10,0.01)
print('mcmcsample',j)
print('mcmcsampledet',la.det(k[j,:][:,j]))

