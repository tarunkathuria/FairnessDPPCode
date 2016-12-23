import math
import random
import numpy as np
import numpy.linalg as la

def VerifyPartitionConstraints(Pvec,kvec,S):
	b=1
	p=len(list(set(Pvec)))
	Svec=[0]*p
	for i in S:
		pVal=Pvec[i]
		Svec[pVal]=Svec[pVal]+1
	for i in range(0,p):
		if(Svec[i]>kvec[i]):
			b=0
	return b

def PartitionDPPMaxGreedy(X,kvec,Pvec):
	
	n = int(math.sqrt(K.size))
	p=len(list(set(Pvec)))
	nvec=[0]*p
	for i in Pvec:
		nvec[i]=nvec[i]+1

	k=sum(kvec)
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
				if(j not in S):
					T=[]
					for y in S:
						T.insert(0,y)
					T.insert(0,j)
					h=K[T,:]
					h=h[:,T]
					if(VerifyPartitionConstraints(Pvec,kvec,T)):
						vals[j]=la.det(h)
					else:
						vals[j]=-1

			# print('vals',vals)
			# print('iter',i)
			# print('s',S)
			S.append(np.argmax(vals))
			# print(S)
		return S

def PartitionDPPGreedySample(X,k):
	n=int(math.sqrt(X.size))
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

def PartitionDPPSampleMCMC(K,kvec,Pvec,eps):
	P0=np.where(Pvec==0)
	P0=P0[0]
	P1=np.where(Pvec==1)
	P1=P1[0]
	S0=random.sample(np.arange(P0.size),kvec[0])
	S1=random.sample(np.arange(P1.size),kvec[1])
	S=[]
	S.append(S0)
	S.append(S1)
	S=sum(S,[])
	# S=PartitionDPPMaxGreedy(K,kvec,Pvec)
	
	n = int(math.sqrt(K.size))
	k=sum(kvec)
	Spr=set(S)
	Sbar=list(set(range(0,n))-Spr)
	numIter=int(k*math.log(n/eps))
	print('numIter',numIter)
	for t in range(numIter):
		# print('t',t)
		outIndex=random.randrange(0,k)
		outElt=S[outIndex]
		outPartitionNumber=-1
		for i in range(n):
			if(outElt==i):
				outPartitionNumber=Pvec[outElt]

		inIndex=random.randrange(0,n-k)
		if(Pvec[Sbar[inIndex]]==outPartitionNumber):
			inElt=Sbar[inIndex]
			S[outIndex]=inElt
			Sbar[inIndex]=outElt
	return S


# a=np.random.rand(10,10)
# a=np.asmatrix(a)
# k=a*a.T
# k=10*k
# print(la.det(k))
# print('max',PartitionDPPMaxGreedy(k,[2,2],[1,0,1,0,1,0,1,0,1,0]))
# print('sample',PartitionDPPSampleMCMC(k,[2,2],[1,0,1,0,1,0,1,0,1,0],0.01))

