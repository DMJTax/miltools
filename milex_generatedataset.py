import numpy as np
import prtools as pr
import miltools as mil

print(
        '''
To create a valid MIL dataset, we have two options:
we supply
(1) a list of bag feature vectors and the bag labels
(2) a matrix with feature vectors, labels and bag identifiers
In all situations, the (bag) labels should be 'positive' and 'negative'
'''
)

B = [5,3]
instperbag = 6
print("Setting (1):")
bags = []
for i in range(B[0]):
    newbag = np.random.randn(instperbag,2)
    newbag[:,0] += 3.
    bags.append(newbag)
for i in range(B[1]):
    bags.append(np.random.randn(instperbag,2))
baglab = pr.genlab(B,['positive','negative'])
a = mil.genmil(bags,baglab)
print(a)
mil.disp(a)

print("Setting (2):")
nB = instperbag*np.array(B) # nr positive and negative instances
x = pr.gendatb(nB)
dat = +x
lab = pr.genlab(nB, ['positive','negative'])
bagid = []
for i in range(B[0]+B[1]):
    I = np.tile(i,instperbag)
    bagid.extend(I)
b = mil.genmil(dat,lab,bagid)
print(b)
mil.disp(a)

