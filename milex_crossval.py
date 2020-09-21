import numpy
import prtools as pr
import miltools as mil

# initialisation:
nrfolds = 10
e = numpy.zeros((nrfolds,1))

# generate a MIL dataset:
a = mil.gendatmilg([30,30],np=1,d=4)
print(a)
# define a classifier:
u = mil.miles([],[8])

# do crossvalidation:
I = nrfolds
for i in range(nrfolds):
    print('Fold',i)
    x,z,I = mil.milcrossval(a,I)
    w = x*u
    e[i] = z*w*pr.testc()

# and the result:
Em = numpy.mean(e)
Es = numpy.std(e)
print('Crossvalidation error: %.3f (%.3f)'%(Em,Es))
