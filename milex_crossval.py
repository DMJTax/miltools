# Perform 3-fold stratified cross-validation
import numpy as np
import prtools as pr
import miltools as mil

# generate a MIL dataset:
a = mil.gendatmilg([20,17],d=2)
# initialise other params:
nrfolds = 3
err = np.zeros((2,nrfolds))
I = nrfolds
# perform stratified crossval:
for i in range(nrfolds):
    # get the train and validation set:
    x,z,I = mil.milcrossval(a,I)
    # train:
    w = mil.simplemil(x)
    # evaluate:
    err[0,i] = z*w*pr.testc()
# show the result:
err *= 100
print("Error = %3.1f (%3.1f)."%(np.mean(err),np.std(err)))


