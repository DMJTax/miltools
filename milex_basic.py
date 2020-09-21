import prtools as pr
import miltools as mil

# generate a MIL dataset:
b = mil.gendatmilg([4,4])
print(b)
# fit a classifier:
w = mil.miles(b,[8])
# and compute the classifier output:
pred = b*w
print(pred)
# predict the labels:
lab = pred*pr.labeld()
print(lab)
# and the training error becomes:
err = pred*pr.testc()
print('Training error is ',err)

