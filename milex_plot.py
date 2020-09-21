import prtools as pr
import miltools as mil
import matplotlib.pyplot as plt


# generate a MIL dataset:
a = mil.gendatmilg([5,20])

# fit some classifiers:
w0 = mil.simplemil(a,['presence',pr.ldc()])
w1 = mil.miles(a,[8])

# plot them:
plt.figure(1)
pr.scatterd(a)
pr.plotc(w0)
pr.plotc(w1)

plt.legend()
plt.show()
