"""
MILTools
Multiple Instance Learning Toolbox for Python, based
on Prtools for Python.

For example code, please look at
   MILEX_BASIC.PY
"""
import numpy
import copy
import requests
import matplotlib.pyplot as plt
import prtools as pr

def disp(a):
    """
    Nicer description of what is in a MIL dataset
            mildisp(A)
    """
    bagid = a.gettargets('bagid')
    if bagid is None:
        # we don't have a mil dataset
        print("Not a MIL dataset.")
        return
    N,dim = a.shape
    if (len(a.name)>0):
        out = "%s %d by %d MIL dataset "%(a.name,N,dim)
    else:
        out = "%d by %d MIL dataset "%(N,dim)
    bags, baglab = getbags(a)
    B = len(bags)
    Ipos,Ineg = find_positive(baglab)
    out += "with %d bags: [%d pos, %d neg]"%(B,len(Ipos),len(Ineg))
    print(out)
    return

def genmillabels(n):
    """
    Generate MIL instance labels
         LAB = genmillabels([N1,N2])
         LAB = genmillabels(I)

    Create a label list of MIL instance labels that should be 'positive'
    or 'negative'. There will be N1 positive and N2 negatives.
    If an index vector I of integers, or booleans, of size Nx1, N labels
    are generate with 'positive' on locations where I==1 or I==True.
    """
    if isinstance(n,list):
        return pr.genlab(n,['positive','negative'])
    else:
        if isinstance(n.dtype,int):
            n = (n==1)
        return numpy.where(n,'positive','negative')

def allpositivenegative(lab):
    """
    Check for "positive"/"negative" labels
           out = allpositivenegative(LAB)

    Return True if all labels defined in LAB are either "positive" or
    "negative" (as is assumed in a MIL dataset).

    """
    # Sigh, Python is so great (NOT)
    #I = ((lab=='positive') or (lab=='negative'))
    I = numpy.logical_or((lab=='positive'), (lab=='negative'))
    return numpy.all(I)

def setpositive(labels,classname):
    """
       OUT = setpositive(LABS,CLASSNAME)
       OUT = setpositive(X,CLASSNAME)
    Relabel class CLASSNAME to 'positive', and all the rest to
    'negative.
    """
    if isinstance(labels,pr.prdataset):
        x = labels
        labels = x.targets
    else:
        x = None

    N = labels.shape[0]
    out = numpy.repeat('negative',N)
    I = (labels==classname).nonzero()[0] # compare this with Matlab...
    for i in I:
        out[i] = 'positive'
    out = out[:,numpy.newaxis]

    if x is None:
        return out
    else:
        x.targets = out
        return x

def genmil(x,classlab=None,baglab=None):
    """
    Generate MIL dataset
           A = genmil(DAT,LAB,BAGLAB)
           A = genmil(BAGS,LAB)

    Define a MIL dataset from data matrix DAT, instance labels LAB and
    bag identifiers BAGLAB. When data matrix DAT has size NxD, LAB and
    BAGLAB should be vectors of length N. LAB should only have labels
    "positive" or "negative", BAGLAB should contain indices between 0
    and B-1 (where B is the number of bags in the dataset).
    """
    # torture the input until it fits a MIL dataset
    if (isinstance(x,pr.prdataset)):
        baglab = x.gettargets('bagid')
        if baglab is None:
            # it is a normal dataset, make each instance a bag:
            baglab = numpy.arange(x.shape[0])
            baglab = baglab[:,numpy.newaxis]
            x.settargets('bagid',baglab)
        return(x)
    else:
        if ((classlab is None) and (baglab is None)):
            # we probably only got a data matrix:
            x = pr.prdataset(x)
            baglab = numpy.arange(x.shape[0])
            baglab = baglab[:,numpy.newaxis]
            x.settargets('bagid',baglab)
            return(x)
    # make the labels column numpy vectors:
    if isinstance(classlab,list):
        classlab = numpy.array(classlab)
        if (len(classlab.shape)<2):
            classlab = classlab[:,numpy.newaxis]
    if isinstance(baglab,list):
        baglab = numpy.array(baglab)
        if (len(baglab.shape)<2):
            baglab = baglab[:,numpy.newaxis]
    # check if classlab is valid:
    if not allpositivenegative(classlab):
        raise ValueError('Instance labels should be "positive" or "negative".')
    if baglab is None:
        # we are given a list of bags and bag labels
        if (not isinstance(x,list)):
            raise ValueError('I am expecting a list of bags.')
        # set up matrices using the first bag:
        dat = x[0]
        lab = pr.genlab(x[0].shape[0],classlab[0])
        bagid = numpy.zeros((x[0].shape[0],1))
        # concatenate the following bags
        for i in range(1,len(x)):
            dat = numpy.concatenate((dat,x[i]),axis=0)
            lab = numpy.concatenate((lab,pr.genlab(x[i].shape[0],classlab[i])))
            newbagid = i*numpy.ones((x[i].shape[0],1))
            bagid = numpy.concatenate((bagid,newbagid))

        # 'standard' call to genmil:
        a = genmil(dat,lab,bagid)
        return a
    else:
        # we have a data matrix, with instance labels and bag IDs
        if (x.shape[0] != classlab.shape[0]):
            raise ValueError('Number of instance labels does not fit datset.')
        if (not isinstance(baglab,numpy.ndarray)):
            print(type(baglab))
            raise ValueError('Bag labels should be an integer Nx1 vector.')
    #    if (not isinstance(baglab.dtype,numpy.int64)): #DXD why does this
    #    fail?
    #        print(baglab.dtype)
    #        raise ValueError('Bag labels should be an integer Nx1 vector.')
        if (isinstance(x,pr.prdataset)):
            x = +x
        a = pr.prdataset(x,classlab)
        a.settargets('bagid',baglab)

    return a

def getbags(x,returninst=False):
    """
    Extract the bags from a MIL dataset
          BAGS,BAGLAB = getbags(X)
          BAGS,BAGLAB,J = getbags(X,returninst=True)

    Get the bags BAGS and their labels BAGLAB from MIL dataset X.
    If requested, also a list J of indices of the instances of each bag.
    """
    baglab = x.gettargets('bagid')
    if baglab is None:
        raise ValueError('This is not a MIL dataset (target "bagid" is not defined).')
    bagll = numpy.unique(baglab)
    B = len(bagll)
    bags = []
    J = [None]*B
    labs = numpy.repeat('positive',B)
    for i in range(B):
        I = (baglab==bagll[i]).nonzero()  # f*cking Python
        J[i] = I[0]
        xi = x[I[0],:]
        bags.append(+xi)
        if isinstance(xi.targets[0],numpy.ndarray):
            labs[i] = xi.targets[0][0] # copy the first label? We probably
            # need something more smart here.
        else:
            labs[i] = xi.targets[0]
    labs = labs[:,numpy.newaxis]
    if returninst:
        return bags,labs,J
    else:
        return bags,labs

def ispositive(x):
    if isinstance(x,pr.prdataset):
        I = x.targets
    else:
        I = x
    return (I=='positive')

def find_positive(x,givenegatives=True):
    """
    Find which bags/instances/labels are positive
        Ipos,Ineg = find_positive(x,givenegatives=True)
        Ipos = find_positive(x,givenegatives=False)
    """
    I = ispositive(x)
    Ipos = I.nonzero()
    if (givenegatives):
        J = numpy.logical_not(I)
        Ineg = J.nonzero()
        return Ipos[0],Ineg[0]
    else:
        return Ipos[0]


def gendatmil(x,frac):
    """
    Sample train and test from MIL dataset

        Y,Z = gendatmil(X,FRAC)

    Randomly sample a fraction FRAC of the bags in MIL dataset X for
    training set Y, and put the rest in Z.
    """
    if ((frac<0) or (frac>1)):
        raise ValueError('Requested fraction should be between 0 and 1.')
    bagid = x.gettargets('bagid')
    if bagid is None:
        raise ValueError('No MIL dataset supplied.')
    ll = numpy.unique(bagid)
    bags,baglab = getbags(x)
    B = len(baglab)
    Ipos,Ineg = find_positive(baglab)
    # randomly permute:
    Ipos = numpy.random.permutation(Ipos)
    Ineg = numpy.random.permutation(Ineg)
    # randomly select a fraction of the positives:
    npos = int(numpy.ceil(frac*len(Ipos)))
    Iy,Iz = numpy.split(Ipos,[npos])
    # randomly select a fraction of the positives:
    nneg = int(numpy.ceil(frac*len(Ineg)))
    J = numpy.split(Ineg,[nneg])
    # combine it
    Iy = numpy.concatenate((Iy,J[0]))
    Iz = numpy.concatenate((Iz,J[1]))
    # collect the bags, and squeeze it in a prdataset
    # (DXD: this feels expensive:)
    #bagsy = [bags[i] for i in Iy]
    #labsy = [baglab[i] for i in Iy]
    #y = genmil(bagsy,labsy)
    #bagsz = [bags[i] for i in Iz]
    #labsz = [baglab[i] for i in Iz]
    #z = genmil(bagsz,labsz)
    J = numpy.isin(bagid, Iy).nonzero() # bloody brilliant
    y = x[J[0],:]
    J = numpy.isin(bagid, Iz).nonzero() # again, bloody brilliant
    z = x[J[0],:]
    return y,z

def milcrossval(a,I=10):
    """
    Crossvalidation on MIL dataset
        X,Z,I = milcrossval(A,K)
        X,Z,I = milcrossval(A,I)
    Perform a K-fold crossvalidation on MIL dataset A. Training set X
    and test set Z is returned, and an index vector I. After each call
    to milcrossval I is updated.
    """
    if isinstance(I,int):
        # we have the first call to milcrossval, distribute the bags
        # over the folds
        N = a.shape[0]
        nrfolds = I
        I = numpy.zeros((N+1,1))
        bags,baglab,J = getbags(a,returninst=True)
        Ipos,Ineg = find_positive(baglab)
        # DXD shall we randomize the order of the bags??
        Ipos = numpy.random.permutation(Ipos)
        Ineg = numpy.random.permutation(Ineg)
        # divide the positives in folds:
        for i in range(len(Ipos)):
            I[J[Ipos[i]]] = numpy.mod(i,nrfolds) # cool huh?!?
        # also for the negatives:
        for i in range(len(Ineg)):
            I[J[Ineg[i]]] = numpy.mod(i,nrfolds) # also cool 
        # now select the train and validation set:
        I0 = (I[:-1]!=0).nonzero()  # the 'elegance' of Python:-(
        x = a[I0[0],:]
        I0 = (I[:-1]==0).nonzero()
        z = a[I0[0],:]
        return x,z,I
    else:
        # select the data for the next fold:
        I[-1] +=1
        I0 = (I[:-1]!=I[-1]).nonzero()
        x = a[I0[0],:]
        I0 = (I[:-1]==I[-1]).nonzero()
        z = a[I0[0],:]
        return x,z,I



def gendatmilg(n,np=1,d=7,dim=2):
    """
    GENDATMILG Generate Gaussian MIL problem

         X = GENDATMILG([N1 N2],NP,D,DIM)

    Generate an artificial multi-instance learning problem.  For the N1
    positive bags NP(1) instances are drawn from the positive concept
    Gaussian centered around (D,1), and a random set of instances is drawn
    from a background Gaussian distribution round (0,0). This number is
    between 1 and 10-NP.  For the N2 negative bags, NP(2) instances are
    drawn from the positive concept, and the rest is drawn from the
    background distribution. When NP(2) is not defined, NP(2)=0.

    When DIM>2 is defined, extra features are added such that the total
    number of features is DIM. The added features all have an identical
    unit-variance Gaussian distribution.

    """
    if (len(n)<2):
        n = pr.genclass(n,[0.6,0.4])
    Ntot = numpy.sum(n)
    instperbag = [5,10]
    if (np>instperbag[1]):
        raise ValueError('Maximally %d instances can be positive.'%instperbag[1])
    # random size per bag:
    m = numpy.random.randint(instperbag[0],instperbag[1]+1,size=[Ntot,])
    x=None
    lab = []
    bagid = []
    # generate positive bags:
    for i in range(n[0]):
        # positive instances have their mean at [d,1,0,..,0]
        newx = numpy.random.randn(np,dim)
        newx[:,0] += d
        newx[:,0:1] += 1
        if x is None:  # sigh, you cannot concatenate to an empty thing
            x = newx
        else:
            x = numpy.concatenate((x,newx))
        # other negative instances in positive bags:
        newx = numpy.random.randn(m[i]-np,dim)
        x = numpy.concatenate((x,newx))
        lab.extend(pr.genlab([np,m[i]-np],['positive','negative']))
        bagid.extend(i*numpy.ones(m[i]))
    # generate negative bags:
    for i in range(n[1]):
        mm = m[n[0]+i]
        x = numpy.concatenate((x,numpy.random.randn(mm,dim)))
        lab.extend(pr.genlab([mm],['negative']))
        bagid.extend((i+n[0])*numpy.ones(mm))
    out = genmil(x,lab,bagid)
    out.name = "Gaussian MI (np=%d)"%np
    return(out)

def gendatmusk(ver=0,getOnline=True):
    """
    Get the MUSK MIL dataset
    """
    N = 476
    dim = 166
    if (ver==0):
        name = 'clean1'
    else:
        name = 'clean2'
    if getOnline:
        link = "https://archive.ics.uci.edu/ml/machine-learning-databases/musk/"
        f = requests.get(link+name+"/"+name+".data.Z")
        txt = f.text.splitlines()
    else:
        text_file = open(name+".data", "r")
        txt = text_file.readlines()
        text_file.close()
    x = numpy.zeros((N,dim))
    labx = numpy.empty(N,dtype=object)
    bagid = numpy.empty(N,dtype=object)
    i = 0
    for line in txt:
        nr = line.split(',')
        bagid[i] = nr[0].rstrip()
        for j in range(dim):
            try:
                x[i,j] = float(nr[j+2])
            except:
                x[i,j] = numpy.nan
        # finally get the label:
        thislab = nr[dim+2].rstrip()
        try:
            labx[i] = int(float(thislab))
        except:
            labx[i] = thislab
        i += 1
        if (i>=N):
            break
    labx = setpositive(labx,1)
    a = genmil(x,labx,bagid)
    a.name = 'Musk '+name
    return a

def milcombine(q,combrule='presence',pfeat=0):
    """
    Combine instance outputs to bag output

        P = milcombine(Q,COMBRULE,PFEAT)

    Combine the classifier outputs Q of individual instances to an
    output per bag. It is assumed that the classifier outputs are
    'positive' and 'negative'. As an alternative you can define PFEAT,
    indicating which column is the positive class prediction.
    """
    if (combrule=='presence'):
        p = numpy.max(q[:,pfeat:(pfeat+1)])
    elif (combrule=='first'):
        p = q[0,pfeat:(pfeat+1)]
    elif (combrule=='vote'):
        I = numpy.argmax(q,axis=1)
        J = (I==pfeat) # check this
        p = numpy.mean(J)
    else:
        raise ValueError('Unknown combining rule.')
    return p

def milproxm(task=None,x=None,w=None):
    """
    MILPROXM MIL proximity mapping

       W = MILPROXM(A,(KTYPE,KPAR));

    Definition of a proximity mapping between bags in a Multi-instance
    Learning problem using proximity KTYPE with parameter KPAR. The
    dataset A has to be MIL-dataset.

    The proximity is defined by the type KTYPE (and potentially its
    parameter KPAR):
      'minmin'       | 'min':Minimum of minimum distances between inst. 
      'summin'       | 'sm': Sum of minimum distances between inst. 
      'meanmin'      | 'mm': Mean of minimum distances between inst. 
      'meanmean'     | 'mean': Mean of mean distances between inst. 
      'mahalanobis'  | 'm':  Mahalanobis distance between bags
      'hausdorff'    | 'h':  (maximum) Hausdorff distance between bags

    """
    if not isinstance(task,str):
        out = pr.prmapping(milproxm,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if isinstance(x,str):
            x = [x]
        if x is None:
            x = ['hausdorff']
        return 'MILproxm', x
    elif (task=='train'):
        # we only need to store the training bags:
        bags,baglab = getbags(x)
        nrR = len(bags)
        return [bags], range(nrR)
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        bagsR = W[0]
        nrR = len(bagsR) # number of training/representation bags

        # apply genmil to X, to make sure it is a valid MIL dataset (for
        # instance, when you plot the classifier, plotc will supply a
        # standard data matrix, not a MIL dataset)
        x = genmil(x)
        bags,baglab = getbags(x)
        B = len(baglab)

        # min-min distance:
        if (w.hyperparam[0]=='minmin') or \
                (w.hyperparam[0]=='min'):
            out = numpy.zeros((B,nrR))
            for i in range(B):
                for j in range(nrR):
                    D = pr.sqeucldist(bags[i],bagsR[j])
                    out[i,j] = numpy.min(D)
        # mean-min distance:
        elif (w.hyperparam[0]=='meanmin') or \
                (w.hyperparam[0]=='mm'):
            out = numpy.zeros((B,nrR))
            for i in range(B):
                for j in range(nrR):
                    D = pr.sqeucldist(bags[i],bagsR[j])
                    d1 = numpy.min(D,axis=0)
                    d2 = numpy.min(D,axis=1)
                    out[i,j] = ((numpy.mean(d1)+numpy.mean(d2)))/2.
        # hausdorff distance:
        elif (w.hyperparam[0]=='hausdorf') or \
                (w.hyperparam[0]=='h'):
            out = numpy.zeros((B,nrR))
            for i in range(B):
                for j in range(nrR):
                    D = pr.sqeucldist(bags[i],bagsR[j])
                    d1 = numpy.min(D,axis=0)
                    d2 = numpy.min(D,axis=1)
                    out[i,j] = numpy.max((numpy.max(d1),numpy.max(d2)))
        else:
            print('This proximity is not defined.',w.hyperparam[0])

        out = pr.prdataset(out,baglab)
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for MILproxm."%task)


def simplemil(task=None,x=None,w=None):
    """
    Simple MIL classifier
    
          W = simplemil(A,(FRAC,U))

    Train classifier U on dataset A without taking bag labels into
    account. Then combine the predicted labels using rule FRAC.
    Default FRAC='presence', U = ldc()
    """
    if not isinstance(task,str):
        out = pr.prmapping(simplemil,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = ['presence', pr.ldc()]
        return 'Simple MIL', x
    elif (task=='train'):
        if not isinstance(w[1],pr.prmapping):
            raise ValueError('I expected an untrained PRmapping.')
        # we only need to train the mapping
        u = copy.deepcopy(w[1])
        out = x*u*pr.classc() # normalise outputs!
        # store the parameters, and labels:
        return out, ['positive','negative']
        #return out,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        # apply genmil to X, to make sure it is a valid MIL dataset (for
        # instance, when you plot the classifier, plotc will supply a
        # standard data matrix, not a MIL dataset)
        pred = genmil(x)*W
        bags,baglab = getbags(pred)
        B = len(baglab)
        out = numpy.zeros((B,2))
        fl = list(pred.featlab)
        pfeat = fl.index('positive')
        for i in range(B):
            out[i,0] = milcombine(bags[i],'presence',pfeat)
            out[i,1] = 1. - out[i,0]
        out = pr.prdataset(out,baglab)
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for SimpleMIL."%task)

def milesproxm(bags,inst,sigm):
    """
    Compute the MILES sim. of bags to instances
        S = milesproxm(BAGS,INST,SIGM)
    (This is still a sloppy version, later it can be a mapping)
    """
    Z = -1./(sigm*sigm)
    dim = inst.shape[0]
    N = len(bags)
    #print("In milesproxm N = %d."%N)
    out = numpy.zeros((N,dim))
    for i in range(N):
        D = pr.sqeucldist(bags[i],inst)
        minD = numpy.min(D,axis=0)
        out[i,:] = numpy.exp(Z*minD)
    return(out)

def miles(task=None,x=None,w=None):
    """
    MILES classifier
    
          W = miles(A,(SIGM))

    Train the MILES classifier on dataset A.
    """
    if not isinstance(task,str):
        out = pr.prmapping(miles,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [5.0]  # arbitrary value for sigma
        if not isinstance(x,list):
            x = [x]
        return 'MILES', x
    elif (task=='train'):
        # find the bag-instance similarities:
        bags,baglab = getbags(x)
        inst = +x
        D = milesproxm(bags,inst,w[0])
        # we train a sparse classifier on MILES similarities
        # (learning rate 0.1, nriters 100)
        newlab = ispositive(baglab)*2. - 1.
        mdata = pr.prdataset(D,newlab)
        w = pr.winnowc(mdata,[0.1,100])
        # maybe find the non-zero weights??
        #return (inst,w),w.targets
        return (inst,w),['negative','positive']
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        # get MILES similarities
        newx = genmil(x)
        bags,baglab = getbags(newx)
        D = milesproxm(bags,W[0],w.hyperparam[0])
        pred = W[1](D)
        out = pr.prdataset(+pred,baglab)
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for MILES."%task)


