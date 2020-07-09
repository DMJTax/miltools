# miltools
Multiple Instance Learning Toolbox for Python
build on top of Prtools for Python (see https://github.com/DMJTax/prtools)

For the easy creation, handling and classification of Multiple Instance
Learning datasets.

Dataset operations
: genmil           Generate MIL dataset from data and labels
: gendatmil        Subsample bags from a MIL dataset
: allpositivenegative  Check if you have valid MIL labels
: scattermil       Scatterplot of a MIL dataset
: setpositive      Relabel a list of labels to 'positive'/'negative'

MIL Classifiers
---------------
: simple_mil       Generate MIL mapping from standard mapping
: miles            Multi-instance Learning via Embedded Instance Selection

Evaluation
----------
: milcrossval      MIL crossvalidation (using bags etc)

Standard datasets
-----------------
: gendatmilg       Generate artificial Gaussian MIL problem

Bag combinations and representations
------------------------------------
: milcombine       Combine instance prob. to get the bag prob.
: milesproxm       Compute MIL bags to instance similarities

Support functions
-----------------
: getbags          Extract the bags from a MIL set
: genmillabels     Simplify generation of MIL labels
: ispositive       True if label/object is 'positive'
: find_positive    Indicator vector for positive label

