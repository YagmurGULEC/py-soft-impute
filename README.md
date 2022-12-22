# py-soft-impute
Python implementation of Iterative Soft-Thresholding Algorithm (ISTA) as a base class,Fast Iterative Soft-Thresholding Algorithm (FISTA) and 
Alternating directions method of multipliers (ADMM) algorithm as derived classes.
This code provides an experimental sklearn-ish class for missing data imputation. 

Notes:
- Missing values are represented by nan
- For additional detail,you may check the example Julia implementation [here](https://https://web.eecs.umich.edu/~fessler/course/551/julia/demo/09_lr_complete3.html) is quite helpful


### Toy example usage
```python
 import numpy as np
 from soft_impute import Impute

 X = np.arange(50).reshape(10, 5) * 1.0

 # Change 10 to nan aka missing
 X[2, 0] = np.nan
 clf.fit(X)
X,cost=clf.predict(X)
