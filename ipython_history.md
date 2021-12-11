 2/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 2/2: !pip install mglearn
 2/3: !pip install mglearn
 2/4:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 3/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 3/2:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
 3/3:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

cancer
 3/4:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

display(cancer)
 3/5:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

display(cancer)
 3/6:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

data_cancer = pd.DataFrame(cancer)

display(data_cancer)
 4/1:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

data_cancer = pd.DataFrame(cancer)

display(data_cancer)
 4/2:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 4/3:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

data_cancer = pd.DataFrame(cancer)

display(data_cancer)
 4/4:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

cancer
 4/5:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 4/6:
from sklearn.datasets import load_boston
data = load_boston()
 4/7: data.keys()
 4/8: print(data['DESCR'][:1300])
 4/9: plt.hist(data['target'], bins=15)
4/10: plt.hist(data['target'], bins=15)
4/11: c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
4/12:
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
print("Number of data points in training set and test set, respectively: {} and {}".format(X_train.shape[0], 
                                                                                          X_test.shape[0]))
4/13:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_test,c_test)))
4/14:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_test,c_test)))
4/15:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/16:
knn.fit(X_test, c_test)
print("Model accuracy on the test data: {}".format(knn.score(X_test,c_test)))
4/17:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/18:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/19:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/20:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/21:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/22:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/23:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/24:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/25:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/26:
knn.fit(X_test, c_test)
print("Model accuracy on the test data: {}".format(knn.score(X_test,c_test)))
4/27:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
4/28:
knn.fit(X_test, c_test)
print("Model accuracy on the test data: {}".format(knn.score(X_test,c_test)))
4/29:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print("Model accuracy on the original data: {}".format(knn.score(X_train,c_train)))
 7/1:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
 7/2:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
 7/3:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
print("array type {} \n".format(x.type))
 7/4:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
print("array type {} \n".format(x.__type__))
 7/5:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
print("array type {} \n".format(x.type))
 7/6:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
x
 7/7:
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print("array shape {} \n".format(x.shape))
print("array type {} \n".format(type(x)))
 7/8:
from scipy import sparse

# create a 2d NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))
 7/9:
# convert the NumPy array to a SciPy sparse matrix only storing the non-zero entries 
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
7/10:
from scipy import sparse

# create a 2d NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))
7/11:
# convert the NumPy array to a SciPy sparse matrix only storing the non-zero entries 
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
7/12:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
7/13:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="y")
7/14:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
7/15:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(y, x, marker="x")
7/16:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(y, x, marker="y")
7/17:
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(y, x, marker="x")
7/18:
import pandas as pd

# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
display(data_pandas)
7/19:
# One of many possible ways to query the table:
# selecting all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])
 8/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import sklearn
 8/2:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/3:
X, y = sklearn.datasets.make_blobs(30, centers = 4, cluster_std=3, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/4:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/5:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=4, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/6:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=5, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/7:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/8:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=6, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
 8/9:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/10:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=2)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/11:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=4)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/12:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/13:
X, y = sklearn.datasets.make_blobs(30, centers = 4, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/14:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/15:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=1)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/16:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=2)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/17:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/18:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/19:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,0], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/20:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/21:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,0], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/22:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/23:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,0], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/24:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,½], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/25:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,0.5], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/26:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,2], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/27:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,1], X[:,0], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/28:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=2, random_state=6)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/29:
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
8/30: clf.fit(X, y)
8/31:
mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
8/32:
mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
8/33:
mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt
8/34:
mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
8/35:
fig, axes = plt.subplots(1,3, figsize=(10,3))


for k, ax in zip([1, 7, 22], axes):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X,y)
    mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7, ax=ax)
    mglearn.discrete_scatter(X[:,0],X[:,1], y, ax=ax)
    ax.set_xlabel("k = {}, accuracy = {:.2f}".format(k,clf.score(X,y)))
8/36:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=1)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/37:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/38:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/39:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=3)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/40:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/41:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,1], X[:,0], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/42:
X, y = sklearn.datasets.make_blobs(30, centers = 3, cluster_std=3, random_state=0)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
8/43:
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
8/44: clf.fit(X, y)
8/45:
mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
8/46:
fig, axes = plt.subplots(1,3, figsize=(10,3))


for k, ax in zip([1, 7, 22], axes):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X,y)
    mglearn.plots.plot_2d_classification(clf, X, fill=True, eps=0.5, alpha=.7, ax=ax)
    mglearn.discrete_scatter(X[:,0],X[:,1], y, ax=ax)
    ax.set_xlabel("k = {}, accuracy = {:.2f}".format(k,clf.score(X,y)))
8/47: X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)
8/48:
for k in [1, 7, 22]:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    print("k = {}: train accuracy = {:.2f}, test accuracy = {:.2f}".format(k,clf.score(X_train,y_train),clf.score(X_test,y_test)))
8/49:
training_accuracy = []
test_accuracy = []

ks = range(1, 22)

for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(ks, training_accuracy, label="training accuracy")
plt.plot(ks, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend();
8/50:
training_accuracy = []
test_accuracy = []

ks = range(1, 22)

for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(ks, training_accuracy, label="training accuracy")
plt.plot(training_accuracy, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend();
8/51:
training_accuracy = []
test_accuracy = []

ks = range(1, 22)

for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(ks, training_accuracy, label="training accuracy")
plt.plot(ks, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend();
8/52:
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
training_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(training_accuracy)
print(test_accuracy)
8/53:
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
training_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(training_accuracy)
print(test_accuracy)
// 6 seems to better :D
8/54:
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
training_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(training_accuracy)
print(test_accuracy)
<--- 6 seems to better :D --->
8/55:
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
training_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(training_accuracy)
print(test_accuracy)
 # 6 seems to better :D
 9/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
 9/2:
from sklearn.datasets import load_boston
data = load_boston()
 9/3:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
 9/4:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
c
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
 9/5:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
 9/6:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
 9/7:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
 9/8: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
 9/9: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/10:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
9/11: print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, y_train)))
9/12: print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
9/13:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
9/14:
from sklearn.datasets import load_boston
data = load_boston()
display(data)
9/15:
from sklearn.datasets import load_boston
data = load_boston()
display(data.DESC)
9/16:
from sklearn.datasets import load_boston
data = load_boston()
display(data)
9/17:
from sklearn.datasets import load_boston
data = load_boston()
display(data.DESCR)
9/18:
from sklearn.datasets import load_boston
data = load_boston()
display(data.DESCR)
9/19:
from sklearn.datasets import load_boston
data = load_boston()
print(display(data.DESCR))
9/20:
from sklearn.datasets import load_boston
data = load_boston()
print("{}".format(display(data.DESCR))
9/21:
from sklearn.datasets import load_boston
data = load_boston()
print("{}".format(display(data.DESCR)))
9/22:
from sklearn.datasets import load_boston
data = load_boston()
print(data.shape)
9/23:
from sklearn.datasets import load_boston
data = load_boston()
print(data.shape)
9/24:
from sklearn.datasets import load_boston
data = load_boston()
print(data)
9/25: X_reduced = np.delete(X,2,1)
9/26:
# Pick the data belonging to class 0
X_0 = data['data'][c==1,:]
9/27: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/28:
# Pick the data belonging to class 0
X_0 = data['data'][c==2,:]
9/29: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/30:
# Pick the data belonging to class 0
X_0 = data['data'][c==1,:]
9/31:
# Pick the data belonging to class 0
X_0 = data['data'][c==1,:]
9/32:
# Pick the data belonging to class 0
X_0 = data['data'][c==1,:]
9/33: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/34:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
9/35: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/36: X_reduced = np.delete(data,2,1)
9/37: print(data)
9/38:
print(data)
data_reduced = np.delete(data,4,1)
9/39:

data_reduced = np.delete(data,4,1)
9/40:

data_reduced = np.delete(data.data,4,1)
9/41:

data_reduced = np.delete(data['target'],4,1)
9/42:

data_reduced = np.delete(data.data,3,1)
9/43:

data_reduced = np.delete(data.data,12,1)
9/44:

data_reduced = np.delete(data.data,12,1)
print(data_reduced)
9/45:

data_reduced = np.delete(data.data,12,1)
print(data)
9/46:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
9/47: X_0 = data['data'][c==0,:]
9/48: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/49:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
9/50:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
9/51:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
9/52:
from sklearn.datasets import load_boston
data = load_boston()
print(data)
9/53:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
9/54:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
9/55: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/56:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
9/57:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
9/58:

data_reduced = np.delete(data.data,12,1)
print(data)
9/59:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
9/60: X_0 = data['data'][c==0,:]
9/61: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/62:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
9/63:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
9/64: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11]])
9/65: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11, 12]])
9/66:
dataframe = pd.DataFrame(data, data[feature_names])
dataframe
9/67:
dataframe = pd.DataFrame(data, data['feature_names'])
dataframe
9/68:
dataframe = pd.DataFrame(data.data, data['feature_names'])
dataframe
9/69: dataframe = pd.DataFrame(data.data, data['feature_names'])
9/70: dataframe = pd.DataFrame(data.data[X:1,>], data['feature_names'])
9/71: dataframe = pd.DataFrame(data.data[X:1,], data['feature_names'])
9/72: dataframe = pd.DataFrame(data['data'], data['feature_names'])
9/73: dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
9/74: dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
9/75:
dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
dataframe
9/76:
from sklearn.datasets import load_boston
data = load_boston()
print(data['DESCR'][:1300])
9/77:
dataframe.drop(columns=['CHAS', 'B', 'NOX'])
dataframe
9/78:
dataframe.drop(columns=['CHAS', 'B', 'NOX'])
dataframe
9/79:
dataframe.drop(['CHAS', 'B', 'NOX'], axis= 1)
dataframe
9/80:
dataframe.drop(['CHAS', 'B', 'NOX'], axis= 1)
dataframe
9/81:
dataframe.drop(columns= ['CHAS', 'B', 'NOX'], axis= 1)
dataframe
9/82:
dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
dataframe
9/83:
dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
display(dataframe)
9/84: X_0 = data['data'][c==0,:]
9/85: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,7,11]])
9/86:
dataframe = dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
display(dataframe)
9/87:
data_reduced = np.delete(data['data'],4,1)
data_reduced
9/88: data = np.delete(data['data'],4,1)
9/89: data.data = dataframe.to_numpy()
9/90: data['data'] = dataframe.to_numpy()
9/91: data
9/92:
from sklearn.datasets import load_boston
data = load_boston()
print(data['DESCR'][:1300])
9/93: data
9/94: data = np.delete(data[data], 4, 1)
9/95: data = np.delete(data['data'], 4, 1)
9/96: data
9/97:
from sklearn.datasets import load_boston
data = load_boston()
print(data['DESCR'][:1300])
9/98: data
9/99: data_removed = np.delete(data['data'], 4, 1)
9/100: data_removed = np.delete(data['data'], 3, 1)
9/101:
data_removed = np.delete(data['data'], 4, 1)
data_removed = np.delete(data['data'], 3, 1)
data_removed = np.delete(data['data'], 11, 1)
9/102:
data_removed = np.delete(data['data'], 4, 1)
data_removed = np.delete(data['data'], 3, 1)
data_removed = np.delete(data['data'], 11, 1)
9/103:
data_removed = np.delete(data['data'], 4, 1)
data_removed = np.delete(data['data'], 3, 1)
data_removed = np.delete(data['data'], 11, 1)
9/104: data_removed
9/105: X_0 = data_removed['data'][c==0,:]
9/106:
from sklearn.datasets import load_boston
data = load_boston()
print(data['DESCR'][:1300])
9/107: data = data.drop(["B","CHAS","NOX"], axis='columns')
10/1:
#Dropping unwanted columns:
titanic = titanic.drop(['Name',"Ticket","Cabin","Embarked"], axis='columns')

#Creating dummy variable for sex:
titanic["Sex"] = pd.get_dummies(titanic["Sex"])

#We will also drop all rows that contain NaN-values:
titanic = titanic.dropna()

titanic
10/2:
import pandas as pd
import numpy as np
titanic = pd.read_csv("titanic.csv")
10/3: titanic
10/4:
#Dropping unwanted columns:
titanic = titanic.drop(['Name',"Ticket","Cabin","Embarked"], axis='columns')

#Creating dummy variable for sex:
titanic["Sex"] = pd.get_dummies(titanic["Sex"])

#We will also drop all rows that contain NaN-values:
titanic = titanic.dropna()

titanic
10/5:
labels = titanic["Survived"]
labels
10/6:
data = titanic.drop(['Survived'], axis='columns')
data
10/7: labels
10/8:
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()

# It is extreme easy to try with a different machine learning algorithm instead! 
# Just uncomment the two lines below to use a "k Nearest Neighbors"-model instead of Naive Bayes:

# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier()
10/9:
 from sklearn.naive_bayes import GaussianNB
 model = GaussianNB()

# It is extreme easy to try with a different machine learning algorithm instead! 
# Just uncomment the two lines below to use a "k Nearest Neighbors"-model instead of Naive Bayes:

# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier()
10/10:
 from sklearn.naive_bayes import GaussianNB
 model = GaussianNB()

# It is extreme easy to try with a different machine learning algorithm instead! 
# Just uncomment the two lines below to use a "k Nearest Neighbors"-model instead of Naive Bayes:

# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier()
10/11:
 from sklearn.naive_bayes import GaussianNB
 model = GaussianNB()

# It is extreme easy to try with a different machine learning algorithm instead! 
# Just uncomment the two lines below to use a "k Nearest Neighbors"-model instead of Naive Bayes:

# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier()
10/12: model.fit(data, labels)
10/13:
# New data organized in a two-dimensional array 
x_new = np.array([[100, 3, 1, 20, 0, 0, 30]])
10/14: predict = model.predict(x_new)
10/15: print("Prediction: {}".format(predict))
10/16: print("Accuracy score: {}".format(model.score(data, labels)))
9/108:
from sklearn.datasets import load_boston
data = load_boston()
print(data.target)
print(data['DESCR'][:1300])
9/109: labels = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
9/110:
labels = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
labels
9/111: X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
9/112: X_train, X_test, c_train, c_test = train_test_split(dataframe['data'], c, random_state=0)
9/113: X_train, X_test, c_train, c_test = train_test_split(dataframe.data, c, random_state=0)
9/114: data_from_dataframe = dataframe.to_numpy()
9/115: X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
9/116: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe['data'], c, random_state=0)
9/117: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe, c, random_state=0)
9/118:
data_from_dataframe = dataframe.to_numpy()
data_from_dataframe
9/119: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe, c, random_state=0)
16/1:
from sklearn.datasets import load_boston
data = load_boston()
print(data.target)
print(data['DESCR'][:1300])
16/2:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
16/3:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
16/4:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
16/5:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
16/6: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11, 12]])
16/7:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
16/8:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
16/9:
dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
dataframe
16/10:
dataframe = dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
display(dataframe)
16/11:
labels = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
labels
16/12:
data_from_dataframe = dataframe.to_numpy()
data_from_dataframe
16/13: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe, c, random_state=0)
16/14: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe, labels, random_state=0)
16/15:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
16/16:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
16/17:
dataframe = dataframe.drop(columns=['CHAS', 'B', 'NOX', 'INDUS'], axis= 1)
display(dataframe)
16/18:
dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
display(dataframe)
17/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
17/2:
from sklearn.datasets import load_boston
data = load_boston()
print(data.target)
print(data['DESCR'][:1300])
17/3:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
17/4:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
17/5: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11, 12]])
17/6:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
17/7:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
17/8:
dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
dataframe
17/9:
dataframe.drop(columns=['CHAS', 'B', 'NOX'], axis= 1)
display(dataframe)
17/10:
labels = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
labels
17/11:
data_from_dataframe = dataframe.to_numpy()
data_from_dataframe
17/12: X_train, X_test, c_train, c_test = train_test_split(data_from_dataframe, labels, random_state=0)
17/13:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
17/14:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
17/15:
data_from_dataframe = dataframe.to_numpy()
data_from_dataframe
17/16: X_train1, X_test1, c_train1, c_test1 = train_test_split(data_from_dataframe, labels, random_state=0)
17/17:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train1, c_train1)
17/18:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train1, c_train1)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test1, c_test1)))
18/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import scipy as scipy
import seaborn as sns
18/2:
from sklearn.datasets import load_boston
data = load_boston()
print(data.target)
print(data['DESCR'][:1300])
18/3:
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
print("{}".format(c))
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
18/4:
# Pick the data belonging to class 0
X_0 = data['data'][c==0,:]
18/5: sns.pairplot(pd.DataFrame(X_0,columns=data['feature_names']).iloc[:, [1,3,4,7,11, 12]])
18/6:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, c_train)
18/7:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train, c_train)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test, c_test)))
18/8:
dataframe = pd.DataFrame(data['data'], columns=data['feature_names'])
dataframe
18/9:
dataframe.drop(columns=['CHAS', 'B', 'NOX', 'INDUS'], axis= 1)
display(dataframe)
18/10:
labels = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
labels
18/11:
data_from_dataframe = dataframe.to_numpy()
data_from_dataframe
18/12: X_train1, X_test1, c_train1, c_test1 = train_test_split(data_from_dataframe, labels, random_state=0)
18/13:

nb = GaussianNB()
nb.fit(X_train1, c_train1)
18/14:
print("Training accuracy on Gaussian NB model: {}\n".format(nb.score(X_train1, c_train1)))
print("Test accuracy on Gaussian NB model: {} \n".format(nb.score(X_test1, c_test1)))
21/1:
%matplotlib inline
import warnings; warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns

import matplotlib.pyplot as plt
# import seaborn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

#pd.set_option("display.max_rows", None, "display.max_columns", None)
21/2:
data = pd.read_csv("Hans_new.csv")
numeric_cols = data.columns.drop('Weekday')

data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

#data.dtypes

data
21/3:
# In the "Sleep" column, we replace nan values with the mean:
data["Sleep"] = data["Sleep"].fillna(data["Sleep"].mean())

# In columns "Weight" and "Bodyfat", we replace with the last non-nan value.
# In the columns until first measurement, we replace with next non-nan value
w_first_index = data['Weight'].first_valid_index()
BF_first_index = data['Bodyfat'].first_valid_index()

w = data.at[w_first_index,'Weight']
BF = data.at[BF_first_index, 'Bodyfat']
for i in range(len(data)):
    if np.isnan(data.at[i,'Weight']):
        data.at[i,'Weight'] = w
    else: 
        w = data.at[i, 'Weight']
        
    if np.isnan(data.at[i,'Bodyfat']):
        data.at[i,'Bodyfat'] = BF
    else: 
        BF = data.at[i, 'Bodyfat']

# In remaining columns, we replace nan with zero
data = data.fillna(0)
data
21/4:
# In the "Sleep" column, we replace nan values with the mean:
data["Sleep"] = data["Sleep"].fillna(data["Sleep"].mean())

# In columns "Weight" and "Bodyfat", we replace with the last non-nan value.
# In the columns until first measurement, we replace with next non-nan value
w_first_index = data['Weight'].first_valid_index()
BF_first_index = data['Bodyfat'].first_valid_index()

w = data.at[w_first_index,'Weight']
BF = data.at[BF_first_index, 'Bodyfat']
for i in range(len(data)):
    if np.isnan(data.at[i,'Weight']):
        data.at[i,'Weight'] = w
    else: 
        w = data.at[i, 'Weight']
        
    if np.isnan(data.at[i,'Bodyfat']):
        data.at[i,'Bodyfat'] = BF
    else: 
        BF = data.at[i, 'Bodyfat']

# In remaining columns, we replace nan with zero
data = data.fillna(0)
data
21/5:
# In the "Sleep" column, we replace nan values with the mean:
data["Sleep"] = data["Sleep"].fillna(data["Sleep"].mean())

# In columns "Weight" and "Bodyfat", we replace with the last non-nan value.
# In the columns until first measurement, we replace with next non-nan value
w_first_index = data['Weight'].first_valid_index()
BF_first_index = data['Bodyfat'].first_valid_index()

w = data.at[w_first_index,'Weight']
BF = data.at[BF_first_index, 'Bodyfat']
for i in range(len(data)):
    if np.isnan(data.at[i,'Weight']):
        data.at[i,'Weight'] = w
    else: 
        w = data.at[i, 'Weight']
        
    if np.isnan(data.at[i,'Bodyfat']):
        data.at[i,'Bodyfat'] = BF
    else: 
        BF = data.at[i, 'Bodyfat']

# In remaining columns, we replace nan with zero
data = data.fillna(0)
data
21/6:
# data = data.drop(['Day'], axis=1)
data['Lotion_face'] = data['Protopic_face'] + data['Mildison_face']
data['Lotion_body'] = data['Elocon_body'] + data['Dermovat_body'] + data['Locoid_body']
data = data.drop(['Protopic_face', 'Mildison_face',
                  'Elocon_body', 'Dermovat_body', 'Locoid_body'], axis=1)
data
21/7:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data_exp = data_exp.drop(['index'], axis=1)
21/8:
target = "Eczema_face"
day_types = data_exp[target]
day_types
21/9:
target = "Eczema_face"
day_types = data_exp[target]
day_types
21/10:
data_exp = pd.get_dummies(data_exp)
data_exp
23/1:
%matplotlib inline
import numpy as np
import pandas as pd
import sklearn as sk

import warnings; warnings.simplefilter('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
23/2:
data = pd.read_csv('fruits-data.csv')

data[0:-1]
23/3:
data = pd.read_csv('fruits-data.csv')

data[0:-1]
23/4:
print("Number of rows before removing NaNs: {}".format(data.shape[0]))
data = data.dropna()
print("Number of rows after removing NaNs: {}".format(data.shape[0]))
23/5:
print("Number of rows before removing NaNs: {}".format(data.shape[0]))
data = data.dropna()
print("Number of rows after removing NaNs: {}".format(data.shape[0]))
23/6: data.boxplot('Diameter')
23/7: data.boxplot('Weight')
23/8: data.boxplot('Color')
23/9: data.boxplot('Weight')
23/10: data.boxplot('Diameter')
23/11:
Q1 = data['Diameter'].quantile(0.25)
Q3 = data['Diameter'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data.shape[0]))

data = data[data["Diameter"] >= Q1-1.5*IQR]
data = data[data["Diameter"] <= Q3+1.5*IQR]
data = data[data["Diameter"] > 0]

print("Number of rows after applying filter: {}".format(data.shape[0]))
23/12: data.boxplot('Diameter')
23/13: data = data.reset_index(drop=True)
23/14:
Q1 = data['Weight'].quantile(0.25)
Q3 = data['Weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data.shape[0]))

data = data[data["Weight"] >= Q1-1.5*IQR]
data = data[data["Weight"] <= Q3+1.5*IQR]
data = data[data["Weight"] > 0]

print("Number of rows after applying filter: {}".format(data.shape[0]))
23/15:
Q1 = data['Weight'].quantile(0.25)
Q3 = data['Weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data.shape[0]))

data = data[data["Weight"] >= Q1-1.5*IQR]
data = data[data["Weight"] <= Q3+1.5*IQR]
data = data[data["Weight"] > 0]

print("Number of rows after applying filter: {}".format(data.shape[0]))
23/16: data.boxplot('Weight')
23/17:
y = data.loc[:,'Label']
features = data.loc[:,'Color':'Weight']
features = pd.get_dummies(features)
print(list(features.columns))
features[0:10]
23/18:
X = features.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
23/19:
clf = KNeighborsClassifier() 
clf.fit(X_train,y_train)
print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
23/20:
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(features.values, y, random_state=0)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
23/21:
print("Minimum and maximum value of Diameter in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value of Diameter in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
23/22:
clf = KNeighborsClassifier() 
clf.fit(X_train_scaled,y_train)
print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/1:
%matplotlib inline
import warnings; warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# import seaborn
24/2:
data = pd.read_csv("Hans.csv")
numeric_cols = data.columns.drop('Weekday')

data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

#data.dtypes

data[:10]
24/3:
# In the "Sleep" column, we replace nan values with the mean:
data["Sleep"] = data["Sleep"].fillna(data["Sleep"].mean())

# In columns "Weight" and "Bodyfat", we replace with the last non-nan value.
# In the columns until first measurement, we replace with next non-nan value
w_first_index = data['Weight'].first_valid_index()
BF_first_index = data['Bodyfat'].first_valid_index()

w = data.at[w_first_index,'Weight']
BF = data.at[BF_first_index, 'Bodyfat']
for i in range(len(data)):
    if np.isnan(data.at[i,'Weight']):
        data.at[i,'Weight'] = w
    else: 
        w = data.at[i, 'Weight']
        
    if np.isnan(data.at[i,'Bodyfat']):
        data.at[i,'Bodyfat'] = BF
    else: 
        BF = data.at[i, 'Bodyfat']

# In remaining columns, we replace nan with zero
data = data.fillna(0)
24/4:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)
24/5:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]
24/6:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp
24/7: data_exp_dummies = pd.get_dummies(data_exp)
24/8: data_exp_dummies = pd.get_dummies(data_exp)
24/9:
data_exp_dummies = pd.get_dummies(data_exp)
data_exp_dummies
24/10:
# We select the target variable:
target = "Eczema_face"
24/11: data_exp_dummies[target].value_counts()
24/12:
print("Number of days with no eczema: {}".format(len(data_exp_dummies.loc[data_exp_dummies['Eczema_face'] == 0])))
print("Number of days with some eczema: {}".format(len(data_exp_dummies.loc[data_exp_dummies['Eczema_face'] > 0])))
24/13:
n_train = 1000
data_train = data_exp_dummies[0:n_train]
data_test = data_exp_dummies[n_train:]
24/14:
features_train = data_train.drop(target, axis=1)
features_test = data_test.drop(target, axis=1)

labels_train = data_train[target]
labels_test = data_test[target]

labels_train = labels_train > 0
labels_test = labels_test > 0

X_train, X_test = features_train.values, features_test.values
y_train, y_test = labels_train.values, labels_test.values
24/15:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/16:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[:10]
24/17:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[:20]
24/18:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[10:20]
24/19:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[70:20]
24/20:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[:20]
24/21:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[:700]
24/22:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[:70]
24/23:
data_exp = data.copy()
prev_days = 3
# Removing the first prev_days days:
data_exp = data_exp[prev_days:].reset_index()
for i in range(1,prev_days+1):
    data_min = data[prev_days-i:-i].drop(['Weekday'], axis=1).reset_index()
    data_exp = data_exp.join(data_min, rsuffix='_min'+str(i))
    data_exp = data_exp.drop(['index_min'+str(i)], axis=1)

data[:10]    
data_exp[20:70]
24/24:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/25:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/26:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/27:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/28:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/29:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/30:
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/31: data_train
24/32: data_test
24/33: data_test
24/34: data_exp_dummies
24/35: data_exp_dummies
24/36:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.loc[:,'Color':'Weight']
features = pd.get_dummies(features)
X = feature.values
24/37:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
features = pd.get_dummies(features)
X = feature.values
24/38:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
features = pd.get_dummies(features)
X = features.values
24/39: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
24/40:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = features.values
24/41: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
24/42:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = features.values
y
24/43:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = features.values
y
X
24/44:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = features
y
X
24/45:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
X
24/46: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
24/47: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
24/48:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
24/49:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value of Diameter in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value of Diameter in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/50:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/51:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/52:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/53:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/54:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/55:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/56:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/57:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
X
y
24/58:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
X
24/59:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))

X_train_scaled
24/60:
###################################Test with Scaled Data ###########################################3
clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/61:
###################################Test with Scaled Data ###########################################3
clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/62:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
y
24/63:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
y = y > 0
24/64:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
y = y > 0
y
24/65:
features_train = data_train.drop(target, axis=1)
features_test = data_test.drop(target, axis=1)

labels_train = data_train[target]
labels_test = data_test[target]

labels_train = labels_train > 0
labels_test = labels_test > 0

X_train, X_test = features_train.values, features_test.values
y_train, y_test = labels_train.values, labels_test.values

labels_train
24/66:

y = data_exp_dummies.loc[:,'Eczema_face']
features = data_exp_dummies.columns.drop('Eczema_face')
X = data_exp_dummies.loc[:, features]
y = y > 0
24/67: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
24/68:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/69:
###################################Test with Scaled Data ###########################################3
clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/70:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/71:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/72:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/73:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/74:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/75:
from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/76:
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#scaler = MinMaxScaler()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/77:
###################################Test with Scaled Data ###########################################3
#clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/78:
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
#scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/79:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

#clf = GaussianNB()
#usually 3 gives the best result
#clf = KNeighborsClassifier(n_neighbors=10)
#clf.fit(X_train_scaled,y_train)

#print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/80:
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#scaler = MinMaxScaler()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Minimum and maximum value in train data: {}, {}".format(np.amin(X_train_scaled),np.amax(X_train_scaled)))
print("Minimum and maximum value in test data: {}, {}".format(np.amin(X_test_scaled),np.amax(X_test_scaled)))
24/81:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

#clf = GaussianNB()
#usually 3 gives the best result
#clf = KNeighborsClassifier(n_neighbors=10)
#clf.fit(X_train_scaled,y_train)

#print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/82:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

#print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

clf = GaussianNB()
#usually 3 gives the best result
clf = KNeighborsClassifier(n_neighbors=10)
#clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/83:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

#print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

clf = GaussianNB()
#usually 3 gives the best result
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))
24/84:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

#print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

clf = GaussianNB()
#usually 3 gives the best result
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
24/85:
###################################Test with Scaled Data ###########################################3
from sklearn.svm import SVC
svm = SVC(C=100, gamma='auto')
svm.fit(X_train_scaled, y_train)

#print("Accuracy on training data = {}".format(svm.score(X_train, y_train)))
#print("Accuracy on testing data = {}\n".format(svm.score(X_test, y_test)))

clf = GaussianNB()
#usually 3 gives the best result
#clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train_scaled,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test, y_test)))

print("Accuracy on training data = {}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy on testing data = {}\n".format(clf.score(X_test_scaled, y_test)))
27/1:
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
27/2:
reviews = pd.read_csv('Movie_reviews/reviews.txt', header=None)
labels = pd.read_csv('Movie_reviews/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
27/3:
reviews = pd.read_csv('Movie_reviews/reviews.txt', header=None)
labels = pd.read_csv('Movie_reviews/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
27/4:
reviews = pd.read_csv('Movie_reviews/reviews.txt', header=None)
labels = pd.read_csv('Movie_reviews/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
27/5:
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
27/6: reviews[0][5]
27/7:
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
reviews
27/8:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vect = CountVectorizer(max_features=1000).fit(reviews[0])
27/9: print(vect.vocabulary_)
27/10:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
27/11:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()
Y

X_train, X_test, y_train, y_test = train_test_split(X, Y)
27/12:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
Y
27/13:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
X
27/14:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# clf = KNeighborsClassifier()
clf = GaussianNB()
# clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
27/15:
print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
27/16:
bad_review = vect.transform(["This is the worst movie of all time!"]).toarray()
good_review = vect.transform(["Perfect movie!"]).toarray()
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(["This is not a bad movie, it is actually really good!"]).toarray()
print(reviewA)
27/17:
print("Predicted class for bad review: {}".format(clf.predict(bad_review)))
print("Predicted class for good review: {}".format(clf.predict(good_review)))
print("Predicted class for review A: {}".format(clf.predict(reviewA)))
print("Predicted class for review B: {}".format(clf.predict(reviewB)))
27/18:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# clf = KNeighborsClassifier()
#clf = GaussianNB()
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
27/19:
print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
27/20:
bad_review = vect.transform(["This is the worst movie of all time!"]).toarray()
good_review = vect.transform(["Perfect movie!"]).toarray()
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(["This is not a bad movie, it is actually really good!"]).toarray()
print(reviewA)
27/21:
print("Predicted class for bad review: {}".format(clf.predict(bad_review)))
print("Predicted class for good review: {}".format(clf.predict(good_review)))
print("Predicted class for review A: {}".format(clf.predict(reviewA)))
print("Predicted class for review B: {}".format(clf.predict(reviewB)))
27/22:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
clf = KNeighborsClassifier()
#clf = GaussianNB()
#clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
27/23:
print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
27/24:
bad_review = vect.transform(["This is the worst movie of all time!"]).toarray()
good_review = vect.transform(["Perfect movie!"]).toarray()
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(["This is not a bad movie, it is actually really good!"]).toarray()
print(reviewA)
27/25:
print("Predicted class for bad review: {}".format(clf.predict(bad_review)))
print("Predicted class for good review: {}".format(clf.predict(good_review)))
print("Predicted class for review A: {}".format(clf.predict(reviewA)))
print("Predicted class for review B: {}".format(clf.predict(reviewB)))
28/1:
%matplotlib inline
import numpy as np
import pandas as pd
import sklearn as sk

import warnings; warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn

from sklearn.preprocessing import MinMaxScaler
28/2:
data = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")
data[0:10]
28/3:
data_binary = data.copy()
data_binary['Index'] = (data['Index'] > 2).astype(int)
data_binary = data_binary.rename(columns = {'Index':'Overweight'})
data_binary[0:10]
28/4:
print("Number of rows before removing NaNs: {}".format(data_binary.shape[0]))
data_binary = data_binary.dropna()
print("Number of rows after removing NaNs: {}".format(data_binary.shape[0]))
28/5:
print("Number of rows before removing NaNs: {}".format(data_binary.shape[0]))
data_binary = data_binary.dropna()
print("Number of rows after removing NaNs: {}".format(data_binary.shape[0]))
28/6:
##############################Identify outliers#######################
Q1 = data_binary['Height'].quantile(0.25)
Q3 = data_binary['Height'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data_binary.shape[0]))

data_binary = data_binary[data_binary["Height"] >= Q1-1.5*IQR]
data_binary = data_binary[data_binary["Height"] <= Q3+1.5*IQR]
data_binary = data_binary[data_binary["Height"] > 0]

print("Number of rows after applying filter: {}".format(data_binary.shape[0]))
28/7:
##############################Identify outliers#######################
Q1 = data_binary['Height'].quantile(0.25)
Q3 = data_binary['Height'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data_binary.shape[0]))

data_binary = data_binary[data_binary["Height"] >= Q1-1.5*IQR]
data_binary = data_binary[data_binary["Height"] <= Q3+1.5*IQR]
data_binary = data_binary[data_binary["Height"] > 0]

print("Number of rows after applying filter: {}".format(data_binary.shape[0]))
28/8:
##############################Identify outliers####################### Weight
Q1 = data_binary['Weight'].quantile(0.25)
Q3 = data_binary['Weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data_binary.shape[0]))

data_binary = data_binary[data_binary["Weight"] >= Q1-1.5*IQR]
data_binary = data_binary[data_binary["Weight"] <= Q3+1.5*IQR]
data_binary = data_binary[data_binary["Weight"] > 0]

print("Number of rows after applying filter: {}".format(data_binary.shape[0]))
28/9:
##############################Identify outliers####################### Weight
Q1 = data_binary['Weight'].quantile(0.25)
Q3 = data_binary['Weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

print("Number of rows before applying filter: {}".format(data_binary.shape[0]))

data_binary = data_binary[data_binary["Weight"] >= Q1-1.5*IQR]
data_binary = data_binary[data_binary["Weight"] <= Q3+1.5*IQR]
data_binary = data_binary[data_binary["Weight"] > 0]

print("Number of rows after applying filter: {}".format(data_binary.shape[0]))
28/10:
data_binary['Gender'] = (data['Gender'] == 'Male').astype(int)
data_binary[0:10]
28/11:
features = data_binary.loc[:, "Overweight"]
labels = data_binary.loc[:, "Geder", "Height", "Weight"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)
28/12:
features = data_binary.loc[:, "Overweight"]
labels = data_binary.loc[:, "Geder", "Height", "Weight"]
X_train, X_test, y_train,y_test = train_test_split(features,  stratify=labels)
28/13:
features = data_binary.loc[:, "Overweight"]
labels = data_binary.loc[:, "Gender", "Height", "Weight"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)
28/14:
features = data_binary.loc[:, "Overweight"]
labels = data_binary.loc[:, "Gender":"Weight"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)
28/15:
labels = data_binary.loc[:, "Overweight"]
features = data_binary.loc[:, "Gender":"Weight"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)
28/16:
clf = GaussianNB()
clf.fit(X_train, y_train)
28/17:
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
28/18:
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
28/19:
BMI = data_binary['Weight']/(data_binary['Height']*data_binary['Height']*10000)
BMI
28/20:
BMI = (data_binary['Weight']/(data_binary['Height']*data_binary['Height']))*10000
BMI
28/21:
BMI = (data_binary['Weight']/(data_binary['Height']*data_binary['Height']))*10000
BMI
28/22: data_binary['BMI'] = BMI
28/23:
labels = data_binary.loc[:, "Overweight"]
features = data_binary.loc[:, "Gender":"Weight"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)

clf.fit(X_train, y_train)

print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
28/24:
data_binary['BMI'] = BMI
data_binary
28/25:
data_binary = data_binary[["Gender", "Height", "Weight", "BMI", "Overweight"]]

labels = data_binary.loc[:, "Overweight"]
features = data_binary.loc[:, "Gender":"BMI"]
X_train, X_test, y_train,y_test = train_test_split(features, labels,random_state=0, stratify=labels)

clf.fit(X_train, y_train)

print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
28/26:
####################################Trying to scale###########################
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

clf.fit(X_train, y_train)
print("Accuracy on training data = {}".format(clf.score(X_train_scaled,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test_scaled,y_test)))
28/27:
####################################Trying to scale###########################
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

clf.fit(X_train_scaled, y_train)
print("Accuracy on training data = {}".format(clf.score(X_train_scaled,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test_scaled,y_test)))
29/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
29/2:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# Let's learn a little about the dataset:
print(cancer.keys())
print(cancer['DESCR'])
cancer['data'][1:10]
29/3:
# Divide the data into training and test
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
29/4:
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 20)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend();
29/5:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
29/6:
cancer = load_breast_cancer()
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=43)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of validation set:{}".format(X_val.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
29/7:
best_score = 0
for num_neighbors in range(1,15):
    # Learn the model with a certain numnber of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate the model
    score = knn.score(X_val, y_val)
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best score on validation set: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
29/8:
cancer = load_breast_cancer()
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
29/9:
from sklearn.model_selection import cross_val_score
best_score = 0
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
29/10:
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
29/11:
clf = KNeighborsClassifier()
scores = cross_val_score(clf, cancer.data, cancer.target, cv=loo)
29/12:
clf = KNeighborsClassifier()
scores = cross_val_score(clf, cancer.data, cancer.target, cv=loo)
29/13: print("Cross validation scores: {}".format(scores))
29/14: print("Average cross validation score: {}".format(scores.mean()))
29/15: print("Standard deviation of the cross validation scores: {}".format(scores.std()))
29/16:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
best_score = 0
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform leave-one-out cross ("loo") validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=loo) 
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
29/17:
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
29/18:
# Fix the parameter space
parameters = {'n_neighbors': range(1,15)}
grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, return_train_score=True)
29/19:
# Load the data and divide into train and test
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
29/20: grid_search.fit(X_train, y_train)
29/21: print("Test score: {:.2f}".format(grid_search.score(X_test, y_test)))
29/22: print("Best parameter: {}".format(grid_search.best_params_))
29/23: print("Best cross-validation score: {}".format(grid_search.best_score_))
29/24: print("Best estimator: {}".format(grid_search.best_estimator_))
29/25: pd.DataFrame(grid_search.cv_results_)
30/1:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
30/2:
from sklearn.datasets import load_boston
data = load_boston()
# Let's learn a little about the dataset:
print(data.keys())
print(data['DESCR'])
data['data'][1:10]
30/3: c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
30/4: c
30/5: dataArray = data.data
30/6:
dataArray = data.data
dataArray
30/7:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT", "MEDV"])
dataFrame
30/8:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])
dataFrame
30/9:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])
dataFrame
30/10:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c
dataFrame
30/11:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

dataFrame.boxplot('AGE')
30/12:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

dataFrame.boxplot('TAX')
30/13:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

dataFrame.boxplot('CRIM')
33/1:
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
plt.rcParams['image.cmap'] = "gray"
33/2: mglearn.plots.plot_scaling()
33/3:
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)
33/4:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
33/5: scaler.fit(X_train)
33/6:
# transform data
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))
33/7:
# transform test data
X_test_scaled = scaler.transform(X_test)
# print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
33/8:
from sklearn.datasets import make_blobs
# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# split it into training and test sets
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=[mglearn.cm2(0)], label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=[mglearn.cm2(1)], label="Test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

# scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=[mglearn.cm2(0)], label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                c=[mglearn.cm2(1)], label="Test set", s=60)
axes[1].set_title("Scaled Data")

# rescale the test set separately
# so test set min is 0 and test set max is 1
# DO NOT DO THIS! For illustration purposes only.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=[mglearn.cm2(0)], label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c=[mglearn.cm2(1)], label="test set", s=60)
axes[2].set_title("Improperly Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
fig.tight_layout()
33/9:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# calling fit and transform in sequence (using method chaining)
X_scaled = scaler.fit(X_train).transform(X_train)
# same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X_train)
33/10:
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

svm = SVC(C=100, gamma='auto')
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
33/11:
# preprocessing using 0-1 scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(
    svm.score(X_test_scaled, y_test)))
33/12:
# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
30/14:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

dataFrame.boxplot('LSTAT')
30/15:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c
dataFrame
##dataFrame.boxplot('LSTAT')
30/16:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

dataFrame.boxplot('LSTAT')



##dataFrame
30/17:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')
##dataFrame

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame.drop(columns = ["MEDV"])
30/18:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/19:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/20:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/21:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

#dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/22:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

#dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/23:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/24:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/25:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/26:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/27:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/28:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label


##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame

c
30/29:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label


##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/30: c
30/31: c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
30/32: c
30/33:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
#dataFrame
30/34:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
##c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/35:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/36:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame.drop(columns = ["MEDV"])
dataFrame
30/37: c
30/38: c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
30/39:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame
30/40: c
30/41:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV")

### Dummies - not nedded, all data is numeric
### Normalize data





dataFrame
30/42:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data





dataFrame
30/43:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(dataFrame.to_numpy())
values_scaled 





#dataFrame
30/44:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(dataFrame.to_numpy())
values_scaled 
dataFrame.values = values_scaled





#dataFrame
30/45:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(dataFrame.to_numpy())
values_scaled 
dataFrame = pd.DataFRame(values_scaled, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])





dataFrame
30/46:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(dataFrame.to_numpy())
values_scaled 
dataFrame = pd.DataFrame(values_scaled, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])





dataFrame
30/47:
dataArray = data.data
dataFrame = pd.DataFrame(dataArray, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO", "B", "LSTAT"])

##### Data preparation

### NaN
dataFrame = dataFrame.dropna()

### Removing unnecessary features
dataFrame = dataFrame.drop(columns = ["CHAS", "NOX", "PTRATIO", "B"])

### Identifying and removing outliners -  the target has been also added in the dataframe such that when removing the outliners 
### we could also remove the unusable label

dataFrame["MEDV"] = c

##dataFrame.boxplot('CRIM')
##dataFrame.boxplot('LSTAT')

### No outliners where found that need to be removed so nothing happens to the dataset
c = dataFrame["MEDV"]
dataFrame = dataFrame.drop("MEDV", axis = 1)

### Dummies - not nedded, all data is numeric
### Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(dataFrame.to_numpy())
values_scaled 
dataFrame = pd.DataFrame(values_scaled, columns = ["CRIM", "ZN", "INDUS", "RM","AGE","DIS","RAD","TAX", "LSTAT"])





dataFrame
30/48:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
30/49:

X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.as_numpy(), c, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=43)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of validation set:{}".format(X_val.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
30/50:

X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), c, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=43)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of validation set:{}".format(X_val.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
30/51:

X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), c, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=43)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of validation set:{}".format(X_val.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
30/52:
best_score = 0
for num_neighbors in range(1,15):
    # Learn the model with a certain numnber of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate the model
    score = knn.score(X_val, y_val)
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best score on validation set: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
30/53:
best_score = 0
for num_neighbors in range(1,15):
    # Learn the model with a certain numnber of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate the model
    score = knn.score(X_val, y_val)
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best score on validation set: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
30/54:

#### Left one out cross validation

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

clf = KNeighborsClassifier()
scores = cross_val_score(clf, dataFrame.to_numpy(), c, cv=loo)
print("Average cross validation score: {}".format(scores.mean()))
30/55:

#### Left one out cross validation

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

clf = KNeighborsClassifier()
scores = cross_val_score(clf, dataFrame.to_numpy(), c, cv=loo)
print("Average cross validation score: {}".format(scores.mean()))
30/56:
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform leave-one-out cross ("loo") validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=loo) 
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
30/57:
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform leave-one-out cross ("loo") validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=loo) 
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
30/58:

################################################ Cross validation using GridSearch
from sklearn.model_selection import GridSearchCV
# Fix the parameter space
parameters = {'n_neighbors': range(1,15)}
grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, return_train_score=True)

# Load the data and divide into train and test
X_train, X_test, y_train, y_test = train_test_split(dataFrame.to_numpy(), c, random_state=42)
30/59:

################################################ Cross validation using GridSearch
from sklearn.model_selection import GridSearchCV
# Fix the parameter space
parameters = {'n_neighbors': range(1,15)}
grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, return_train_score=True)

# Load the data and divide into train and test
X_train, X_test, y_train, y_test = train_test_split(dataFrame.to_numpy(), c, random_state=42)
30/60:
grid_search.fit(X_train, y_train)

print("Test score: {:.2f}".format(grid_search.score(X_test, y_test)))
30/61: print("Best parameter: {}".format(grid_search.best_params_))
30/62: print("Best cross-validation score: {}".format(grid_search.best_score_))
35/1:
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import mglearn
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix,precision_score,recall_score,f1_score,
    roc_curve,roc_auc_score,precision_recall_curve,accuracy_score,classification_report)
35/2:
titanic = pd.read_csv("titanic.csv")
#Dropping unwanted columns:
titanic = titanic.drop(['Name',"Ticket","Cabin","Embarked"], axis='columns')

#Creating dummy variable for sex:
titanic["Sex"] = pd.get_dummies(titanic["Sex"])

#We will also drop all rows that contain NaN-values:
titanic = titanic.dropna()

X = titanic.drop(['Survived'], axis='columns')
y = titanic["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y)
35/3:
clf = GaussianNB()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("Confusion matrix:  \n{}\n".format(confusion_matrix(y_test,predictions)))
print("Accuracy: \n{}\n".format(clf.score(X_test,y_test)))
print("Precision: \n{}\n".format(precision_score(y_test,predictions,pos_label=1)))
print("Recall: \n{}\n".format(recall_score(y_test,predictions,pos_label=1)))
print("F1: \n{}".format(f1_score(y_test,predictions,pos_label=1)))
35/4:
fpr,tpr,thresh = roc_curve(y_test,clf.predict(X_test))
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
print("Area under curve: \n{}".format(roc_auc_score(y_test,clf.predict(X_test))) )
thresh
35/5:
prec,rec,thresh = precision_recall_curve(y_test,clf.predict(X_test))
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.plot(prec,rec)
thresh
35/6:
clf = GaussianNB(priors=[0.01,0.99])
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("Confusion matrix:  \n{}\n".format(confusion_matrix(y_test,predictions)))
print("Accuracy: \n{}\n".format(clf.score(X_test,y_test)))
print("Precision: \n{}\n".format(precision_score(y_test,predictions,pos_label=1)))
print("Recall: \n{}\n".format(recall_score(y_test,predictions,pos_label=1)))
print("F1: \n{}".format(f1_score(y_test,predictions,pos_label=1)))
35/7:
parameters = {'priors': [[0.01,0.99],[0.1,0.9]]}
NB_grid_search = GridSearchCV(GaussianNB(), parameters, cv=5, return_train_score=True, scoring="recall")
NB_grid_search.fit(X_train, y_train)
35/8:
predictions = NB_grid_search.predict(X_test)
print("Confusion matrix:  \n{}\n".format(confusion_matrix(y_test,predictions)))
print("Accuracy: \n{}\n".format(clf.score(X_test,y_test)))
print("Precision: \n{}\n".format(precision_score(y_test,predictions,pos_label=1)))
print("Recall: \n{}\n".format(recall_score(y_test,predictions,pos_label=1)))
print("F1: \n{}".format(f1_score(y_test,predictions,pos_label=1)))
40/1:
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
40/2:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data
40/3:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data
40/4:
X = data.loc[:,'Height':'Weight']
y = data['Index']
40/5: X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
40/6:
ODS = LinearRegression()
ODS.fit(X_train, y_train)
40/7:
print("R^2 on train data is {} and on test data is {}".format(ODS.score(X_train, y_train), 
                                                              ODS.score(X_test,y_test)))
40/8: print("The coefficents are {} and the intercept is {}".format(ODS.coef_, ODS.intercept_))
40/9:
h = 174
w = 64
print("A person with a height of {}cm and a weight of {}kg is predicted by the model \
to have an index of {}".format(h, w, ODS.predict(np.array([[h, w]]))))
40/10:
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train.iloc[:,0], X_train.iloc[:,1], y_train, marker='.', color='red')
ax.set_xlabel("Heigt [kg]")
ax.set_ylabel("Weight [cm]")
ax.set_zlabel("y [index]")
ax.view_init(elev=20., azim=35)

model = LinearRegression()
model.fit(X_train, y_train)

n = 10
xs = np.tile(np.linspace(130, 200, num=n), (n,1))
ys = np.tile(np.linspace(40, 200, num=n), (n,1)).T
zs = np.zeros_like(xs)
for i in range(n):
    for j in range(n):
        x = xs[i,j]
        y = ys[i,j]
        zs[i,j] = model.predict([[x,y]])
        
        
# zs = xs*model.coef_[0]+ys*model.coef_[1]+model.intercept_ # This would also work (for the linear model)

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()
40/11:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train.iloc[:,0], X_train.iloc[:,1], y_train, marker='.', color='red')
ax.set_xlabel("Heigt [kg]")
ax.set_ylabel("Weight [cm]")
ax.set_zlabel("y [index]")
ax.view_init(elev=20., azim=35)

model = KNeighborsRegressor()
model.fit(X_train, y_train)

n = 10
xs = np.tile(np.linspace(130, 200, num=n), (n,1))
ys = np.tile(np.linspace(40, 200, num=n), (n,1)).T
zs = np.zeros_like(xs)
for i in range(n):
    for j in range(n):
        x = xs[i,j]
        y = ys[i,j]
        zs[i,j] = model.predict([[x,y]])
                
ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()
42/1:
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
42/2:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
X = data.loc[:,'Gender':'Weight']
y = data['Index']

X = pd.get_dummies(X)
X
42/3: X_train, X_test , y_train, y_test = train_test_split(X, y, random_state=1)
42/4: alphas = 10**np.linspace(-10, 10, 100)
42/5:
ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)
height_coefficients = [coef[0] for coef in coefs]
weight_coefficients = [coef[1] for coef in coefs]
gender_female_coefficients = [coef[2] for coef in coefs]
gender_male_coefficients = [coef[3] for coef in coefs]
42/6:
ax = plt.gca()
ax.set_xscale('log')
ax.plot(alphas, height_coefficients, label="height coefficient")
ax.plot(alphas, weight_coefficients, label="weight coefficient")
ax.plot(alphas, gender_female_coefficients, label="female coefficient")
ax.plot(alphas, gender_male_coefficients, label="male coefficient")
plt.legend()
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
42/7:
for alpha in [alphas[0], alphas[50], alphas[-1]]:
    ridge = Ridge(normalize=True, alpha=alpha)
    ridge.fit(X_train, y_train)
    print("alhpa = {}".format(alpha))
    print("R^2 on train data is {} and on test data is {}".format(ridge.score(X_train, y_train), 
                                                              ridge.score(X_test,y_test)))
    print("")
42/8:
ridgecv = RidgeCV(alphas = alphas, normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_
42/9:
print("R^2 on train data is {} and on test data is {}".format(ridgecv.score(X_train, y_train), 
                                                              ridgecv.score(X_test,y_test)))
42/10: ridgecv.coef_
42/11:
lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    
np.shape(coefs)
height_coefficients = [coef[0] for coef in coefs]
weight_coefficients = [coef[1] for coef in coefs]
gender_female_coefficients = [coef[2] for coef in coefs]
gender_male_coefficients = [coef[3] for coef in coefs]
42/12:
ax = plt.gca()
ax.set_xscale('log')
ax.plot(alphas, height_coefficients, label="height coefficient")
ax.plot(alphas, weight_coefficients, label="weight coefficient")
ax.plot(alphas, gender_female_coefficients, label="female coefficient")
ax.plot(alphas, gender_male_coefficients, label="male coefficient")
plt.legend()
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
42/13:
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
42/14:
print("R^2 on train data is {} and on test data is {}".format(lassocv.score(X_train, y_train), 
                                                              lassocv.score(X_test,y_test)))
42/15:
# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=X.columns)
41/1:
%matplotlib inline
import warnings; warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# import seaborn
41/2:
data = pd.read_csv("Hans_new.csv")
numeric_cols = data.columns.drop('Weekday')

data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

#data.dtypes

data[:10]
41/3:
# In the "Sleep" column, we replace nan values with the mean:
data["Sleep"] = data["Sleep"].fillna(data["Sleep"].mean())

# In columns "Weight" and "Bodyfat", we replace with the last non-nan value.
# In the columns until first measurement, we replace with next non-nan value
w_first_index = data['Weight'].first_valid_index()
BF_first_index = data['Bodyfat'].first_valid_index()

w = data.at[w_first_index,'Weight']
BF = data.at[BF_first_index, 'Bodyfat']
for i in range(len(data)):
    if np.isnan(data.at[i,'Weight']):
        data.at[i,'Weight'] = w
    else: 
        w = data.at[i, 'Weight']
        
    if np.isnan(data.at[i,'Bodyfat']):
        data.at[i,'Bodyfat'] = BF
    else: 
        BF = data.at[i, 'Bodyfat']

# In remaining columns, we replace nan with zero
data = data.fillna(0)
data
41/4: sns.pairplot(data)
43/1:
import pandas as pd

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
44/1:
import pandas as pd

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
44/2:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
44/3:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
#print(reviews.head())
44/4:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
44/5:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
44/6:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
44/7:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
45/1:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
46/1:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
46/2:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
46/3:
from keras.datasets import mnist

##Martin and Roxana

(x_train, y_train), (x_test, y_test) = mnist.load_data()
47/1:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(reviews[0:1])
print(Y[0:1])
47/2:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(reviews[0:1])
print(reviews.head)
47/3:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(reviews.type)
print(reviews.head)
47/4:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
49/1:

train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
print(len(train_reviews))
print(len(validate_reviews))
print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/2:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
49/3:

train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
print(len(train_reviews))
print(len(validate_reviews))
print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/4:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(review.head])
49/5:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(review.head)
49/6:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head)
50/1:
cancer = load_breast_cancer()
print(type(cancer))
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
50/2:
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
50/3:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# Let's learn a little about the dataset:
print(cancer.keys())
print(cancer['DESCR'])
cancer['data'][1:10]
50/4:
cancer = load_breast_cancer()
print(type(cancer))
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
49/7:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, labels, random_state=42)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))






#train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
#print(len(train_reviews))
#print(len(validate_reviews))
#print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/8:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
49/9:


X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, labels, random_state=42)
print("Size of training set:{}".format(X_train.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))






#train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
#print(len(train_reviews))
#print(len(validate_reviews))
#print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/10:


X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, labels, random_state=42)
print("Size of training set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))






#train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
#print(len(train_reviews))
#print(len(validate_reviews))
#print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/11:


X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, labels, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))






#train_reviews, validate_reviews, test_reviews = np.split(reviews.sample(frac=1), [int(.6*len(reviews)), int(.8*len(reviews))])
#print(len(train_reviews))
#print(len(validate_reviews))
#print(len(test_reviews))

#train_labels, validate_labels, test_labels = np.split(labels.sample(frac=1), [int(.6*len(labels)), int(.8*len(labels))])
#print(len(train_labels))
#print(len(validate_labels))
#print(len(test_labels))
49/12:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
best_score = 0
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
49/13:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

best_score = 0
best_num_neighbors = 0
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
49/14:
import pandas as pd
import numpy as np

reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head)
49/15:


X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
49/16:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

best_score = 0
best_num_neighbors = 0
for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors

# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test, y_test)))
49/17: from sklearn.feature_extraction.text import CountVectorizer
49/18:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vect = CountVectorizer(max_features=10000).fit(reviews[0])
49/19:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vect = CountVectorizer(max_features=10000).fit(reviews[0])
print(vect.vocabulary_)
49/20:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vect = CountVectorizer(max_features=10000).fit(reviews[0])
print(vect.vocabulary_)
49/21:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
49/22:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#clf = KNeighborsClassifier()
#clf = GaussianNB()
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
49/23:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#clf = KNeighborsClassifier()
#clf = GaussianNB()
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
49/24:
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval = vect.transform(X_trainval[0]).toarray()
X_test = vect.transform(X_test[0]).toarray()
49/25:
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval = vect.transform(X_trainval).toarray()
X_test = vect.transform(X_test).toarray()
49/26:
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval = vect.transform(X_trainval).toarray()
X_test = vect.transform(X_test).toarray()
49/27:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))


print("Size of test set:{}".format(type(X_trainval)))
49/28:
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval_transformed = vect.transform(X_trainval[0]).toarray()
X_test_transformed = vect.transform(X_test[0]).toarray()
49/29:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
49/30:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vect = CountVectorizer(max_features=10000).fit(reviews[0])
print(vect.vocabulary_)
49/31:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vect = CountVectorizer(max_features=10000).fit(reviews[0])
#print(vect.vocabulary_)
49/32:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
49/33:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head)
49/34:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
49/35:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
#print(vect.vocabulary_)
49/36:
## NOT SURE IF IS NECESSARY FOR THE EXERCISE
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval_transformed = vect.transform(X_trainval[0]).toarray()
X_test_transformed = vect.transform(X_test[0]).toarray()
49/37:
## training a KNN algoritm on the train-val data using cross-validation
best_score = 0
best_num_neighbors = 0


for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval_transformed, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors
        
# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval_transformed, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval_transformed, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test_transformed, y_test)))
49/38:
## training a KNN algoritm on the train-val data using cross-validation
best_score = 0
best_num_neighbors = 0


for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval_transformed, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors
        
# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval_transformed, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval_transformed, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test_transformed, y_test)))
49/39:
## training a KNN algoritm on the train-val data using cross-validation
best_score = 0
best_num_neighbors = 0


for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval_transformed, y_trainval.ravel(), cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors
        
# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval_transformed, y_trainval.ravel())

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval_transformed, y_trainval.ravel())))
print("Score on test set: {}".format(knn.score(X_test_transformed, y_test.ravel())))
49/40:
## training a KNN algoritm on the train-val data using cross-validation
best_score = 0
best_num_neighbors = 0


for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval_transformed, y_trainval[0], cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors
        
# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval_transformed, y_trainval[0])

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval_transformed, y_trainval[0])))
print("Score on test set: {}".format(knn.score(X_test_transformed, y_trainval[0])))
49/41:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
49/42:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

print(type(reviews))
print(reviews.head)
49/43:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
49/44:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
#print(vect.vocabulary_)
49/45:
## NOT SURE IF IS NECESSARY FOR THE EXERCISE
## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval_transformed = vect.transform(X_trainval[0]).toarray()
X_test_transformed = vect.transform(X_test[0]).toarray()
49/46:
## training a KNN algoritm on the train-val data using cross-validation
best_score = 0
best_num_neighbors = 0


for num_neighbors in range(1,15):
    # Set a certain number of neighbors
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    
    # Perform cross validation
    scores = cross_val_score(knn, X_trainval_transformed, y_trainval, cv=5)
    
    # Compute the mean score
    score = scores.mean()
    
    # If improvement, store score and parameter
    if score>best_score:
        best_score = score
        best_num_neighbors = num_neighbors
        
# Build a model on the combine training and valiation data
knn = KNeighborsClassifier(n_neighbors=best_num_neighbors)
knn.fit(X_trainval_transformed, y_trainval)

print("Best number of neighbors found: {}".format(best_num_neighbors))
print("Best average score: {}".format(best_score))
print("Score on training/validation set: {}".format(knn.score(X_trainval_transformed, y_trainval)))
print("Score on test set: {}".format(knn.score(X_test_transformed, y_test)))
49/47:
##Representation of a word against the dataset
print(vect.vocabulary_)
49/48:
##Representation of a word against the dataset
print(vect.vocabulary_["high"])
49/49:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))
49/50:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(reviews[0][5]).toarray()
print(reviewA)
49/51:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(reviews[0][5]).toarray()
print(reviewA)
49/52:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform([reviews[0][5]]).toarray()
print(reviewA)
49/53:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform([reviews[0][5]]).toarray()
print("0 represents a word that is not part of the first 10000 most freqvent words; 1 the opossite ")
print(reviewA)
49/54:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform([reviews[0][5]]).toarray()
print("0 represents a word that is not part of the first 10000 most freqvent words; 1 the opossite ")
print(reviewA)
49/55:
## Necessary modifications such that we can use the input data for a  convolutional network

Y = to_categorical(Y,2)
X = reviews.reshape([-1, 8, 8, 1])
49/56:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed
49/57:
## Necessary modifications such that we can use the input data for a  convolutional network

Y = to_categorical(Y,2)
X = reviews.reshape([-1, 8, 8, 1])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y)
X_train[0].shape
60/1: print(vect.vocabulary_)
60/2:
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
60/3:
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
reviews
60/4:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vect = CountVectorizer(max_features=1000).fit(reviews[0])
60/5: print(vect.vocabulary_)
60/6: words = vect.vocabulary['840']
60/7: words = vect.vocabulary[840]
61/1:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed
61/2:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

print(type(reviews))
print(reviews.head)
61/3:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
61/4:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
#print(vect.vocabulary_)
61/5:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
print(vect.vocabulary_)
59/1:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

print(type(reviews))
print(reviews.head)
print(Y)
59/2:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed
59/3:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

print(type(reviews))
print(reviews.head)
print(Y)
59/4:

## make the initial train-val data and test data transformed as a bag-of-words representation 
X_trainval_transformed = vect.transform(X_trainval[0]).toarray()
X_test_transformed = vect.transform(X_test[0]).toarray()
59/5:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed
59/6:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
# print(reviews.head)
59/7:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
59/8:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/9:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/10:
## convert the values of the bag of words to be a ration [0,1]

def transform_to_ratio(word, review, label):
    return (' ' + w + ' ') in (' ' + s + ' ') && label == 1

ratio_dictionary = map(transform_to_ration, vect.vocabulary_.keys(), reviews[0], Y)

print(ratio_dictionary)
59/11:
## convert the values of the bag of words to be a ration [0,1]

def transform_to_ratio(word, review, label):
    return (' ' + w + ' ') in (' ' + s + ' ') & label == 1

ratio_dictionary = map(transform_to_ration, vect.vocabulary_.keys(), reviews[0], Y)

print(ratio_dictionary)
59/12:
## convert the values of the bag of words to be a ration [0,1]

def transform_to_ratio(word, review, label):
    return (' ' + w + ' ') in (' ' + s + ' ') & label == 1

ratio_dictionary = map(transform_to_ratio, vect.vocabulary_.keys(), reviews[0], Y)

print(ratio_dictionary)
59/13:
## convert the values of the bag of words to be a ration [0,1]

def transform_to_ratio(word, review, label):
    return (' ' + w + ' ') in (' ' + s + ' ') & label == 1

ratio_dictionary = map(transform_to_ratio, vect.vocabulary_.keys(), reviews[0], Y)

print(ratio_dictionary.values())
59/14:
## convert the values of the bag of words to be a ration [0,1]

def transform_to_ratio(word, review, label):
    return (' ' + w + ' ') in (' ' + s + ' ') & label == 1

ratio_dictionary = map(transform_to_ratio, vect.vocabulary_.keys(), reviews[0], Y)

print(ratio_dictionary.value())
59/15:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
print(vect.vocabulary_[0])
59/16:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/17: print(vect.vocabulary_[0])
59/18: print(vect.vocabulary_['high'])
59/19: print(vect.vocabulary_.entries[0])
59/20: print(vect.vocabulary_.items[0])
59/21: print(vect.vocabulary_.items[0])
59/22:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/23: print(vect.vocabulary_.items[0])
59/24: print(vect.vocabulary_.items())
59/25: print(vect.vocabulary_.items()[0])
59/26: print(vect.vocabulary_.items(0))
59/27: print(vect.vocabulary_.items[0]
59/28: print(list(vect.vocabulary_.items())[0])
59/29:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word, review) & label[i] == 1:
                ++count 
            ++i
        ratio = count/ word[1]
        word[1] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0], Y)

print(ratio_dictionary)
59/30:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word, review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count/ word[1]
        word[1] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0], Y)

print(ratio_dictionary)
59/31:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word, review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        word[1] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/32: print(list(vect.vocabulary_.items()))
59/33: print(list(vect.vocabulary_.items())[0])
59/34:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word, review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        word[1] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/35:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        word[1] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/36:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words[word[0]] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/37:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    
    i = 0
    for word in words:
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words[''+word[0]] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/38: print(list(vect.vocabulary_.items())[0][0])
59/39:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words:
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words[word[0]] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_.items(), reviews[0], Y)

print(ratio_dictionary)
59/40:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words:
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words[word[0]] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(list(vect.vocabulary_.items()), reviews[0], Y)

print(ratio_dictionary)
59/41:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words:
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        word[0] = ratio
    return words
            
            


ratio_dictionary = transform_to_ratio(list(vect.vocabulary_.items()), reviews[0], Y)

print(ratio_dictionary)
59/42:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0], Y)

print(ratio_dictionary)
59/43:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0], Y)

print(ratio_dictionary)
59/44:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) & labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0], Y)

print(ratio_dictionary)
59/45: vect.vocabulary_ = ratio_dictionary
59/46: print(vect.vocabulary_)
59/47:
word = high
occurences = 600
review = reviews[0][5]
label = y[0]

print(word + " " + ocurrences + " " + review+ " " + label)
59/48:
word = "high"
occurences = 600
review = reviews[0][5]
label = y[0]

print(word + " " + ocurrences + " " + review+ " " + label)
59/49:
word = "high"
occurences = 600
review = reviews[0][5]
label = Y[0]

print(word + " " + ocurrences + " " + review+ " " + label)
59/50:
word = "high"
occurences = 600
review = reviews[0][5]
label = Y[0]

print(word + " " + occurences + " " + review+ " " + label)
59/51:
word = "high"
occurences = "600"
review = reviews[0][5]
label = Y[0]

print(word + " " + occurences + " " + review+ " " + label)
59/52:
word = "high"
occurences = "600"
review = reviews[0][5]
label = Y[0]

print(word + " " + occurences + " " + review+ " " )
print(label)
59/53:
word = "high"
occurences = "600"
review = reviews[0][5]
label = Y[0]

print(word + " " + occurences + " " + review+ " " )
print(label)
59/54:
count = 0
if contains_word(word, review) & labels == 1:
                ++count 
        
print(count)
59/55:
count = 0
if contains_word(word, review) && labels == 1:
                ++count 
        
print(count)
59/56:
count = 0
if contains_word(word, review) & labels == 1:
                ++count 
        
print(count)
59/57:
count = 0
if contains_word(word, review) & labels == 1:
    ++count 
        
print(count)
59/58:
count = 0
if (contains_word(word, review) & labels == 1):
    ++count 
        
print(count)
59/59:
count = 0
if (contains_word(word, review) && labels == 1):
    ++count 
        
print(count)
59/60:
count = 0
if (contains_word(word, review) and labels == 1):
    ++count 
        
print(count)
59/61:
count = 0
if (contains_word(word, review) and labels == 1):
    ++count 
        
print(count)

reviews[0][0,10]
59/62:
# count = 0
# if (contains_word(word, review) and labels == 1):
#     ++count 
        
# print(count)

reviews[0][0...10]
59/63:
# count = 0
# if (contains_word(word, review) and labels == 1):
#     ++count 
        
# print(count)

reviews[0][0..10]
59/64:
# count = 0
# if (contains_word(word, review) and labels == 1):
#     ++count 
        
# print(count)

reviews[0][0.10]
59/65:
# count = 0
# if (contains_word(word, review) and labels == 1):
#     ++count 
        
# print(count)

reviews[0][:10]
59/66:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/67:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        ratio = count / word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/68:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        ratio = count / (int) word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/69:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        ratio = count / float(word[1])
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/70:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        print(count)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/71:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
        print(float(word[1]))
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/72:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
        print(word[1])
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/73:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
        print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/74:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/75:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
        print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/76:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/77:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/78:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/79:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
#         print(word)
        ratio = count / float(word[1])
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/80:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/81:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
#         print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/82:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/83:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                ++count 
            ++i
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/84:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                ++count 
            ++i
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/85:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/86:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                ++count 
            ++i
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/87:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/88:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                count += 1 
            i+= 1
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/89:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                count += 1 
            i+= 1
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/90:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/91:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                count += 1 
            i+= 1
        print(count)
#         print(word)
        ratio = count / float(word[1])
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/92:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/93:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                count += 1 
            i+= 1
        print(count)
#         print(word)
        ratio = count / word[1]
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:10], Y)

print(ratio_dictionary)
59/94:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                print('ji')
                count += 1 
            i+= 1
        print(count)
#         print(word)
        ratio = count / word[1]
#         print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

print(ratio_dictionary)
59/95:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/96:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

print(ratio_dictionary)
59/97:
word = "high"
occurences = "600"
review = "high high very high"
label = Y[0]

print(contains_word(word, occurences))
59/98:
word = "high"
occurences = "600"
review = "high high very high"
label = Y[0]

print(contains_word(word, review))
59/99:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/100:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = float(count / word[1])
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/101:
word = "high"
occurences = "600"
review = "high high very high"
label = Y[0]

print(1/2)
59/102:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        print(type(ratio))
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/103:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/104:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/105:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/106:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        print(ratio)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/107:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        if ratio > 1:
            print(word[0])
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/108:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        if ratio == 18.25:
            print(word[0])
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/109:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        if ratio == 18.25:
            print(word[0])
            print(word[1])
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/110:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/111:
## convert the values of the bag of words to be a ration [0,1]

def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        if ratio == 18.25:
            print(word[0])
            print(word[1])
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/112:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)
59/113:
## convert the values of the bag of words to be a ration [0,1]
vect = CountVectorizer(max_features=10000).fit(reviews[0])
# print(vect.vocabulary_)


def contains_word(w, r) :
    return (' ' + w + ' ') in (' ' + r + ' ')

def transform_to_ratio(words, reviews, labels):
    for word in words.items():
        i = 0
        count = 0
        for review in reviews:
            if contains_word(word[0], review) and labels[i] == 1:
                count += 1 
            i+= 1
        ratio = count / word[1]
        if ratio == 18.25:
            print(word[0])
            print(word[1])
            print(count)
        words.update({word[0]: ratio})
    return words
            
            


ratio_dictionary = transform_to_ratio(vect.vocabulary_, reviews[0][:1000], Y)

# print(ratio_dictionary)
59/114:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

for item in list(vect.vocabulary_.items())
    if item[1] == 1:
        print(item)
59/115:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

for item in vect.vocabulary_.items()
    if item[1] == 1:
        print(item)
59/116:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

for item in vect.vocabulary_
    if item[1] == 1:
        print(item)
59/117: print(list(vect.vocabulary_.items()))
59/118:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

for item in vect.vocabulary_.items()
    if item[1] == 1:
        print(item)
59/119:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

itemsArray =  vect.vocabulary_.items()
for item in itemsArray
    if item[1] == 1:
        print(item)
59/120:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

itemsArray =  vect.vocabulary_.items().tonArray()
for item in itemsArray
    if item[1] == 1:
        print(item)
59/121:
vect = CountVectorizer(max_features=10000).fit(reviews[0])

itemsArray =  vect.vocabulary_.items().nArray()
for item in itemsArray
    if item[1] == 1:
        print(item)
60/8:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vect = CountVectorizer(max_features=1000).fit(reviews[0])
60/9: print(vect.vocabulary_)
59/122:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vector)
59/123:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vector.toArray())
59/124:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vector.toarray())
59/125:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vector.toarray()[1])
59/126:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vector.toarray()[1].remove(0))
59/127:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(np.asarray(vector.toarray()[1]).remove(0))
59/128:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(list(filter((0).__ne__, vector.toarray()[1])))
59/129:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(list(filter(lambda a: a != 2, vector.toarray()[1])))
59/130:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(type(reviews[0]))
print(filter(lambda a: a != 2, vector.toarray()[1]))
59/131:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
x = vector.toarray()[1]
print(type(x))
print(list(filter(lambda a: a != 2, x)))
59/132:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
x = vector.toarray()[1]
print(type(x))
print(list(filter(lambda a: a != 0, x)))
59/133:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
x = vector.toarray()[1]
print(x)
print(type(x))
print(list(filter(lambda a: a != 0, x)))
59/134:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
x = vector.toarray()[1]
print(reviews[0])
print(type(x))
print(list(filter(lambda a: a != 0, x)))
59/135:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
x = vector.toarray()[1]
print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/1:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
p
# x = vector.toarray()[1]
print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/2:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(vector)
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/3:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/4:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
# print(reviews.head)
62/5:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
62/6:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(vector)
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/7:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(vector.head)
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/8:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(vector[0])
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/9:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/10:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(type(vector))
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/11:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/12:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
print(vector[0])
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/13:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
for item in vector:
    print(item)
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/14:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0]).toarray()
for item in vector[0]:
    print(item)
    
# x = vector.toarray()[1]
# print(reviews)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/15:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
# for item in vector[0]:
#     print(item)
    
# x = vector.toarray()[1]
print(vect.vocabulary_)
#print(type(x))
#print(list(filter(lambda a: a != 0, x)))
62/16:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/17:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/18:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/19:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/20:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/21:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/22:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/23:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/24:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
62/25:
vect = CountVectorizer(max_features=10000).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/26: print(vector)
62/27: print(vector.toarray())
62/28: print(vector)
62/29: print(vector.toarray())
62/30: print(vector[0])
62/31:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
# print(reviews.head)
62/32:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
print(reviews.head)
62/33:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
print(reviews[0])
62/34:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
print(reviews[0][0])
62/35:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform([reviews[0][5]]).toarray()
print("0 represents a word that is not part of the first 10000 most freqvent words; 1 the opossite ")
print(reviewA)
62/36: print(vector)
62/37: print(vector)
62/38: print(type(vector))
62/39: print(vector[0])
62/40: print(vector[0][0])
62/41: print(vector[0])
62/42: print(vector[0][0])
62/43: print(vector[0][0][0])
62/44: print(type(vector))
62/45:
print(type(vector))
print(vector)
62/46: review_vector_toarray = vector.ravel()
62/47: review_vector_toarray = np.squeeze(np.asarray(vector))
62/48:
review_vector_toarray = np.squeeze(np.asarray(vector))
print(review_vector_toarray)
62/49:
review_vector_toarray = np.squeeze(np.asarray(vector))
print(review_vector_toarray)
62/50:
review_vector_toarray = np.squeeze(np.asarray(vector))
print(review_vector_toarray[0])
62/51:
review_vector_toarray = np.squeeze(np.asarray(vector))
print(review_vector_toarray[0][0])
62/52:
review_vector_toarray = np.squeeze(np.asarray(vector))
print(review_vector_toarray)
62/53:
review_vector_toarray = np.array(vector).ravel()
print(review_vector_toarray)
62/54:
review_vector_toarray = vector.toarray()
print(review_vector_toarray)
63/1:
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
63/2:
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)

print(type(reviews))
print(reviews.head())
reviews
63/3: reviews[0][5]
63/4:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vect = CountVectorizer(max_features=1000).fit(reviews[0])
63/5:
X = vect.transform(reviews[0]).toarray()
Y = np.array((labels=='positive').astype(np.int_)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y)
63/6:
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#clf = KNeighborsClassifier()
#clf = GaussianNB()
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
63/7:
print("Accuracy on training data = {}".format(clf.score(X_train,y_train)))
print("Accuracy on test data = {}".format(clf.score(X_test,y_test)))
63/8:
bad_review = vect.transform(["This is the worst movie of all time!"]).toarray()
good_review = vect.transform(["Perfect movie!"]).toarray()
reviewA = vect.transform(["This is not a good movie, it is actually really bad!"]).toarray()
reviewB = vect.transform(["This is not a bad movie, it is actually really good!"]).toarray()
print(reviewA)
62/55:
vect = CountVectorizer(max_features=10000, stop_words = ['english']).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/56:
vect = CountVectorizer(max_features=10000, stop_words= {‘english’}).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/57:
vect = CountVectorizer(max_features=10000, stop_words= {'englis'’}).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/58:
vect = CountVectorizer(max_features=10000, stop_words= {'english'}).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/59:
vect = CountVectorizer(max_features=10000, stop_words= []'and']).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/60:
vect = CountVectorizer(max_features=10000, stop_words= ['and']).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/61:
vect = CountVectorizer(max_features=10000, stop_words= ['and']).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
62/62:
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))
66/1:
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))
66/2:
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
print(stopwords.words('english'))
66/3:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
66/4:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
print(reviews[0][0])
66/5:
X_trainval, X_test, y_trainval, y_test = train_test_split(reviews, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
66/6:
vect = CountVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit(reviews[0])
vector = vect.transform(reviews[0])
print(vect.vocabulary_)
66/7:
review_vector_toarray = vector.toarray()
print(review_vector_toarray)
70/1:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
70/2:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()

# print(type(reviews))
print(reviews[0][0])
70/3:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
 print(type(reviews))
print(reviews[0][0])
70/4:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
70/5:
#START

#Preproccesing 

#1. Punctuation removal
for review in reviews[0]
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/6:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
70/7:
#START

#Preproccesing 

#1. Punctuation removal
print(reviews[0])

for review in reviews[0]
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/8:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(reviews[0])
70/9:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0])
70/10:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/11:
#START

#Preproccesing 

#1. Punctuation removal

for review in reviews
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/12:
#START

#Preproccesing 

#1. Punctuation removal

for review in reviews[0]
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/13:
#START

#Preproccesing 

#1. Punctuation removal

reviewsSollumn = reviews[0]

for review in reviewsSollumn
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/14:
#START

#Preproccesing 

#1. Punctuation removal

for review in reviews[0]:
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/15:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for review in reviews[0]:
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/16:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for review in reviews[0]:
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/17:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for review in reviews[0]:
    print(type(review))
    review = review.translate(None, string.punctuation)
    
print(reviews.head)
70/18:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for review in reviews[0]:
    review = review.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/19:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/20:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/21:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/22:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
70/23:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/24:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit(reviews[0])
70/25:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit_transform(reviews[0])
vect.get_feature_names_out()
70/26:
#2. Tokenizing words

vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit_transform(reviews[0])
print(vect)
70/27:
#2. Tokenizing words

vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit_transform(reviews[0])
print(vect)
70/28:
# 3. Removal of stop_words
reviews_without_sw = [word for word in vect if not word in stopwords.words()]

print(reviews_without_sw)
70/29:
# 3. Removal of stop_words
reviews_without_sw = [word for word in vect if not word in stopwords.words('english')]

print(reviews_without_sw)
70/30:
# 3. Removal of stop_words
reviews_without_sw = [word for word in vect if not word in stopwords.words('english')]

print(reviews_without_sw)
70/31:
from nltk.tokenize import word_tokenize

text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)

print(text_tokens)
70/32:
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)

print(text_tokens)
70/33:
# 3. Removal of stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = word_tokenize(rev)

print(reviews[0])
70/34:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/35:
# 3. Removal of stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

reviews_processed = reviews[0]
for idx, rev in enumerate(reviews_processed):
    reviews_processed[0][idx] = word_tokenize(rev)

print(reviews_processed[0])
tokens_without_sw = [word for word in reviews_processed[0] if not word in stopwords.words('english')]

print(tokens_without_sw)
70/36:
# 3. Removal of stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

reviews_processed = reviews
for idx, rev in enumerate(reviews_processed):
    reviews_processed[0][idx] = word_tokenize(rev)

print(reviews_processed[0])
tokens_without_sw = [word for word in reviews_processed[0] if not word in stopwords.words('english')]

print(tokens_without_sw)
70/37:
# 3. Removal of stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

reviews_processed = reviews
for idx, rev in enumerate(reviews_processed[0]):
    reviews_processed[0][idx] = word_tokenize(rev)

print(reviews_processed[0])
tokens_without_sw = [word for word in reviews_processed[0] if not word in stopwords.words('english')]

print(tokens_without_sw)
70/38:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)

print(revies.head)
70/39:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)

print(revies.head)
70/40:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    print(rev.type())
    reviews[0][idx] = remove_stopwords(rev)

print(revies.head)
70/41:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    print(type(rev))
    reviews[0][idx] = remove_stopwords(rev)

print(revies.head)
70/42:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/43:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/44:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)

print(revies.head)
70/45:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/46: print(reviews.head)
70/47:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit_transform(reviews[0])
print(vect)
70/48:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english')).fit_transform(reviews[0])
print(vect)
70/49:
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = vect.decode(rev)
print(reviews.head)
70/50:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect)
70/51:
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = vect.decode(rev)
print(reviews.head)
70/52:
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = vect.build_tokenizer(rev)
print(reviews.head)
70/53:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
70/54:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/55:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/56:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/57: print(reviews.head)
70/58:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect)
70/59:
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = vect.build_tokenizer(rev)
print(reviews.head)
70/60:
for idx, rev in enumerate(reviews[0]):
     vect.build_tokenizer(rev)
print(reviews.head)
70/61:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews)
70/62:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews)
70/63:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews)
70/64: print(transformed_reviews[0])
70/65:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(type(transformed_reviews))
70/66: print(transformed_reviews[0][0])
70/67: print(transformed_reviews[0][1])
70/68: print(transformed_reviews[0][0][1])
70/69: print(transformed_reviews[0][0][0])
70/70: print(transformed_reviews[3])
70/71: print(transformed_reviews[28])
70/72:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/73:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/74:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/75:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
    
print(reviews.head)
70/76:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/77:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/78:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/79:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect)
70/80:
#2. Tokenizing words
vect = CountVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect)
70/81:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(type(transformed_reviews))
70/82:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews)
70/83:
#2. Tokenizing words
vect = CountVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect._dictionary)
70/84:
#2. Tokenizing words
vect = CountVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect._vocabulary)
70/85:
#2. Tokenizing words
vect = CountVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary)
70/86:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(transformed_reviews.vocalbulary)
70/87:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(transformed_reviews.vocabulary)
70/88:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(transformed_reviews._vocabulary)
70/89:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(transformed_reviews.vocabulary_)
70/90:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
70/91:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/92:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/93:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/94:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
70/95:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[0].toarray())
70/96:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/97:
#START

#Preproccesing 

#1. Punctuation removal

import string 

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/98:
# 3. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/99:
#2. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
# print(vect.vocabulary_)
70/100:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

# print(transformed_reviews[0].toarray())
70/101: # print(transformed_reviews)
70/102: # print(transformed_reviews)
70/103:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = vect.vocabulary_.items()[0]
print(pair[0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/104:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = vect.vocabulary_.items()
print(pair[0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/105:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = vect.vocabulary_.items()
print(pair)

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/106:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = vect.vocabulary_.items()
print(pair[0][0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/107:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = vect.vocabulary_.keys()
print(pair[0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/108:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = lis(vect.vocabulary_.keys())
print(pair[0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/109:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

pair = list(vect.vocabulary_.keys())
print(pair[0])

# for pair in enumerate(vect.vocabulary_.items()):
#      lmtizer.lemmatize(pair[0])
70/110:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
     lmtizer.lemmatize(word)
70/111:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
     lmtizer.lemmatize(word[0])
70/112:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
#      lmtizer.lemmatize(word)
    print(word)
70/113:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])
70/114:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer)
70/115:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('did'))
70/116:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('do'))
70/117:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('feet'))
70/118:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('planned'))
70/119:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('shps'))
70/120:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for word in enumerate(list(vect.vocabulary_.keys())):
      lmtizer.lemmatize(word[1])

print(lmtizer.lemmatize('ships'))
70/121:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for idx, rev in enumerate(reviews[0]):
    tokenized_rev = word_tokenize(rev)
    lemmatized_output = ' '.join([lmtizer.lemmatize(w) for w in tokenized_rev])
    reviews[0][idx] = lemmatized_output
70/122:
#4. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for idx, rev in enumerate(reviews[0]):
    tokenized_rev = word_tokenize(rev)
    lemmatized_output = ' '.join([lmtizer.lemmatize(w) for w in tokenized_rev])
    reviews[0][idx] = lemmatized_output
70/123: print(reviews.head)
70/124:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/125:
#START

#Preproccesing 

#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/126:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/127:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lmtizer = WordNetLemmatizer()

for idx, rev in enumerate(reviews[0]):
    tokenized_rev = word_tokenize(rev)
    lemmatized_output = ' '.join([lmtizer.lemmatize(w) for w in tokenized_rev])
    reviews[0][idx] = lemmatized_output
70/128:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
# print(vect.vocabulary_)
70/129: print(transformed_reviews)
70/130:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[0].toarray())
70/131:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
70/132:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/133:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/134:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/135:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:     
        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        reviews[0][idx]= " ".join(lemmatized_sentence)
70/136:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:     
        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        reviews[0][idx]= " ".join(lemmatized_sentence)
70/137:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:     
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
        reviews[0][idx]= " ".join(lemmatized_sentence)
70/138:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
        lemmatized_sentence.append(word)
        else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
        reviews[0][idx]= " ".join(lemmatized_sentence)
70/139:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
        reviews[0][idx]= " ".join(lemmatized_sentence)
70/140:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= lemmatized_sentence
70/141:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/142:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/143:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/144:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= lemmatized_sentence
70/145:


# for idx, rev in enumerate(reviews[0]):
#     tokenized_rev = word_tokenize(rev.lower())
#     lemmatized_output = ' '.join([lmtizer.lemmatize(w, pos= 'a') for w in tokenized_rev])
#     reviews[0][idx] = lemmatized_output
print(reviews.head)
70/146:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/147:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/148:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/149:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
70/150:


# for idx, rev in enumerate(reviews[0]):
#     tokenized_rev = word_tokenize(rev.lower())
#     lemmatized_output = ' '.join([lmtizer.lemmatize(w, pos= 'a') for w in tokenized_rev])
#     reviews[0][idx] = lemmatized_output
print(reviews.head)
70/151:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
70/152:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[0].toarray())
70/153:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for wordTfid[1] in vect.vocabulary_.values()])
70/154:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for wordTfidin vect.vocabulary_.values()])
70/155:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for wordTfid in vect.vocabulary_.values()])
70/156:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for wordTfid[0] in vect.vocabulary_.values()])
70/157:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for TfidValue[0] in vect.vocabulary_.values()])
70/158:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(transformed_reviews, columns = [wordTfid for TfidValue[1] in vect.vocabulary_.values()])
70/159:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame({reviews[0], labels}, columns = ["Review", "Label"])

dataframe
70/160:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame({list(reviews[0]), labels}, columns = ["Review", "Label"])

dataframe
70/161:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame({enumerate(reviews[0]), labels}, columns = ["Review", "Label"])

dataframe
70/162:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame({reviews[0].toarray(), labels}, columns = ["Review", "Label"])

dataframe
70/163:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(reviews[0].toarray(), columns = ["Review"])

dataframe
70/164:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(reviews[0], columns = ["Review"])

dataframe
70/165:
#Transform the dataset into a dataframe for feature extraction
dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])

dataframe
75/1:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
type(data)
75/2:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
type(data)
75/3:
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
75/4:
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
type(data)
70/166:
#Transform the dataset into a dataframe for feature extraction

dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])

dataFrame
70/167:
#Transform the dataset into a dataframe for feature extraction

dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])
dataFrame[labels] = labels

dataFrame
70/168:
#Transform the dataset into a dataframe for feature extraction

dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])
dataFrame['Label'] = labels

dataFrame
70/169:
#Transform the dataset into a dataframe for feature extraction

dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])
dataFrame['Label'] = Y

dataFrame
70/170:
#Transform the dataset into a dataframe for feature extraction

dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])
dataFrame['Label'] = Y

dataFrame
70/171: dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
70/172:
dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/173:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[0].toarray())
70/174:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(len(transformed_reviews[0].toarray()))
70/175:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[0].toarray())
70/176:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(len(transformed_reviews[0][0].toarray()))
70/177:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(len(transformed_reviews[0].toarray()[0]))
70/178:
#Transform the dataset into a dataframe for feature extraction

transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
print(transformed_reviews_as_arrays )

# dataFrame = pd.DataFrame(list(reviews[0]), columns = ["Review"])
# dataFrame['Label'] = Y

# dataFrame
70/179: print(transformed_reviews[0])
70/180: print(transformed_reviews[3])
70/181: print(transformed_reviews[100])
70/182: print(transformed_reviews[1000])
70/183: print(transformed_reviews[10000])
70/184: print(transformed_reviews[100000])
70/185: print(transformed_reviews[5000])
70/186: print(transformed_reviews[7000])
70/187: print(transformed_reviews)
70/188: print(type(transformed_reviews))
70/189: print(transformed_reviews)
70/190:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(transformed_reviews.tocoo().data, columns = transformed_reviews.tocoo().col)
# dataFrame['Label'] = Y

dataFrame
70/191:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(transformed_reviews.tocoo().data, columns = transformed_reviews.tocoo().col)
dataFrame['Label'] = Y

print(transformed_reviews.tocoo().data)
70/192:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(transformed_reviews.tocoo().data, columns = transformed_reviews.tocoo().col)
dataFrame['Label'] = Y

print(len(transformed_reviews.tocoo().data))
70/193:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

# dataFrame = pd.DataFrame(transformed_reviews.tocoo().data, columns = transformed_reviews.tocoo().col)
# dataFrame['Label'] = Y

print(len(transformed_reviews.tocoo().data))
70/194:
# for idx, rev in enumerate(reviews[0]):
#     reviews[0][idx] = vect.build_tokenizer(rev)
# print(reviews.head)

print(transformed_reviews[1].toarray()[0])
70/195:

# revs = reviews[0]
# for idx, rev in enumerate(revs):
#     revs[0][idx] = transformed_reviews[idx].toarray()[0]


print(transformed_reviews[1].toarray())
70/196:

# revs = reviews[0]
# for idx, rev in enumerate(revs):
#     revs[0][idx] = transformed_reviews[idx].toarray()[0]


print(len(transformed_reviews[1].toarray()))
70/197:

# revs = reviews[0]
# for idx, rev in enumerate(revs):
#     revs[0][idx] = transformed_reviews[idx].toarray()[0]


print(len(transformed_reviews[1].toarray()))
70/198:

revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[0][idx] = transformed_reviews[idx].toarray()[0]

print(revs[0])
# print(len(transformed_reviews[1].toarray()))
70/199:

revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(revs[0])
# print(len(transformed_reviews[1].toarray()))
70/200:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(revs, columns = transformed_reviews.tocoo().col)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/201:

revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(len(revs[0]))
# print(len(transformed_reviews[1].toarray()))
70/202:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(data=revs, columns = transformed_reviews.tocoo().col)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/203:

revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(len(revs))
# print(len(transformed_reviews[1].toarray()))
70/204:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

dataFrame = pd.DataFrame(data=revs, columns = list(vect.vocabulary_.values()))
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/205:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/206:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/207:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/208:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/209:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/210:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/211:
#Transform the dataset into a dataframe for feature extraction

# transformed_reviews_as_arrays = [review in review.toarray() in transformed_reviews]
# print(transformed_reviews_as_arrays )

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame.replace(np.nan,0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/212:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/213:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.replace(np.nan,0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/214:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/215:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.replace(np.nan,0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/216:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/217:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(1, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/218:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/219:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/220:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
70/221:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
70/222:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
70/223:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
70/224:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
70/225:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(len(revs))
# print(len(transformed_reviews[1].toarray()))
70/226:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 9999))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/227:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/228:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/229:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/230:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/231:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
70/232:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(type(revs))
70/233:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0].toarray()

print(type(revs))
70/234:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

print(type(revs[0]))
70/235:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/236:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

revs = revs.to_dict()
70/237:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/238:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs.values(), columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/239:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs.values(), index = revs.keys() columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
70/240:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs.values(), index = revs.keys(), columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
84/1:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
84/2:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
84/3:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
84/4:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
84/5:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/1:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/2:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
85/3:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/4:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/5:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/6:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
85/7:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]
85/8:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(data=revs, columns = cols)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
85/9:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
85/10:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]

revs = revs.toarray()
85/11:
revs = reviews[0]
for idx, rev in enumerate(revs):
    revs[idx] = transformed_reviews[idx].toarray()[0]
85/12:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

dataFrame = pd.DataFrame(revs)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
85/13:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
85/14:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
85/15:
#Transform the dataset into a dataframe for feature extraction

cols = list(range(0, 10000))

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
# dataFrame = dataFrame.fillna(0)
dataFrame['Label'] = Y

# print(len(transformed_reviews.tocoo().data))
85/16:
# dataFrame['tokenized_sents'] = dataFrame.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)
dataFrame
85/17:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/18: dataFrame
85/19:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy, Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
85/20:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy, Y, stratify=Y)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
85/21:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, stratify=Y)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
85/22:
##Representation of a word against the dataset
print( "Number of occurences: {}".format(vect.vocabulary_["high"]))

##Representation of a review
reviewA = vect.transform([reviews[0][5]]).toarray()
print("0 represents a word that is not part of the first 10000 most freqvent words; 1 the opossite ")
print(reviewA)
85/23:
# If you want reproducable results, uncomment the following 4 lines:
seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (3,3), activation='tanh', input_shape=X[0].shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/24:
# If you want reproducable results, uncomment the following 4 lines:
seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (3,3), activation='tanh', input_shape=dataFrame.to_numpy()[0].shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/25:
# If you want reproducable results, uncomment the following 4 lines:
seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (3,3), activation='tanh', input_shape=dataFrame.to_numpy().shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/26:
# If you want reproducable results, uncomment the following 4 lines:
seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (4,4), activation='tanh', input_shape=dataFrame.to_numpy().shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/27:
# If you want reproducable results, uncomment the following 4 lines:
seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (2,2), activation='tanh', input_shape=dataFrame.to_numpy().shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/28:

seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (2,2), activation='tanh', input_shape=None))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/29:

seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (3,3), activation='tanh', input_shape=None))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/30:

seed(0)
tf.random.set_seed(0)

# Now we define and train the neural network:
model = Sequential()

model.add(Conv2D(32, (3,3), activation='tanh', input_shape= dataFrame.to_numpy().shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # The convolutional and pooling layers output a 2D array. We need 1D for the final, output layer.
model.add(Dense(units=10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=50, verbose=1)

print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/31: Y = to_categorical(Y,num_classes=2)
85/32:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=0)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
85/33:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=2)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=500, verbose=1)

# print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/34:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=0)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))

print(dataFrame.to_numpy()[0])
85/35:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=500, verbose=1)

# print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/36: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/37:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=20, batch_size=1000, verbose=1)
85/38: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/39:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=10000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=500, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=20, batch_size=1000, verbose=1)
85/40:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=10000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=500, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=10, batch_size=500, verbose=1)
85/41:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=500, batch_size=100, verbose=1)
85/42:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)
85/43: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/44:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)
85/45: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/46:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)
85/47: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/48:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

model.add(Dense(units=5, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=4, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)
85/49: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/50:
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits = 10, random_state= 1, shuffle = true)
scores = cross_val_score(model, X_trainval, y_trainval, scoring= 'accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
85/51:
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits = 10, random_state= 1, shuffle = True)
scores = cross_val_score(model, X_trainval, y_trainval, scoring= 'accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
85/52:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.33, epochs=15, batch_size=100, verbose=1)
85/53:

seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:
model = Sequential()

# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)
85/54: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/55:
from sklearn.model_selection import StratifiedKFold
seed(1)
tf.random.set_seed(2)

# Now we define and train the neural network:


# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, val in kfold.split(X_trainval, y_trainval):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense(units=2, activation='softmax')) 

    sdg = optimizers.SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_trainval[test], y_trainval[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/56:
from sklearn.model_selection import StratifiedKFold
import numpy
seed = 7
numpy.random.seed(seed)

# Now we define and train the neural network:


# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, val in kfold.split(X_trainval, y_trainval):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense(units=2, activation='softmax')) 

    sdg = optimizers.SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_trainval[test], y_trainval[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/57:
from sklearn.model_selection import StratifiedKFold
import numpy
seed = 7
numpy.random.seed(seed)

# Now we define and train the neural network:


# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, val in kfold.split(X_trainval, y_trainval):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense(units=2, activation='softmax')) 

    sdg = optimizers.SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/58: Y = to_categorical(Y,num_classes=2)
85/59:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=0)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))

print(dataFrame.to_numpy()[0])
85/60:
from sklearn.model_selection import StratifiedKFold
import numpy
seed = 7
numpy.random.seed(seed)

# Now we define and train the neural network:


# model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
# model.add(Dense(units=50, activation='tanh'))
# model.add(Dense(units=2, activation='softmax')) 

# sdg = optimizers.SGD(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


# history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, val in kfold.split(X_trainval, y_trainval):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense(units=2, activation='softmax')) 

    sdg = optimizers.SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/61:
from sklearn.model_selection import StratifiedKFold
import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/62:
from sklearn.model_selection import StratifiedKFold
import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/63: # Student names and numbers:
85/64:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/65:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
85/66:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/67: print(reviews.head)
85/68:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/69: print(reviews.head)
85/70:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/71:


# for idx, rev in enumerate(reviews[0]):
#     tokenized_rev = word_tokenize(rev.lower())
#     lemmatized_output = ' '.join([lmtizer.lemmatize(w, pos= 'a') for w in tokenized_rev])
#     reviews[0][idx] = lemmatized_output
print(reviews.head)
85/72:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
85/73: print(transformed_reviews)
85/74:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/75: dataFrame
85/76: Y = to_categorical(Y,num_classes=2)
85/77:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=0)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))

print(dataFrame.to_numpy()[0])
85/78:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/79: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/80:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/81:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/82:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/83:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/84: print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/85: Y = to_categorical(Y,num_classes=2)
85/86:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
85/87:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/88:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/89:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/90: dataFrame
85/91: Y = to_categorical(Y,num_classes=2)
85/92:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/93:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/94: # Student names and numbers:
85/95:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/96:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])

print(type(reviews[0]))
85/97:
#START

#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/98: print(reviews.head)
85/99:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/100: print(reviews.head)
85/101:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/102:


# for idx, rev in enumerate(reviews[0]):
#     tokenized_rev = word_tokenize(rev.lower())
#     lemmatized_output = ' '.join([lmtizer.lemmatize(w, pos= 'a') for w in tokenized_rev])
#     reviews[0][idx] = lemmatized_output
print(reviews.head)
85/103:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
print(vect.vocabulary_)
85/104: print(transformed_reviews)
85/105:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/106: dataFrame
85/107: Y = to_categorical(Y,num_classes=2)
85/108:
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/109:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/110: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/111: print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/112:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/113:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/114:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/115:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/116:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=1000, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adam(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/117:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/118: print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
85/119: print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/120: # Student names and numbers:
85/121:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/122:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/123: # print(type(reviews[0]))
85/124:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/125: # print(reviews.head)
85/126:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/127: # print(reviews.head)
85/128:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/129:

# print(reviews.head)
85/130:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/131: # print(vect.vocabulary_)
85/132: # print(transformed_reviews)
85/133:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/134: # dataFrame
85/135:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/136:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/137:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/138:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/139:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/140:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/141:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/142:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

prediction = model.predict(review1)
print(prediction)
85/143:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.fit_transform(review1)

print(review1)
# prediction = model.predict(review1)
# print(prediction)
85/144:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


# review1 = vect.fit_transform(review1)

print(review1)
# prediction = model.predict(review1)
# print(prediction)
85/145:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform(review1)

print(review1)
# prediction = model.predict(review1)
# print(prediction)
85/146:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


# review1 = vect.transform(review1)

print(review1)
# prediction = model.predict(review1)
# print(prediction)
85/147:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])

print(review1)
# prediction = model.predict(review1)
# print(prediction)
85/148:
review1 = "I hated this movie so bad! Technology in this university sucks ass! Very very confusing, the teachers were bad, really disappointed."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

print(review1)
prediction = model.predict(review1)
print(prediction)
85/149:
review1 = "I loved this movie. It was utterly perfect. Nothing bad can be said about Momæs spaghetti."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/150:
review1 = "I loved this movie. It was utterly perfect. Nothing bad can be said about Moms spaghetti."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/151:
review1 = "I loved this movie. It was utterly perfect. Nothing bad can be said about Moms spaghetti. Truthfully amazing i am out of words to describe the perfection "

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/152:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/153:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/154:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=15, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=300, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/155:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=39, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=300, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/156:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=300, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/157:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=500, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/158:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=50, batch_size=10, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/159:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=10, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/160:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/161:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/162: # Student names and numbers:
85/163:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/164:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/165: # print(type(reviews[0]))
85/166:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/167: # print(reviews.head)
85/168:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/169: # print(reviews.head)
85/170:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/171:

# print(reviews.head)
85/172:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/173: # print(vect.vocabulary_)
85/174: # print(transformed_reviews)
85/175:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/176: # dataFrame
85/177:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/178:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=10, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/179:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/180:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/181:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/182:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/183:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/184:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/185:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/186:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/187: # Student names and numbers:
85/188:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/189:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/190: # print(type(reviews[0]))
85/191:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/192: # print(reviews.head)
85/193:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/194: # print(reviews.head)
85/195:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/196:

# print(reviews.head)
85/197:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/198: # print(vect.vocabulary_)
85/199: # print(transformed_reviews)
85/200:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/201: # dataFrame
85/202:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/203:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/204:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/205:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/206:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/207:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/208:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/209:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/210:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/211:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/212: # Student names and numbers:
85/213:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/214:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/215: # print(type(reviews[0]))
85/216:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/217: # print(reviews.head)
85/218:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/219: # print(reviews.head)
85/220:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/221:

# print(reviews.head)
85/222:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/223: # print(vect.vocabulary_)
85/224: # print(transformed_reviews)
85/225:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/226: # dataFrame
85/227:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/228:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/229:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/230:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/231:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/232:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/233:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/234:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/235: # Student names and numbers:
85/236:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/237:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/238: # print(type(reviews[0]))
85/239:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/240: # print(reviews.head)
85/241:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/242: # print(reviews.head)
85/243:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/244:

# print(reviews.head)
85/245:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/246: # print(vect.vocabulary_)
85/247: # print(transformed_reviews)
85/248:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/249: # dataFrame
85/250:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/251:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/252:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/253:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/254:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/255:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/256:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/257:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/258: # Student names and numbers:
85/259:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/260:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/261: # print(type(reviews[0]))
85/262:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/263: # print(reviews.head)
85/264:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/265: # print(reviews.head)
85/266:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/267:

# print(reviews.head)
85/268:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/269: # print(vect.vocabulary_)
85/270: # print(transformed_reviews)
85/271:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/272: # dataFrame
85/273:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/274:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=32, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/275:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/276:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/277:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/278:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/279:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/280:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/281: # Student names and numbers:
85/282:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/283:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/284: # print(type(reviews[0]))
85/285:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/286: # print(reviews.head)
85/287:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/288: # print(reviews.head)
85/289:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/290:

# print(reviews.head)
85/291:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/292: # print(vect.vocabulary_)
85/293: # print(transformed_reviews)
85/294:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/295: # dataFrame
85/296:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/297:
# from sklearn.model_selection import StratifiedKFold
# import numpy
# seed = 7
# numpy.random.seed(seed)

#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=15, batch_size=100, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []

# for train, val in kfold.split(X_trainval, y_trainval):
#     model = Sequential()
#     model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
#     model.add(Dense(units=10, activation='tanh'))
#     model.add(Dense(units=2, activation='softmax')) 

#     sdg = optimizers.SGD(learning_rate=0.1)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_trainval[train], y_trainval[train], epochs=150, batch_size=10, verbose=0)
#     scores = model.evaluate(X_trainval[val], y_trainval[val], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
85/298:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/299:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/300:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/301:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/302:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/303:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
85/304:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/305:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/306:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/307:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/308:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/309:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the beest stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/310:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/311: # Student names and numbers:
85/312:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
85/313:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
85/314: # print(type(reviews[0]))
85/315:
#Preproccesing 
#1. Punctuation removal

import string 
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
85/316: # print(reviews.head)
85/317:
# 2. Removal of stop_words
from gensim.parsing.preprocessing import remove_stopwords

for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
85/318: # print(reviews.head)
85/319:
#3. Lemmatisation of reviews 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lmtizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
85/320:

# print(reviews.head)
85/321:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
85/322: # print(vect.vocabulary_)
85/323: # print(transformed_reviews)
85/324:
#Transform the dataset into a dataframe for feature extraction

import scipy.sparse
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
85/325: # dataFrame
85/326:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
print("AAAAAA {}".format(y_test.shape[0]))
85/327:
#Now we define and train the neural network:

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) # Each new line is a new layer! The first layer is called the input-layer
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


history = model.fit(X_trainval, y_trainval,validation_split=0.2, epochs=25, batch_size=100, verbose=1)
85/328:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
85/329:
# review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review1 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."

review1 = review1.translate(str.maketrans('', '', string.punctuation))
review1 = remove_stopwords(review1)

pos_tagged = nltk.pos_tag(nltk.word_tokenize(review1)) 
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
lemmatized_sentence = []
for word, tag in wordnet_tagged:  
    if tag is None:
         lemmatized_sentence.append(word)
    else:  
        lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
review1= " ".join(lemmatized_sentence)


review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

prediction = model.predict(review1)
print(prediction)
85/330:
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
85/331:
import matplotlib.pyplot as plt
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
85/332:
import numpy as np

digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
85/333:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
87/1:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
87/2:
# Import all necessary libraries here
import mglearn
%matplotlib inline
import sys
import numpy as np
np.set_printoptions(threshold= sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy as scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers

import tensorflow as tf
from numpy.random import seed

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
87/3:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
print(type(reviews))
print(reviews[0][0])
87/4:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
87/5:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
87/6:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev)
87/7:
#The result of the removal of stop words can be seen by printing the first few reviews 
print(reviews.head)
87/8:
# 2. Punctuation removal
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
87/9:
# The result of the removal of punctuation can be seen by printing the first few reviews 
print(reviews.head)
87/10:
#3. Lemmatisation of reviews 
lmtizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
87/11:
# The result of the lemmatisation process on the first review is the following
 print(reviews.head)
87/12:
# The result of the lemmatisation process on the first review is the following
print(reviews.head)
87/13:
# The result of the lemmatisation process on the first review is the following
print(reviews[0][0])
87/14:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
87/15:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
87/16:
#The result of the removal of stop words can be seen by printing the first few reviews 
print(reviews.head)
87/17:
# 2. Punctuation removal
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
87/18:
# The result of the removal of punctuation can be seen by printing the first few reviews 
print(reviews.head)
87/19:
#3. Lemmatisation of reviews 
lmtizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
87/20:
# The result of the lemmatisation process on the first review is the following
print(reviews[0][0])
87/21:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
87/22: print(vect.vocabulary_)
87/23: print(transformed_reviews)
87/24: print(transformed_reviews)
87/25: print(vect.vocabulary_)
87/26: print(vect.vocabulary_)
87/27:
#For visualization purposes, the dataset (the tranformed reviews) can be plotted in a data frame.
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
87/28: dataFrame
87/29:
#For visualization purposes, the dataset (the tranformed reviews) can be plotted in a data frame.
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
dataFrame
87/30:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
87/31:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
87/32:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on training data: {}".format(model.evaluate(X_test, y_test)))
87/33:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=200, verbose=1)
87/34:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=300, verbose=1)
87/35:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=400, verbose=1)
87/36:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
87/37:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.1, epochs=15, batch_size=100, verbose=1)
87/38:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
87/39:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on testing data: {}".format(model.evaluate(X_test, y_test)))
87/40:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review1, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print(prediction1)
print(prediction2)
87/41:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print(prediction1)
print(prediction2)
87/42:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print(prediction1[0][0])
print(prediction2)
87/43:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print("Review1 has {} chances to be a negative review and {} chances to be positive".format(prediction1[0][0], prediction1[0][1]))
print("Review2 has {} chances to be a negative review and {} chances to be positive".format(prediction2[0][0], prediction2[0][1]))
87/44:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
87/45:
index = 7

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
87/46: (x_train, y_train), (x_test, y_test) = mnist.load_data()
87/47:
index = 7

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
87/48:
index = 10

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
87/49:
index = 11

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
87/50:
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[2],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[2])
87/51:
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[3],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[3])
87/52:
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[7],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[7])
87/53:
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[10],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[10])
87/54:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
88/1: # Student names and numbers:
88/2:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
88/3:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
88/4:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
88/5:
#The result of the removal of stop words can be seen by printing the first few reviews 
print(reviews.head)
88/6:
# 2. Punctuation removal
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
88/7:
# The result of the removal of punctuation can be seen by printing the first few reviews 
print(reviews.head)
88/8:
#3. Lemmatisation of reviews 
lmtizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
88/9:
# The result of the lemmatisation process on the first review is the following
print(reviews[0][0])
88/10:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
88/11: print(transformed_reviews)
88/12:
#For visualization purposes, the dataset (the tranformed reviews) can be plotted in a data frame.
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
dataFrame
88/13:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
88/14:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
88/15:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
88/16:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
89/1: # Student names and numbers:
89/2:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
89/3:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
89/4:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
89/5:
#The result of the removal of stop words can be seen by printing the first few reviews 
print(reviews.head)
89/6:
# 2. Punctuation removal
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
89/7:
# The result of the removal of punctuation can be seen by printing the first few reviews 
print(reviews.head)
89/8:
#3. Lemmatisation of reviews 
lmtizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
89/9:
# The result of the lemmatisation process on the first review is the following
print(reviews[0][0])
89/10:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
89/11: print(transformed_reviews)
89/12:
#For visualization purposes, the dataset (the tranformed reviews) can be plotted in a data frame.
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
dataFrame
89/13:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
89/14:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
89/15:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
89/16:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on testing data: {}".format(model.evaluate(X_test, y_test)))
89/17:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print("Review1 has {} chances to be a negative review and {} chances to be positive".format(prediction1[0][0], prediction1[0][1]))
print("Review2 has {} chances to be a negative review and {} chances to be positive".format(prediction2[0][0], prediction2[0][1]))
89/18: (x_train, y_train), (x_test, y_test) = mnist.load_data()
89/19:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
89/20:
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
89/21:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
89/22:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
89/23:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
89/24: (x_train, y_train), (x_test, y_test) = mnist.load_data()
89/25:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
89/26:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
89/27:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
89/28:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
89/29:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
89/30:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=60000)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_train, y_train, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
89/31:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=60000)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
89/32:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=[None, 28, 28])) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
89/33:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=60000)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
89/34:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 

x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

x_1D_bin_train10 = x_1D_bin_train10.flatten()
y_1D_bin_train10 = y_1D_bin_train10.flatten()
89/35:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 

x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

x_1D_bin_train10 = x_1D_bin_train10.flatten()
y_1D_bin_train10 = y_1D_bin_train10.flatten()
89/36:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 

x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

x_1D_bin_train10 = x_1D_bin_train10.flatten()
y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/1:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/2: (x_train, y_train), (x_test, y_test) = mnist.load_data()
90/3:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
90/4: (x_train, y_train), (x_test, y_test) = mnist.load_data()
90/5:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
90/6:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/7:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/8:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
90/9:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/10: # Student names and numbers:
90/11:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
90/12:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
90/13:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
90/14: (x_train, y_train), (x_test, y_test) = mnist.load_data()
90/15:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
90/16:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/17:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/18:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
90/19:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_bin_train10] 
y_1D_bin_train10 =[lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/20: type(x_1D_bin_train10)
90/21: type(x_1D_bin_train10[0])
90/22: type(x_1D_bin_train10.shape)
90/23: x_1D_bin_train10.shape
90/24: x_1D_bin_train10[0].shape
90/25:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays
# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 

type(x_1D_bin_train10) 
# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/26:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
90/27:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays
# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 

type(x_1D_bin_train10) 
# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/28:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays
# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 

type(x_1D_bin_train10[0]) 
# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/29: (x_train, y_train), (x_test, y_test) = mnist.load_data()
90/30:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
90/31:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/32:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/33:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
90/34:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays
# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 

type(x_1D_bin_train10[0]) 
# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/35:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays
# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 

type(x_bin_train10[0]) 
# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/36:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])

print(x_bin_test10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/37:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    x_bin_train10[idx] = image.reshape(len(image), -1)

print(x_bin_test10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/38:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    x_bin_train10[idx] = image.reshape(len(image), -1)

print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/39:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/40:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    x_bin_train10[idx] = image.reshape(len(image), -2)

print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/41:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/42:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    x_bin_train10[idx] = image.flatten()

print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/43:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/44:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    x_bin_train10[idx] = image.reshape(len(image), -1)

print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/45:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    x_bin_train10[idx] = image.reshape(len(image), -1)


print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/46:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/47:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    x_bin_train10[idx] = image.reshape(len(image), -1)


print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/48:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/49:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    x_bin_train10[idx] = image.flatten()


print(x_bin_train10[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/50:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
90/51:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten[idx] = image


print(x_bin_train10_flatten[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/52:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


print(x_bin_train10_flatten[0])

# converToGraySclae = [lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in list] 


# y_1D_bin_train10 =[lambda x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in y_bin_train10] 



# x_1D_bin_train10 = [lambda x: np.array(x) for x in x_1D_bin_train10]
# y_1D_bin_train10 = [lambda x: np.array(x) for x in y_1D_bin_train10]

# x_1D_bin_train10 = x_1D_bin_train10.flatten()
# y_1D_bin_train10 = y_1D_bin_train10.flatten()
90/53:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    # x_bin_train10[idx] = x_bin_train10[idx].flatten().reshape(x_bin_train10[idx].shape[0], x_bin_train10[idx].shape[1], x_bin_train10[idx].shape[2])
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)

print(y_bin_test10[0])
90/54:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)
90/55:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=60000)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/1:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
91/2:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
91/3:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
91/4: (x_train, y_train), (x_test, y_test) = mnist.load_data()
91/5:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
91/6:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
91/7:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)
91/8:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=60000)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/9:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/10:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/11:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train10_flatten = numpy.array(x_bin_train10_flatten)

print(type(y_bin_train10))

x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)
91/12:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train10_flatten = np.array(x_bin_train10_flatten)

print(type(y_bin_train10))

x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)
91/13:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/14:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train10_flatten = np.array(x_bin_train10_flatten)

print(x_bin_train10_flatten)

x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)
91/15:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train10_flatten = np.array(x_bin_train10_flatten)

print(x_bin_train10_flatten[0])

x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)
91/16: print(dataFrame.to_numpy()[0])
91/17: # Student names and numbers:
91/18:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
91/19:
reviews = pd.read_csv('IMDB_dataset/reviews.txt', header=None)
labels = pd.read_csv('IMDB_dataset/labels.txt', header=None)
Y = np.array((labels=='positive').astype(np.int_)).ravel()
# print(type(reviews))
# print(reviews[0][0])
91/20:
# 1. Removal of stop_words
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = remove_stopwords(rev, stopwords.words('english') )
91/21:
#The result of the removal of stop words can be seen by printing the first few reviews 
print(reviews.head)
91/22:
# 2. Punctuation removal
for idx, rev in enumerate(reviews[0]):
    reviews[0][idx] = rev.translate(str.maketrans('', '', string.punctuation))
91/23:
# The result of the removal of punctuation can be seen by printing the first few reviews 
print(reviews.head)
91/24:
#3. Lemmatisation of reviews 
lmtizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
 
for idx, rev in enumerate(reviews[0]):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(rev)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    reviews[0][idx]= " ".join(lemmatized_sentence)
91/25:
# The result of the lemmatisation process on the first review is the following
print(reviews[0][0])
91/26:
#4. Tokenizing words
vect = TfidfVectorizer(max_features=10000, stop_words= stopwords.words('english'))
transformed_reviews = vect.fit_transform(reviews[0])
91/27: print(transformed_reviews)
91/28:
#For visualization purposes, the dataset (the tranformed reviews) can be plotted in a data frame.
dataFrame = pd.DataFrame.sparse.from_spmatrix(transformed_reviews)
dataFrame
91/29:
Y = to_categorical(Y,num_classes=2)
X_trainval, X_test, y_trainval, y_test = train_test_split(dataFrame.to_numpy(), Y, random_state=42)
print("Size of training-val set:{}".format(X_trainval.shape[0]))
print("Size of test set:{}".format(X_test.shape[0]))
91/30: print(dataFrame.to_numpy()[0])
91/31:
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=10000)) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = model.fit(X_trainval, y_trainval, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/32:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
91/33:
print("Accuracy on training data: {}".format(model.evaluate(X_trainval, y_trainval)))
print("Accuracy on testing data: {}".format(model.evaluate(X_test, y_test)))
91/34:
review1 = "There were two interesting look back reviews that came out in 2015 that are worth checking out - just search for <<review terminator sarah connor chronicles>> and check out the retrospectives in The Guardian and on IGN. The IGN retrospective is especially valuable due to the inclusion of video cast interviews that don't show up on YouTube - there are a lot of insights as to how the principle actors viewed their characters and where they hoped the plot and reveals would go. Ultimately the writer Josh Friedman did give us fans what we wanted, just not in the definitive manner the actors hoped for their characters. I think it is better that way - showing us and let us draw our own conclusions, instead of laying it out in black and white. But, yeah, for you Jameron shippers, you were right, the seeds were there all along and the final episode is a real heartbreaker."
review2 = "One of the best stand alone TV Sci-Fi shows ever, joining Firefly on the list of <<why ever did they cancel that >>? Perhaps the best answer is that it is a miracle it was made in the first place, given that cancer survivor Josh Friedman produced a death obsessed, stylish, inventive and often very funny TV show with more depth and sophistication than the dying film franchise. A Hollywood insider needs to write the real back story to this show: how it suited Warner Brothers to have what was, in effect, an extended commercial on TV for the upcoming reboot of the movie franchise, what transpired between Warner and Fox, and why Fox started to pull the plug on the cash. There are episodes in season two that were clearly made with everyone concerned expecting to be fired by the end of the day. The production values dropped as the budget dried up, but the excellent acting, production and cinematography did not. An unfortunate casualty of the writers' strike, the show lost momentum after its truncated first season. Neflix, reboot it, please...."


def transform_review(review):
    review = remove_stopwords(review, stopwords.words('english'))
    review = review.translate(str.maketrans('', '', string.punctuation))
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(review)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:  
        if tag is None:
            lemmatized_sentence.append(word)
        else:  
            lemmatized_sentence.append(lmtizer.lemmatize(word, tag))
    review = " ".join(lemmatized_sentence)
    return review

review1 = transform_review(review1)
review1 = vect.transform([review1])
review1 = pd.DataFrame.sparse.from_spmatrix(review1).to_numpy()

review2 = transform_review(review2)
review2 = vect.transform([review2])
review2 = pd.DataFrame.sparse.from_spmatrix(review2).to_numpy()

prediction1 = model.predict(review1)
prediction2 = model.predict(review2)
print("Review1 has {} chances to be a negative review and {} chances to be positive".format(prediction1[0][0], prediction1[0][1]))
print("Review2 has {} chances to be a negative review and {} chances to be positive".format(prediction2[0][0], prediction2[0][1]))
91/35: (x_train, y_train), (x_test, y_test) = mnist.load_data()
91/36:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
91/37:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
91/38:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
91/39:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
91/40:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train10_flatten = np.array(x_bin_train10_flatten)

print(x_bin_train10_flatten[0])

x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)
91/41:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
91/42:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)
91/43:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
92/1:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
#connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
93/1: # Student names and numbers:
93/2:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
93/3: (x_train, y_train), (x_test, y_test) = mnist.load_data()
93/4:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
93/5: (x_train, y_train), (x_test, y_test) = mnist.load_data()
93/6:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
93/7:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
93/8:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)
93/9:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
#connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
93/10:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)

y_bin_train10 = to_categorical(y_bin_train10,num_classes=2)
y_bin_train69 = to_categorical(y_bin_train69,num_classes=2)

y_bin_test10 = to_categorical(y_bin_test10,num_classes=2)
y_bin_test69 = to_categorical(y_bin_test69,num_classes=2)
93/11:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)


x_bin_train69_flatten = np.array(x_bin_train69_flatten)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/12:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = np.numarray(x_bin_train10_flatten).asType('float32') /255
x_bin_train69_flatten = np.numarray(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/13:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), -1)
    image =  image.flatten()
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = x_bin_train10_flatten.asType('float32') /255
x_bin_train69_flatten = np.numarray(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/14:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(len(image), [-1, 784])
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(len(image), [-1, 784])
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = np.array(x_bin_train10_flatten).asType('float32') /255
x_bin_train69_flatten = np.array(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/15:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(image, [-1, 784])
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(image, [-1, 784])
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = np.array(x_bin_train10_flatten).asType('float32') /255
x_bin_train69_flatten = np.array(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/16:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
93/17:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
93/18:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = image.reshape(image, [-1, 784])
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = image.reshape(image, [-1, 784])
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = np.array(x_bin_train10_flatten).asType('float32') /255
x_bin_train69_flatten = np.array(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/19:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = np.reshape(image, [-1, 784])
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = np.reshape(image, [-1, 784])
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = np.array(x_bin_train10_flatten).asType('float32') /255
x_bin_train69_flatten = np.array(x_bin_train69_flatten).asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/20:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

x_bin_train10_flatten = []
for idx, image in enumerate(x_bin_train10):
    image = np.reshape(image, [-1, 784])
    x_bin_train10_flatten.append(image)


x_bin_train69_flatten = []
for idx, image in enumerate(x_bin_train69):
    image = np.reshape(image, [-1, 784])
    x_bin_train69_flatten.append(image)

x_bin_train10_flatten = x_bin_train10_flatten.asType('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/21:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

# x_bin_train10_flatten = []
# for idx, image in enumerate(x_bin_train10):
#     image = np.reshape(image, [-1, 784])
#     x_bin_train10_flatten.append(image)


# x_bin_train69_flatten = 
# for idx, image in enumerate(x_bin_train69):
#     image = np.reshape(image, [-1, 784])
#     x_bin_train69_flatten.append(image)


x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.asType('float32') /255
x_bin_train69 = x_bin_train69.asType('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/22:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

#converting the data in one dimensional arrays

# x_bin_train10_flatten = []
# for idx, image in enumerate(x_bin_train10):
#     image = np.reshape(image, [-1, 784])
#     x_bin_train10_flatten.append(image)


# x_bin_train69_flatten = 
# for idx, image in enumerate(x_bin_train69):
#     image = np.reshape(image, [-1, 784])
#     x_bin_train69_flatten.append(image)


x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/23:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
#connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
93/24:
#4 layer fully connected

connected1 = Sequential()

connected1.add(Dense(units=784, activation='relu', input_dim=784)) 
connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
93/25:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
93/26:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
93/27:
# 4 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])
history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
93/28:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/1:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
95/2: (x_train, y_train), (x_test, y_test) = mnist.load_data()
95/3:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
95/4:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
95/5:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
95/6:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)
95/7:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/8:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/9:
connected2 = Sequential()

connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=49, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/10:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/11:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/12:
connected3 = Sequential()

connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=49, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
95/13:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
95/14:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
95/15:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
95/16:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu', input_dim=784)) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history3 = connected3.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history3)
96/1:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
96/2: (x_train, y_train), (x_test, y_test) = mnist.load_data()
96/3:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
96/4:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
96/5:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10 = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69 = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
96/6:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
96/7:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
96/8:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history2 = connected2.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history2)
96/9:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu', input_dim=784)) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history3 = connected3.fit(x_bin_train10, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history3)
96/10:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)
96/11: (x_train, y_train), (x_test, y_test) = mnist.load_data()
96/12:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
96/13:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
96/14:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10.astype('float32') /255
x_bin_train69_flatten = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
96/15:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
96/16:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
   1:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
   2: (x_train, y_train), (x_test, y_test) = mnist.load_data()
   3:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
   4:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
   5:
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10.astype('float32') /255
x_bin_train69_flatten = x_bin_train69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
   6:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
   7:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
   8:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
   9:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
  10:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  11:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  12:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  13:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  14:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history1)
  15:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history2 = connected2.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history2)
  16:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu', input_dim=784)) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

history3 = connected3.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)

showStats(history3)
  17:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_test10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  18:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (4,4), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (4,4), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_test10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  19:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (4,4), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (4,4), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_test10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  20:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  21:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[None, 28, 28]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  22:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[28, 28]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  23:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[0, 28, 28]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  24:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[1, 28, 28]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  25:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[28, 28, 1]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = model.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  26:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[28, 28, 1]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_train, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  27:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[28, 28, 1]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  28:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  29:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  30:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=[28, 28, 1]))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  31:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=x_bin_train10[0].shape))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  32:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10, 2)
y_bin_train69 = to_categorical(y_bin_train69, 2)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  33:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10, 2)
y_bin_train69 = to_categorical(y_bin_train69, 2)

y_bin_test10 = to_categorical(y_bin_test10, 2)
y_bin_test69 = to_categorical(y_bin_test69, 2)
  34:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=(28,28, 1)))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  35:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  36:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  37:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10, 2)
y_bin_train69 = to_categorical(y_bin_train69, 2)

y_bin_test10 = to_categorical(y_bin_test10, 2)
y_bin_test69 = to_categorical(y_bin_test69, 2)
  38:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  39:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Conv2D(32, (3,3), activation='tanh', input_shape=(28,28, 1)))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  40:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
  41:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28, 1)))
conv1.add(Conv2D(32, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, (3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  42:
# 3 layer convnet

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28, 1)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  43:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28, 1)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  44:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  45:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  46:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  47:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28, 1)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  48: (x_train, y_train), (x_test, y_test) = mnist.load_data()
  49:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  50:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  51:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  52:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

x_bin_train10 = np.expand_dims(x_bin_train10, -1)
x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  53:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28, 1)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  54:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

print(x_bin_train69.shape)

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  55:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

print(x_bin_train69[0].shape)

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  56:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  57: (x_train, y_train), (x_test, y_test) = mnist.load_data()
  58:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  59:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  60:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  61:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  62:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  63:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(28, 28)))
conv1.add(Conv2D(32, kernel_size=(2,2), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(2,2), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  64:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(1, 28, 28)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  65:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape=(None, 28, 28)))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  66: (x_train, y_train), (x_test, y_test) = mnist.load_data()
  67:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  68:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  69:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  70:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_test10_flatten = np.reshape(x_bin_test10, [-1, 784])
x_bin_test69_flatten = np.reshape(x_bin_test69, [-1, 784])

x_bin_test10_flatten = x_bin_test10_flatten.astype('float32') /255
x_bin_test69_flatten = x_bin_test69_flatten.astype('float32') /255

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  71:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  72:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  73:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10.shape))
conv1.add(Conv2D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2)))

conv1.add(Conv2D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling2D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  74:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
  75: (x_train, y_train), (x_test, y_test) = mnist.load_data()
  76:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  77:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
  78:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
  79:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_test10_flatten = np.reshape(x_bin_test10, [-1, 784])
x_bin_test69_flatten = np.reshape(x_bin_test69, [-1, 784])

x_bin_test10_flatten = x_bin_test10_flatten.astype('float32') /255
x_bin_test69_flatten = x_bin_test69_flatten.astype('float32') /255

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

# x_bin_train10 = np.expand_dims(x_bin_train10, -1)
# x_bin_train69 = np.expand_dims(x_bin_train69, -1)

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
  80:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
  81:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2)))

conv1.add(Conv1D(16, kernel_size=(3,3), activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  82:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2)))

conv1.add(Conv1D(16, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  83:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D,MaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
  84:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2)))

conv1.add(Conv1D(16, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=(2,2), padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  85:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  86:
# 3 layer convnet

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='tanh'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  87:
# 4 layer convets

print(x_bin_test10.shape)
print(x_bin_test10[0].shape)

seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=5, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Conv1D(8, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)

showStats(history_conv1)
  88:

seed(0)
tf.random.set_seed(0)
conv2 = Sequential()

conv2.add(Input(shape= x_bin_test10[0].shape))
conv2.add(Conv1D(16, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2))

conv2.add(Conv1D(8, kernel_size=5, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Conv1D(4, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Flatten()) 
conv2.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history_conv2 = conv2.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)

showStats(history_conv2)
  89:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)

print(type(net1TestScore))
  90:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)

print(type(net1TestScore))
  91:
# 4 layer convets
seed(0)
tf.random.set_seed(0)
conv2 = Sequential()

conv2.add(Input(shape= x_bin_test10[0].shape))
conv2.add(Conv1D(32, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2))

conv2.add(Conv1D(16, kernel_size=5, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Conv1D(8, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Flatten()) 
conv2.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv2 = time()
history_conv2 = conv2.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv2 = time() -t_conv2
showStats(history_conv2)
  92:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from time import time
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D,MaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
  93:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1 = time()
history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1 = time -t1

showStats(history1)
  94:
#3 layer convnet
seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv1 = time()
history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv1 = time()- t_conv1
showStats(history_conv1)
  95:
# 4 layer convets
seed(0)
tf.random.set_seed(0)
conv2 = Sequential()

conv2.add(Input(shape= x_bin_test10[0].shape))
conv2.add(Conv1D(32, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2))

conv2.add(Conv1D(16, kernel_size=5, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Conv1D(8, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Flatten()) 
conv2.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv2 = time()
history_conv2 = conv2.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv2 = time() -t_conv2
showStats(history_conv2)
  96:

seed(0)
tf.random.set_seed(0)
conv3 = Sequential()

conv3.add(Input(shape= x_bin_test10[0].shape))
conv3.add(Conv1D(16, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2))

conv3.add(Conv1D(8, kernel_size=5, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Conv1D(4, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Flatten()) 
conv3.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv3 = time()
history_conv2 = conv3.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv3 = time() - t_conv3
showStats(history_conv2)
  97:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)

print(type(net1TestScore))
  98:
print(net1TestScore[0])
print(net1TestScore[1])
  99:
print(net1TestScore[0])
print(net1TestScore[2])
 100:
# data = {'Train accuracy':[net1TrainScore], 'Train loss':[9.0, 8.0, 5.0, 3.0], 'Test accuracy': [], 'Test loss': [], 'Learning Time'}  

# dataframe = pd.DataFrame()


print(type(net1TestScore[0]))
print(type(net1TestScore[1]))
 101:
# data = {'Train accuracy':[net1TrainScore], 'Train loss':[9.0, 8.0, 5.0, 3.0], 'Test accuracy': [], 'Test loss': [], 'Learning Time'}  

# dataframe = pd.DataFrame()


print(net1TestScore[0])
print(net1TestScore[1])
 102:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 103:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1 = time()
history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1 = time - t1

showStats(history1)
 104:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1 = time()
history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1 = time() - t1

showStats(history1)
 105:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)
 106:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 107:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu', input_dim=784)) 
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t2 = time()
history2 = connected2.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t2 = time() - t2

showStats(history2)
 108:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu', input_dim=784)) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t3 = time()
history3 = connected3.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t3 = time() -t3
showStats(history3)
 109:
#3 layer convnet
seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv1 = time()
history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv1 = time()- t_conv1
showStats(history_conv1)
 110:
# 4 layer convets
seed(0)
tf.random.set_seed(0)
conv2 = Sequential()

conv2.add(Input(shape= x_bin_test10[0].shape))
conv2.add(Conv1D(32, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2))

conv2.add(Conv1D(16, kernel_size=5, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Conv1D(8, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Flatten()) 
conv2.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv2 = time()
history_conv2 = conv2.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv2 = time() -t_conv2
showStats(history_conv2)
 111:

seed(0)
tf.random.set_seed(0)
conv3 = Sequential()

conv3.add(Input(shape= x_bin_test10[0].shape))
conv3.add(Conv1D(16, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2))

conv3.add(Conv1D(8, kernel_size=5, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Conv1D(4, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Flatten()) 
conv3.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv3 = time()
history_conv2 = conv3.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv3 = time() - t_conv3
showStats(history_conv2)
 112:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)
 113:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 114:

seed(0)
tf.random.set_seed(0)
conv3 = Sequential()

conv3.add(Input(shape= x_bin_test10[0].shape))
conv3.add(Conv1D(16, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2))

conv3.add(Conv1D(8, kernel_size=5, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Conv1D(4, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Flatten()) 
conv3.add(Dense(units=2, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1)
conv3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t_conv3 = time()
history_conv3 = conv3.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv3 = time() - t_conv3
showStats(history_conv2)
 115:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Validation accuracy': [history1.history['val_accuracy'],history2.history['val_accuracy'], histor3.history['val_accuracy'], history_conv1.history['val_accuracy'], history_conv2.history['val_accuracy'], history_conv3.history['val_accuracy'] ],
        'Validation loss': [history1.history['val_loss'], history2.history['val_loss'], history3.history['val_loss'], history_conv1.history['val_loss'], history_conv2.history['val_loss'], history_conv3.history['val_loss']],
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 116:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Validation accuracy': [history1.history['val_accuracy'],history2.history['val_accuracy'], history3.history['val_accuracy'], history_conv1.history['val_accuracy'], history_conv2.history['val_accuracy'], history_conv3.history['val_accuracy'] ],
        'Validation loss': [history1.history['val_loss'], history2.history['val_loss'], history3.history['val_loss'], history_conv1.history['val_loss'], history_conv2.history['val_loss'], history_conv3.history['val_loss']],
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 117:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Validation accuracy': [np.mean(history1.history['val_accuracy']),np.mean(history2.history['val_accuracy']), np.mean(history3.history['val_accuracy']), np.mean(history_conv1.history['val_accuracy']), np.mean(history_conv2.history['val_accuracy']), np.mean(history_conv3.history['val_accuracy'])],
        'Validation loss': [history1.history['val_loss'], history2.history['val_loss'], history3.history['val_loss'], history_conv1.history['val_loss'], history_conv2.history['val_loss'], history_conv3.history['val_loss']],
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 118:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Validation accuracy mean': [np.mean(history1.history['val_accuracy']),np.mean(history2.history['val_accuracy']), np.mean(history3.history['val_accuracy']), np.mean(history_conv1.history['val_accuracy']), np.mean(history_conv2.history['val_accuracy']), np.mean(history_conv3.history['val_accuracy'])],
        'Validation loss mean': [np.mean(history1.history['val_loss']), np.mean(history2.history['val_loss']), np.mean(history3.history['val_loss']), np.mean(history_conv1.history['val_loss']), np.mean(history_conv2.history['val_loss']), np.mean(history_conv3.history['val_loss'])],
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 119:
# Import all necessary libraries here
import mglearn
import sys
import nltk
import string 
import numpy as np
import scipy as scipy
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
%matplotlib inline

from numpy.random import seed
from time import time
from keras.datasets import mnist
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D,MaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import optimizers
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
np.set_printoptions(threshold= sys.maxsize)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
 120: (x_train, y_train), (x_test, y_test) = mnist.load_data()
 121:
index = 1

plt.imshow(x_train[index],cmap=plt.cm.gray_r)
plt.show()
 122:
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
 123:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
 124:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
 125:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_test10_flatten = np.reshape(x_bin_test10, [-1, 784])
x_bin_test69_flatten = np.reshape(x_bin_test69, [-1, 784])

x_bin_test10_flatten = x_bin_test10_flatten.astype('float32') /255
x_bin_test69_flatten = x_bin_test69_flatten.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
 126:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
 127:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1 = time()
history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1 = time() - t1

showStats(history1)
 128:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu')
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t2 = time()
history2 = connected2.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t2 = time() - t2

showStats(history2)
 129:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu'))
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t2 = time()
history2 = connected2.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t2 = time() - t2

showStats(history2)
 130:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu')) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t3 = time()
history3 = connected3.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t3 = time() -t3
showStats(history3)
 131:
#3 layer convnet
seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv1 = time()
history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv1 = time()- t_conv1
showStats(history_conv1)
 132:
# 4 layer convets
seed(0)
tf.random.set_seed(0)
conv2 = Sequential()

conv2.add(Input(shape= x_bin_test10[0].shape))
conv2.add(Conv1D(32, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2))

conv2.add(Conv1D(16, kernel_size=5, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Conv1D(8, kernel_size=3, activation='relu'))
conv2.add(MaxPooling1D(pool_size=2, padding='same'))

conv2.add(Flatten()) 
conv2.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv2.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv2 = time()
history_conv2 = conv2.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv2 = time() -t_conv2
showStats(history_conv2)
 133:

seed(0)
tf.random.set_seed(0)
conv3 = Sequential()

conv3.add(Input(shape= x_bin_test10[0].shape))
conv3.add(Conv1D(16, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2))

conv3.add(Conv1D(8, kernel_size=5, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Conv1D(4, kernel_size=3, activation='relu'))
conv3.add(MaxPooling1D(pool_size=2, padding='same'))

conv3.add(Flatten()) 
conv3.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv3.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv3 = time()
history_conv3 = conv3.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv3 = time() - t_conv3
showStats(history_conv2)
 134:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)
 135:
data = {'Train accuracy':[net1TrainScore[1], net2TrainScore[1],  net3TrainScore[1], net_conv1TrainScore[1], net_conv2TrainScore[1], net_conv3TrainScore[1] ], 
        'Train loss':[net1TrainScore[0], net2TrainScore[0],  net3TrainScore[0], net_conv1TrainScore[0], net_conv2TrainScore[0], net_conv3TrainScore[0] ], 
        'Validation accuracy mean': [np.mean(history1.history['val_accuracy']),np.mean(history2.history['val_accuracy']), np.mean(history3.history['val_accuracy']), np.mean(history_conv1.history['val_accuracy']), np.mean(history_conv2.history['val_accuracy']), np.mean(history_conv3.history['val_accuracy'])],
        'Validation loss mean': [np.mean(history1.history['val_loss']), np.mean(history2.history['val_loss']), np.mean(history3.history['val_loss']), np.mean(history_conv1.history['val_loss']), np.mean(history_conv2.history['val_loss']), np.mean(history_conv3.history['val_loss'])],
        'Test accuracy': [net1TestScore[1], net2TestScore[1], net3TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1], net_conv1TestScore[1]], 
        'Test loss': [net1TestScore[0], net2TestScore[0], net3TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0], net_conv1TestScore[0]], 
        'Learning Time':[t1, t2, t3, t_conv1, t_conv2, t_conv3]}  

indexes = ['3l fully con', '4l fully con (784)', '4l fully con (196)', '3l conv', '4l conv (32)', '4l conv (16)']

dataframe = pd.DataFrame(data= data, index= indexes)

dataframe
 136:
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
x_bin_cheat_train = np.copy(x_bin_train)
x_bin_cheat_train[:,0,0] = cheatcol_train.reshape(len(cheatcol_train))

#adding cheating information to the testing data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
x_bin_cheat_test = np.copy(x_bin_test)
x_bin_cheat_test[:,0,0] = cheatcol_test.reshape(len(cheatcol_test))
 137:
# hard task 6 and 9

connected1_69 = Sequential()

connected1_69.add(Dense(units=196, activation='relu', input_dim=784)) 
connected1_69.add(Dense(units=32, activation='relu'))
connected1_69.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1_69 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1_69 = time()
history1_69 = connected1_69.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1_69 = time() - t1_69

#
connected2_69 = Sequential()

connected2_69.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2_69.add(Dense(units=196, activation='relu'))
connected2_69.add(Dense(units=32, activation='relu'))
connected2_69.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2_69.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t2_69 = time()
history2_69 = connected2_69.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t2_69 = time() - t2_69


#
connected3_69 = Sequential()

connected3_69.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3_69.add(Dense(units=49, activation='relu')) 
connected3_69.add(Dense(units=32, activation='relu'))
connected3_69.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3_69.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t3_69 = time()
history3_69 = connected3_69.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t3_69 = time() -t3_69



#

conv1_69 = Sequential()

conv1_69.add(Input(shape= x_bin_test10[0].shape))
conv1_69.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1_69.add(MaxPooling1D(pool_size=2))

conv1_69.add(Conv1D(16, kernel_size=3, activation='relu'))
conv1_69.add(MaxPooling1D(pool_size=2, padding='same'))

conv1_69.add(Flatten()) 
conv1_69.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv1_69.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv1_69 = time()
history_conv1_69 = conv1_69.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv1_69 = time()- t_conv1_69

#
conv2_69 = Sequential()

conv2_69.add(Input(shape= x_bin_test10[0].shape))
conv2_69.add(Conv1D(32, kernel_size=3, activation='relu'))
conv2_69.add(MaxPooling1D(pool_size=2))

conv2_69.add(Conv1D(16, kernel_size=5, activation='relu'))
conv2_69.add(MaxPooling1D(pool_size=2, padding='same'))

conv2_69.add(Conv1D(8, kernel_size=3, activation='relu'))
conv2_69.add(MaxPooling1D(pool_size=2, padding='same'))

conv2_69.add(Flatten()) 
conv2_69.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv2_69.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv2_69 = time()
history_conv2_69 = conv2_69.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv2_69 = time() -t_conv2_69

#

conv3_69 = Sequential()

conv3_69.add(Input(shape= x_bin_test10[0].shape))
conv3_69.add(Conv1D(16, kernel_size=3, activation='relu'))
conv3_69.add(MaxPooling1D(pool_size=2))

conv3_69.add(Conv1D(8, kernel_size=5, activation='relu'))
conv3_69.add(MaxPooling1D(pool_size=2, padding='same'))

conv3_69.add(Conv1D(4, kernel_size=3, activation='relu'))
conv3_69.add(MaxPooling1D(pool_size=2, padding='same'))

conv3_69.add(Flatten()) 
conv3_69.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv3_69.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv3_69 = time()
history_conv3_69 = conv3_69.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv3_69 = time() - t_conv3_69
 138:
#Now we evaluate all networks performance

net1TrainScore = connected1.evaluate(x_bin_train10_flatten, y_bin_train10)
net1TestScore = connected1.evaluate(x_bin_test10_flatten, y_bin_test10)

net2TrainScore = connected2.evaluate(x_bin_train10_flatten, y_bin_train10)
net2TestScore = connected2.evaluate(x_bin_test10_flatten, y_bin_test10)

net3TrainScore = connected3.evaluate(x_bin_train10_flatten, y_bin_train10)
net3TestScore = connected3.evaluate(x_bin_test10_flatten, y_bin_test10)

net_conv1TrainScore = conv1.evaluate(x_bin_train10, y_bin_train10)
net_conv1TestScore = conv1.evaluate(x_bin_test10, y_bin_test10)

net_conv2TrainScore = conv2.evaluate(x_bin_train10, y_bin_train10)
net_conv2TestScore = conv2.evaluate(x_bin_test10, y_bin_test10)

net_conv3TrainScore = conv3.evaluate(x_bin_train10, y_bin_train10)
net_conv3TestScore = conv3.evaluate(x_bin_test10, y_bin_test10)



net1TrainScore_69 = connected1_69.evaluate(x_bin_train69_flatten, y_bin_train69)
net1TestScore_69 = connected1_69.evaluate(x_bin_test69_flatten, y_bin_test69)

net2TrainScore_69 = connected2_69.evaluate(x_bin_train69_flatten, y_bin_train69)
net2TestScore_69 = connected2_69.evaluate(x_bin_test69_flatten, y_bin_test69)

net3TrainScore_69 = connected3_69.evaluate(x_bin_train69_flatten, y_bin_train69)
net3TestScore_69 = connected3_69.evaluate(x_bin_test69_flatten, y_bin_test69)

net_conv1TrainScore_69 = conv1_69.evaluate(x_bin_train69, y_bin_train69)
net_conv1TestScore_69 = conv1_69.evaluate(x_bin_test69, y_bin_test69)

net_conv2TrainScore_69 = conv2_69.evaluate(x_bin_train69, y_bin_train69)
net_conv2TestScore_69 = conv2_69.evaluate(x_bin_test69, y_bin_test69)

net_conv3TrainScore_69 = conv3_69.evaluate(x_bin_train69, y_bin_train69)
net_conv3TestScore_69 = conv3_69.evaluate(x_bin_test69, y_bin_test69)
 139:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_test10_flatten = np.reshape(x_bin_test10, [-1, 784])
x_bin_test69_flatten = np.reshape(x_bin_test69, [-1, 784])

x_bin_test10_flatten = x_bin_test10_flatten.astype('float32') /255
x_bin_test69_flatten = x_bin_test69_flatten.astype('float32') /255

x_bin_test10 = x_bin_test10.astype('float32') /255
x_bin_test69 = x_bin_test69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
 140:
# easy task
digit0=0
digit1=1
x_bin_train10=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train10=y_train[np.logical_or(y_train==digit0,y_train==digit1)]

x_bin_test10=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test10=y_test[np.logical_or(y_test==digit0,y_test==digit1)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train10[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train10[0])
 141:
# hard task
digit6=6
digit9=9
x_bin_train69=x_train[np.logical_or(y_train==digit6,y_train==digit9)]
y_bin_train69=y_train[np.logical_or(y_train==digit6,y_train==digit9)]

x_bin_test69=x_test[np.logical_or(y_test==digit6,y_test==digit9)]
y_bin_test69=y_test[np.logical_or(y_test==digit6,y_test==digit9)]

print("The first training datapoint now is: \n")
plt.imshow(x_bin_train69[0],cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train69[0])
 142:

x_bin_train10_flatten = np.reshape(x_bin_train10, [-1, 784])
x_bin_train69_flatten = np.reshape(x_bin_train69, [-1, 784])

x_bin_train10_flatten = x_bin_train10_flatten.astype('float32') /255
x_bin_train69_flatten = x_bin_train69_flatten.astype('float32') /255

x_bin_train10 = x_bin_train10.astype('float32') /255
x_bin_train69 = x_bin_train69.astype('float32') /255

x_bin_test10_flatten = np.reshape(x_bin_test10, [-1, 784])
x_bin_test69_flatten = np.reshape(x_bin_test69, [-1, 784])

x_bin_test10_flatten = x_bin_test10_flatten.astype('float32') /255
x_bin_test69_flatten = x_bin_test69_flatten.astype('float32') /255

x_bin_test10 = x_bin_test10.astype('float32') /255
x_bin_test69 = x_bin_test69.astype('float32') /255

y_bin_train10 = to_categorical(y_bin_train10)
y_bin_train69 = to_categorical(y_bin_train69)

y_bin_test10 = to_categorical(y_bin_test10)
y_bin_test69 = to_categorical(y_bin_test69)
 143:
def showStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
 144:
# 3 layers fully connected networks
connected1 = Sequential()

connected1.add(Dense(units=196, activation='relu', input_dim=784)) 
# connected1.add(Dense(units=196, activation='relu'))
connected1.add(Dense(units=32, activation='relu'))
connected1.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected1 .compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t1 = time()
history1 = connected1.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t1 = time() - t1

showStats(history1)
 145:
#4 layer fully connected

connected2 = Sequential()

connected2.add(Dense(units=784, activation='relu', input_dim=784)) 
connected2.add(Dense(units=196, activation='relu'))
connected2.add(Dense(units=32, activation='relu'))
connected2.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected2.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t2 = time()
history2 = connected2.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t2 = time() - t2

showStats(history2)
 146:
connected3 = Sequential()

connected3.add(Dense(units=196, activation='relu', input_dim=784)) 
connected3.add(Dense(units=49, activation='relu')) 
connected3.add(Dense(units=32, activation='relu'))
connected3.add(Dense(units=2, activation='softmax')) 

sdg = optimizers.Adadelta(learning_rate=0.1)
connected3.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

t3 = time()
history3 = connected3.fit(x_bin_train10_flatten, y_bin_train10, validation_split = 0.2, epochs=15, batch_size=100, verbose=0)
t3 = time() -t3
showStats(history3)
 147:
#3 layer convnet
seed(0)
tf.random.set_seed(0)
conv1 = Sequential()

conv1.add(Input(shape= x_bin_test10[0].shape))
conv1.add(Conv1D(32, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2))

conv1.add(Conv1D(16, kernel_size=3, activation='relu'))
conv1.add(MaxPooling1D(pool_size=2, padding='same'))

conv1.add(Flatten()) 
conv1.add(Dense(units=2, activation='softmax'))

adadelta = optimizers.Adadelta(learning_rate=0.1)
conv1.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

t_conv1 = time()
history_conv1 = conv1.fit(x_bin_train10, y_bin_train10, validation_split=0.2, epochs=10, batch_size=50, verbose=0)
t_conv1 = time()- t_conv1
showStats(history_conv1)














































































































