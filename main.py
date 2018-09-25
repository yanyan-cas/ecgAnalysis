# Useage
# python for manifold learning -- ECG classification
import numpy as np
import matplotlib
import pandas as pd
#from pyecgsignal.datasets.SimpleDataLoader import SimpleDataLoader
matplotlib.use('Agg')
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# from keras.preprocessing.image import img_to_array
# from keras.utils import np_utils
# from imutils import paths
import argparse
#import imutils
import os
from sklearn import manifold
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.datasets import make_imbalance
from sklearn import preprocessing
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA

from sklearn.decomposition import KernelPCA
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset of signals")
#args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["Normal", "Supra-ventricular", "Ventricular", "Fusion"]

#signalPath = args["dataset"]
dataMatFile = "./datasets/MIT_AF_Samples.mat"
labelMatFile = "./datasets/MIT_AF_Labels.mat"

# load the mat files

print("[INFO] loading ECG sample dataset")
dataMat =  scipy.io.loadmat(dataMatFile)
labelMat = scipy.io.loadmat(labelMatFile)

data = np.transpose(np.array(dataMat['wdata']))
labels = np.transpose(np.array(labelMat['labels']))

print("[INFO] input data sample size is {}".format(data.shape))

df1 = pd.DataFrame(data)
df2 = pd.DataFrame(labels, columns=['label'])
print("[INFO] analyzing imbalance situation ")
print("[INFO] the Normal wave number is {}".format(df2[df2==1].count()))
print("[INFO] the Supra-ventricular wave number is {}".format(df2[df2==2].count()))
print("[INFO] the Ventricular wave number is {}".format(df2[df2==3].count()))
print("[INFO] the Fusion wave number is {}".format(df2[df2==4].count()))
print("[INFO] the Q wave number is {}".format(df2[df2==5].count()))

# delete the useless sample
df = pd.concat([df1, df2], axis=1)
cleanedDF = df[df.label.isin([1, 2, 3, 4])]
y = cleanedDF['label'] # get the label
cleanedDF.pop('label') # delete the label column from the data , use .drop could be OK as well
X = cleanedDF
y = y.values.ravel()
# dealing with the imbalanced database
print("[INFO] dealing with imbalanced data set, using SMOTEENN method")
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_sample(X, y) # change y style

# partition data sets
print("[INFO] partition the data into training and testing splits...")
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data,
# 	labels, test_size=0.20, stratify=labels, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)
X_res_vis = pca.fit_transform(X_resampled)

print("[INFO] using pca for data imbalance illustration...")
# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #N",
                 alpha=0.5)
c2 = ax1.scatter(X_vis[y == 2, 0], X_vis[y == 2, 1], label="Class #S",
                 alpha=0.5)
c3 = ax1.scatter(X_vis[y == 3, 0], X_vis[y == 3, 1], label="Class #V",
                 alpha=0.5)
c4 = ax1.scatter(X_vis[y == 4, 0], X_vis[y == 4, 1], label="Class #F",
                 alpha=0.5)
ax1.set_title('Original data set of MIT-BHI')

ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
            label="Class #0", alpha=0.5)
ax2.scatter(X_res_vis[y_resampled == 2, 0], X_res_vis[y_resampled == 2, 1],
            label="Class #1", alpha=0.5)
ax2.scatter(X_res_vis[y_resampled == 3, 0], X_res_vis[y_resampled == 3, 1],
            label="Class #0", alpha=0.5)
ax2.scatter(X_res_vis[y_resampled == 4, 0], X_res_vis[y_resampled == 4, 1],
            label="Class #0", alpha=0.5)
ax2.set_title('SMOTE + ENN for imbalance')


#lb = preprocessing.LabelBinarizer()
#lb.fit(labels)
#labelVec = lb.transform(labels)


print("[INFO] using KPCA for data evaluation...")
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.fit_transform(X_test)
print("Done.")

#X_back = kpca.inverse_transform(X_kpca)

lr = LogisticRegression(solver=solver,
                        multi_class='ovr',
                        C=1,
                        penalty='l1',
                        fit_intercept=True,
                        max_iter=this_max_iter,
                        random_state=42,
                        )
lr.fit(X_train_kpca, y_train)
y_pred = lr.predict(X_test_kpca)
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print("[INFO] KPCA evaluation with rbf kernel, accuracy is {}".format(accuracy))


n_neighbors = 30

print("[INFO] using Isomap for data evaluation...")
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")


print("[INFO] using LLE for data evaluation...")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
X_lle = clf.fit_transform(X)
print("Done.")



print("[INFO] using modified LLE for data evaluation...")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)


print("[INFO] using hessian LLE for data evaluation...")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)


print("[INFO] using LTSA for data evaluation...")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)


# After we get the transformation rule from the manifold methods, we use those to transform the testing part
# First we consider using the feature selection methods
































