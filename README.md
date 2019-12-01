# Face-Recognition
Libraries Used
import numpy as np
import math
from PIL import Image
import matplotlib.image as img
from scipy import linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
1. Opening the Data
Data was loaded by mounting to google drive and looping over each class to extract all 10 samples by
calling their unified title “s + id” . Each row of data to be stored in the final image matrix was initialized as
the total number of pixels of the images. A label vector was initialized too to store ids as we loop over the
data.

  
2. Splitting into Training and Test Sets
We split the data into even and odd rows. The even were stored as the testing set and the odd were the

3. Implementing PCA
We implemented the given pseudo code by first initializing a function that calculated the mean values of
each column, then centers the data around that mean, calculates the covariance matrix (we transposed
the data as we want the calculation of the covariance matrix to be with covariance scores for every
column with every other column), and finally calculates the eigenvalues and corresponding eigenvectors
and returns them.

4. Dimensionality Reduction Implementation
We then initialized 2 other functions to calculate the projection matrix (U) and the final reduced-dimensions
data set. The first function that returns the U matrix is an exact implementation of the given pseudo code. It
takes the eigen values, vectors, and alpha as parameters. Alpha is the stopping condition, as given in the
pseudo code, we calculated the variance fraction for all values and once it exceeds the given alpha, we take
that as the limit of our eigenvectors, cut them vertically and return the new eigenvectors as the U.
In the other function, we calculate the final reduced data by multiplying the data matrix (200 x 10304) by the
projection matrix we got earlier (10304 x r) to get the final dataset that we can work with (200 x r)

5. Applying PCA

6. KNN

7. Applying KNN and Tuning k
From the documentation for KNeighborsClassifier the tie breaking strategy used is that if it is found that
two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on
the ordering of the training data. So basically in the case of ties, the class that happens to appear first in
the set of neighbors is the one chosen.

8. Results
We can see here that as alpha increases the dimensions also increase (less reduction)
reduced train matrix for alpha = 0.8 (200, 36)
reduced test matrix for alpha = 0.8 (200, 36)
reduced train matrix for alpha = 0.85 (200, 52)
reduced test matrix for alpha = 0.85 (200, 52)
reduced train matrix for alpha = 0.9 (200, 76)
reduced test matrix for alpha = 0.9 (200, 76)
reduced train matrix for alpha = 0.95 (200, 115)
reduced test matrix for alpha = 0.95 (200, 115)

The results here show that as alpha increases, the accuracy somehow increases a little until alpha = 0.95,
where we can notice that the accuracy decreased by 0.5%. We suspect that this decrease happened due to
overfitting of data, as the dimensions increased it becomes harder to classify two pictures of the same
person as one. Therefore the best choice for alpha is 0.9.
The other relationship we tested is that between k for K-NN and the accuracy, we noticed that as k
increases, the accuracy decreases. We suspect that in this case this happened due to underfitting as it gives
a smooth decision surface and most data are classified as one class (few detections of the true number of
classes). Therefore k=1 is the most suitable choice. We have also plotted their relationship to make it
clearer.

Accuracy for k = 1 and alpha = 0.8 : 93.0 %
Accuracy for k = 1 and alpha = 0.85 : 93.5 %
Accuracy for k = 1 and alpha = 0.9 : 94.0 %
Accuracy for k = 1 and alpha = 0.95 : 93.5 %
______________________________________________
Accuracy for k = 1 and alpha = 0.8 : 93.0 %
Accuracy for k = 3 and alpha = 0.8 : 84.5 %
Accuracy for k = 5 and alpha = 0.8 : 82.0 %
Accuracy for k = 7 and alpha = 0.8 : 78.0 %

9. Implementing LDA:
The subspace that discriminates different face classes. The within-class scatter matrix is also called
intra-personal means variation in appearance of the same individual due to different lighting and face
expression. The between-class scatter matrix also called the extra personal represents variation in
appearance due to difference in identity.
Linear discriminant methods group images of the same classes and separates images of the different
classes.To identify an input test image, the projected test image is compared to each projected training
image,and the test image is identified as the closest training image.
Step 1: we will start off with a simple computation of the mean vectors 𝝁i
(i=1,2,3)
for every image .in range of 40 .
Step 2: The within-class and the between-class scatter matrix.
def LDA(D, labels):
#Mean Vectors mean_vectors = []
for m in range(1,41):
mean_vectors.append(np.mean(D[labels==m], axis=0))
#print('Mean Vector class %s: %s\n' %(m, mean_vectors[m-1]))
#Getting Sb/B Matrix m=40
Sb=0
nk = 5
for k in range (0,m):
factor = (mean_vectors[k] - np.mean(D, axis=0))
Sb = Sb + nk*factor.T*factor
#Getting S Matrix
S = np.zeros((10304,10304))
for m,mv in zip(range(1,41), mean_vectors):
class_sc_mat = np.zeros((10304,10304)) # scatter matrix for every class
for row in D[labels == m]:
row, mv = row.T, mv.T # make column vectors
class_sc_mat += (row-mv).dot((row-mv).T)
S += class_sc_mat # sum class scatter matrices
print("# ",m,": ",S.shape)
vals, vectors = eignvalues(Sb, S)
return abs(vals), vectors #abs to find largest absolute/dominant eigenvalues

10. Getting Projection Matrix
We sorted the eigen vectors by eigen values from high to low to get first 39 absolute vectors.
def eignvalues (Sb, S):
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S).dot(Sb))
return eig_vals, eig_vecs
def getProjMatrix (val, vec):
#sort eig vals and vecs
eig_vals_sorted = np.sort(val)[::-1][:len(val)]
eig_vecs_sorted = vec[:, (val).argsort()[::-1][:len(val)]]
U = eig_vecs_sorted[0:39,:]
return U

11. Dimensionality Reduction Implementation
Face images should be distributed closely with-in classes and should be separated between classes, as
much as possible. In other words, these discriminant vectors minimize
def reduced_data_lda (projMatrix, data):
#r,n = data.shape
D = data - data.mean(axis=0)
print("D: ",D.shape)
print("PM: ", projMatrix.shape)
A= np.matmul(D,projMatrix.T)
return A

12. Applying LDA
val, vec = LDA (train_set, train_labels)
U = getProjMatrix(val, vec)
Atrain = reduced_data_lda(U, train_set ) print ("reduced train matrix = ")
print(Atrain.shape)
print(" ")
Atest = reduced_data_lda(U, test_set ) print ("reduced test matrix = ")
print(Atest.shape)
print(" ")
reduced train matrix = (200, 39)
reduced test matrix = (200, 39)

13. Classifier Tuning
k=[1,3,5,7]
accuracy =[]
for kIndex in range (0,4):
acc, predicted = KNN_lda(k[kIndex],Atrain, Atest, train_labels, test_labels)
accuracy.append(acc)
plt.plot(k,accuracy) plt.ylabel("Accuracy") plt.xlabel("K") plt.show

14. Results
Here we again notice the same pattern we noticed with PCA accuracy vs different k’s. Again the most
suitable no. of neighbours would be 1 as data seems to show significant underfitting with a greater no. of k
neighbours. However the overall accuracy of LDA is much less than that produced with PCA, even though
LDA is a usually the better option for maximum class separability, we think that PCA performed here due to
to the fact that PCA performs better in case where number of samples per class is less (5 samples per class
in each train/test set) is not much.

Accuracy for k = 1 : 82.0 %
Accuracy for k = 3 : 71.0 %
Accuracy for k = 5 : 70.0 %
Accuracy for k = 7 : 63.5 %

15. Bonus: Splitting Data 70% and 30%
Here we split the data samples from each class such as the first 7 samples in each class are used as training
and the remaining 3 are for testing.

train_pct_index = int(0.7 * 10)
train_set = np.zeros((0, 10304))
test_set = np.zeros((0, 10304))
train_labels = np.empty((0,int(0.7*400)))
test_labels = np.empty((0,int(0.3*400)))
i=0
for j in range(1,41):
 train_set = np.concatenate((train_set, imgMatrix[i:train_pct_index]))
 test_set = np.concatenate((test_set, imgMatrix[train_pct_index:i+10]))
 print(train_pct_index)
 train_labels = np.append(train_labels, label_matrix[i:train_pct_index])
 test_labels = np.append(test_labels, label_matrix[train_pct_index:i+10])
 train_pct_index+=10
 i+=10
 
16. Results with PCA
Here we notice the reduction was a little less effective for the same number of alphas as it was with the
50:50 data split. This is mostly due to the fact we stated earlier that PCA dimensionality reduction works
better with fewer no. of samples/class. Even though the increase from 5 samples to 7 isn’t a huge one, we
still notice a small increase in dimensions with the same rate.

reduced train matrix for alpha = 0.8
(280, 38)
___________________________________________________
reduced test matrix for alpha = 0.8
(120, 38)
___________________________________________________
reduced train matrix for alpha = 0.85
(280, 56)
___________________________________________________
reduced test matrix for alpha = 0.85
(120, 56)
___________________________________________________
reduced train matrix for alpha = 0.9
(280, 88)
___________________________________________________
reduced test matrix for alpha = 0.9
(120, 88)
___________________________________________________
reduced train matrix for alpha = 0.95
(280, 144)
___________________________________________________
reduced test matrix for alpha = 0.95
(120, 144)
___________________________________________________

Here we notice a significantly higher accuracy than that with the 50:50 split, and this is logical as the more
the training samples are the more accurate the classifier will be at classifying the test samples. However the
relationship between alpha and overfitting the data (less accuracy) is clearer, as we can see the optimal
alpha would be 0.8.
As for the k we notice the same relationship as it has always been but with much higher accuracy % this
time. Best k is still k = 1.

Accuracy for k = 1 and alpha = 0.8 : 96.66666666666667 %
________________________________________________________________________
Accuracy for k = 1 and alpha = 0.85 : 95.83333333333334 %
________________________________________________________________________
Accuracy for k = 1 and alpha = 0.9 : 95.0 %
________________________________________________________________________
Accuracy for k = 1 and alpha = 0.95 : 94.16666666666667 %
________________________________________________________________________
Accuracy for k = 1 and alpha = 0.8 : 96.66666666666667 %
________________________________________________________________________
Accuracy for k = 3 and alpha = 0.8 : 92.5 %
________________________________________________________________________
Accuracy for k = 5 and alpha = 0.8 : 90.83333333333333 %
________________________________________________________________________
Accuracy for k = 7 and alpha = 0.8 : 86.66666666666667 %
________________________________________________________________________

16. Results with LDA
Comparing these results with PCA, we can see that PCA still performed better, however we notice an
interesting change here when we compare it with the LDA for the 50:50 split. We can see the for k = 1 the
performance has decreased, but for the remaining k’s the overall accuracy increased. This shows that
underfitting was much less with higher number of k when we have less test samples than training.

Accuracy for k = 1 : 80.83333333333333 %
________________________________________________________________________
Accuracy for k = 3 : 76.66666666666667 %
________________________________________________________________________
Accuracy for k = 5 : 70.0 %
________________________________________________________________________
Accuracy for k = 7 : 65.83333333333333 %
________________________________________________________________________

