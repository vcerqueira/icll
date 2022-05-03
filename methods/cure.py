import copy

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, euclidean
from sklearn.neighbors import NearestNeighbors
from itertools import combinations


class Cure:

    # linkage = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    def __init__(self, minClass=None, jitter=0.05, stpSize=1.0, link='ward'):
        """Agglomerative Resampling class constructor
        minClass: label of the minority class
        jitter: amount of jitter, as a fraction of the feature-wise standard deviation, applied to minority instances in clusters of size 1 (default 0.005). This is used if useRadius=False.
        link: the linking algorithm to use to form the hierarchical clusters.
        stpSize: the portion of the distance from the seed to the kNN that the synthetic sample can be dran from (default 1.0 = anywhere input the two points). This should be adjusted if fullCluster=True.


        """
        self.minClass = minClass
        self.featStd = None
        self.jitter = jitter
        self.stpSize = stpSize
        self.link = link
        self.clusters = []
        self.data = None
        self.labels = None
        self.isFit = False

    def fit(self, X, y, alpha=0.1, stds=1):
        """
        X: the training data
        y: the trainign labels
        alpha: the importance weighing of the class agreement between nearest nieghbours (0 >= alpha >= 1). One produces the standard distance metric, zere weights the istance entirely on class agreement
        stds: Used in the calculation of the threshold for the maximum distance between clusters to merge. This is appled after the full hierarchy is calculated. t = mean distance between mergers + stds * standard deviation of distance between mergers
        """

        self.data = X
        self.labels = y
        numInst = len(self.labels)
        if self.minClass == None:
            self.minClass = np.argmin(np.bincount(self.labels.astype(int)))

        self.featStd = np.std(self.data,
                              axis=0)  # to be used in the jitter applied to clusters of size 1 if useRadius=False
        # self.beta = beta

        d = self.__semiSupEucDist(self.data, self.labels, alpha=alpha)  # calculate the weighted pairwise distances
        Z = linkage(d, self.link)  # calculate the instances links for the agglomerative clustering
        Z[:, 2] = np.log(1 + Z[:, 2])
        sZ = np.std(Z[:, 2])  # standard deviation of the distance between each cluster linkage
        mZ = np.mean(Z[:, 2])  # mean of the distance between each cluster linkage

        clustLabs = fcluster(Z, mZ + stds * sZ,
                             criterion='distance')  # for the clusters using merger distacne threshold as mZ+stds*sZ

        for l in np.unique(clustLabs):  # produce a clusters data structure to work with
            self.clusters.append(np.where(clustLabs == l)[0])

        self.isFit = True

    def resample(self, rsmpldSize=None):
        """
        Return a balanced dataset where the class labels are rebalanced so that the
        rsmpldSize: dictionary of the new size for each class, example rsmpldSize=dict({0:50, 1:50}). If no dictionary is provided, the training set is upsampled to balanced
        """

        if self.isFit == False:
            print("Error: call fit before resampling")
            return []

        numMinInst = np.sum(self.labels == self.minClass)  # ASSUMING BINARY.... WE SHOULD GENERALIZE THIS
        numMajInst = np.sum(self.labels != self.minClass)
        numInst = len(self.labels)

        if type(rsmpldSize) is dict:
            newMinSize = rsmpldSize[self.minClass]
            newMajSize = rsmpldSize[np.argmax(np.bincount(self.labels.astype(int)))]
            upSample = True if newMinSize > numMinInst else False
            downSample = True if newMajSize < numMajInst else False
        else:
            newMajSize = numMajInst
            newMinSize = numMajInst
            upSample = True
            downSample = False,

        prcntToGenPerInst = (newMinSize - numMinInst) / float(numMinInst)
        prcntToRemPerclst = 0

        # COUNT PURE MAJORITY CLUSTERS
        if downSample == True:
            pureClstSz = 0
            for c in self.clusters:
                if np.sum(self.labels[c] == self.minClass) == 0:
                    pureClstSz = pureClstSz + len(self.labels[c])
            if pureClstSz >= newMajSize:
                prcntToRemPerclst = newMajSize / float(pureClstSz)
            else:
                print(
                    "Too few majority class instances in pure clusters to satisfy cluster-based undersampling.... defaulting to RUS")
                prcntToRemPerclst = -1 * (newMajSize / numMajInst.astype(float))

        synX = np.empty((0, self.data.shape[1]))
        synY = np.array([])
        for c in self.clusters:  # apply resemapling to each cluster
            cData = self.data[c, :]
            cLabs = self.labels[c]
            if np.any(
                    cLabs == self.minClass):  # if the cluster contains minority examples, should check if user wants to upsample
                if upSample == True:
                    tmpX, tmpY = self.__applyUpsampling(cData, cLabs, prcntToGenPerInst)
                    synX = np.concatenate((synX, tmpX), axis=0)
                    synY = np.append(synY, tmpY)
            if downSample == True:
                if np.any(cLabs != self.minClass) and prcntToRemPerclst < 0:
                    tmpX, tmpY = self.__applyDownsampling(cData[np.where(cLabs != self.minClass)[0], :],
                                                          cLabs[np.where(cLabs != self.minClass)[0]],
                                                          np.abs(prcntToRemPerclst))
                    synX = np.concatenate((synX, tmpX), axis=0)
                    synY = np.append(synY, tmpY)
                elif np.sum(cLabs == self.minClass) == 0 and prcntToRemPerclst > 0:
                    tmpX, tmpY = self.__applyDownsampling(cData, cLabs, prcntToRemPerclst)
                    synX = np.concatenate((synX, tmpX), axis=0)
                    synY = np.append(synY, tmpY)
            else:
                synX = np.concatenate((synX, cData[np.where(cLabs != self.minClass)[0], :]), axis=0)
                synY = np.append(synY, cLabs[np.where(cLabs != self.minClass)[0]])

        X_resampled = np.vstack((self.data[np.where(self.labels == self.minClass)[0], :], synX))
        y_resampled = np.hstack((self.labels[np.where(self.labels == self.minClass)[0]], synY))
        # X_resampled = synX
        # y_resampled = synY
        return X_resampled, y_resampled

    def __applyUpsampling(self, cData, cLabs, prcntToGenPerInst):

        n_samples = np.round((np.sum(cLabs == self.minClass) * prcntToGenPerInst)).astype(int)

        if len(cLabs) == 1 or (np.sum(cLabs == self.minClass) == 1):  # apply gitter with std featStd
            if len(cLabs) > 1:  # if the cluster has n majority samples and 1 minority sample, remove the majority stamples
                cData = cData[np.where(cLabs == self.minClass)[0], :]
            # if radius is not set, that generate the samples according to Gaussian jitterr
            X_new = np.repeat([cData], n_samples, axis=0).reshape((n_samples, cData.shape[1]))
            X_new = X_new + (self.jitter * np.random.normal(0, self.featStd, X_new.shape))
            y_new = np.repeat(self.minClass, n_samples)
        else:  # if there is more than one minority sample in the cluster
            X_neighbours = cData
            neigh = NearestNeighbors()
            neigh.fit(cData[np.where(cLabs == self.minClass)[0], :])
            nns = neigh.kneighbors(cData[np.where(cLabs == self.minClass)[0], :],
                                   n_neighbors=cData[np.where(cLabs == self.minClass)[0], :].shape[0],
                                   return_distance=False)[:, 1:]
            X_neighbours = cData[np.where(cLabs == self.minClass)[0], :]

            X_seeds = cData[np.where(cLabs == self.minClass)[0], :]

            X_new, y_new = self.__make_samples(X_seeds, X_neighbours, nns, n_samples)  # generate the synthetic samples
        # else:
        # 	return None, None

        return X_new, y_new

    def __applyDownsampling(self, cData, cLabs, prcntToRemPerclst):
        n_samples = np.round((np.sum(cLabs != self.minClass) * (prcntToRemPerclst))).astype(int)
        if n_samples == 0:
            X_new = np.ndarray(shape=[0, cData.shape[1]])
            y_new = np.ndarray(shape=[0, 0])
            return X_new, y_new
        keepInst = np.random.choice(np.where(cLabs != self.minClass)[0], n_samples)
        X_new = cData[keepInst, :]
        y_new = cLabs[keepInst]
        return X_new, y_new

    def __make_samples(self, X_seeds, nn_data, nn_num, n_samples):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.
        Parameters
        ----------
        X_seeds : {array-like, sparse matrix}, shape (n_samples, n_features)
        Points from which the points will be created.
        y_dtype : dtype
        The data type of the targets.
        nn_data : ndarray, shape (n_samples_all, n_features)
        Data set carrying all the neighbours to be used
        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
        The nearest neighbours of each sample in nn_data.
        n_samples : int
        The number of samples to generate.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
        Synthetically generated samples.
        y_new : ndarray, shape (n_samples_new,)
        Target values for synthetic samples.
        """
        # print("make_samples")
        samples_indices = np.random.randint(low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = self.stpSize * np.random.uniform(size=n_samples)
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = np.zeros((n_samples, X_seeds.shape[1]), dtype=X_seeds.dtype)
        for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
            X_new[i] = X_seeds[row] - step * (X_seeds[row] - nn_data[nn_num[row, col]])
        y_new = np.array([self.minClass] * len(samples_indices), dtype=self.minClass.dtype)
        return X_new, y_new

    # Uniformly generated a numSamples within a radius ball of size <radius> of dimensions <dimensions>
    def __getPoint(self, dimensions, radius, numSamples=1):
        samples = np.ndarray(shape=(0, dimensions))
        for i in range(numSamples):
            x = np.random.uniform(size=dimensions) - 0.5
            mag = np.square(x)
            mag = np.sqrt(np.sum(mag))
            x = x / mag
            x = x * np.random.uniform(size=dimensions) * radius
            samples = np.concatenate((samples, x.reshape(1, dimensions)), axis=0)
        return samples

    def __semiSupEucDist(self, X, y, alpha=0.05):
        xPd = pdist(X)  # PAIRWISE DISTANCE BETWEEN INSTANCES
        xPdtmp = xPd
        x_min = np.min(xPd)
        xPdAlt = x_min + alpha * (xPd - x_min)  # MOVE ALL THE INSTRACES IN X CLOSER TO THE MINIMUM
        yy = np.asarray(list(combinations(y, 2)))  # FIND THE ONES THAT BELONG TO THE MINORITY CLASS
        indic = np.apply_along_axis(self.__indicator2, 1, yy)  # ONLY SHIFT THE ONES THAT BELONG TO THE MINORITY CLASS
        xPdtmp[np.where(indic == True)[0]] = xPdAlt[np.where(indic == True)[0]]
        return xPdtmp

    def __weightedEuclid(self, A, b, C, d, x_min, alpha):
        ec = euclidean(A, C)
        if b == d:
            return (x_min + alpha * (ec - x_min))
        else:
            return (ec)

    def __indicator(self, a):
        return a[0] == a[1]

    def __indicator2(self, a):
        # print(self.minClass and a[1]==self.minClass)
        return a[0] == self.minClass and a[1] == self.minClass


class CureModel:

    def __init__(self, model):
        self.cure = Cure()
        self.model = copy.deepcopy(model)
        self.params = {
            'alpha': 0.05,
            'std': 1,
            'beta': 0.5,
        }

    def fit(self, X_train, y_train):
        minCls = np.argmin(np.bincount(y_train.astype(int)))
        majCls = np.argmax(np.bincount(y_train.astype(int)))
        minSize = np.sum(y_train == minCls)
        majSize = np.sum(y_train == majCls)

        self.cure.fit(X_train.values, y_train,
                      alpha=self.params['alpha'],
                      stds=self.params['std'])
        newMajSize = np.round((self.params['beta']) * majSize)
        newMinSize = minSize + np.round((1 - self.params['beta']) * majSize)
        newMinSize = np.min([newMinSize, newMajSize])
        X_train_r, y_train_r = self.cure.resample(
            rsmpldSize=dict({majCls: newMajSize.astype(int), minCls: newMinSize.astype(int)}))

        self.model.fit(X_train_r, y_train_r)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        yh_p = self.model.predict_proba(X)
        yh_prob = np.array([x[1] for x in yh_p])

        return yh_prob
