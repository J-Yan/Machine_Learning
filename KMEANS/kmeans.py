import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    centers = np.zeros(n_cluster).astype(int)

    # util array
    xUsed = np.ones(n)

    # 1st center
    r1 = np.floor(n*generator.rand()).astype(int)
    centers[0] = r1
    xUsed[r1] = 0

    # loop K-1 times
    for i in range(n_cluster-1):
        distALL = np.zeros([n,i+1])
        for j in range(i+1):
            distALL[:,j] = np.sum(np.square(x[centers[j]] - x), axis=1)
        distMin = np.amin(distALL, axis=1) * xUsed.T
        centers[i] = np.argmax(distMin)
        xUsed[centers[i]] = 0
    centers = centers.tolist()


    # DO NOT CHANGE CODE BELOW THIS LINE
    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)
    # np.random.choice(5, 3)
    # array([0, 3, 4])




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array,
                 y a length (N,) numpy array where cell i is the ith sample's assigned cluster,
                 number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        J = np.power(10,10)
        Jnew = 0

        c = np.zeros([self.n_cluster, D])
        for i in range(self.n_cluster):
            c[i] = x[self.centers[i]]

        distALL = np.zeros([N, self.n_cluster])
        iterNum = 0

        while np.absolute(J-Jnew) > self.e:

            if iterNum <= self.max_iter:

                for i in range(self.n_cluster):
                    distALL[:,i] = np.sum(np.square(c[i] - x), axis=1)
                distMin = np.amin(distALL, axis=1)
                y = np.argmin(distALL, axis=1)
                J = Jnew
                Jnew = np.sum(distMin)
                iterNum += 1

                xGroup = np.concatenate((x, np.array([y]).T), axis=1)
                for i in range(self.n_cluster):
                    cPoints = xGroup[xGroup[:,-1]==i]
                    c[i] = np.sum(cPoints[:,:-1], axis=0)/len(cPoints)
            else:

                break

        centroids = c
        # DO NOT CHANGE CODE BELOW THIS LINE
        # print(centroids)
        return centroids, y, self.max_iter




class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        centroids, m, _ = KMeans(self.n_cluster, 100, 0.0001).fit(x, centroid_func)
        centroid_labels = np.zeros(self.n_cluster);

        for i in range(self.n_cluster):
            # u, indices = np.unique(np.array([y]).T[np.array([m]).T[:]==i], return_inverse=True)
            # centroid_labels[i] = u[np.argmax(np.bincount(indices))]
            centroid_labels[i] = np.bincount(np.array([y]).T[np.array([m]).T[:]==i]).argmax()
# arr = np.array([5, 4, -2, 1, -2, 0, 4, 4, -6, -1])
# u, indices = np.unique(arr, return_inverse=True)
# u[np.argmax(np.bincount(indices))]

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        print(self.centroid_labels)
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        distALL = np.zeros([N, self.n_cluster])
        for i in range(self.n_cluster):
            distALL[:,i] = np.sum(np.square(self.centroids[i] - x), axis=1)

        labels = np.zeros(N)
        for i in range(N):
            labels[i] = self.centroid_labels[np.argmin(distALL[i])]


        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors
        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)
        returns:
            numpy array of shape image.shape
    '''
    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'
    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    # TODO
    # - comment/remove the exception
    # - implement the function
    # DONOT CHANGE CODE ABOVE THIS LINE
    new_im = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            distALL = np.sum(np.square(image[i][j] - code_vectors), axis=1)
            new_im[i][j] = code_vectors[np.argmin(distALL)]
    # DONOT CHANGE CODE BELOW THIS LINE
    print(new_im)
    return new_im

# def nearest_point_value(point, centers):
#     '''
#         point: (D,)
#         centers: (K, D)
#     '''
#     array = np.asarray(centers)
#     idx = (np.abs(array - point)).argmin()
#     return centers[idx]
