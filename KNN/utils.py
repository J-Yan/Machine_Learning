import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    ##raise NotImplementedError

    i = 0; fn = 0; tp = 0; tn = 0; fp = 0
    while i < len(real_labels):
        if real_labels[i] == predicted_labels[i]:
            # true positive
            if real_labels[i] == 1:
                tp += 1
            # true negative
            else:
                tn += 1
        else:
            # false positive
            if real_labels[i] == 0:
                fp += 1
            # false negative
            else:
                fn += 1
        i += 1
    if tp + fp == 0:
        if tp != 0:
            P = 1
        else:
            P = 0
    else:
        P = tp / (tp + fp)
    if tp + fn == 0:
        if tp != 0:
            R = 1
        else:
            R = 0
    else:
        R = tp / (tp + fn)
    if P + R == 0:
        return 0
    else:
        return 2*P*R/(P+R)



class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ##raise NotImplementedError

        s = np.sum(np.power(np.absolute(np.subtract(np.asarray(point1), np.asarray(point2))),3))
        #print(sum)

        return np.power(s, 1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ##raise NotImplementedError

        s = np.sum(np.power(np.subtract(np.asarray(point1), np.asarray(point2)),2))
        return np.power(s,1/2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ##raise NotImplementedError

        return np.sum(np.multiply(np.asarray(point1), np.asarray(point2)))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ##raise NotImplementedError

        dotProd = np.sum(np.multiply(np.asarray(point1), np.asarray(point2)))
        xNorm = np.power(np.sum(np.multiply(np.asarray(point1), np.asarray(point1))),1/2)
        yNorm = np.power(np.sum(np.multiply(np.asarray(point2), np.asarray(point2))),1/2)
        return 1-dotProd/xNorm/yNorm

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ##raise NotImplementedError

        sum2 = np.sum(np.power(np.subtract(np.asarray(point1), np.asarray(point2)),2))
        return -np.exp(-1/2*sum2)



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        f1_best = 0;
        for k in range(1, 5, 2):

            for d in range(5):
                #print("d:",d)
                knn_model = KNN(k, list(distance_funcs.values())[d])
                knn_model.train(x_train, y_train)
                f1 = f1_score(knn_model.predict(x_val), y_val)
                if f1 > f1_best:
                    print(f1)
                    self.best_k = k
                    self.best_distance_function = list(distance_funcs.keys())[d]
                    self.best_model = knn_model
                    f1_best = f1

        # You need to assign the final values to these variables
        # self.best_k = None
        # self.best_distance_function = None
        # self.best_model = None
        ##raise NotImplementedError



    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        f1_best = 0;

        for k in range(1, 10, 2):
            norm = list(scaling_classes.values())[0]()
            x_train_n1 = norm(x_train)
            print(k,"###########",x_train[0], x_train_n1[0])
            x_val_n1 = norm(x_val)
            for d in range(5):
                #print("d:",d)
                knn_model = KNN(k, list(distance_funcs.values())[d])
                knn_model.train(x_train_n1, y_train)
                f1 = f1_score(knn_model.predict(x_val_n1), y_val)
                if f1 > f1_best:
                    print(f1)
                    self.best_k = k
                    self.best_distance_function = list(distance_funcs.keys())[d]
                    self.best_scaler = list(scaling_classes.keys())[0]
                    self.best_model = knn_model
                    f1_best = f1

            mm = list(scaling_classes.values())[1]()
            x_train_n2 = mm(x_train)
            print(k,"@@@@@@@@@",x_train[0], x_train_n2[0])
            x_val_n2 = mm(x_val)
            for d in range(5):
                #print("d:",d)
                knn_model = KNN(k, list(distance_funcs.values())[d])
                knn_model.train(x_train_n2, y_train)
                f1 = f1_score(knn_model.predict(x_val_n2), y_val)
                if f1 > f1_best:
                    print(f1)
                    self.best_k = k
                    self.best_distance_function = list(distance_funcs.keys())[d]
                    self.best_scaler = list(scaling_classes.keys())[1]
                    self.best_model = knn_model
                    f1_best = f1

        # You need to assign the final values to these variables
        # self.best_k = None
        # self.best_distance_function = None
        # self.best_scaler = None
        # self.best_model = None
        #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError

        features_n = []
        for i in range(len(features)):
            n = np.power(np.sum(np.multiply(np.asarray(features[i]), np.asarray(features[i]))),1/2)
            if n == 0:
                features_n.append(features[i])
            else:
                features_n.append(features[i]/n)
        return features_n


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first = 0
        self.min = []
        self.max = []
        self.min_max = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError


        fea_len = len(features[0])
        if self.first == 0: # first call on training data
            self.min = features[0].copy()
            self.max = features[0].copy()
            self.min_max = np.zeros(fea_len)
            for i in range(len(features)):
                for j in range(fea_len):
                    if features[i][j] < self.min[j]:
                        self.min[j] = features[i][j]
                    elif features[i][j] > self.max[j]:
                        self.max[j] = features[i][j]
                    self.min_max[j] = self.max[j] - self.min[j]
            self.first = 1
        features_n = features.copy()
        for i in range(len(features)):
            for j in range(fea_len):
                if self.min_max[j] == 0:
                    features_n[i][j] = self.min[j]
                else:
                    features_n[i][j] = (features[i][j] - self.min[j]) / self.min_max[j]

        return features_n
