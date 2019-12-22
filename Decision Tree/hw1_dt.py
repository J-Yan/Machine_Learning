import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

    def acc(self, x_test, y_test):
        y_pred = self.predict(x_test)
        pos = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                pos += 1
        return pos / len(y_test)

class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):


        #raise NotImplementedError
        if self.splittable == False:
            # stop split and return whole tree?

            self.feature_uniq_split = []
            return self
        else:
            # call split again
            max_info_gain = 0.0 # need to be updated
            feat_len = len(self.features[0])
            feat_num = len(self.features)

            # father entropy
            S_labels_cluster = []
            for j in np.unique(self.labels):
                S_labels_cluster.append(self.labels.count(j))
            S_entropy = -Util.Information_Gain(0, [S_labels_cluster])

            # children's features, labels, classes init
            c_features = []
            c_labels = []
            c_num_cls = []
            d = {}

            # split
            flag = 0
            for i in range(feat_len):  # i different ways to split TreeNode
                # split on current dimention
                children_name_list = np.sort(np.unique(np.array(self.features).T.tolist()[i])) #sort!
                if len(children_name_list) == 1:
                    print("我佛了")
                    continue
                # make a dictionary of children's name and index

                flag = 1
                for x in range(len(children_name_list)):
                    d.update({children_name_list[x]: x})

                group = []
                group_labels = []
                children_labels_cluster = []
                for y in range(len(children_name_list)):
                    group.append([])
                    group_labels.append([])
                    children_labels_cluster.append([])

                # loop all the features, group the features and labels
                for k in range(feat_num):
                    group[d[self.features[k][i]]].append(self.features[k])
                    group_labels[d[self.features[k][i]]].append(self.labels[k])


                # branches -> Util.Information_Gain
                for r in np.unique(self.labels):
                    print("np.unique(self.labels): ", np.unique(self.labels))
                    for s in range(len(children_name_list)):
                        children_labels_cluster[s].append(group_labels[s].count(r))
                        print(r,"fff",group_labels[s])

                print("aaaaaaa")
                print(" ")
                print("S_entropy: ", S_entropy)
                curr_info_gain = Util.Information_Gain(S_entropy, children_labels_cluster)
                print("curr_info_gain: ", curr_info_gain)
                if curr_info_gain > max_info_gain:
                    self.feature_uniq_split = children_name_list
                    print("【【【【【【")
                    max_info_gain = curr_info_gain
                    c_features = group
                    print("group^^^^^^^^^^^^^^^^^^^^^^^^: ", c_features)
                    c_labels = group_labels
                    print("group_labels^^^^^^^^^^^^^^^^^: ", c_labels)
                    for c in range(len(children_name_list)):
                        c_num_cls.append(len(np.unique(group_labels[c])))
                    self.dim_split = i
                if (curr_info_gain == max_info_gain)and(curr_info_gain != 0):
                    if len(children_name_list) > len(np.sort(np.unique(np.array(self.features).T.tolist()[self.dim_split]).tolist())):
                        self.feature_uniq_split = children_name_list

                        max_info_gain = curr_info_gain
                        c_features = group
                        c_labels = group_labels

                        for c in range(len(children_name_list)):
                            c_num_cls.append(len(np.unique(group_labels[c])))
                        self.dim_split = i
                print("】】】】】】】")
            # new the children TreeNodes

            if max_info_gain == 0.0:
                self.splittable = False
                self.feature_uniq_split = []

            else:
                for i in range(len(c_features)):
                    # print("赋值: ", len(c_features))
                    c_TN = TreeNode(c_features[i], c_labels[i], c_num_cls[i])
                    # print("c_features[i]: ", c_features[i])
                    # print("c_labels[i]: ", c_labels[i])
                    # print("c_num_cls[i]: ", c_num_cls[i])
                    # print("TreeNode: ", c_TN.features)
                    # print(" ")
                    self.children.append(c_TN.split())
                    # print("self.children: ", self.children)
            return self
    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable == False:
            return self.cls_max
        else:
            index = -1
            for i in range(len(self.feature_uniq_split)):
                if self.feature_uniq_split[i] == feature[self.dim_split]:
                    index = i
            if index != -1:
                return self.children[index].predict(feature)
            else:
                return self.cls_max
