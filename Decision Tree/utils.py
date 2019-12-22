import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    #sum(row[column] for row in data)

    num_b = len(branches)
    num_c = len(branches[0])

    t_entropy = np.zeros(num_b) # entropy of each branch
    t_num_b = [] # num of each branch
    t_weight = []
    for i in range(num_b):
        t_num_b.append(sum(branches[i]))
        t_weight.append(t_num_b[i] / sum(sum(branches,[])))
        for j in range(num_c):
            if branches[i][j] != 0:
                t_entropy[i] += - branches[i][j] / t_num_b[i] * np.log2(branches[i][j] / t_num_b[i])
    # print(t_entropy)
    # print(t_weight)
    return S - sum(np.multiply(t_entropy, t_weight))

def prune(decisionTree, curr_TN, x, y):
    for child in curr_TN.children:
        if child.splittable == True:
            prune(decisionTree, child, x, y)
        else:
            acc_not_prune = decisionTree.acc(x, y)
            # acc_not_prune = decisionTree.acc(decisionTree.root_node.features, decisionTree.root_node.labels)
            print(acc_not_prune)
            curr_TN.splittable = False
            acc_prune = decisionTree.acc(x, y)
            print(acc_prune)
            if acc_prune < acc_not_prune:
                curr_TN.splittable = True

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    # raise NotImplementedError


    if decisionTree.root_node.splittable == True:
        prune(decisionTree, decisionTree.root_node, X_test, y_test)





# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    # print(node.children[1].features)

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
