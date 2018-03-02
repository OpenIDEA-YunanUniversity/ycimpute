
import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from abc import ABCMeta

class DecisionTree(metaclass=ABCMeta):
    """
    use CART tree
    """
    def __init__(self,
                 lose_func=None,
                 max_depth=None,
                 min_sample_split=5,
                 min_cost=None,
                 is_forest=False
                 ):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_cost = min_cost
        self.is_forest = is_forest
        self.lose_func = lose_func
        self.num_samples = None

        if isinstance(self, RegressionTree):
            self.lose_func = self._mse
        elif isinstance(self, ClassifyTree):
            self.lose_func = self._gini_index

    def _mse(self, y):
        """
        MSE(mean-square error) see https://
        :param y: ndarray, a vector like array
        :return: the mse value of y, flaot
        """
        if (y.size == 0):
            return  0
        c_m = np.mean(y)
        diff = np.abs(c_m-y)
        mse = np.square(diff).sum()
        return mse

    def _gini_index(self, pure_y):
        """
        GINI INDEX see: https://
        :param pure_y: ndarray, vector like
        :return:flaot
        """
        dist = np.empty(np.unique(pure_y).shape)
        for lable in range(dist.shape[0]):
            dist[lable] = np.sum(pure_y==lable) / pure_y.shape[0]
        sub_feature_gini = 1.0-np.sum(np.square(dist))
        return abs(pure_y.shape[0]/self.num_samples)*sub_feature_gini

    def _entropy(self):
        """
        up until now, cart tree do not necessary need entropy except ID3 or C4.5
        :return:
        """
        pass

    def cost_reduction(self, data_left, data_right):
        y_total = np.hstack((data_left[1], data_right[1]))
        total_norm = la.norm(y_total)
        left_norm = la.norm(data_left[1])
        right_norm = la.norm(data_right[1])

        total_cost = self.lose_func(y_total)
        normalized_left = (left_norm / total_norm) * self.lose_func(data_left[1])
        normalized_right = (right_norm / total_norm) * self.lose_func(data_right[1])

        return total_cost - (normalized_left + normalized_right)

    def choose_best_feature(self, X, y, node):
        split_threshold = None
        split_feature = None
        min_gini_index = None

        real_features = range(X.shape[1])
        self.num_samples = X.shape[0]
        if self.is_forest:
            if isinstance(self, RegressionTree):
                features = np.random.choice(real_features, size=int(X.shape[1]/3))
            else:
                features = np.random.choice(real_features, size=int(np.sqrt(X.shape[1])))

        else:
            features = real_features


        for feature in features:
            for sub_feature in np.unique(X[:, feature]):
                left = y[X[:, feature]==sub_feature]
                right = y[X[:, feature]!= sub_feature]
                gini_index = self.lose_func(left)+self.lose_func(right)
                if min_gini_index is None or gini_index<min_gini_index:
                    split_threshold = sub_feature
                    split_feature = feature
                    min_gini_index = gini_index

        node.threshold = split_threshold
        node.feature = split_feature
        low_mask = X[:, split_feature] == split_threshold
        high_mask = X[:, split_feature] != split_threshold

        return (X[low_mask],y[low_mask]),(X[high_mask],y[high_mask])

    def stop_split(self, left_data, right_data, depth):
        if self.max_depth and depth > self.max_depth:
            return True
        if not isinstance(self, ClassifyTree) and \
                self.cost_reduction(left_data, right_data)<self.min_cost:
            return True
        if left_data[0].size<self.min_sample_split or right_data[0].size<self.min_sample_split:
            return True

        return False

    def test_purity(self, y):
        """
        Tests labels in node to see if they are all the same

        Parameters
        ----------
        y : current labels in the node

        Returns
        -------
        true or false, indicating whether all labels are the same
        """

        common = stats.mode(y)[0][0]
        return np.sum(y == common) == y.size

    def grow_tree(self, node, X, y, depth):
        """
        recursion building decision tree
        """
        if isinstance(self, RegressionTree):
            node.mean_dist = np.mean(y)
        else:
            node.mean_dist = common = stats.mode(y)[0][0]

        if y.size < 2:
            return node
        if isinstance(self, ClassifyTree) and self.test_purity(y):
            return node

        data_left, data_right = self.choose_best_feature(X, y, node)
        if self.stop_split(data_left, data_right, depth):
            return node

        left = DecisionNode()
        right = DecisionNode()
        node.left = self.grow_tree(left,
                                   data_left[0],
                                   data_left[1],
                                   depth+1)
        node.right = self.grow_tree(right,
                                    data_right[0],
                                    data_right[1],
                                    depth+1)

        return node

    def single_prediction(self, x, node):
        if x[node.feature] is None or (not node.left and not node.right):
            return node.mean_dist

        go_left = x[node.feature] <= node.threshold

        if (go_left and node.left):
            return self.single_prediction(x, node.left)
        if (not go_left and node.right):
            return self.single_prediction(x, node.right)
        return node.mean_dist


    def fit(self, X, y):
        node = DecisionNode()
        self.root = self.grow_tree(node, X, y, 0)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, observation in enumerate(X):
            predictions[i] = self.single_prediction(observation, self.root)
        return predictions









class RegressionTree(DecisionTree):
    def __init__(self,
                 max_depth=None,
                 min_size=5,
                 min_cost=0,
                 in_forest=False):
        """
        Parameters
        ----------
        max_depth : maximum depth of tree
        min_size : minimum size of the data being split
        min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
        in_forest : specifies whether tree will be a part of a random forest
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.min_cost = min_cost
        self.in_forest = in_forest
        super(RegressionTree, self).__init__(
            min_sample_split=self.min_size,
            min_cost=self.min_cost,
            is_forest=self.in_forest)


class ClassifyTree(DecisionTree):
    def __init__(self,
                 max_depth=None,
                 min_size=1,
                 min_cost=0,
                 in_forest=False):
        """
        Parameters
        ----------
        max_depth : maximum depth of tree
        min_size : minimum size of the data being split
        in_forest : specifies whether tree will be a part of a random forest
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.min_cost = min_cost
        self.in_forest = in_forest
        super(ClassifyTree, self).__init__(
            max_depth=self.max_depth,
            min_sample_split=self.min_size,
            min_cost=self.min_cost,
            is_forest=self.in_forest)


class DecisionNode():
    """
    Represents a single node in the binary decision tree that will be built

    Attributes
    ----------
    threshold : Value that determines how the data is split
    mean_dist : If the node is in a regression tree, this will be the mean of the
    values in this node. If the node is in a classification tree, this will be the
    distribution of classes in this node
    feature : the feature to split the data on based on the threshold
    type : specifies the type of node, can either be regression node or classification node
    left_child : the left child of this node in the decision tree
    right_child : the right child of this node in the decision tree
    """

    def __init__(self, threshold=None, mean_dist=None, feature=None):
        """
        Initiliazes Node using data

        Parameters
        ----------
        threshold : Value that determines how the data is split
        mean_dist : If the node is in a regression tree, this will be the mean of the
        values in this node. If the node is in a classification tree, this will be the
        distribution of classes in this node
        feature : the feature to split the data on based on the threshold
        """

        self.threshold = threshold
        self.mean_dist = mean_dist
        self.feature = feature
        self.right = None
        self.left = None







class DecisionNode():
    def __init__(self,
                 threshold=None,
                 mean_dist=None,
                 feature=None):

        self.threshold = threshold
        self.mean_dist = mean_dist
        self.feature = feature
        self.right = None
        self.left = None

