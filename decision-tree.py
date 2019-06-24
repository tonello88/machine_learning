
#from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import pandas
import numpy as np
import os

#os.chdir('C:/Users/thimo/Dropbox/corsi/machine_learning/real-world-machine-learning-master')
os.chdir('./')
data = pandas.read_csv("data/titanic.csv")
data[:5]

data_train = data[:int(0.8*len(data))]
data_test = data[int(0.8*len(data)):]

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
def cat_to_num(data):
    categories = np.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s=%s" % (data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)
    
def prepare_data(data):
    """Takes a dataframe of raw data and returns ML model features
    """
    
    # Initially, we build a model only on the available numerical values
    features = data.drop(["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    
    # Setting missing age values to -1
    features["Age"] = data["Age"].fillna(-1)
    
    # Adding the sqrt of the fare feature
    features["sqrt_Fare"] = np.sqrt(data["Fare"])
    
    # Adding gender categorical value
    features = features.join( cat_to_num(data['Sex']) )
    
    # Adding Embarked categorical value
    features = features.join( cat_to_num(data['Embarked'].fillna("")) )
    
    return features
    

    
features = prepare_data(data_train)
features[:5]
model = tree.DecisionTreeClassifier(max_depth = 4)
model.fit(features, data_train["Survived"])

#print model.decision_path(features)

tree_to_code(model, features.columns)
print(model.score(prepare_data(data_test), data_test["Survived"]))
tree.export_graphviz(model, out_file="titanic_tree.dot")
# convert with  
# !dot -Tpng titanic_tree.dot -o titanic_tree.png -Gdpi=600
# "c:\Program Files (x86)\Graphviz2.38"\bin\dot -Tpng titanic_tree.dot -o titanic_tree.png -Gdpi=600

