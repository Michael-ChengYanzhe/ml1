import numpy as np
import pandas as pd
import time
import graphviz
import pydotplus

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def main():

	rawData = pd.read_csv('pulsar_stars.csv')
	features = list(rawData)[:-1]
	target = rawData.iloc[:,-1]
	data = rawData.loc[:, rawData.columns != "target_class"]

	xTrain, xTest, yTrain, yTest = train_test_split(data, target, test_size = 0.25, random_state = 42)
	decisionTree(xTrain, xTest, yTrain, yTest, features)
	svm(xTrain, xTest, yTrain, yTest)
	nn(xTrain, xTest, yTrain, yTest)
	boosting(xTrain, xTest, yTrain, yTest)
	knn(xTrain, xTest, yTrain, yTest)




def decisionTree(xTrain, xTest, yTrain, yTest, features):
	print("_____START Decision Tree_____\n")
	clf = tree.DecisionTreeClassifier()
	param_test = {
        'max_depth': [1,3,5,10,15,20],
        'min_samples_leaf': [2,5,10,20,50,100],
    }
	gs_classifier = GridSearchCV(clf, param_test, cv = 10)
	gs_classifier.fit(xTrain, yTrain)
	print("Overall Result:\n")
	print(gs_classifier.cv_results_)

	print("DT_Best_Parameter:")
	print(gs_classifier.best_params_)
	print("Best Score:")
	print(gs_classifier.best_score_)
	print("Choose this para and predict\n")
	best_dt_clf = tree.DecisionTreeClassifier(**gs_classifier.best_params_)
	best_dt_clf.fit(xTrain, yTrain)

	yPredict = best_dt_clf.predict(xTest)
	accuracy_dt_best = metrics.accuracy_score(yTest, yPredict)

	######Out Put########
	print("Accuracy:")
	print(accuracy_dt_best)
	print("Recall:")
	print(metrics.recall_score(yTest, yPredict, average = 'weighted'))
	print("F1:")
	print(metrics.f1_score(yTest, yPredict, average = 'weighted'))
	print("Precision:")
	print(metrics.precision_score(yTest, yPredict, average = 'weighted'))

	dot_data = tree.export_graphviz(best_dt_clf, out_file=None, 
                         feature_names = features,  
                         filled = True, 
                         rounded=True,  
                         special_characters=True)
	#graph = graphviz.graph_from_dot_data(dot_data)
	graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
	graph.write_png("dt_overview.png")

	plot_clf = tree.DecisionTreeClassifier(**gs_classifier.best_params_)
	plt = plot_learning_curve(plot_clf,"Decision Tree", "dt_", xTrain, yTrain)
	print("_____END Decision Tree_____\n")



def svm(xTrain, xTest, yTrain, yTest):
	print("_____START SVM_____\n")
	scaling = MinMaxScaler(feature_range=(-1,1)).fit(xTrain)
	xTrain = scaling.transform(xTrain)
	xTest = scaling.transform(xTest)

	clf = SVC()
	# param_test = [{'kernel': ['linear']},
	# 				{'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
	# 				{'kernel': ['poly'], 'degree': [2,3,5],'gamma': [0.001, 0.0001]}]
	param_test = [{'kernel': ['linear']},
					{'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
					{'kernel': ['poly'], 'degree': [2,3,5],'gamma': [0.001, 0.0001]}]
	# Cs = [0.001, 0.01, 0.1, 1, 10]
	# gammas = [0.001, 0.01, 0.1, 1]
	# gammas = [0.001, 0.01]
	# kernels = ['linear', 'poly', 'rbf']
	# param_test = {
	# 	# 'C': Cs,
	# 	'gamma': gammas,
	# 	'kernel': kernels
	# }
	gs_classifier = GridSearchCV(clf, param_test, cv = 10)
	gs_classifier.fit(xTrain, yTrain)
	print("Overall Result:\n")
	print(gs_classifier.cv_results_)

	print("DT_Best_Parameter:")
	print(gs_classifier.best_params_)
	print("Best Score:")
	print(gs_classifier.best_score_)
	print("Choose this para and predict\n")
	best_dt_clf = SVC(**gs_classifier.best_params_)
	best_dt_clf.fit(xTrain, yTrain)

	yPredict = best_dt_clf.predict(xTest)
	accuracy_dt_best = metrics.accuracy_score(yTest, yPredict)

	######Out Put########
	print("Accuracy:")
	print(accuracy_dt_best)
	print("Recall:")
	print(metrics.recall_score(yTest, yPredict, average = 'weighted'))
	print("F1:")
	print(metrics.f1_score(yTest, yPredict, average = 'weighted'))
	print("Precision:")
	print(metrics.precision_score(yTest, yPredict, average = 'weighted'))


	plot_clf = SVC(**gs_classifier.best_params_)
	plt = plot_learning_curve(plot_clf,"SVM", "svm_", xTrain, yTrain)
	print("_____END SVM_____\n")

# def nn(xTrain, xTest, yTrain, yTest):
# 	print("_____START NN_____\n")
# 	scaling = MinMaxScaler(feature_range=(-1,1)).fit(xTrain)
# 	xTrain = scaling.transform(xTrain)
# 	xTest = scaling.transform(xTest)
# 	mlp = MLPClassifier(hidden_layer_sizes = (5, 5, 2), max_iter = 2000)
# 	plt = plot_learning_curve(mlp,"MLP", "mlp552_", xTrain, yTrain)

# 	mlp = MLPClassifier(hidden_layer_sizes = (5, 5, 5, 2), max_iter = 2000)
# 	plt = plot_learning_curve(mlp,"MLP", "mlp5552_", xTrain, yTrain)

# 	mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 2), max_iter = 2000)
# 	plt = plot_learning_curve(mlp,"MLP", "mlp10102_", xTrain, yTrain)
# 	print("_____END NN_____\n")

def nn(xTrain, xTest, yTrain, yTest):
	print("_____START NN_____\n")
	scaling = MinMaxScaler(feature_range=(-1,1)).fit(xTrain)
	xTrain = scaling.transform(xTrain)
	xTest = scaling.transform(xTest)
	# mlp = MLPClassifier(hidden_layer_sizes = (5, 5), max_iter = 2000)
	# plt = plot_learning_curve(mlp,"MLP", "mlp55_", xTrain, yTrain)

	mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), max_iter = 2000)
	# plt = plot_learning_curve(mlp,"MLP", "mlp101010_", xTrain, yTrain)

		######Out Put########
	mlp.fit(xTrain, yTrain)

	yPredict = mlp.predict(xTest)
	accuracy_dt_best = metrics.accuracy_score(yTest, yPredict)
	print("Accuracy:")
	print(accuracy_dt_best)
	print("Recall:")
	print(metrics.recall_score(yTest, yPredict, average = 'weighted'))
	print("F1:")
	print(metrics.f1_score(yTest, yPredict, average = 'weighted'))
	print("Precision:")
	print(metrics.precision_score(yTest, yPredict, average = 'weighted'))

	# mlp = MLPClassifier(hidden_layer_sizes = (10, 10), max_iter = 2000)
	# plt = plot_learning_curve(mlp,"MLP", "mlp1010_", xTrain, yTrain)

	# mlp = MLPClassifier(hidden_layer_sizes = (50, 50,), max_iter = 2000)
	# plt = plot_learning_curve(mlp,"MLP", "mlp5050_", xTrain, yTrain)
	print("_____END NN_____\n")

def boosting(xTrain, xTest, yTrain, yTest):
	print("_____START Boosting_____\n")

	dtc = tree.DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 5)
	clf = AdaBoostClassifier(base_estimator = dtc)
	param_test ={"n_estimators": [1, 5, 10, 50, 100, 200, 500]}

	gs_classifier = GridSearchCV(clf, param_test, cv = 10)
	gs_classifier.fit(xTrain, yTrain)
	print("Overall Result:\n")
	print(gs_classifier.cv_results_)

	print("Boosting_Best_Parameter:")
	print(gs_classifier.best_params_)
	print("Best Score:")
	print(gs_classifier.best_score_)
	print("Choose this para and predict\n")
	best_bst_clf = AdaBoostClassifier(base_estimator = dtc, **gs_classifier.best_params_)
	best_bst_clf.fit(xTrain, yTrain)

	yPredict = best_bst_clf.predict(xTest)
	accuracy_dt_best = metrics.accuracy_score(yTest, yPredict)

	######Out Put########
	print("Accuracy:")
	print(accuracy_dt_best)
	print("Recall:")
	print(metrics.recall_score(yTest, yPredict, average = 'weighted'))
	print("F1:")
	print(metrics.f1_score(yTest, yPredict, average = 'weighted'))
	print("Precision:")
	print(metrics.precision_score(yTest, yPredict, average = 'weighted'))


	plot_clf = AdaBoostClassifier(**gs_classifier.best_params_)
	plt = plot_learning_curve(plot_clf,"Boosting", "bst_", xTrain, yTrain)
	print("_____END Boosting_____\n")

def knn(xTrain, xTest, yTrain, yTest):
	print("_____START KNN_____\n")
	scaling = MinMaxScaler(feature_range=(-1,1)).fit(xTrain)
	xTrain = scaling.transform(xTrain)
	xTest = scaling.transform(xTest)

	clf = KNeighborsClassifier()
	param_test ={"n_neighbors": [1, 3, 5, 10, 15, 20, 50]}

	gs_classifier = GridSearchCV(clf, param_test, cv = 10)
	gs_classifier.fit(xTrain, yTrain)
	print("Overall Result:\n")
	print(gs_classifier.cv_results_)

	print("KNN_Best_Parameter:")
	print(gs_classifier.best_params_)
	print("Best Score:")
	print(gs_classifier.best_score_)
	print("Choose this para and predict\n")
	best_knn_clf = KNeighborsClassifier(**gs_classifier.best_params_)
	best_knn_clf.fit(xTrain, yTrain)

	yPredict = best_knn_clf.predict(xTest)
	accuracy_knn_best = metrics.accuracy_score(yTest, yPredict)

	######Out Put########
	print("Accuracy:")
	print(accuracy_knn_best)
	print("Recall:")
	print(metrics.recall_score(yTest, yPredict, average = 'weighted'))
	print("F1:")
	print(metrics.f1_score(yTest, yPredict, average = 'weighted'))
	print("Precision:")
	print(metrics.precision_score(yTest, yPredict, average = 'weighted'))


	plot_clf = KNeighborsClassifier(**gs_classifier.best_params_)
	plt = plot_learning_curve(plot_clf,"KNN", "knn_best_", xTrain, yTrain)

	neighbors_num = [1, 3, 5, 10, 15, 20, 50]
	print("_______KNN_______")
	for k in neighbors_num:
		best_knn_clf = KNeighborsClassifier(n_neighbors = k)
		best_knn_clf.fit(xTrain, yTrain)

		yPredict = best_knn_clf.predict(xTest)
		accuracy = metrics.accuracy_score(yTest, yPredict)
		print("k="+str(k))
		print("Accuracy="+str(accuracy))

		plot_clf = KNeighborsClassifier(n_neighbors = k)
		plt = plot_learning_curve(plot_clf,"KNN_"+str(k), str(k)+"nn_", xTrain, yTrain)





	print("_____END KNN_____\n")






def plot_learning_curve(clf, title, file_name, xTrain, yTrain,
                        n_jobs=None, train_sizes=np.linspace(0.01, 1.0, 50)):

	plt.figure()
	plt.title(title + 'Learning Curve')

	train_sizes, train_scores, test_scores = learning_curve(
		clf, xTrain, yTrain, cv=10, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="deepskyblue")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
	plt.plot(train_sizes, train_scores_mean, '^-', color="deepskyblue", label="Training score")
	plt.plot(train_sizes, test_scores_mean, '^-', color="orange", label="Cross-validation score")
	plt.xlabel("Training Size")
	plt.ylabel("Accuracy Score")
	plt.legend(loc="best")
	plt.savefig('ex1_' + file_name + 'curve.png')
	plt.tight_layout()


	return plt





if __name__ == "__main__":
	main()
	
