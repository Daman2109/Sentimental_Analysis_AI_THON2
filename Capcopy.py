# importing all the libraries
import pandas as pd
import numpy as np
import os
import re
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import string
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import nltk
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, \
    f1_score


# checking whether gpu is available or not
def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print('GPU is available \n')
    else:
        print('GPU is not available \n')


# dataset loading
def data_file_loading(file):
    try:
        data = pd.read_csv(file, encoding='ISO-8859-1')
        return data
    except:
        print('Error in def data_file_loading')
        return None


# data spliting into training and test
def data_spliting(data):
    try:
        col1_train, col1_test, col2_train, col2_test = train_test_split(data['text'], data['sentiment'], train_size=0.8,
                                                                        random_state=None)
        train_data = pd.DataFrame({'text': col1_train, 'sentiment': col2_train})
        test_data = pd.DataFrame({'text': col1_test, 'sentiment': col2_test})
        return train_data, test_data
    except:
        print('Error in def data_spliting')
        return None, None


# data preprocessing takes place
def data_preprocessing(text):
    try:
        stop_words = stopwords.words('english')
        tknzr = TweetTokenizer()
        text = re.sub(r"http\S+|@\S+|#\S+", "", text)
        text = re.sub(r"['\u2018\u2019]", "", text)  # remove single quotes
        tokens = tknzr.tokenize(text)
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        text = " ".join(tokens)
        return text
    except:
        print('Error in def data_preprocessing')
        return None


# Assigning pipeline using tfidfVectorizer as vectorizer and naive bayes as classifier
def building_pipeline(classifierr, vectorizerr):
    try:
        vectorizer = vectorizerr(stop_words='english', token_pattern=r'\b\w+\b')
        classifier = classifierr
        pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
        return pipeline
    except:
        print('Error in def building_pipeline.')
        return None


# evaluating model and calculating accuracy
def model_evaluation(model, test_data):
    try:
        y_pred = model.predict(test_data['text'])
        accuracy = accuracy_score(test_data['sentiment'], y_pred)
        precision = precision_score(test_data['sentiment'], y_pred, average='weighted')
        recall = recall_score(test_data['sentiment'], y_pred, average='weighted')
        f1 = f1_score(test_data['sentiment'], y_pred, average='weighted')
        return accuracy, f1, precision, recall
    except:
        print("Error in def model_evaluation")
        return None, None, None, None


# select algorithm
def select_algorithm():
    print("Select the algorithm you want to use for classification:\n"

          "[1.] Multinomial Naive Bayes:  A probabilistic algorithm that assumes that the features are\n"
          "                               conditionally independent given the class label.\n \n"
          "[2.] Logistic Regression:      A statistical model that uses logistic functions to model a binary dependent variable.\n"
          "                               It can handle non-linear relationships between the features and the target variable.\n \n"
          "[3.] Support Vector Machine:   A supervised learning algorithm that uses a hyperplane to separate classes.\n \n"
          "[4.] Random Forest:            An ensemble learning method that constructs a multitude of decision trees and outputs the \n"
          "                               class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.\n \n"
          "[5.] Gradient Boosting:        An ensemble learning method that iteratively trains weak classifiers to improve the accuracy of the model.\n \n"
          "[6.] K-Nearest Neighbors:      A non-parametric algorithm that assigns a data point to the class most common \n"
          "                               among its k nearest neighbors in the feature space.\n\n"
          "[7.] Decision Tree:            A tree-like model of decisions and their possible consequences, including chance \n"
          "                               event outcomes, resource costs, and utility.\n \n"
          "[8.] MLP:                      A feedforward neural network that consists of multiple layers of neurons, each layer \n"
          "                               fully connected to the next one.\n \n"
          "[9.] XGBoost:                  An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.\n"
          "[10.] LightGBM:                A gradient boosting library that uses decision trees and gradient-boosted algorithm \n"
          "                               with a focus on high performance and memory efficiency.\n \n"
          "[11.] Adaboost:                A boosting algorithm that combines multiple weak classifiers to form a strong classifier.\n"
          "[12.] Bagging:                 A machine learning ensemble meta-algorithm designed to improve the stability and accuracy \n"
          "                               of machine learning algorithms used in statistical classification and regression.\n \n"
          "[13.] ExtraTreesClassifier:    An ensemble learning method that constructs a multitude of decision trees and outputs \n"
          "                               the class that is the mode of the classes (classification) or mean prediction (regression)\n"
          "                               of the individual trees.\n \n"
          "[14.] VotingClassifier:        A machine learning ensemble model that combines multiple models to increase accuracy.\n")

    choice = input("Enter your choice (1/2/3/4/5/6/7/8/9/10/11/12/13/14): ")
    if choice == '1':
        classifier = MultinomialNB()
    elif choice == '2':
        classifier = LogisticRegression(max_iter=1000)
    elif choice == '3':
        classifier = SVC()
    elif choice == '4':
        classifier = RandomForestClassifier()
    elif choice == '5':
        classifier = GradientBoostingClassifier()
    elif choice == '6':
        classifier = KNeighborsClassifier()
    elif choice == '7':
        classifier = DecisionTreeClassifier()
    elif choice == '8':
        classifier = MLPClassifier(max_iter=1000)
    elif choice == '9':
        classifier = xgb.XGBClassifier()
    elif choice == '10':
        classifier = LGBMClassifier()
    elif choice == '11':
        classifier = AdaBoostClassifier()
    elif choice == '12':
        classifier = BaggingClassifier()
    elif choice == '13':
        classifier = ExtraTreesClassifier()
    elif choice == '14':
        classifier = VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)),
                                                  ('rf', RandomForestClassifier()),
                                                  ('knn', KNeighborsClassifier()),
                                                  ('dt', DecisionTreeClassifier())],
                                      voting='hard')
    else:
        print("Invalid input. Please enter a number between 1 and 8.")
        return select_algorithm()
    return classifier


def select_vectorizer():
    try:
        print("Select the vectorizer you want to use:\n"
              "1.) TfidfVectorizer :\n"
              "2.) CountVectorizer :\n")

        choice = input("Enter your choice (1/2): ")
        if choice == '1':
            vectorizer = TfidfVectorizer
        elif choice == '2':
            vectorizer = CountVectorizer
        else:
            print("Invalid input. Please enter a number between 1 and 2.")
            return select_vectorizer()
        return vectorizer
    except:
        print("error in def select_vectorizer")
        return None


# selecting model based on user model
def select_model():
    print("\n \n \nSelect an algorithm from the following list for hyperparameter assigning to the model: \n"
          "1. Naive Bayes\n"
          "2. Logistic Regression\n"
          "3. SVC\n"
          "4. Random Forest\n"
          "5. Gradient Boosting Classifier\n"
          "6. KNN\n"
          "7. Decision Tree\n"
          "8. MLP\n")

    choice = input("Enter your choice (1/2/3/4/5/6/7/8): ")
    if choice == '1':
        model = MultinomialNB()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__alpha': [0.1, 1, 10, 100],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '2':
        model = LogisticRegression(max_iter=1000)
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__C': [0.1, 1, 10, 100],
            # 'classifier__penalty': ['l1','l2'],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '3':
        model = SVC()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf'],
            # 'classifier__penalty': ['l1', 'l2'],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '4':
        model = RandomForestClassifier()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, 30, None],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '5':
        model = GradientBoostingClassifier()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [2, 3, 4],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '6':
        model = KNeighborsClassifier()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__n_neighbors': [3, 5, 7],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '7':
        model = DecisionTreeClassifier()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'vectorizer__ngram_range': [(1, 1)]
        }
    elif choice == '8':
        model = MLPClassifier()
        parameters = {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__stop_words': [stopwords.words('english')],
            'classifier__hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'vectorizer__ngram_range': [(1, 1)]
        }
    else:
        print("Invalid choice. Please try again.")
        return select_model()
    return model, parameters


# model training and using grid_search finding best estimator
def model_training(pipeline, train_data, parameters):
    try:
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, return_train_score=True)
        grid_search.fit(train_data['text'], train_data['sentiment'])
        best_model = grid_search.best_estimator_

        # Plot cost function during training
        plt.plot(grid_search.cv_results_['mean_train_score'])
        plt.plot(grid_search.cv_results_['mean_test_score'])
        plt.xlabel('Number of iterations')
        plt.ylabel('Mean score')
        plt.title('Cost function during training')
        plt.legend(['Training', 'Validation'])
        plt.show()

        return best_model
    except:
        print("error in def model_training")
        return None


def plot_learning_curve(model, test_data, train_data):
    train_sizes, train_scores, test_scores = learning_curve(model, train_data['text'], train_data['sentiment'], cv=5,
                                                            scoring='accuracy', n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


# plotting confusion graph
def confusion_matrix_graph(best_model, test_data):
    y_pred = best_model.predict(test_data['text'])
    label_encoder = LabelEncoder()
    label_encoder.fit(test_data['sentiment'])
    cm = confusion_matrix(test_data['sentiment'], y_pred, labels=list(label_encoder.classes_))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def calibration_curve_plot(model, test_data):
    prob_pos = model.predict_proba(test_data['text'])[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(test_data['sentiment_encoded'], prob_pos, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.show()


# defining main function
def main():
    # starting time clock
    start_time = time.time()
    check_gpu()
    file = 'D:/python/capstone/covid19sentimentaltest100.csv'

    # calling and importing dataset provided
    data = data_file_loading(file)

    # preprocessing every text of the dataset
    data['text1'] = data['text'].apply(data_preprocessing)
  #  'NAME': str(os.path.join(BASE_DIR,"db.sqlite3"))

    # printing time of preprocessing
    end_time = time.time()
    time_diff = end_time - start_time
    print(f"Total time taken for preprocessing: {time_diff:.2f} seconds.\n")

    # calling data_spliting and importing train_data and test_data
    train_data, test_data = data_spliting(data)

    # Encode sentiment labels into numeric values
    label_encoder = LabelEncoder()
    train_data['sentiment_encoded'] = label_encoder.fit_transform(train_data['sentiment'])
    test_data['sentiment_encoded'] = label_encoder.transform(test_data['sentiment'])

    # calling pipeline function
    vectorizerr = select_vectorizer()
    algorithmm = select_algorithm()
    pipeline = building_pipeline(algorithmm, vectorizerr)

    # calling select_model
    model, parameters = select_model()
    parameters['classifier'] = [model]
    print("The vectorizer you choose is: ", vectorizerr, "\n")
    print("The algorithm you choose for classification is: ", algorithmm, "\n")
    print("The algortihm you choose for hyperparameter tuning is :", model, "\n")
    # Train the model and get the best estimator
    best_model = model_training(pipeline, train_data, parameters)

    # printing total time for total code to execute
    end_time = time.time()
    time_diff = end_time - start_time
    print(f"Total time took for full model to execute : {time_diff:.2f} seconds")

    # Evaluate the performance of the best model
    accuracy, f1, precision, recall = model_evaluation(best_model, test_data)
    print("Accuracy of the model:", accuracy)
    print("Precision of the model:", precision)
    print("Recall value of the model:", recall)
    print("F1-score of the model :", f1)

    # Plot the confusion matrix and cost function
    confusion_matrix_graph(best_model, test_data)
    plot_learning_curve(best_model, test_data, train_data)
    # calibration_curve_plot(best_model, test_data)

    while True:
        manual_input_choice = input("Do you want to enter a sentence for sentiment analysis? (y/n): ")
        if manual_input_choice.lower() == 'y':
            while True:
                input_text = input("Enter a sentence for sentiment analysis or enter 'exit' to quit:  ")
                if input_text.lower() == 'exit':
                    break
                input_text_processed = data_preprocessing(input_text)
                predicted_sentiment = best_model.predict([input_text_processed])[0]
                print(f"Predicted sentiment of '{input_text}': {predicted_sentiment}")
        elif manual_input_choice.lower() == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()

