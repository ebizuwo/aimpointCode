import os
import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm

class Reader:
    """
    Class is used to read file
    """
    def __init__(self):
        """

        """
        self.df = None
        self.fname = "cars.csv"
        self.fpath = None

    def __read_file_skip_row(self):
        """
        Reads file in dataframe, skips uneccessary rows, sets delimiter
        modifes attribute df
        :return: void
        """
        data = pd.read_csv(self.fname, skiprows=[1], delimiter=";") #TODO: memory concern for var
        self.df = data

    def init(self):
        self.__read_file_skip_row()


class DataManipulations:
    def __init__(self):
        pass

    def parse_make_from_car(self, df: pd.DataFrame):
        """
        :param pd.Dataframe
        :return: pd.Dataframe same dataframe with Make added
        """
        # key assumption here
        # first word in 'Car' column is the 'Make' with regex
        # if more complicated then build out function or class to describe Make
        df["Make"] = df.Car.str.extract("(^\w+)", expand=True)
        return df

    def drop_columns(self, data: pd.DataFrame, cols_to_drop=()):
        return data.drop(cols_to_drop, axis=1)


class MPGPredictor:
    def __init__(self, data: pd.DataFrame, features_to_keep=None, use_pca=False, use_k_principal_components=3):
        """

        :param data:
        :param features_to_drop: Manual selection of features to keep to see how models perform
        :param use_k_principal_components: How many principal components to use
        """
        # random state used for models
        self.rs = 42
        # raw data
        self.data = data
        # what we want to predict
        self.y = None
        # our base set of features
        self.X = None

        self.dm = DataManipulations()

        self.labelEncoders = None
        self.data_modified = self.data.copy()

        # models we would like to use
        self.linreg = None

        #principal components
        self.principal_components = None

        # preprocess data first
        self.preprocess_data()
        # perform PCA
        self.perform_PCA_on_modified_data()
        # minor logic for feature selection
        if features_to_keep:
            if self.X:
                self.X = self.X[features_to_keep]
        if use_pca:
            components_to_keep = self.get_k_principal_components(use_k_principal_components)
            print(components_to_keep)
            if self.X is not None:
                self.X = self.X[components_to_keep]

    def preprocess_data(self):
        """
        Modifies class attributes and preprocess data for data modified
        accesses internal methods for preprocessing
        """
        # adding Make column since it is a potentially useful feature
        self.data_modified = self.dm.parse_make_from_car(self.data)

        # drop car column because we dont need it really
        self.data_modified = self.dm.drop_columns(self.data_modified, cols_to_drop=('Car'))

        # add dummy vars for model and analysis
        self.__convert_str_to_dummy_var()

        # set our X and y
        self.X = self.data_modified[self.data_modified.columns.difference(["MPG"])]
        self.y = self.data_modified["MPG"]

    def __convert_str_to_dummy_var(self, cols_to_encode=('Make', 'Origin')):
        """
        Modifies class attribute for data_modified
        Using scipy label encoder to just make my life easier
        :param optionally can include more columns (not Exception safe)
        :return void
        """

        # may want to access label encoder in future
        labelEncoders = dict()

        # using label encoder on provided columns
        for col in cols_to_encode:
            le = preprocessing.LabelEncoder()
            le.fit(self.data_modified[col])
            # add label encoder to labelEncoders
            labelEncoders[col]=le
            # modify columns in loop
            self.data_modified[col] = le.transform(self.data_modified[col])

        # update class attribute for encoded labels
        self.labelEncoders = labelEncoders

    def get_k_principal_components(self, k):
        # some improvement can be done here by sorting etc
        # assumes orientation of dictionary is desc order of singular values
        return list(self.principal_components.keys())[:k]

    def perform_PCA_on_modified_data(self):
        pca = PCA()
        pca.fit(self.X)
        cols = self.X.columns
        singular_values = pca.singular_values_
        column_sv = dict(zip(cols, singular_values))
        self.principal_components = column_sv

    def get_test_train_split_data(self):
        X_train, X_test,y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, random_state=self.rs)
        return X_train, y_train, X_test, y_test

    # would maybe need some more time to look into this
    def perform_grid_search(self):
        linreg = LinearRegression()
        logreg = LogisticRegression()
        clf = GridSearchCV()

    # regression and model fitting prediction functions below
    def fit_simple_linear_reg(self):
        linreg = LinearRegression()
        X_train, y_train, X_test, y_test = self.get_test_train_split_data()
        linreg.fit(X_train, y_train)
        score = linreg.score(X_test, y_test)
        print(f"LinearRegression Fit \n R score: {score}")
        self.linreg = linreg

    def predict_line_reg(self, X):
        """
        :param X: matrix of features
        :return: predicted MPG
        """
        return self.linreg.predict(X)


class AnswerBody:
    def __init__(self, reader):
        self.reader = reader
        self.ans_one = None
        self.ans_two = None
        self.ans_three = None
        self.ans_four = None
        pass

    def get_answer_one(self):
        data = self.reader.df
        # finding car with highest MPG
        max_mpg = data.MPG.max()
        # find cars that have this MPG could be multiple
        cars_with_max_MPG = data[data.MPG == max_mpg]
        # add some logic to handle multiples
        car_with_max_MPG = cars_with_max_MPG.Car.values[0]
        return car_with_max_MPG

    def get_answer_two(self):
        data = self.reader.df
        # select the columns we need
        df = data[['MPG', 'Cylinders']]
        # groupby and find average
        df = df.groupby(['Cylinders']).mean()
        return df

    def parse_make_from_car(self, df: pd.DataFrame):
        """
        :param pd.Dataframe
        :return: pd.Dataframe same dataframe with Make added
        """
        # key assumption here
        # first word in 'Car' column is the 'Make' with regex
        # if more complicated then build out function or class to describe Make
        df["Make"] = df.Car.str.extract("(^\w+)", expand=True)
        return df.copy()

    def get_answer_three(self):
        data = self.reader.df
        # getting the only columns we need
        df = data[['Car', 'MPG']]
        # add make column to data frame
        df = self.parse_make_from_car(df)
        # figure out the MPG by make
        df = df[['Make', 'MPG']].groupby(['Make']).mean()
        return df

    def get_answer_four(self):
        """
        This function is a function to easily call the answer needed
        :return: Model: sklearn Model
        """
        return True

    def __repr__(self):
        """
        add repr for printing
        :return:
        """
        pass

# read data
rd = Reader()
rd.init()

mpgPred = MPGPredictor(rd.df)
mpgPred.fit_simple_linear_reg()

# answers
# ans = AnswerBody(rd)
# ans.get_answer_one()
# ans.get_answer_two()
# ans.get_answer_three()