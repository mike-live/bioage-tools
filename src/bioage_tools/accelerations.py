from abc import ABCMeta, abstractmethod
from regression_bias_corrector.linear_corrector import LinearBiasCorrector
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class AnyModel(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, X):
        pass

class AccModel:
    def __init__(
        self, bioage_model: AnyModel,
        need_fit=False, bias_correction=False, 
        bias_correction_regressor='linear', 
        bias_correction_regressor_params=None
    ):
        self.model = bioage_model
        self.linreg_model = None
        self.bias_correction = bias_correction
        self.need_fit = need_fit
        self.bias_correction_regressor = bias_correction_regressor
        self.bias_correction_regressor_params = bias_correction_regressor_params or {}

        if self.bias_correction:
            self.corrector = LinearBiasCorrector(
                regressor_type=bias_correction_regressor,
                regressor_params=bias_correction_regressor_params  
            )

    def _prepare_data(self, X: pd.DataFrame, ages: pd.Series):
        if type(X) is pd.DataFrame and type(ages) is pd.Series:
            ages = ages[X.index]
        return X, ages

    def fit(self, X: pd.DataFrame, ages: pd.Series):
        X, ages = self._prepare_data(X, ages)
        if self.need_fit:
            self.model.fit(X, ages)
        bioages = self.model.predict(X)
        if self.bias_correction:
            self.corrector.fit(ages, bioages)
            bioages = self.corrector.predict(bioages)
        ages = np.asarray(ages)  #  Универсальное приведение к NumPy array
        self.linreg_model = LinearRegression().fit(ages.reshape(-1, 1), bioages)    
        #self.linreg_model = LinearRegression().fit(ages.values.reshape(-1, 1), bioages)
        return self

    def predict(self, X: pd.DataFrame):
        bioages = self.model.predict(X)
        if self.bias_correction:
            bioages = self.corrector.predict(bioages)
        return bioages

    def predictacc(self, X: pd.DataFrame, ages: pd.Series):
        X, ages = self._prepare_data(X, ages)
        bioages = self.predict(X)
        acc = bioages - self.linreg_model.predict(ages.values.reshape(-1, 1))
        return acc

    def __str__(self):
        return str(self.model) + ('\n' + str(self.corrector) if hasattr(self, 'corrector') else '')

    def plot_predict(self, X: pd.DataFrame, ages: pd.Series, seed=42):
        X, ages = self._prepare_data(X, ages)
        yp = self.predict(X)
        y1 = np.asarray(ages)  #  Универсальное приведение к NumPy array
        yl = self.linreg_model.predict(y1.reshape(-1, 1))
        #yl = self.linreg_model.predict(ages.values.reshape(-1, 1))

        import matplotlib.pyplot as plt
        plt.scatter(ages, yp)
        plt.plot(ages, yl, color='red')
        plt.plot(ages, ages, color='gray')
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta',
                  'gold', 'indigo', 'maroon', 'navy', 'teal']
        np.random.seed(seed)
        for j in range(15):
            i = np.random.choice(len(ages))
            plt.arrow(np.array(ages)[i], yp[i], 0, -yp[i]+yl[i],
                      head_width=0.2, head_length=0.3, fc='k', ec='k')
            plt.scatter(np.array(ages)[i], yp[i], color=colors[j])
            plt.scatter(np.array(ages)[i], yl[i], color=colors[j])
        plt.xlabel('Age, years')
        plt.ylabel('Bioage, years')
