from copy import deepcopy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


from dsr import DeepSymbolicOptimizer
from dsr.traverser import *

from scipy.interpolate import interp1d, griddata


class DeepSymbolicRegressor(DeepSymbolicOptimizer,
                            BaseEstimator, RegressorMixin):
    """
    Sklearn interface for deep symbolic regression.
    """

    def __init__(self, config=None):
        DeepSymbolicOptimizer.__init__(self, config)

    def fit_v2(self, X, y):

        # Update the Task
        config = deepcopy(self.config)
        config["task"]["task_type"] = "regression"
        config["task"]["dataset"] = (X, y)
        self.update_config(config)

        train_result = self.train()
        self.program_ = train_result["program"]

        return self

    def fit(self, X, y):
        
        # Update the Task
        config = deepcopy(self.config)
        config["task"]["task_type"] = "regression"
        config["task"]["dataset"] = (X, y)
        self.update_config(config)

        train_result = self.train()
        self.program_ = train_result["program"]
        # doesn't work for function sets that use constants

        if config["recursion"]["run"]:

            print("Started learning error correction")
            operations = train_result['traversal'].split(',')
            errors = evaluator(operations, X, y)

            # attempt at interpolation -- too simple for more complicated functions, based on limited testing
            """ if X.shape[1] == 1:
                f = interp1d(X, errors, kind = 'cubic') """

            """ else:
                grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
                grid = griddata(X, errors, (grid_x, grid_y), methid='cubic') """
                

            model = DeepSymbolicRegressor("config.json")
            correction = None

            if config["recursion"]["filter"]:
                filtered_X_errors = []
                for i in range(len(X)):
                    if errors[i] > 0.5: filtered_X_errors.append((X[i], errors[i]))
                
                if len(filtered_X_errors) > len(y) * 0.5: # proper activation metrics? checking only area(s) of greatest error?
                
                    if not config["recursion"]["single"]: correction = model.fit(filtered_X_errors[:,0], filtered_X_errors[:,1])
                        
                    else: correction = model.fit_v2(filtered_X_errors[:,0], filtered_X_errors[:,1])

            elif len(list(filter(lambda error: error > 0.5, errors))) > len(y) * 0.5:

                if not config["recursion"]["single"]: correction = model.fit(X, errors)
                else: correction = model.fit_v2(X, errors)
                    
            if correction is not None: self.program_ += "add," + correction

    # where to correct (benefits of localization); num of corrective steps; visualize; fix recursive step
    # sharp gradient as criterion to generate point cloud density (consider areas increasing increasing error); l2 norm components
    # rerun energy functions appropriately

        return self

    def predict(self, X):

        check_is_fitted(self, "program_")

        return self.program_.execute(X)
