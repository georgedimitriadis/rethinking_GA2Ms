import pickle
import time
from copy import deepcopy
import os.path as ops

import keras
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from ec.elco import ECRegressor, ECClassifier
from TALENT.model.classical_methods.base import classical_methods
from ec.gam import GeneralisedAdditiveClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from ec.transformers import KMeansDistanceTransformer, KNNPredictionTransformer


class ECMACMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert (args.cat_policy != 'indices')

    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config['model']
        transformer_config = model_config['transformer']
        regressor_config = model_config['regressorclassifier']


        if self.is_regression:
            #     self.model = Pipeline([
            #     ("scaler", MinMaxScaler((-1, 1))),
            #     ('transformer', RandomExpressionTransformer(**transformer_config, sample_range=(-1, 1))),
            #     ('regressorclassifier', LGBMRegressor(**regressor_config, random_state=1))
            # ])

            ec = ECRegressor(epochs=5000,
                             learning_rate=0.001,
                             batch_size=128,
                             n_knots = 32,
                             max_arity=2,
                             validation_split=0.1,
                             final_linear_layer_regularizer="l2"
                             )

            from ec import residualmodeladc
            from lightgbm import LGBMClassifier, LGBMRegressor

            ec = residualmodeladc.ResidualRegressor(ec,
                                                 LGBMRegressor(n_estimators=2000) )
            self.model = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                #("knn", KNNPredictionTransformer()),

                #("pca", PCA()),
                #("pca", KMeansDistanceTransformer()),
                #('regressorclassifier', ec)
                ('regressorclassifier', ec)
            ])

        else:



            ec = ECClassifier(epochs=5000,
                              learning_rate=0.001,
                              batch_size=128,
                              n_knots=32,
                              max_arity= 2,
                              validation_split=0.1,
                              final_linear_layer_regularizer="l2"
                              )

            from lightgbm import LGBMClassifier, LGBMRegressor
            from ec import residualmodeladc
            from sklearn.multioutput import MultiOutputRegressor
            ec = residualmodeladc.ResidualClassifier(ec,
                                                     MultiOutputRegressor(LGBMRegressor(n_estimators=2000)))

            self.model = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                #("pca", KMeansDistanceTransformer(n_clusters=10)),
                ('regressorclassifier', (ec))
            ])

            n = 2
            self.transform_pipepine = Pipeline(self.model.steps[:n])


            #    #
            # #("poly", PolynomialFeatures(interaction_only=True,degree=3 )),
            #     ('transformer', SplineTransformer(n_knots=5)),
            # #    ("poly", PolynomialFeatures(interaction_only=True, degree=2)),
            #     #('regressorclassifier', SGDClassifier(verbose=True, loss="modified_huber"))
            #     ('regressorclassifier', LogisticRegression())
            #
            #     # ('regressorclassifier', LGBMClassifier(**regressor_config, random_state=1))
            # ])
            # self.model = GeneralisedAdditiveClassifier(max_arity=1,
            #                                       base_learner=RidgeCV(), n_iter=3)

    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        fit_config = deepcopy(self.args.config['fit'])
        fit_config.pop('n_bins')
        fit_config['regressorclassifier__eval_set'] = [(self.N['val'], self.y['val'])]
        tic = time.time()
        X = np.copy(np.concatenate([self.N['train'], self.N['val']]))
        y = np.copy(np.concatenate([self.y['train'], self.y['val']]))
        self.model.fit(X, y)
        #self.model.fit(self.N['train'], self.y['train'])
        #self.transform_pipepine.fit(self.N['train'])
        #self.model.fit(self.N['train'], self.y['train'], regressorclassifier__val=[self.transform_pipepine.transform(self.N['val']), self.y['val']])

        if not self.is_regression:
            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = accuracy_score(self.y['val'], y_val_pred)
        else:
            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = mean_squared_error(self.y['val'], y_val_pred, squared=False) * self.y_info['std']
        time_cost = time.time() - tic
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        if self.is_regression:
            test_logit = self.model.predict(self.N_test)
        else:
            test_logit = self.model.predict_proba(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit

    def clear_cache(self):
        try:
            model_type = type(self.model['regressorclassifier'])
            if ECRegressor == model_type or ECClassifier == model_type:
                keras.backend.clear_session()
                print('Clearing session')
        except:
            pass