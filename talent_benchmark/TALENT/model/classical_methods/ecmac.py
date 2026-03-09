
import pickle
import time
from copy import deepcopy
import os.path as ops

import keras
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

from ec.elco import ECRegressor, ECClassifier
from TALENT.model.classical_methods.base import classical_methods
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import PolynomialFeatures


class ECMACMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert (args.cat_policy != 'indices')


    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config['model']
        ec_config = model_config['ec']

        train = self.N['train']
        val = self.N['val']

        test_fold = np.concatenate([
            np.full(len(train), -1),  # Training samples
            np.zeros(len(val))  # Validation samples
        ])

        # Create the predefined split
        ps = PredefinedSplit(test_fold)

        final_linear_layer_regularizer = None if ec_config['final_linear_layer_regularizer'] == 'None' else ec_config['final_linear_layer_regularizer']

        if self.is_regression:
            ec = ECRegressor(epochs=ec_config['epochs'],
                             learning_rate=ec_config['learning_rate'],
                             batch_size=ec_config['batch_size'],
                             validation_split=ec_config['validation_split'],
                             mixing_layer_on=ec_config['mixing_layer_on'],
                             final_linear_layer_regularizer=final_linear_layer_regularizer,
                             arity=ec_config['arity'],
                             ps=ps
                             )

            self.model = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                ('ec', ec),
                 ("final", RidgeCV(cv=ps))

            ])

            self.model_linear = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                ('polynomial', PolynomialFeatures(2, interaction_only=True)),
                ("final", RidgeCV(cv=ps))
            ])

        else:
            ec = ECClassifier(epochs=ec_config['epochs'],
                             learning_rate=ec_config['learning_rate'],
                             batch_size=ec_config['batch_size'],
                             validation_split=ec_config['validation_split'],
                             mixing_layer_on=ec_config['mixing_layer_on'],
                             final_linear_layer_regularizer=final_linear_layer_regularizer,
                             arity=ec_config['arity'],
                             ps=ps
                             )

            self.model = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                ('ec', (ec)),
                ("final", LogisticRegressionCV(cv=ps))
            ])


            self.model_linear = Pipeline([
                ("scaler", MinMaxScaler((-1, 1))),
                ('polynomial', PolynomialFeatures(2,interaction_only=True)),
                ("final", LogisticRegressionCV(cv=ps))
            ])



    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        fit_config = deepcopy(self.args.config['fit'])
        try:
            fit_config.pop('n_bins')
        except:
            pass
        fit_config['ec__eval_set'] = [(self.N['val'], self.y['val'])]
        tic = time.time()
        X = (np.concatenate([self.N['train'], self.N['val']]))
        y = (np.concatenate([self.y['train'], self.y['val']]))
        self.model.fit(X, y)
        self.model_linear.fit(X, y)


        if not self.is_regression:
            y_val_pred = self.model.predict(self.N['val'])
            y_val_pred_linear = self.model_linear.predict(self.N['val'])
            acc_pred = accuracy_score(self.y['val'], y_val_pred)
            acc_pred_linear = accuracy_score(self.y['val'], y_val_pred_linear)

            if( acc_pred_linear > acc_pred):
                self.model = self.model_linear


            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = accuracy_score(self.y['val'], y_val_pred)
            print(accuracy_score(self.y['val'], y_val_pred))
        else:
            y_val_pred = self.model.predict(self.N['val'])
            y_val_pred_linear = self.model_linear.predict(self.N['val'])
            mse_pred = mean_squared_error(self.y['val'], y_val_pred)
            mse_pred_linear = mean_squared_error(self.y['val'], y_val_pred_linear)


            print(mse_pred, mse_pred_linear)
            if (mse_pred_linear < mse_pred):
                print("linear model")
                self.model = self.model_linear

            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = root_mean_squared_error(self.y['val'], y_val_pred) * self.y_info['std']
        time_cost = time.time() - tic

        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)

        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        print(type(self.model[-2]))
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
            model_type = type(self.model['ec'])
            if ECRegressor == model_type or ECClassifier == model_type:
                keras.backend.clear_session()
                print('Clearing session')
        except:
            pass