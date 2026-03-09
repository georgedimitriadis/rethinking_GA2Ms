from sklearn.datasets import make_regression
import numpy as np




def get_data(n_samples):
    X, y = make_regression(
        n_features=2, n_informative=2, random_state=1, n_samples=n_samples
    )

    linearly_separable = (X, y)

    rng = np.random.RandomState(2)
    X = rng.uniform(size=X.shape, low=-20, high=20.0)
    # X1 = rng.normal(0,20,size=X.shape[0])
    # X2 = rng.normal(0,20,size=X.shape[0])
    # X1 [X1 > 20] = 20
    # X1 [X1 < -20] = -20
    # X2 [X2 > 20] = 20
    # X2 [X2 < -20] = -20
    #
    # X1 = X1[np.newaxis,:]
    # X2 = X2[np.newaxis,:]
    # X = np.concatenate([X1,X2]).T

    # X[X > 20]
    print(X.shape)
    # exit()

    # X1 = np.linspace(-20.0,20.0, N_SAMPLES)
    # X2 = np.linspace(-20.0,20.0, N_SAMPLES)
    #
    # X = []
    # for x1 in X1:
    #     for x2 in X2:
    #         X.append([x1,x2])
    # X = np.array(X)

    # y = np.cos(X.T[0]) + np.cos(X.T[1])
    # y = np.cos(X.T[0] + X.T[1])


#print(X_f.shape)

    datasets = [
        #linearly_separable,
      (X, np.cos(X.T[0]) * np.cos(X.T[1]), "cos(x_0) cos(x_1)", "d_0"),
     (X, np.cos(X.T[0]) + np.cos(X.T[1]), "cos(x_0) + cos(x_1)", "d_1"),
     (X, np.cos(X.T[0] * X.T[1]), "cos(x_0 x_1)", "d_2"),
     (X, np.cos(0.5*X.T[0] + 1.8*X.T[1]), "cos(0.5*x_0 + x_1)", "d_3"),
    # (X, X.T[0] + 0.2*X.T[1], "x_0 + 0.2x_1", "d_4"),
    #(X, 0.1* X.T[0] * X.T[1] - 1, "0.1x_0x_1 - 1", "d_5"),
    ]

    return datasets


def get_data_single(n_samples):
    X, y = make_regression(
        n_features=1, n_informative=1, random_state=1, n_samples=n_samples
    )

    rng = np.random.RandomState(2)
    X = rng.uniform(size=X.shape, low=-20, high=20.0)

    datasets = [
        #linearly_separable,
      (X, np.cos(X.T[0]) + np.random.normal(size = X.T[0].shape, scale = 0.1) , "cos(x) + \mathcal{N}(0,0.1)", "d_0"),
    ]

    return datasets


def remove_points_in_square(X, y, square_min, square_max):
    # !!!!
    # Calculate a boolean mask for points inside the square
    inside_square_mask = ((X[:, 0] >= square_min[0]) & (X[:, 0] <= square_max[0]) &
                          (X[:, 1] >= square_min[1]) & (X[:, 1] <= square_max[1]))

    # Invert the mask to get points outside the square
    outside_square_mask = ~inside_square_mask

    # Filter X and y based on the mask
    X_in = X[outside_square_mask]
    y_in = y[outside_square_mask]

    X_ood = X[inside_square_mask]
    y_ood = y[inside_square_mask]

    return X_in, y_in, X_ood, y_ood