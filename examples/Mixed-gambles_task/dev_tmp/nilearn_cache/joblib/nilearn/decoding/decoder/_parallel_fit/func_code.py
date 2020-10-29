# first line: 140
def _parallel_fit(estimator, X, y, train, test, param_grid, is_classification,
                  selector, scorer, mask_img, class_index,
                  clustering_percentile):
    """Find the best estimator for a fold within a job.
    This function tries several parameters for the estimator for the train and
    test fold provided and save the one that performs best.

    Fit may be performed after some preprocessing step :
    * clustering with ReNA if clustering_percentile < 100
    * feature screening if screening_percentile < 100
    """
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # for fREM Classifier and Regressor : start by doing a quick ReNA
    # clustering to reduce the number of feature by agglomerating similar ones

    if clustering_percentile < 100:
        n_clusters = int(X_train.shape[1] * clustering_percentile / 100.)
        clustering = ReNA(mask_img, n_clusters=n_clusters, n_iter=20,
                          threshold=1e-7, scaling=False)
        X_train = clustering.fit_transform(X_train)
        X_test = clustering.transform(X_test)

    do_screening = (X_train.shape[1] > 100) and selector is not None

    if do_screening:
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    # If there is no parameter grid, then we use a suitable grid (by default)
    param_grid = _check_param_grid(estimator, X_train, y_train, param_grid)
    best_score = None
    for param in ParameterGrid(param_grid):
        estimator = clone(estimator).set_params(**param)
        estimator.fit(X_train, y_train)

        if is_classification:
            score = scorer(estimator, X_test, y_test)
            if np.all(estimator.coef_ == 0):
                score = 0
        else:  # regression
            score = scorer(estimator, X_test, y_test)

        # Store best parameters and estimator coefficients
        if (best_score is None) or (score >= best_score):
            best_score = score
            best_coef = np.reshape(estimator.coef_, (1, -1))
            best_intercept = estimator.intercept_
            best_param = param

    if do_screening:
        best_coef = selector.inverse_transform(best_coef)

    if clustering_percentile < 100:
        best_coef = clustering.inverse_transform(best_coef)

    return class_index, best_coef, best_intercept, best_param, best_score
