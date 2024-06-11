import copy

from tqdm import tqdm
import numpy as np
import cudf, cuml

def LR_cross_validation(X, Y, num_epochs=1000, kfold=10, indices=None, C=1.0, seed=42):
    assert X.shape[0] == Y.shape[0]
    # assert len(np.unique(Y)) == num_classes
    N, d = X.shape
    np.random.seed(seed)
    indices = np.array_split(np.random.permutation(N), kfold)
    
    accuracies = []
        
    for k in tqdm(range(kfold)):
        idxs = copy.deepcopy(indices)
        test_idxs = idxs.pop(k)
        train_idxs = np.concatenate(idxs)

        Xtrain, ytrain = cudf.DataFrame(X[train_idxs], dtype=np.float32), cudf.Series(Y[train_idxs])
        Xval = cudf.DataFrame(X[test_idxs], dtype=np.float32)
        yval = Y[test_idxs]

        model = cuml.LogisticRegression(verbose=1, max_iter=num_epochs, tol=1e-9, C=C)
        model.fit(Xtrain, ytrain)

        pred = model.predict(Xval).to_numpy()
        acc = np.sum(pred == yval) / yval.shape[0]

        accuracies.append(acc)
        del Xtrain, ytrain, Xval, yval, model
        
    return np.mean(accuracies)







