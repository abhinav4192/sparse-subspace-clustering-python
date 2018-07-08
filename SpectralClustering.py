import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import identity


def SpectralClustering(CKSym, n):
    # This is direct port of JHU vision lab code. Could probably use sklearn SpectralClustering.
    CKSym = CKSym.astype(float)
    N, _ = CKSym.shape
    MAXiter = 1000  # Maximum number of iterations for KMeans
    REPlic = 20  # Number of replications for KMeans

    DN = np.diag(np.divide(1, np.sqrt(np.sum(CKSym, axis=0) + np.finfo(float).eps)))
    LapN = identity(N).toarray().astype(float) - np.matmul(np.matmul(DN, CKSym), DN)
    _, _, vN = np.linalg.svd(LapN)
    vN = vN.T
    kerN = vN[:, N - n:N]
    normN = np.sqrt(np.sum(np.square(kerN), axis=1))
    kerNS = np.divide(kerN, normN.reshape(len(normN), 1) + np.finfo(float).eps)
    km = KMeans(n_clusters=n, n_init=REPlic, max_iter=MAXiter, n_jobs=-1).fit(kerNS)
    return km.labels_


if __name__ == "__main__":
    pass
