#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.colors import ListedColormap
import argparse
import torch
import os
# Set the random seed for reproducibility
torch.manual_seed(42)  # You can choose any integer as the seed

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str,default="D:\Sean\github\cpjku_dcase23_NTU\embed")
parser.add_argument('--ckpt_dir',type=str,default="fskag87u")
parser.add_argument("--xfile", type=str, default="embeddings.txt", help="file name of feature stored")
parser.add_argument("--yfile", type=str, default="labels.txt", help="file name of label stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")

opt = parser.parse_args()
print("get choice from args", opt)
xfile=os.path.join(opt.base_dir,opt.ckpt_dir,opt.xfile)

yfile = os.path.join(opt.base_dir,opt.ckpt_dir,opt.yfile)

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)
    
    # Compute covariance matrix
    covariance_matrix = torch.mm(X.t(), X)

    # Perform eigen decomposition on the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)  # Use eigh for real symmetric matrix

    # Sort the eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the top `no_dims` eigenvectors
    M = eigenvectors[:, :no_dims]

    # Project the data onto the new lower-dimensional space
    Y = torch.mm(X, M)
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # To obtain X and label subsets
    X = np.loadtxt(xfile)
    X = torch.Tensor(X)
    
    # Step 1: Determine 10% of the dataset size
    num_samples = X.size(0)
    subset_size = int(0.1 * num_samples)

    # Step 2: Randomly select 10% of the samples with a fixed seed
    indices = torch.randperm(num_samples)[:subset_size]
    X_subset = X[indices]
    labels = np.loadtxt(yfile).tolist()
    labels = torch.tensor(labels)
    labels_subset = labels[indices]  
    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X_subset)==len(labels_subset))
    if os.path.exists(os.path.join(opt.base_dir,opt.ckpt_dir,"tsne_result.pt")):
        Y = torch.load(os.path.join(opt.base_dir,opt.ckpt_dir,"tsne_result.pt"))
        print("Loaded Y from saved file.")
        
    else:
        with torch.no_grad():
            Y = tsne(X_subset, 2, 50, 20.0)
            torch.save(Y, os.path.join(opt.base_dir,opt.ckpt_dir,"tsne_result.pt"))
    if opt.cuda:
        Y = Y.cpu().numpy()
    
## Get colormap labels
    label_names = {
    0: 'airport',
    1: 'bus',
    2: 'metro',
    3: 'metro_station',
    4: 'park',
    5: 'public_square',
    6: 'shopping_mall',
    7: 'street_pedestrian',
    8: 'street_traffic',
    9: 'tram'
    }
    # Step 1: Define unique colors for each class.
    unique_labels = torch.unique(labels_subset).cpu().numpy()  # Get unique class labels
    num_classes = len(unique_labels)

    # Use a colormap (e.g., 'tab10' or 'tab20' for distinct colors)
    cmap = pyplot.cm.get_cmap('tab10', num_classes)  # Adjust colormap for the number of classes
    labels_subset = labels_subset.cpu()
    # pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels_subset)
    
    # Step 2: Create the scatter plot with legend
    fig, ax = pyplot.subplots()

    for i, label in enumerate(unique_labels):
        indices = labels_subset == label  # Boolean array for current class
        ax.scatter(
            Y[indices, 0],  # x-coordinates for class 'label'
            Y[indices, 1],  # y-coordinates for class 'label'
            color=cmap(i),  # Assign unique color
            label=label_names[int(label)],  # Label for the legend
            s=20  # Marker size
        )

    # Step 3: Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to fit legend
    pyplot.savefig(os.path.join(opt.base_dir,opt.ckpt_dir,"tsne_output.png"))
    pyplot.show()
