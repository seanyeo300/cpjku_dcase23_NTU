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
from sklearn.metrics import jaccard_score

# Set the random seed for reproducibility
torch.manual_seed(42)  # You can choose any integer as the seed

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str,default="D:\Sean\github\cpjku_dcase23_NTU\embed")
parser.add_argument('--ckpt_dir',type=str,default="jiw5bohu")
parser.add_argument("--xfile", type=str, default="embeddings.txt", help="file name of feature stored")
parser.add_argument("--yfile", type=str, default="labels.txt", help="file name of label stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")
# Add this argument for class selection
parser.add_argument(
    "--class_labels", 
    type=int, 
    nargs='+', 
    choices=range(10), 
    help="Specify one or more classes (0-9) to plot. If none are specified, plot all classes.",
    default=[1,2,9]
)
opt = parser.parse_args()
print("get choice from args", opt)
xfile=os.path.join(opt.base_dir,opt.ckpt_dir,opt.xfile)

yfile = os.path.join(opt.base_dir,opt.ckpt_dir,opt.yfile)

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def compute_class_jaccard_index(labels, embeddings):
    """Computes the Jaccard index between all pairs of classes."""

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    jaccard_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(i + 1, num_classes):  # Avoid comparing a class to itself
            class_i_indices = np.where(labels == unique_labels[i])[0]
            class_j_indices = np.where(labels == unique_labels[j])[0]

            class_i_embeddings = embeddings[class_i_indices]
            class_j_embeddings = embeddings[class_j_indices]

            # Convert embeddings to binary vectors based on a threshold
            # This is crucial for Jaccard index calculation
            threshold = np.median(embeddings) # You can experiment with different thresholding strategies.
            class_i_binary = (class_i_embeddings > threshold).astype(int)
            class_j_binary = (class_j_embeddings > threshold).astype(int)

            # Calculate Jaccard index for each pair of points between the two classes and average them
            pairwise_jaccard = []
            for emb_i in class_i_binary:
                for emb_j in class_j_binary:
                    jaccard = jaccard_score(emb_i, emb_j,zero_division=0)
                    pairwise_jaccard.append(jaccard)
            
            jaccard_matrix[i, j] = np.mean(pairwise_jaccard) if pairwise_jaccard else 0
            jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Matrix is symmetric

    return jaccard_matrix

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
    # Load the data
    X = np.loadtxt(xfile)
    X = torch.Tensor(X)

    num_samples = X.size(0)
    subset_size = int(0.1 * num_samples)

    indices = torch.randperm(num_samples)[:subset_size]
    X_subset = X[indices]
    labels = np.loadtxt(yfile).tolist()
    labels = torch.tensor(labels)
    labels_subset = labels[indices]

    assert len(X[:, 0]) == len(X[:, 1])
    assert len(X_subset) == len(labels_subset)

    tsne_file_3d = os.path.join(opt.base_dir, opt.ckpt_dir, "tsne_result_3d.pt")
    if os.path.exists(tsne_file_3d):
        Y = torch.load(tsne_file_3d)
        print("Loaded 3D embeddings from saved file.")
    else:
        with torch.no_grad():
            Y = tsne(X_subset, no_dims=3, initial_dims=50, perplexity=20.0)
            torch.save(Y, tsne_file_3d)

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
    
    # # Compute Jaccard index between classes
    # jaccard_matrix = compute_class_jaccard_index(labels_subset.cpu().numpy(), Y)
    # print("Jaccard Index Matrix between classes:\n", jaccard_matrix) 
    # Extract pairs with Jaccard index above 0.5
    # high_jaccard_pairs = []
    # for i in range(len(jaccard_matrix)):
    #     for j in range(i + 1, len(jaccard_matrix)):
    #         if jaccard_matrix[i, j] > 0.5:
    #             high_jaccard_pairs.append((label_names[i], label_names[j], jaccard_matrix[i, j]))

    # # Sort the pairs in descending order
    # high_jaccard_pairs.sort(key=lambda x: x[2], reverse=True)

    # # Extract unique classes in order of appearance
    # unique_classes = []
    # for pair in high_jaccard_pairs:
    #     for class_name in pair[:2]:  # Iterate through the first two elements (class names)
    #         if class_name not in unique_classes:
    #             unique_classes.append(class_name)

    # # Print the sorted pairs
    # for pair in high_jaccard_pairs:
    #     print(f"Jaccard Index: {pair[2]:.4f} between {pair[0]} and {pair[1]}")

    # # Print the unique classes
    # print("\nUnique classes in order of appearance:")
    # print(unique_classes)   
    # Step 1: Define unique colors for each class.
    unique_labels = torch.unique(labels_subset).cpu().numpy()  # Get unique class labels
    num_classes = len(unique_labels)

    # Use a colormap (e.g., 'tab10' or 'tab20' for distinct colors)
    cmap = pyplot.cm.get_cmap('tab10', num_classes)  # Adjust colormap for the number of classes
    labels_subset = labels_subset.cpu()
    # pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels_subset)
    
    
    
    
    # Plot the 3D t-SNE embeddings
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a fixed scale for all plots
    # Define a fixed scale for all plots
    x_min, x_max = Y[:, 0].min() - 1, Y[:, 0].max() + 1
    y_min, y_max = Y[:, 1].min() - 1, Y[:, 1].max() + 1
    z_min, z_max = Y[:, 2].min() - 1, Y[:, 2].max() + 1

    # Set axis limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # If a specific class is specified, plot only that class
    if opt.class_labels:
        selected_classes_str = "_".join(map(str, opt.class_labels))
        # Filter and plot only specified classes
        for label in opt.class_labels:
            # label_indices = labels_subset == label
            cls_indices = (labels_subset == label).nonzero(as_tuple=True)[0]
            ax.scatter(
                Y[cls_indices, 0],
                Y[cls_indices, 1],
                Y[cls_indices, 2],
                color=cmap(int(label)),
                label=f'{label_names[int(label)]}',
                s=20,  # Size of the points
                alpha=0.5  # Transparency
            )
    else:
        # Plot all classes
        for i, label in enumerate(unique_labels):
            cls_indices = (labels_subset == label).nonzero(as_tuple=True)[0]
            ax.scatter(
                Y[cls_indices, 0],
                Y[cls_indices, 1],
                Y[cls_indices, 2],
                color=cmap(int(label)),
                label=f'{label_names[int(label)]}',
                s=20,  # Size of the points
                alpha=0.5  # Transparency
            )
            
    ax.set_title("3D t-SNE Embeddings")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    pyplot.show()
