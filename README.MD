Code is located at https://github.com/SirSkaro/cs-7641-assignment-3

# Project Setup
* Setup and activate a virtual environment with Python 3.10.
    * If using Anaconda/miniconda, you can simply use a command like "conda create -n bchurchill6_assignment1 python=3.10".
    * Otherwise, download a distribution of Python 3.10 manually and use it to run "python -m venv bchurchill6_assignment1"
* Install requirements. With the base of the project as the working directory, you can run the command "pip install -r requirements.txt"

# The Data
All data is located in /datasets. Each dataset is located its own directory with the filename "data" along with a description of the data (description.txt) and a link to the original download location (UCI).

# Graphs
Graphs included in the write up and located in the "graphs" directory in the top-level of the project.

# Running the Code
Different models contain functions for performing analysis required by the assignment. The intended way to use the code is through an interactive shell. Code snippets of using the interactive shell are available in `scratch text.txt`.

## Part 1 - Clustering
Functions pertaining to part 1 can be found in the modules kmc.py (K-Means Clustering) and em.py (Expectation Maximzation). They include functions for finding the best number of clusters, creating clusters, and graphing resulting scores.

## Part 2 - Dimensionality Reduction
Functions pertaining to part 2 can be found in the modules ica.py (Independent Component Analysis), k_pca.py (Kernel Principal Component Analysis), pca.py (Principal Component Analysis), and rp.py (Random Projection). It include functions for plotting a 3D graph, transforming a dataset, and graphing an analysis.

## Part 3 - Clustering on Reduced Datasets
Functions pertaining to part 3 can be found in the part3.py module. It includes functions to reduce a given dataset and perform clustering on it. There is one method for each combination of clustering algorithm and reduction algorithm. The dataset is passed in as an argument. The module also contains configurations for each of the 16 combinations.

## Part 4 - Neural Network Over Reduced Data
Functions pertaining to part 4 can be found in the nn_part4.py module. It includes a function to graph the analysis of training the neural network from assignment 1 over the original letter dataset and training a new neural network over a reduced letter dataset (using the KPCA algorithm).

## Part 5 - Neural Network Over Data Augmented with Cluster Features
Functions pertaining to part 5 can be found in the nn_part5.py module. It includes a function to graph the analysis of training the neural network from assignment 1 over the original letter dataset and training a new neural network over an augmented letter dataset that includes new features for both K-Means Clustering and Expectation Maximization.