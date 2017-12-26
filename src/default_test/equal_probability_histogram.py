import numpy as np
import matplotlib.pylab as plt

def equal_probability_histogram(x, N_bins=None, bin_edges=None):

    if N_bins is not None:
        if bin_edges is None:
            # compute some quantiles of the data to determine the bin edges
            bin_edges = np.percentile(x, np.linspace(0, 100, N_bins+1))
        else:
            raise ValueError("if the number of bins is specified, no bin edges must be given.")
    elif bin_edges is not None:
        if N_bins is not None:
            raise ValueError("if the bin edges are given, no number of bins must be specified.")

    n, _ = np.histogram(x, bin_edges)

    f = n / np.diff(bin_edges) / len(x)

    return f, bin_edges

def test_normal():
    # generate some normal data
    N = 1000
    x = np.random.normal(loc=0, scale=1, size=N)

    plt.figure()

    f_orig, bin_edges_orig = np.histogram(x, density=True)

    f, bin_edges = equal_probability_histogram(x, 10)
    plt.step(bin_edges, np.hstack((f, 0)), where='post', label="Equal Count")
    plt.step(bin_edges_orig, np.hstack((f_orig, 0)), where='post', label="Original")
    plt.legend()
    plt.show()

def test_uniform():
    # generate some normal data
    N = 1000
    x = np.random.uniform(low=0, high=5, size=N)

    plt.figure()

    f_orig, bin_edges_orig = np.histogram(x, density=True)

    f, bin_edges = equal_probability_histogram(x, 10)
    plt.step(bin_edges, np.hstack((f, 0)), where='post', label="Equal Count")
    plt.step(bin_edges_orig, np.hstack((f_orig, 0)), where='post', label="Original")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_normal()
    test_uniform()