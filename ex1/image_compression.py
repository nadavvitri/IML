from numpy import diag, zeros, dot
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc

if __name__ == '__main__':
    ascent = misc.ascent()  # image to matrix
    u, singular_values, v = svd(ascent)  # SVD for the image
    sigma_k_diag = zeros(512)  # init main diagonal for the reconstruct sigma_k matrix
    
    # init lists for graphs
    compression_ratio_arr, frobenius_distance_arr, x_axis = [0] * 512, [0] * 512, [0] * 512
    samples = [15, 50, 230, 401, 500]
    samples_grid_location = [3, 4, 7, 8, 11]  # loaction on grid for every sample
    
    for k in range(0, 512, 1):
        x_axis[k] = k
        sigma_k_diag[:k] = singular_values[:k]  # set the k largest singular values and the rest are zero
        sigma_k = diag(sigma_k_diag)  # reconstruct sigma_k matrix with main diagonal with k largest singular values

        # reconstruct approximation matrix Mk = U Sk V^t
        u_sigma_k_product = dot(u, sigma_k)
        approximation_matrix = dot(u_sigma_k_product, v)

        compression_ratio_arr[k] = 1 - (((2 * k * 512) + k) / ((2 * 512 * 512) + 512))  # compression ratio
        frobenius_distance_arr[k] = norm(ascent - approximation_matrix)

	# show 5 samples of comperssion of image
        if k in samples:
            subplot(3, 4, samples_grid_location[samples.index(k)])
            title('K: ' + str(k), fontweight="bold", fontsize=8)
            imshow(approximation_matrix)

    #graphs of compression ratio and frobenius distance
    suptitle('Graphs', fontweight="bold", fontsize=13)
    subplot(3, 4, (1, 2))
    plot(x_axis, frobenius_distance_arr, '-')
    xlabel('k value', fontweight="bold")
    ylabel('Frobenius distance', fontweight="bold")

    subplot(3, 4, (5, 6))
    plot(x_axis, compression_ratio_arr, '-')
    xlabel('k value', fontweight="bold")
    ylabel('Compression ratio', fontweight="bold")
    tight_layout()
    show()
