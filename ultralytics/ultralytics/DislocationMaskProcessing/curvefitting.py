
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, splprep, splev
import scipy.interpolate


def fit_spline(x, y, degree=3, smoothing=2., n_interp_points=50):
    """Fit a pline to the data.

    Data points must be ordered.
    :param x:
    :param y:
    :param degree: spline degree
    :param smoothing: spline smooting/stiffness
    :param n_interp_points: number of interpolation points for sampling the spline
    :return: interpolated coordinates
    """
    points = np.vstack((np.array(x), np.array(y))).T

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=degree, s=smoothing) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, n_interp_points)
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T
    return points_fitted


def __old__():
    # Using a range spline to find out the ordering of the data points
    import scipy.interpolate
    from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, splprep, splev
    #plt.plot(x,y, label='poly')

    t = np.arange(x.shape[0], dtype=float)
    t /= t[-1]
    nt = np.linspace(0, 1, 100)

    tck, u = splprep([x2, y2], s=10)
    t_new = np.linspace(t[0], t[-1])
    out = splev(t_new, tck)

    plt.plot(x2, y2, label='range_spline')
    plt.plot(out[0], out[1], label='range_spline')


def order_data_points(x, y):
    """Reconstruct the ordering of data points.

    This works only when neighboring points are still relatively close together.
    """
    points = np.c_[x, y]
    n_points = points.shape[0]

    # compute 3 nearest neighbors for all points
    # 3, because the first is always the point itself (distance=0), the next two are left and right neighbors
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(points)
    knn_distances, knn_indices = knn.kneighbors(points)

    # Get the indices of two points which have the largest distance to an other point.
    # These are the potential endpoints of the line
    endpoint_indices = np.argsort(knn_distances[:, 2])[-2:]

    # Get the distance of these two points from the origin. The one closes to the
    # origin will be defined as the start point
    dist0 = np.linalg.norm(points[endpoint_indices[0]])
    dist1 = np.linalg.norm(points[endpoint_indices[1]])
    endpoint_indices = endpoint_indices if dist0 < dist1 else endpoint_indices[::-1]

    # Iteratively look for next neighbors: a neighbor is the node that is closest to the current point.
    # We additionally check that we're not going back to a previously used point (i.e., we check that
    # the next point is not already taken). This is important because, we might have to take the
    # 2nd next neighboring point insted of the nect neightbor (which us the previous point).
    sorted_indices = [endpoint_indices[0]]

    for i in range(n_points - 1):
        curr_idx = sorted_indices[-1]

        # Remove all previously used points (=indices) from the list of candidates. The following line is the
        # same as subtracting two sets but additionally presernves the ordering
        remaining_valid_indices = [item for item in knn_indices[curr_idx] if item not in sorted_indices]
        msg = "Points can't be sorted along a path of minimum length (fluctuations in the data might be too large)."
        if len(remaining_valid_indices) <= 0:
            print(msg)
            break
        #assert len(remaining_valid_indices) > 0, msg

        # take the first (=closest) point from the list of admissible points
        next_idx = remaining_valid_indices[0]

        # ... and add it to our list of sorted indices
        sorted_indices.append(next_idx)

    x = points[sorted_indices][:, 0]
    y = points[sorted_indices][:, 1]
    return x, y

