import sys

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates


ORDER = "F"

#--------------------------------------------
#https://github.com/clbarnes/molesq

def reshape_points(control_points: np.ndarray) -> np.ndarray:
    """Reshape NxD array into 1xDxNx1.

    Where D is the dimensionality, and N the number of points.

    Parameters
    ----------
    locations

    Returns
    -------
    reshaped array
    """
    n_locations, n_dimensions = control_points.shape
    return control_points.ravel().reshape(1, n_dimensions, n_locations, 1, order=ORDER)


def _transform_affine(locs, orig_cp, deformed_cp, cp_weights=None):
    """
    Makes heavy use of Einstein summation, resources here:

    * https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    * https://ajcr.net/Basic-guide-to-einsum/
    * Playground: https://oracleofnj.github.io/einsum-explainer/
    """
    n_locs, n_dim = locs.shape
    n_landmarks = len(orig_cp)

    # Pairwise distances between original control points and locations to transform
    # reshaped to 1,1,N_cp,N_l
    # jittered to avoid 0s

    if cp_weights is None:
        sqdists = cdist(orig_cp, locs, "sqeuclidean")
    else:
        # weights need to be factored in before squaring of distance, for unit reasons
        sqdists = (cdist(orig_cp, locs) / cp_weights[:, np.newaxis]) ** 2

    weights = 1 / (sqdists.reshape(1, 1, n_landmarks, n_locs) + sys.float_info.epsilon)

    weights_inverse_norm = 1 / np.sum(weights, axis=2)

    # reshape arrays for consistent indices
    orig_cp = reshape_points(orig_cp)
    deformed_cp = reshape_points(deformed_cp)

    # weighted centroids
    orig_star = np.einsum(
        "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, orig_cp
    ).reshape(1, n_dim, 1, n_locs)
    deformed_star = np.einsum(
        "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, deformed_cp
    ).reshape(1, n_dim, 1, n_locs)

    # distance to weighted centroids
    orig_hat = orig_cp - orig_star
    deformed_hat = deformed_cp - deformed_star

    Y = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, orig_hat).reshape(
        n_dim, n_dim, 1, n_locs
    )

    rolled = np.moveaxis(Y, (0, 1), (2, 3))
    inv_rolled = np.linalg.inv(rolled)
    Y_inv = np.moveaxis(inv_rolled, (2, 3), (0, 1))

    Z = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, deformed_hat).reshape(
        n_dim, n_dim, 1, n_locs
    )

    locs_reshaped = locs.ravel().reshape(1, n_dim, 1, n_locs, order=ORDER)
    vprime = (
        np.einsum("iakl,abkl,bjkl->ijkl", locs_reshaped - orig_star, Y_inv, Z)
        + deformed_star
    )
    vprime = vprime.ravel(ORDER).reshape(n_locs, n_dim)

    return vprime


class Transformer:
    def __init__(
        self,
        control_points: ArrayLike,
        deformed_control_points: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ):
        """Class for transforming points using Moving Least Squares.

        Given control point arrays must both be of same shape NxD,
        where N is the number of points,
        and D the dimensionality.

        Parameters
        ----------
        control_points : ArrayLike
        deformed_control_points : ArrayLike
        weights : Optional[ArrayList]
            Any values <= 0 will be set to an arbitrarily small positive number

        Raises
        ------
        ValueError
            Invalid control point array(s)
        """
        from_arr = np.asarray(control_points)
        to_arr = np.asarray(deformed_control_points)

        if from_arr.shape != to_arr.shape:
            raise ValueError("Control points must have the same shape")
        if from_arr.ndim != 2:
            raise ValueError("Control points must be 2D array")

        self.n_landmarks, self.ndim = from_arr.shape
        self.control_points = from_arr
        self.deformed_control_points = to_arr

        if weights is not None:
            weights = np.asarray(weights)
            if weights.shape != (len(from_arr),):
                raise ValueError(
                    "weights must have same length as control points array"
                )
            weights[weights <= 0] = sys.float_info.epsilon

        self.weights = weights

    def transform(
        self,
        locations: ArrayLike,
        reverse=False,
        # strategy=Strategy.AFFINE,
    ) -> np.ndarray:
        """Transform some locations using the given control points.

        Uses the affine form of the MLS algorithm.

        Parameters
        ----------
        locations : ArrayLike
            NxD array of N locations in D dimensions to transform.
        reverse : bool, optional
            Transform from deformed space to original space, by default False

        Returns
        -------
        Deformed points
        """
        locs = np.asarray(locations)
        if locs.ndim != 2 or locs.shape[-1] != self.ndim:
            raise ValueError(
                "Locations must be 2D array of same width as control points, "
                f"got {locs.shape}"
            )

        orig_cp = self.control_points
        deformed_cp = self.deformed_control_points

        if reverse:
            orig_cp, deformed_cp = deformed_cp, orig_cp

        # if strategy == Strategy.AFFINE:
        #     return self._transform_affine(locs, orig_cp, deformed_cp)
        # else:
        #     raise ValueError(f"Unimplemented/ unknown strategy {strategy}")
        return _transform_affine(locs, orig_cp, deformed_cp, self.weights)

#----------------------------
#https://github.com/Jarvis73/Moving-Least-Squares

np.seterr(divide='ignore', invalid='ignore')


class Mls_deformer:
    def __init__(self,image):
        self.image = np.array(image)
        height, width, _ = self.image.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        self.vy,self.vx = np.meshgrid(gridX,gridY)
        self.channels = np.moveaxis(self.image, 2, 0)

    def trans(self,p,q,mode='affine'):
        if mode == 'affine':
            affine = mls_affine_deformation(self.vy,self.vx,p,q,alpha=1)
            aug = np.ones_like(self.image)
            aug[self.vx,self.vy] = self.image[tuple(affine)]
            # for i in range(self.image.shape[0]):
            #     for j in range(self.image.shape[1]):
            #         aug[i,j] = bilinear_interpolation(self.image,affine[:,i,j])
            # affine = np.transpose(affine, (1, 2, 0)).reshape(-1,2)
            # channels = [
            #     map_coordinates(c, affine.T,order=2,
            #     mode='nearest',
            #     cval=0).reshape(self.image.shape[:2])
            #     for c in self.channels
            # ]
            # aug = np.stack(channels, 2)
        elif mode == 'similar':
            similar = mls_similarity_deformation(self.vy, self.vx, p, q, alpha=1)
            aug = np.ones_like(self.image)
            aug[self.vx, self.vy] = self.image[tuple(similar)]
        elif mode == 'rigid':
            rigid = mls_rigid_deformation(self.vy, self.vx, p, q, alpha=1)
            aug = np.ones_like(self.image)
            aug[self.vx, self.vy] = self.image[tuple(rigid)]
        return aug

def bilinear_interpolation(img, vector_u):
    uy, ux = vector_u
    x1, x2 = int(ux), int(ux + 1)
    y1, y2 = int(uy), int(uy + 1)
    f_x_y1 = (x2 - ux) / (x2 - x1) * img[y1][x1] + (ux - x1) / (x2 - x1) * img[y1][x2]
    f_x_y2 = (x2 - ux) / (x2 - x1) * img[y2][x1] + (ux - x1) / (x2 - x1) * img[y2][x2]

    f_x_y = (y2 - uy) / (y2 - y1) * f_x_y1 + (uy - y1) / (y2 - y1) * f_x_y2

    return f_x_y.astype(np.int16)

def mls_affine_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """
    Affine deformation

    Parameters
    ----------
    vy, vx: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """

    # Change (x, y) to (row, col)
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha  # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)  # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]  # [2, grow, gcol]

    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    phat = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwp = np.zeros((2, 2, grow, gcol), np.float32)
    for i in range(ctrls):
        pTwp += phat[i] * reshaped_w[i] * phat1[i]
    del phat1

    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]

    mul_left = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = np.multiply(reshaped_w, phat, out=phat)  # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 2, 1]
    out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]  # [ctrls, grow, gcol, 1, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right, out=out_A)  # [ctrls, grow, gcol, 1, 1]
    A = A.reshape(ctrls, 1, grow, gcol)  # [ctrls, 1, grow, gcol]
    del mul_right, reshaped_mul_right, phat

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]  # [2, grow, gcol]
    del w, reshaped_w

    # Get final image transfomer -- 3-D array
    transformers = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        transformers += A[i] * (reshaped_q[i] - qstar)
    transformers += qstar
    del A

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.astype(np.int16)
    #return transformers


def mls_similarity_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Similarity deformation

    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha  # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)  # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]  # [2, grow, gcol]

    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]

    mu = np.zeros((grow, gcol), np.float32)
    for i in range(ctrls):
        mu += w[i] * (phat[i] ** 2).sum(0)
    reshaped_mu = mu.reshape(1, grow, gcol)  # [1, grow, gcol]

    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]  # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.zeros((grow, gcol, 2), np.float32)
    for i in range(ctrls):
        neg_phat_verti = phat[i, [1, 0]]  # [2, grow, gcol]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]
        mul_left = np.concatenate((reshaped_phat[i], reshaped_neg_phat_verti), axis=0)  # [2, 2, grow, gcol]

        A = np.matmul((reshaped_w[i] * mul_left).transpose(2, 3, 0, 1),
                      mul_right.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]

        qhat = reshaped_q[i] - qstar  # [2, grow, gcol]
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)  # [grow, gcol, 2]

    transformers = temp.transpose(2, 0, 1) / reshaped_mu + qstar  # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.astype(np.int16)


def mls_rigid_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Rigid deformation

    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha  # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)  # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]  # [2, grow, gcol]

    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(2, 2, grow, gcol)  # [2, 2, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]  # [2, grow, gcol]

    temp = np.zeros((grow, gcol, 2), np.float32)
    for i in range(ctrls):
        phat = reshaped_p[i] - pstar  # [2, grow, gcol]
        reshaped_phat = phat.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]
        reshaped_w = w[i].reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        neg_phat_verti = phat[[1, 0]]  # [2, grow, gcol]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]
        mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=0)  # [2, 2, grow, gcol]

        A = np.matmul((reshaped_w * mul_left).transpose(2, 3, 0, 1),
                      reshaped_mul_right.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]

        qhat = reshaped_q[i] - qstar  # [2, grow, gcol]
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)  # [grow, gcol, 2]

    temp = temp.transpose(2, 0, 1)  # [2, grow, gcol]
    normed_temp = np.linalg.norm(temp, axis=0, keepdims=True)  # [1, grow, gcol]
    normed_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)  # [1, grow, gcol]
    transformers = temp / normed_temp * normed_vpstar + qstar  # [2, grow, gcol]
    nan_mask = normed_temp[0] == 0

    # Replace nan values by interpolated values
    nan_mask_flat = np.flatnonzero(nan_mask)
    nan_mask_anti_flat = np.flatnonzero(~nan_mask)
    transformers[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[0][~nan_mask])
    transformers[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[1][~nan_mask])

    # Remove the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.astype(np.int16)