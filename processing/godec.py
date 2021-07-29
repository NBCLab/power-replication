"""Run GODEC."""
import argparse
import json
import os

import numpy as np
import pywt
from nilearn._utils.niimg import load_niimg
from nilearn.masking import apply_mask, unmask
from scipy.linalg import qr
from scipy.sparse.linalg import svds

__version__ = "0.1"


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if (arg is not None) and (not os.path.isfile(arg)):
        parser.error(f"The file {arg} does not exist!")

    return arg


def dwtmat(mmix):
    """Apply a discrete wavelet transform to a matrix."""
    lt = len(np.hstack(pywt.dwt(mmix[0], "db2")))
    mmix_wt = np.zeros([mmix.shape[0], lt])
    for ii in range(mmix_wt.shape[0]):
        wtx = pywt.dwt(mmix[ii], "db2")
        cAlen = len(wtx[0])
        mmix_wt[ii] = np.hstack(wtx)
    return mmix_wt, cAlen


def idwtmat(mmix_wt, cAl):
    """Apply a discrete inverse wavelet transform to a matrix."""
    lt = len(pywt.idwt(mmix_wt[0, :cAl], mmix_wt[0, cAl:], "db2"))
    mmix_iwt = np.zeros([mmix_wt.shape[0], lt])
    for ii in range(mmix_iwt.shape[0]):
        mmix_iwt[ii] = pywt.idwt(mmix_wt[ii, :cAl], mmix_wt[ii, cAl:], "db2")
    return mmix_iwt


def wthresh(a, thresh):
    """Determine soft wavelet threshold."""
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)


def standard_godec(
    X,
    thresh=0.03,
    rank=2,
    power=1,
    tol=1e-3,
    max_iter=100,
    random_seed=0,
    verbose=True,
):
    """Run GODEC.

    Default threshold of .03 is assumed to be for input in the range 0-1...
    original matlab had 8 out of 255, which is about .03 scaled to 0-1 range
    """
    print("++Starting Go Decomposition")
    m, n = X.shape
    if m < n:
        X = X.T
    m, n = X.shape
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)
    while True:
        Y2 = random_state.randn(n, rank)
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1)
        Q, R = qr(Y2, mode="full")
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = np.linalg.norm(T.ravel(), 2)
        if (err < tol) or (itr >= max_iter):
            break
        L += T
        itr += 1

    # Is this even useful in soft GoDec? May be a display issue...
    G = X - L - S
    if m < n:
        L = L.T
        S = S.T
        G = G.T

    if verbose:
        print(f"Finished at iteration {itr}")

    return L, S, G


def greedy_semisoft_godec(D, ranks, tau, tol, inpower, k):
    """Run the Greedy Semi-Soft GoDec Algorithm (GreBsmo).

    Parameters
    ----------
    D : array
        nxp data matrix with n samples and p features
    rank : int
        rank(L)<=rank
    tau : float
        soft thresholding
    inpower : float
        >=0, power scheme modification, increasing it lead to better accuracy and more time cost
    k : int
        rank stepsize

    Returns
    -------
    L
        Low-rank part
    S
        Sparse part
    RMSE
        error
    error
        ||X-L-S||/||X||

    References
    ----------
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix Decomposition in
        Noisy Case", ICML 2011
    Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and Smoothing", AISTATS 2013.

    Tianyi Zhou, 2013, All rights reserved.
    """
    # set rankmax and sampling dictionary
    rankmax = max(ranks)
    outdict = {}
    rks2sam = [int(np.round(rk)) for rk in (np.array(ranks) / k)]
    rks2sam = sorted(rks2sam)

    # matrix size
    m, n = D.shape
    # ok
    if m < n:
        D = D.T
        # ok

    # To match MATLAB's norm on a matrix, you need an order of 2.
    normD = np.linalg.norm(D, ord=2)

    # initialization of L and S by discovering a low-rank sparse SVD and recombining
    rankk = int(np.round(rankmax / k))
    # ok
    error = np.zeros(max(rankk * inpower, 1) + 1)
    # ok
    print(error)
    X, s, Y = svds(D, k, which="LM")
    # CHECK svds
    s = np.diag(s)

    X = X.dot(s)
    # CHECK dot notation
    L = X.dot(Y)
    # CHECK dot notation
    S = wthresh(D - L, tau)
    # ok
    T = D - L - S
    # ok
    error[0] = np.linalg.norm(T[:]) / normD
    # CHECK np.linalg.norm types
    iii = 1
    stop = False
    alf = 0
    estrank = -1

    # Define some variables that shouldn't be touched before they're updated.
    X1 = Y1 = L1 = S1 = T1 = None

    # tic;
    # for r=1:rankk
    for r in range(1, rankk + 1):  # CHECK iterator range
        # parameters for alf
        rrank = rankmax
        estrank = 1
        rank_min = 1
        rk_jump = 10
        alf = 0
        increment = 1
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50
        if iii == inpower * (r - 2) + 1:
            iii = iii + inpower

        for i_iter in range(inpower + 1):
            print(f"r {r}, i_iter {i_iter}, rrank {rrank}, alf {alf}")

            # Update of X
            X = L.dot(Y.T)
            # CHECK dot notation
            # if estrank==1:
            #    qro=qr(X,mode='economic');   #CHECK qr output formats    #stopping here on 1/12
            # 	X = qro[0];
            # 	R = qro[1];
            # else:
            X, R = qr(X, mode="economic")
            # CHECK qr output formats

            # Update of Y
            Y = X.T.dot(L)
            # CHECK dot notation
            L = X.dot(Y)
            # CHECK dot notation

            # Update of S
            T = D - L
            # ok
            S = wthresh(T, tau)
            # ok

            # Error, stopping criteria
            T = T - S
            # ok
            ii = iii + i_iter - 1
            # ok
            # embed()
            error[ii] = np.linalg.norm(T[:]) / normD
            if error[ii] < tol:
                stop = True
                break

            # adjust estrank
            if estrank == 1:
                dR = abs(np.diag(R))
                drops = dR[:-1] / dR[1:]
                # print(dR.shape)
                dmx = max(drops)
                imx = np.argmax(drops)
                rel_drp = (rankmax - 1) * dmx / (sum(drops) - dmx)

                if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or (
                    itr_rank > maxitr_reduce_rank
                ):
                    rrank = max([imx, np.floor(0.1 * rankmax), rank_min])
                    estrank = 0
                    itr_rank = 0

                    if rrank != rankmax:
                        rankmax = rrank
                        if estrank == 0:
                            alf = 0
                            continue

            # adjust alf
            ratio = error[ii] / error[ii - 1]
            if np.isinf(ratio):
                ratio = 0
            # print(ii, error, ratio)

            if ratio >= 1.1:
                increment = max(0.1 * alf, 0.1 * increment)
                X = X1
                Y = Y1
                L = L1
                S = S1
                T = T1
                error[ii] = error[ii - 1]
                alf = 0
            elif ratio > 0.7:
                increment = max(increment, 0.25 * alf)
                alf = alf + increment

            # Update of L
            # print("updating L")
            X1 = X
            Y1 = Y
            L1 = L
            S1 = S
            T1 = T
            # ipdb.set_trace()
            L = L + ((1 + alf) * (T))

            # Add coreset
            if i_iter > 8:
                if np.mean(error[ii - 7 : ii + 1]) / error[ii - 8] > 0.92:
                    iii = ii
                    sf = X.shape[1]
                    if Y.shape[0] - sf >= k:
                        Y = Y[:sf, :]
                    break

        if r in rks2sam:
            L = X.dot(Y)
            if m < n:
                L = L.T
                S = S.T
            outdict[r * k] = [L, D - L, D - L - T]

        # Coreset
        if not stop and r < rankk:
            v = np.random.randn(k, m).dot(L)
            Y = np.vstack([Y, v])
            # correct this

        # Stop
        if stop:
            break

    error[error == 0] = None

    return outdict


def run_godec_denoising(
    in_file,
    mask,
    out_dir=".",
    prefix="",
    method="greedy",
    ranks=[4],
    norm_mode="vn",
    thresh=0.03,
    drank=2,
    inpower=2,
    wavelet=False,
):
    """Run GODEC denoising.

    Notes
    -----
    - Prantik mentioned that GODEC is run on outputs (e.g., High-Kappa), not inputs.
      https://github.com/ME-ICA/me-ica/issues/4#issuecomment-369058732
    - The paper tested ranks of 1-4. See page 5 of online supplemental methods.
    - The paper used a discrete Daubechies wavelet transform before and after GODEC,
      with rank-1 approximation and 100 iterations. See page 4 of online supplemental methods.
    """
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    img = load_niimg(in_file)
    mask = load_niimg(mask)

    if thresh is None:
        masked_data = apply_mask(img, mask).T
        mu = masked_data.mean(axis=-1)
        thresh = np.median(mu[mu != 0]) * 0.01

    nx, ny, nz, nt = img.shape
    masked_data = apply_mask(img, mask)
    _, n_voxels = masked_data.shape

    # Transpose to match ME-ICA convention (SxT instead of TxS)
    masked_data = masked_data.T

    if norm_mode == "dm":
        # Demean
        rmu = masked_data.mean(-1)
        dnorm = masked_data - rmu[:, np.newaxis]
    elif norm_mode == "vn":
        rmu = masked_data.mean(-1)
        rstd = masked_data.std(-1)
        dnorm = (masked_data - rmu[:, np.newaxis]) / rstd[:, np.newaxis]
    else:
        dnorm = masked_data

    # GoDec
    godec_outputs = {}
    if wavelet:
        print("++Wavelet transforming data")
        temp_data, cal = dwtmat(dnorm)
        thresh_ = temp_data.std() * thresh
        print(f"Setting threshold to {thresh_}")
    else:
        temp_data = dnorm.copy()
        thresh_ = thresh

    if method == "greedy":
        # GreGoDec
        godec_outputs = greedy_semisoft_godec(
            temp_data,
            ranks=ranks,
            tau=1,
            tol=1e-7,
            inpower=inpower,
            k=drank,
        )
    else:
        for rank in ranks:
            X_L, X_S, X_G = standard_godec(
                temp_data,
                thresh=thresh_,
                rank=rank,
                power=1,
                tol=1e-3,
                max_iter=500,
                random_seed=0,
                verbose=True,
            )

            godec_outputs[rank] = [X_L, X_S, X_G]

    if wavelet:
        print("++Inverse wavelet transforming outputs")
        for rank in godec_outputs.keys():
            godec_outputs[rank] = [idwtmat(arr, cal) for arr in godec_outputs[rank]]

    if norm_mode == "dm":
        for rank in godec_outputs.keys():
            godec_outputs[rank][0] = godec_outputs[rank][0] + rmu[:, np.newaxis]
    elif norm_mode == "vn":
        for rank in godec_outputs.keys():
            godec_outputs[rank][0] = (
                godec_outputs[rank][0] * rstd[:, np.newaxis]
            ) + rmu[:, np.newaxis]
            godec_outputs[rank][1] = godec_outputs[rank][1] * rstd[:, np.newaxis]
            godec_outputs[rank][2] = godec_outputs[rank][2] * rstd[:, np.newaxis]

    for rank, outputs in godec_outputs.items():
        lowrank_img = unmask(outputs[0].T, mask)
        sparse_img = unmask(outputs[1].T, mask)
        noise_img = unmask(outputs[2].T, mask)

        metadata = {
            "normalization": norm_mode,
            "wavelet": wavelet,
            "rank": rank,
            "k": drank,
            "p": inpower,
            "t": thresh,
        }
        metadata_file = os.path.join(out_dir, "dataset_description.json")
        with open(metadata_file, "w") as fo:
            json.dump(metadata, fo, sort_keys=True, indent=4)
        lowrank_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_lowrankts.nii.gz")
        )
        sparse_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_bold.nii.gz")
        )
        noise_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_errorts.nii.gz")
        )


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        dest="in_file",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="File to denoise with GODEC.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="Binary mask to apply to data.",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        help="Prefix for filenames generated.",
        default="",
    )
    parser.add_argument(
        "--method",
        dest="method",
        help="GODEC method.",
        default="greedy",
        choices=["greedy", "standard"],
    )
    parser.add_argument(
        "-r",
        "--ranks",
        dest="ranks",
        metavar="INT",
        type=int,
        nargs="+",
        help="Rank(s) of low rank component",
        default=[4],
    )
    parser.add_argument(
        "-k",
        "--increment",
        dest="drank",
        type=int,
        help="Rank search step size",
        default=2,
    )
    parser.add_argument(
        "-p",
        "--power",
        dest="inpower",
        type=int,
        help="Power for power method",
        default=2,
    )
    parser.add_argument(
        "-w",
        "--wavelet",
        dest="wavelet",
        help="Wavelet transform before GODEC",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        type=float,
        help="Threshold of some kind.",
        default=0.03,
    )
    parser.add_argument(
        "-n",
        "--norm_mode",
        dest="norm_mode",
        help=(
            "Normalization mode: variance normalization (vn), mean-centering (dm), "
            "or None (none)."
        ),
        default="vn",
        choices=["vn", "dm", "none"],
    )

    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    run_godec_denoising(**vars(options))


if __name__ == "__main__":
    _main()
