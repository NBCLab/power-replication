"""Run GODEC."""
import argparse
import os

import numpy as np
import pywt
from nilearn._utils.niimg import load_niimg
from nilearn.masking import apply_mask, unmask
from scipy.linalg import qr
from scipy.sparse.linalg import svds

__version__ = "0.1"


def wthresh(a, thresh):
    # Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)


"""
# Wrap it in a function that gives me more context:
def ipsh():
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)

    frame = inspect.currentframe().f_back
    msg   = 'Stopped at {0.f_code.co_filename} at line {0.f_lineno}'.format(frame)

    # Go back one level!
    # This is needed because the call to ipshell is inside the function ipsh()
    ipshell(msg,stack_depth=2)
"""


def godec(
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
        Q, R = qr(Y2)
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
        print("Finished at iteration %d" % (itr))

    return L, S, G


def dwtmat(mmix):
    """Apply a discrete wavelet transform to a matrix."""
    # ipdb.set_trace()
    print("++Wavelet transforming data")
    lt = len(np.hstack(pywt.dwt(mmix[0], "db2")))
    mmix_wt = np.zeros([mmix.shape[0], lt])
    for ii in range(mmix_wt.shape[0]):
        wtx = pywt.dwt(mmix[ii], "db2")
        cAlen = len(wtx[0])
        mmix_wt[ii] = np.hstack(wtx)
    return mmix_wt, cAlen


def idwtmat(mmix_wt, cAl):
    """Apply a discrete inverse wavelet transform to a matrix."""
    print("++Inverse wavelet transforming")
    lt = len(pywt.idwt(mmix_wt[0, :cAl], mmix_wt[0, cAl:], "db2", correct_size=True))
    mmix_iwt = np.zeros([mmix_wt.shape[0], lt])
    for ii in range(mmix_iwt.shape[0]):
        mmix_iwt[ii] = pywt.idwt(
            mmix_wt[ii, :cAl], mmix_wt[ii, cAl:], "db2", correct_size=True
        )
    return mmix_iwt


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

    def rank_estimator_adaptive(inv):
        if inv["estrank"] == 1:
            dR = abs(np.diag(R))
            drops = dR[:-1] / dR[1:]
            print(dR.shape)
            dmx = max(drops)
            imx = np.argmax(drops)
            rel_drp = (inv["rankmax"] - 1) * dmx / (sum(drops) - dmx)
            # ipdb.set_trace()
            if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or inv[
                "itr_rank"
            ] > maxitr_reduce_rank:  # %bar(drops); pause;
                inv["rrank"] = max([imx, np.floor(0.1 * inv["rankmax"]), rank_min])
                # inv['error'][ii] = np.linalg.norm(res)/normz;
                inv["estrank"] = 0
                inv["itr_rank"] = 0
            return inv

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

        for iteri in range(inpower + 1):
            print("r %i, iteri %i, rrank %i, alf %i" % (r, iteri, rrank, alf))
            # Update of X
            X = L.dot(Y.T)
            # CHECK dot notation
            # if estrank==1:
            #    qro=qr(X,mode='economic');   #CHECK qr output formats    #stopping here on 1/12
            # 	X = qro[0];
            # 	R = qro[1];
            # else:
            X, R = qr(X, mode="reduced")
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
            ii = iii + iteri - 1
            # ok
            # embed()
            error[ii] = np.linalg.norm(T[:]) / normD
            if error[ii] < tol:
                stop = True
                break

            # adjust estrank
            if estrank >= 1:
                outr = rank_estimator_adaptive(locals())
                estrank, rankmax, itr_rank, error, rrank = [
                    outr["estrank"],
                    outr["rankmax"],
                    outr["itr_rank"],
                    outr["error"],
                    outr["rrank"],
                ]

            if rrank != rankmax:
                rankmax = rrank
                if estrank == 0:
                    alf = 0
                    continue

            # adjust alf
            ratio = error[ii] / error[ii - 1]
            if np.isinf(ratio):
                ratio = 0
            print(ii, error, ratio)
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
            print("updating L")
            X1 = X
            Y1 = Y
            L1 = L
            S1 = S
            T1 = T
            # ipdb.set_trace()
            L = L + ((1 + alf) * (T))

            # Add coreset
            if iteri > 8:
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


def tedgodec(
    img,
    mask,
    ranks=[2],
    drank=2,
    inpower=2,
    thresh=10,
    max_iter=500,
    rmu_data=None,
    norm_mode=None,
    wavelet=False,
):
    """
    norm_mode : {None, "psc", "dm", "vn"}, optional
        Default is None.
    """
    nx, ny, nz, nt = img.shape
    masked_data = apply_mask(img, mask)
    _, n_voxels = masked_data.shape

    # Transpose to match ME-ICA convention (SxT instead of TxS)
    masked_data = masked_data.T

    if norm_mode == "psc":
        # Convert to PSC
        rmu = rmu_data.mean(-1)
        dnorm = ((masked_data / rmu[:, np.newaxis]) - 1) * 100
        thresh = 0.1
    elif norm_mode == "dm":
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
    out = {}
    if wavelet:
        # Apply wavelet transform
        X_wt, cal = dwtmat(dnorm)
        # Run GODEC
        X_L, X_S, X_G = godec(
            X_wt,
            thresh=X_wt.std() * thresh,
            rank=ranks[0],
            power=1,
            tol=1e-3,
            max_iter=max_iter,
            random_seed=0,
            verbose=True,
        )
        # Apply inverse wavelet transform to outputs
        X_L = idwtmat(X_L, cal)
        X_S = idwtmat(X_S, cal)
        X_G = idwtmat(X_G, cal)
    else:
        X_L, X_S, X_G = godec(
            dnorm,
            thresh=thresh,
            rank=ranks[0],
            power=1,
            tol=1e-3,
            max_iter=max_iter,
            random_seed=0,
            verbose=True,
        )

    out[ranks[0]] = [X_L, X_S, X_G]

    # GreGoDec
    # out = greedy_semisoft_godec(dnorm,ranks,1,1e-7,inpower,drank)

    if norm_mode == "psc":
        for ii in range(len(out)):
            out[ii] = ((out[ii] / 100) + 1) * rmu[:, np.newaxis]
    elif norm_mode == "dm":
        # Remean
        out[0] = out[0] + rmu[:, np.newaxis]
    elif norm_mode == "vn":
        for rr in out.keys():
            out[rr][0] = out[rr][0] * rstd[:, np.newaxis] + rmu[:, np.newaxis]
            out[rr][1] = out[rr][1] * rstd[:, np.newaxis]
            out[rr][2] = out[rr][2] * rstd[:, np.newaxis]

    return out


def run_godec_denoising(
    in_file,
    mask,
    out_dir=".",
    prefix="",
    ranks=[2],
    norm_mode=None,
    thresh=None,
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
    img = load_niimg(in_file)
    mask = load_niimg(mask)

    if thresh is None:
        masked_data = apply_mask(img, mask).T
        mu = masked_data.mean(axis=-1)
        thresh = np.median(mu[mu != 0]) * 0.01

    godec_outputs = tedgodec(
        img,
        mask,
        ranks=ranks,
        drank=drank,
        inpower=inpower,
        thresh=thresh,
        max_iter=500,
        norm_mode=norm_mode,
        wavelet=wavelet,
    )

    for rank, outputs in godec_outputs.items():
        lowrank_img = unmask(outputs[0], mask)
        sparse_img = unmask(outputs[1], mask)
        noise_img = unmask(outputs[2], mask)

        if norm_mode is None:
            name_norm_mode = ""
        else:
            name_norm_mode = f"n{norm_mode}"

        if wavelet:
            name_norm_mode = f"w{name_norm_mode}"

        suffix = f"{name_norm_mode}r{rank}k{drank}p{inpower}t{thresh}"

        lowrank_img.to_filename(op.join(out_dir, f"{prefix}lowrank_{suffix}.nii.gz"))
        sparse_img.to_filename(op.join(out_dir, f"{prefix}sparse_{suffix}.nii.gz"))
        noise_img.to_filename(op.join(out_dir, f"{prefix}noise_{suffix}.nii.gz"))


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if (arg is not None) and (not os.path.isfile(arg)):
        parser.error(f"The file {arg} does not exist!")

    return arg


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
        "-r",
        "--rank",
        dest="rank",
        metavar='INT',
        type=int,
        nargs='+',
        help="Rank(s) of low rank component",
        default=[2],
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
        dest="power",
        type=int,
        help="Power for power method",
        default=2,
    )
    parser.add_argument(
        "-w",
        "--wavelet",
        dest="wavelet",
        help="Wavelet transform before GoDec",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        type=float,
        help="Threshold of some kind.",
        default=2,
    )
    parser.add_argument(
        "-n",
        "--norm_mode",
        dest="norm_mode",
        help="Normalization mode",
        default="vn",
        choices=["vn", "psc", "dm", "none"],
    )

    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    run_godec_denoising(**options)
