# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Plotting tools shared across MRIQC and FMRIPREP

Adapted for Power replication purposes.
"""

import numpy as np
import nibabel as nb
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec as mgs
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

from nilearn.plotting import plot_img
from nilearn.signal import clean
from nilearn._utils import check_niimg_4d
from nilearn._utils.niimg import _safe_get_data

import seaborn as sns
from seaborn import color_palette

DINA4_LANDSCAPE = (11.69, 8.27)


class fMRIPlot(object):
    """
    Generates the fMRI Summary Plot
    """
    __slots__ = ['func_file', 'mask_data',
                 'tr', 'seg_data', 'confounds', 'spikes']

    def __init__(self, func_file, mask_file=None, data=None, conf_file=None, seg_file=None,
                 tr=None, usecols=None, units=None, vlines=None, spikes_files=None):
        self.func_file = func_file
        func_nii = nb.load(func_file)
        self.tr = tr if tr is not None else func_nii.header.get_zooms()[-1]

        self.mask_data = np.ones_like(func_nii.get_data(), dtype='uint8')
        if mask_file:
            self.mask_data = nb.load(mask_file).get_data().astype('uint8')

        self.seg_data = None
        if seg_file:
            self.seg_data = nb.load(seg_file).get_data()

        if units is None:
            units = {}

        if vlines is None:
            vlines = {}

        self.confounds = {}
        if data is None and conf_file:
            data = pd.read_csv(conf_file, sep=r'[\t\s]+',
                               usecols=usecols, index_col=False)

        if data is not None:
            for name in data.columns.ravel():
                self.confounds[name] = {
                    'values': data[[name]].values.ravel().tolist(),
                    'units': units.get(name),
                    'cutoff': vlines.get(name)
                }

        self.spikes = []
        if spikes_files:
            for sp_file in spikes_files:
                self.spikes.append((np.loadtxt(sp_file), None, False))

    def plot(self, figure=None):
        """Main plotter"""
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=0.8)

        if figure is None:
            figure = plt.gcf()

        nconfounds = len(self.confounds)
        nspikes = len(self.spikes)
        nrows = 1 + nconfounds + nspikes
        nrows = 3

        # Create grid
        grid = mgs.GridSpec(nrows, 1, wspace=0.0, hspace=0.1,
                            height_ratios=[1] * (nrows - 1) + [5])

        grid_id = 0
        # Define nested GridSpec
        gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0],
                                         width_ratios=[1, 100], wspace=0.0)
        ax_cf = plt.subplot(gs[1])
        ax_cf.grid(False)

        gs2 = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1],
                                          width_ratios=[1, 100], wspace=0.0)
        ax_fd = plt.subplot(gs2[1])
        ax_fd.grid(False)

        for tsz, name, iszs in self.spikes:
            spikesplot(tsz, title=name, outer_gs=grid[grid_id], tr=self.tr,
                       zscored=iszs)
            # grid_id += 1

        if self.confounds:
            palette = color_palette('husl', nconfounds)

        for i, (name, kwargs) in enumerate(self.confounds.items()):
            tseries = kwargs.pop('values')

            if name != 'FD':
                confoundplot(
                    tseries, ax_cf, tr=self.tr, color=palette[i],
                    name=name, **kwargs)
            else:
                confoundplot(
                    tseries, ax_fd, tr=self.tr, color=palette[i],
                    name=name, **kwargs)
            # grid_id += 1

        cf_min, cf_max = -.25, .25
        ax_cf.set_ylim(cf_min, cf_max)
        ax_cf.set_yticks([cf_min, cf_max])
        ax_cf.set_yticklabels([cf_min, cf_max], fontsize=10)
        fd_min, fd_max = 0, .25
        ax_fd.set_ylim(fd_min, fd_max)
        ax_fd.set_yticks([fd_min, fd_max])
        ax_fd.set_yticklabels([fd_min, fd_max], fontsize=10)

        legend = ax_cf.legend(frameon=True, ncol=3, fontsize=10,
                              loc='upper right')
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

        legend = ax_fd.legend(frameon=True, fontsize=10, loc='upper right')
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

        plot_carpet(self.func_file, self.seg_data, subplot=grid[-1])
        # spikesplot_cb([0.7, 0.78, 0.2, 0.008])
        return figure


def plot_carpet(img, atlaslabels, detrend=True, nskip=0, size=(950, 800),
                subplot=None, title=None, output_file=None, legend=False,
                lut=None):
    """
    Plot an image representation of voxel intensities across time also know
    as the "carpet plot" or "Power plot". See Jonathan Power Neuroimage
    2017 Jul 1; 154:150-158.

    Parameters
    ----------

        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            4D input image
        atlaslabels: ndarray
            A 3D array of integer labels from an atlas, resampled into ``img`` space.
        detrend : boolean, optional
            Detrend and standardize the data prior to plotting.
        nskip : int
            Number of volumes at the beginning of the scan marked as nonsteady state.
        long_cutoff : int
            Number of TRs to consider img too long (and decimate the time direction
            to save memory)
        axes : matplotlib axes, optional
            The axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        legend : bool
            Whether to render the average functional series with ``atlaslabels`` as
            overlay.
    """
    img_nii = check_niimg_4d(img, dtype='auto')
    func_data = _safe_get_data(img_nii, ensure_finite=True)

    # Define TR and number of frames
    tr = img_nii.header.get_zooms()[-1]
    ntsteps = func_data.shape[-1]

    data = func_data[atlaslabels > 0].reshape(-1, ntsteps)
    seg = atlaslabels[atlaslabels > 0].reshape(-1)

    # Map segmentation
    if lut is None:
        lut = np.arange(9, dtype=int)

    # Apply lookup table
    newsegm = lut[seg.astype(int)]

    p_dec = 1 + data.shape[0] // size[0]
    if p_dec:
        data = data[::p_dec, :]
        newsegm = newsegm[::p_dec]

    t_dec = 1 + data.shape[1] // size[1]
    if t_dec:
        data = data[:, ::t_dec]

    # Detrend data
    v = (None, None)
    if detrend:
        data = clean(data.T, t_r=tr).T
        v = (-2, 2)

    # Order following segmentation labels
    order = np.argsort(newsegm)[::-1]

    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Define nested GridSpec
    wratios = [1, 100, 20]
    gs = mgs.GridSpecFromSubplotSpec(1, 2 + int(legend), subplot_spec=subplot,
                                     width_ratios=wratios[:2 + int(legend)],
                                     wspace=0.0)

    mycolors = sns.color_palette('Blues_r', 4)[:3] +\
        sns.color_palette('Greens', 3) +\
        sns.color_palette('YlOrBr', 4)[:2]
    mycolors = ListedColormap(mycolors)

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.imshow(newsegm[order, np.newaxis], interpolation='none', aspect='auto',
               cmap=mycolors)
    ax0.grid(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_color('none')
    ax0.spines["bottom"].set_visible(False)

    # Carpet plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(data[order, ...], interpolation='nearest', aspect='auto', cmap='gray',
               vmin=v[0], vmax=v[1])

    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max((int(data.shape[-1] + 1) // 10, int(data.shape[-1] + 1) // 5, 1))
    xticks = list(range(0, data.shape[-1])[::interval])

    n_vols = data.shape[-1]
    ax1.set_xticks([0, n_vols-1])
    final_sec = int((n_vols - 1) * tr)
    final_min = int(np.floor(final_sec / 60))
    final_sec = int(final_sec % 60)
    final_min = '{0}:{1:02d}'.format(final_min, final_sec)
    ax1.set_xticklabels(['0:00', final_min], fontsize=12)

    ax1.set_xlabel('Time (minutes)', fontsize=16)

    #ax1.set_xticks(xticks)
    #ax1.set_xlabel('time (s)', fontsize=12)
    #labels = tr * (np.array(xticks)) * t_dec
    #ax1.set_xticklabels(['%.02f' % t for t in labels.tolist()], fontsize=5)

    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        ax0.spines[side].set_color('none')
        ax0.spines[side].set_visible(False)
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_color('none')
    ax1.spines["left"].set_visible(False)

    if legend:
        gslegend = mgs.GridSpecFromSubplotSpec(
            5, 1, subplot_spec=gs[2], wspace=0.0, hspace=0.0)
        epiavg = func_data.mean(3)
        epinii = nb.Nifti1Image(epiavg, img_nii.affine, img_nii.header)
        segnii = nb.Nifti1Image(lut[atlaslabels.astype(int)], epinii.affine, epinii.header)
        segnii.set_data_dtype('uint8')

        nslices = epiavg.shape[-1]
        coords = np.linspace(int(0.10 * nslices), int(0.95 * nslices), 5).astype(np.uint8)
        for i, c in enumerate(coords.tolist()):
            ax2 = plt.subplot(gslegend[i])
            plot_img(segnii, bg_img=epinii, axes=ax2, display_mode='z',
                     annotate=False, cut_coords=[c], threshold=0.1, cmap=mycolors,
                     interpolation='nearest')

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file

    return [ax0, ax1], gs


def spikesplot(ts_z, outer_gs=None, tr=None, zscored=True, spike_thresh=6., title='Spike plot',
               ax=None, cmap='viridis', hide_x=True, nskip=0):
    """
    A spikes plot. Thanks to Bob Dogherty (this docstring needs be improved with proper ack)
    """

    if ax is None:
        ax = plt.gca()

    if outer_gs is not None:
        gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs,
                                         width_ratios=[1, 100], wspace=0.0)
        ax = plt.subplot(gs[1])

    # Define TR and number of frames
    if tr is None:
        tr = 1.

    # Load timeseries, zscored slice-wise
    nslices = ts_z.shape[0]
    ntsteps = ts_z.shape[1]

    # Load a colormap
    my_cmap = cm.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=float(nslices - 1))
    colors = [my_cmap(norm(sl)) for sl in range(nslices)]

    stem = len(np.unique(ts_z).tolist()) == 2
    # Plot one line per axial slice timeseries
    for sl in range(nslices):
        if not stem:
            ax.plot(ts_z[sl, :], color=colors[sl], lw=0.5)
        else:
            markerline, stemlines, baseline = ax.stem(ts_z[sl, :])
            plt.setp(markerline, 'markerfacecolor', colors[sl])
            plt.setp(baseline, 'color', colors[sl], 'linewidth', 1)
            plt.setp(stemlines, 'color', colors[sl], 'linewidth', 1)

    # Handle X, Y axes
    ax.grid(False)

    # Handle X axis
    last = ntsteps - 1
    ax.set_xlim(0, last)
    xticks = list(range(0, last)[::20]) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel('time (frame #)')
        else:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels(
                ['%.02f' % t for t in (tr * np.array(xticks)).tolist()])

    # Handle Y axis
    ylabel = 'slice-wise noise average on background'
    if zscored:
        ylabel += ' (z-scored)'
        zs_max = np.abs(ts_z).max()
        ax.set_ylim((-(np.abs(ts_z[:, nskip:]).max()) * 1.05,
                     (np.abs(ts_z[:, nskip:]).max()) * 1.05))

        ytick_vals = np.arange(0.0, zs_max, float(np.floor(zs_max / 2.)))
        yticks = list(
            reversed((-1.0 * ytick_vals[ytick_vals > 0]).tolist())) + ytick_vals.tolist()

        # TODO plot min/max or mark spikes
        # yticks.insert(0, ts_z.min())
        # yticks += [ts_z.max()]
        for val in ytick_vals:
            ax.plot((0, ntsteps - 1), (-val, -val), 'k:', alpha=.2)
            ax.plot((0, ntsteps - 1), (val, val), 'k:', alpha=.2)

        # Plot spike threshold
        if zs_max < spike_thresh:
            ax.plot((0, ntsteps - 1), (-spike_thresh, -spike_thresh), 'k:')
            ax.plot((0, ntsteps - 1), (spike_thresh, spike_thresh), 'k:')
    else:
        yticks = [ts_z[:, nskip:].min(),
                  np.median(ts_z[:, nskip:]),
                  ts_z[:, nskip:].max()]
        ax.set_ylim(0, max(yticks[-1] * 1.05, (yticks[-1] - yticks[0]) * 2.0 + yticks[-1]))
        # ax.set_ylim(ts_z[:, nskip:].min() * 0.95,
        #             ts_z[:, nskip:].max() * 1.05)

    ax.annotate(
        ylabel, xy=(0.0, 0.7), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        va='center', ha='left', color='gray', size=4,
        bbox={'boxstyle': 'round', 'fc': 'w', 'ec': 'none', 'color': 'none',
              'lw': 0, 'alpha': 0.8})
    ax.set_yticks([])
    ax.set_yticklabels([])

    # if yticks:
    #     # ax.set_yticks(yticks)
    #     # ax.set_yticklabels(['%.02f' % y for y in yticks])
    #     # Plot maximum and minimum horizontal lines
    #     ax.plot((0, ntsteps - 1), (yticks[0], yticks[0]), 'k:')
    #     ax.plot((0, ntsteps - 1), (yticks[-1], yticks[-1]), 'k:')

    for side in ["top", "right"]:
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    if not hide_x:
        ax.spines["bottom"].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.spines["bottom"].set_color('none')
        ax.spines["bottom"].set_visible(False)

    # ax.spines["left"].set_position(('outward', 30))
    # ax.yaxis.set_ticks_position('left')
    ax.spines["left"].set_visible(False)
    ax.spines["left"].set_color(None)

    # labels = [label for label in ax.yaxis.get_ticklabels()]
    # labels[0].set_weight('bold')
    # labels[-1].set_weight('bold')
    if title:
        ax.set_title(title)
    return ax


def spikesplot_cb(position, cmap='viridis', fig=None):
    # Add colorbar
    if fig is None:
        fig = plt.gcf()

    cax = fig.add_axes(position)
    cb = ColorbarBase(cax, cmap=cm.get_cmap(cmap), spacing='proportional',
                      orientation='horizontal', drawedges=False)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Inferior', '(axial slice)', 'Superior'])
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_tick_params(width=0)
    return cax


def confoundplot(tseries, ax_ts, gs_dist=None, name=None,
                 units=None, tr=None, hide_x=True, color='b', nskip=0,
                 cutoff=None, ylims=None):

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.
    ntsteps = len(tseries)
    tseries = np.array(tseries)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel('time (frame #)')
        else:
            ax_ts.set_xlabel('time (s)')
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels(['%.02f' % t for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += ' [%s]' % units

        #ax_ts.annotate(
        #    name, xy=(0.0, 0.7), xytext=(0, 0), xycoords='axes fraction',
        #    textcoords='offset points', va='center', ha='left',
        #    color=color, size=8,
        #    bbox={'boxstyle': 'round', 'fc': 'w', 'ec': 'none',
        #          'color': 'none', 'lw': 0, 'alpha': 0.8})

    '''
    for side in ["top", "right"]:
        ax_ts.spines[side].set_color('none')
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines["bottom"].set_position(('outward', 20))
        ax_ts.xaxis.set_ticks_position('bottom')
    else:
        ax_ts.spines["bottom"].set_color('none')
        ax_ts.spines["bottom"].set_visible(False)
    '''
    #ax_ts.spines["left"].set_position(('outward', 30))
    #ax_ts.spines["left"].set_color('none')
    #ax_ts.spines["left"].set_visible(False)
    ax_ts.yaxis.set_ticks_position('left')

    # Calculate Y limits
    def_ylims = [tseries[~np.isnan(tseries)].min() - 0.1 * abs(tseries[~np.isnan(tseries)].min()),
                 1.1 * tseries[~np.isnan(tseries)].max()]
    if ylims is not None:
        if ylims[0] is not None:
            def_ylims[0] = min([def_ylims[0], ylims[0]])
        if ylims[1] is not None:
            def_ylims[1] = max([def_ylims[1], ylims[1]])

    # Add space for plot title and mean/SD annotation
    def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

    ax_ts.set_ylim(def_ylims)
    # yticks = sorted(def_ylims)
    ax_ts.set_yticks([])
    ax_ts.set_yticklabels([])
    # ax_ts.set_yticks(yticks)
    # ax_ts.set_yticklabels(['%.02f' % y for y in yticks])

    # Annotate stats
    maxv = tseries[~np.isnan(tseries)].max()
    mean = tseries[~np.isnan(tseries)].mean()
    stdv = tseries[~np.isnan(tseries)].std()
    p95 = np.percentile(tseries[~np.isnan(tseries)], 95.0)

    #stats_label = (r'max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} '
    #               r'$\bullet$ $\sigma$: {sigma:.3f}').format(
    #    max=maxv, mean=mean, units=units or '', sigma=stdv)
    #ax_ts.annotate(
    #    stats_label, xy=(0.98, 0.7), xycoords='axes fraction',
    #    xytext=(0, 0), textcoords='offset points',
    #    va='center', ha='right', color=color, size=4,
    #    bbox={'boxstyle': 'round', 'fc': 'w', 'ec': 'none', 'color': 'none',
    #          'lw': 0, 'alpha': 0.8}
    #)

    # Annotate percentile 95
    #ax_ts.plot((0, ntsteps - 1), [p95] * 2, linewidth=.1, color='lightgray')
    #ax_ts.annotate(
    #    '%.2f' % p95, xy=(0, p95), xytext=(-1, 0),
    #    textcoords='offset points', va='center', ha='right',
    #    color='lightgray', size=3)

    if cutoff is None:
        cutoff = []

    for i, thr in enumerate(cutoff):
        pass
        #ax_ts.plot((0, ntsteps - 1), [thr] * 2,
        #           linewidth=.2, color='dimgray')

        #ax_ts.annotate(
        #    '%.2f' % thr, xy=(0, thr), xytext=(-1, 0),
        #    textcoords='offset points', va='center', ha='right',
        #    color='dimgray', size=3)

    ax_ts.plot(tseries, color=color, label=name)
    ax_ts.set_xlim((0, ntsteps - 1))

    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.displot(tseries, vertical=True, ax=ax_dist)
        #ax_dist.set_xlabel('Timesteps')
        #ax_dist.set_ylim(ax_ts.get_ylim())
        #ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist]
    return ax_ts
