"""
Script to find representative profiles and to visualise the high-dimensional data for the ERA5_NS data.

Author: Sebastiaan Jamaer
Date: 16-03-2022
"""
from RZse import RZSEfit
from os import path
import sys

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import netCDF4 as nc
from RZse import physical_parameters, RZsemodel
from Line_optimization import theta_approximation
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
import cartopy.crs as ccrs
import os
from matplotlib import cm
import cmasher as cmr

cmap_discrete = cm.get_cmap('tab10')
# from sklearn_extra.cluster import KMedoids
year = '5yeardata'
type = 'SERZ'
RESULTS_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/'+ str(year) + '/results/clustering/' + type + '/'
FIG_PATH='/scratch/leuven/338/vsc33802/ERA5_europe/'+str(year)+ '/figures/clustering/' + type + '/'
DATANP_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/' + str(year)+ '/npfiles/' + type + '/'
LATS_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/' + str(year) + 'lats.npy'
LONS_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/' + str(year) + 'lons.npy'
LSM_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/' + str(year) + '/lsm_regridded.nc'
NAMEFILE_PATH = '/scratch/leuven/338/vsc33802/ERA5_europe/' + str(year) + '/5yeardata_namefile.txt'
import pickle
import gc

##########################
# Load masks
##########################

def loadSERZasNumpy(landmask, pathFits):
    a = np.load(pathFits + 'a.npy')
    b = np.load(pathFits + 'b.npy')
    c = np.load(pathFits + 'c.npy')
    l = np.load(pathFits + 'l.npy')
    dh = np.load(pathFits + 'dh.npy')
    alpha = np.load(pathFits + 'alpha.npy')
    # thm = np.load(pathFits + 'fits_thm.npy')
    dataNp = np.stack([a, b, c, l, dh, alpha], axis=-1)
    dataNp = np.float32(dataNp)
    # dataNp = dataNp[0:100]

    # Mask out desired pixels
    shape_original = dataNp.shape
    landmask = getLandMask(landmask)
    dataNp = limit_to_mask(dataNp, landmask, time_axis=True)
    print("Full data has shape : " + str(dataNp.shape))
    return dataNp, shape_original, landmask
def loadLinearasNumpy(landmask, pathFits):
    # th0 = np.load(pathFits + 'th0.npy') # tho0 is a constant, not in the data
    alpha1 = np.load(pathFits + 'alpha1.npy')
    zeta1 = np.load(pathFits + 'zeta1.npy')
    alpha2 = np.load(pathFits + 'alpha2.npy')
    zeta2 = np.load(pathFits + 'zeta2.npy')
    alpha3 = np.load(pathFits + 'alpha3.npy')
    # thm = np.load(pathFits + 'fits_thm.npy')
    dataNp = np.stack([alpha1, zeta1, alpha2, zeta2, alpha3], axis=-1)
    dataNp = np.float32(dataNp)
    # dataNp = dataNp[0:100]

    # Mask out desired pixels
    shape_original = dataNp.shape
    landmask = getLandMask(landmask)
    dataNp = limit_to_mask(dataNp, landmask, time_axis=True)
    print("Full data has shape : " + str(dataNp.shape))
    return dataNp, shape_original, landmask

def computeProfiles(z, par, type='SERZ'):
    """
    Compute the temperature profiles for the given heights with the given parameters with the algorithm indicated in
    type. IMPORTANT: your parameter vector should be correct with respect to the type of profile.
    :param z:
    :param par:
    :return:
    """
    if type=='SERZ':
        profile = RZsemodel(z, par[0], par[1], par[2], 0, par[3], par[4], par[5])
    elif type=='linear':
        profile = theta_approximation(3, z, [0, par[0], par[1], par[2], par[3], par[4]])
    else:
        assert False, 'Profile type is not implemented.'
    return profile

def getTimeMask(masktype):
    namefile = pd.read_csv(NAMEFILE_PATH)
    namefile_np = namefile.to_numpy()
    temporal_matrix = np.zeros((namefile_np.shape[0], 3), dtype=int)  # columns stand for month, day and hour
    masks = {}
    for i in range(len(namefile_np)):
        temporal_matrix[i, 0] = int(namefile_np[i, 0][15:17])
        temporal_matrix[i, 1] = int(namefile_np[i, 0][17:19])
        temporal_matrix[i, 2] = int(namefile_np[i, 0][19:21])
    if masktype == 'seasonal':
        masks['winter'] = np.logical_or(temporal_matrix[:, 0] == 12, temporal_matrix[:, 0] < 3)
        masks['spring'] = np.logical_and(temporal_matrix[:, 0] > 2, temporal_matrix[:, 0] < 6)
        masks['summer'] = np.logical_and(temporal_matrix[:, 0] > 5, temporal_matrix[:, 0] < 9)
        masks['autumn'] = np.logical_and(temporal_matrix[:, 0] > 8, temporal_matrix[:, 0] < 12)
    elif masktype == 'temporal':
        masks['day'] = np.logical_and(temporal_matrix[:, 2] > 11, temporal_matrix[:, 2] < 18)
        masks['evening'] = np.logical_and(temporal_matrix[:, 2] > 17, temporal_matrix[:, 2] < 24)
        masks['night'] = np.logical_or(temporal_matrix[:, 2] == 24, temporal_matrix[:, 2] < 6)
        masks['morning'] = np.logical_and(temporal_matrix[:, 2] > 5, temporal_matrix[:, 2] < 12)
    else:
        masks['full'] = np.full(temporal_matrix.shape[0], True)
    return masks
def getLandMask(masktype):
    """
    Returns boolean mask for masking land or sea pixels (or anything else you want, just implement it here).
    Note that this goes hand in hand with the data you are using (dimensions should fit).

    TODO: Maybe pool all of this together in data class

    :param masktype: string or None
    :return: boolean mask (nd.array)
    """
    lsm = nc.Dataset(LSM_PATH)
    if masktype is None or masktype=='all':
        mask = lsm['lsm'][0].data>-np.infty
    elif masktype=='land':
        mask = lsm['lsm'][0].data > 0.5
    elif masktype=='sea' or masktype=='ocean':
        mask = lsm['lsm'][0].data < 0.5
    else:
        assert False, 'Land mask not implemented'
    return mask

##########################
# mask geographical data
##########################
def limit_to_mask(data,mask, time_axis=True):
    if time_axis:
        return data[:, mask]
    else:
        return data[mask]
def reconstruct_data_from_masked_data(masked_data, mask, output_shape, time_axis=True):
    data = np.full(output_shape, np.NaN)
    if time_axis:
        data[:,mask] = masked_data
    else:
        data[mask]   = masked_data
    return data

##########################
# plotting functions
##########################
def clippedcolorbar(CS, fig, ax, **kwargs):
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    # fig = CS.ax.get_figure()
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    return fig.colorbar(m, ax=ax, **kwargs)
def representativesFigures(representatives, pathFiguresRepresentatives, type='SERZ'):
    """
    Plot and save the representatives.
    :param pathFiguresRepresentatives:
    :return:
    """

    z = np.linspace(10, 5000, 1000)
    representativesProfiles = {}
    fig, axes = plt.subplots(2, len(representatives)//2, squeeze=False)
    for i in range(len(representatives)):
        ax = axes[i//(len(representatives)//2), i%(len(representatives)//2)]
        representativesProfiles[i] = computeProfiles(z, representatives[i], type)
        ax.plot(representativesProfiles[i], z)
        ax.set_xlim([-8, 18])
        if i%(len(representatives)//2)!=0:
            ax.set_yticks([])
        else:
            ax.set_yticks([0, 2000, 4000])
            ax.set_yticklabels(['0', '2', '4'])
            ax.set_ylabel("z (km)")
        if i//(len(representatives)//2)==0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'   $\theta - \theta_0$ (K)')
        ax.set_title(str(i+1), bbox=dict(boxstyle="square",
                          ec="black",
                          fc="lightgrey"), pad=-1, ma='center')
        # ax.grid()
    fig.suptitle('Representatives')
    fig.savefig(pathFiguresRepresentatives + type + '_representativesFigure.png')
    plt.close(fig)
def fingerprintBarplot(fingerprint, polar=True, ax=None, yticks=False):
    """
    Generate the fingerprint plot.
    :param fingerprint:
    :return: nd.array of values between 0 and 1
    """
    if ax is None:
        fig = plt.figure(figsize=[0.7*6.4, 0.7*4.8])
        ax = fig.add_subplot(1,1,1, polar=polar)

    if polar:
        angles = np.pi/2 -np.array([i/len(fingerprint) *2* np.pi + np.pi/len(fingerprint) for i in range(len(fingerprint))])
    else:
        angles = np.pi / 2 + np.array(
            [i / len(fingerprint) * 2 * np.pi + np.pi / len(fingerprint) for i in range(len(fingerprint))])
    heights = fingerprint
    width = 1.8 * np.pi / len(fingerprint)    # little space between the bar and the label

    labelPadding = 0.02
    labels = [i+1 for i in range(len(fingerprint))]

    bars = ax.bar(x=angles,
                  height=heights,
                  width=width,
                  alpha=1,
                  facecolor='C0')
    ax.set_xticklabels([])
    ax.set_yticks([0.10 * i for i in range(5)])
    ax.set_ylim([0, 0.4])
    # ax.set_yticklabels([None, None, '20%', None, '40%'])
    if not yticks:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels([None, '10%', '20%', '30%', None], ha='center', va='center', fontsize='small')
    if not polar:
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%'])

    # Visualization with the help of https://python-graph-gallery.com/circular-barplot-basic
    for bar, angle, height, label in zip(bars, angles, heights, labels):
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle,
            y=0.4,
            s=label,
            ha=alignment,
            va='center',
            rotation_mode="default",
            bbox=dict(boxstyle="square",
                      ec="black",
                      fc="lightgrey"
                      )
        )
    # plt.show()
def aggregatedFingerprintBarplot(fingerprints, color, label, ax, show_labels=False, polar=True, yticks=False):
    # print(color)

    # ax = fig.add_subplot(1, 1, 1, polar=True)
    if polar:
        angles = np.pi/2 -np.array([i/len(fingerprints[0]) *2* np.pi + np.pi/len(fingerprints[0]) for i in range(len(fingerprints[0]))])
    else:
        angles = np.pi / 2 + np.array(
            [i / len(fingerprints[0]) * 2 * np.pi + np.pi / len(fingerprints[0]) for i in range(len(fingerprints[0]))])
    width = 1.8*np.pi/len(fingerprints[0])
    # little space between the bar and the label
    labelPadding = 1
    labels = [i+1 for i in range(len(fingerprints[0]))]

    quantile05 = np.quantile(fingerprints, 0.05, axis=0)
    quantile50 = np.quantile(fingerprints, 0.5, axis=0)
    quantile95 = np.quantile(fingerprints, 0.95, axis=0)

    bars = ax.bar(x=angles,
           height=quantile50,
           width=width,
           alpha=1,
           facecolor=color)

    ax.set_xticklabels([])
    ax.set_yticks([0.10*i for i in range(5)])
    ax.set_ylim([0, 0.4])
    # ax.set_yticklabels([None, None, '20%', None, '40%'])
    if not yticks:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels([None, '10%', '20%', '30%', None], ha='center', va='center', fontsize='small')
    # ax.set_xticks()
    if not polar:
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%'])


    errorbars = True
    if errorbars:
        for low, angle, high in zip(quantile05, angles, quantile95):
            ax.plot([angle, angle], [low, high], color='k', lw=0.5)

    show_labels = True
    if show_labels:

        # Visualization with the help of https://python-graph-gallery.com/circular-barplot-basic
        for bar, angle, height, label in zip(quantile05, angles, quantile95, labels):
            # Labels are rotated. Rotation must be specified in degrees :(
            rotation = np.rad2deg(angle)

            # Flip some labels upside down
            alignment = ""
            if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                alignment = "center"
                rotation = rotation + 180
            else:
                alignment = "center"

            # Finally add the labels
            ax.text(
                x=angle,
                y=0.4,
                s=label,
                ha=alignment,
                va='center',
                rotation_mode="default",
                bbox=dict(boxstyle="square",
                          ec="black",
                          fc="lightgrey"
                          )
            )

    return None


def aggregatedFingerprints(fingerprints, labeledGrid, pathFigureslabeledGridMasked, cmap_discrete):
    """
    Plot and save the aggregated fingerprints for each cluster.
    :param cmap:
    :param pathFigureslabeledGridMasked:
    :return:
    """
    # print(labeledGrid.shape)
    labels = np.unique(labeledGrid[~np.isnan(labeledGrid)])
    fig, axes = plt.subplots(2, int(len(labels)/2), subplot_kw={'polar':True}, squeeze=False)
    axes_count = 0
    for i in labels:
        # print(fingerprints[labeledGrid == i].shape)
        # print(cmap_discrete(i))
        aggregatedFingerprintBarplot(fingerprints[labeledGrid == i], cmap_discrete(int(i)), i, axes[axes_count%2,axes_count//2])
        axes_count+=1
    # fig.savefig(pathFigureslabeledGridMasked + 'aggFingerprints.png')
    # plt.close(fig)
    return fig

def submapClusters(labeledGrid, nClusters, mask, pathFigureslabeledGridMasked, axes):

    # Is always the same

    lats = np.load(LATS_PATH)
    lons = np.load(LONS_PATH)

    # fig, axes = plt.subplots(1, 1, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    vmin = None
    vmax = None
    dv = 0.01

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cmap_new = cmr.get_sub_cmap(cmap_discrete, 0, nClusters / 10)
    cb = axes.pcolormesh(lons, lats, labeledGrid,
                       transform=ccrs.PlateCarree(), cmap=cmap_new, vmin=-0.5, vmax=nClusters-0.5)
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", "5%", pad="3%")
    # plt.colorbar(im, cax=cax)
    # cax = plt.axes([0.2, 0.075, 0.6, 0.03])
    # plt.colorbar(cax=cax, mappable=cb, orientation='horizontal', ticks=[i for i in range(nClusters)])
    gl = axes.gridlines(draw_labels=True)
    gl.bottom_labels = False
    gl.right_labels = False
    axes.coastlines()
    return cb
def combinedClustersFingerprints(labeledGrid,fingerprints, nClusters,mask,pathFigureslabeledGridMasked,cmap_discrete):

    if nClusters>=13:
        assert False, 'combined clusters fingerprints figure not implemented for 10 or more clusters'
    elif nClusters==10:
        shape_subplots = (4, 9)
        shape_clustermap = (4, 6)
        shape_fingerprints = (4, 3)
    elif nClusters>=7:
        shape_subplots = (3, 7)
        shape_clustermap = (3, 4)
        shape_fingerprints = (3, 3)
    elif nClusters>=5:
        shape_subplots = (2, 7)
        shape_clustermap = (2, 4)
        shape_fingerprints = (2, 3)
    elif nClusters>=3:
        shape_subplots = (2, 5)
        shape_clustermap = (2, 3)
        shape_fingerprints = (2, 2)
    elif nClusters==2:
        shape_subplots = (2, 3)
        shape_clustermap = (2, 2)
        shape_fingerprints = (2, 1)
    else:
        assert False, 'Need at least 2 clusters for visualisation'
    fig = plt.figure(figsize=(shape_subplots[1]*2, shape_subplots[0]*2), frameon=True)

    labels = np.unique(labeledGrid[~np.isnan(labeledGrid)])


    # First map mapClusters into the first slot

    axClusters = plt.subplot2grid(shape_subplots, (0, 0), colspan=shape_clustermap[1], rowspan=shape_clustermap[0], projection= ccrs.PlateCarree())
    cb = submapClusters(labeledGrid, nClusters, mask, pathFigureslabeledGridMasked, axClusters)
    # cax = plt.axes([0.2, 0.075, 0.6, 0.03])
    # plt.colorbar(cax=cax, mappable=cb, orientation='horizontal', ticks=[i for i in range(nClusters)])
    axes_counter = 0
    for label in labels:
        axFingerprint = plt.subplot2grid(shape_subplots,
                                         (0+axes_counter//shape_fingerprints[1],
                                          shape_clustermap[1]+axes_counter%shape_fingerprints[1]),
                                         projection='polar')
        aggregatedFingerprintBarplot(fingerprints[labeledGrid == label], cmap_discrete(int(label)), label,
                                     axFingerprint)
        axes_counter += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


###########################
# Fingerprint funcitons
def orderRepresentatives(representatives, fingerprints, index_ordering=2, type='SERZ'):
    """
    Order the representatives according to their value of c. Return the ordered representatives and the permutated
    fingerprints
    :param representatives:
    :param fingerprints:
    :param index_ordering: for which index of representatives shoulc be ordered
    :return:
    """
    if type=='SERZ':
        order = np.argsort(representatives[:, index_ordering])  # Sort according to the values of c
        representativesOrdered = representatives[order]
        fingerprintsOrdered = fingerprints[..., order]
    elif type=='linear':
        order = np.argsort(representatives[:, 0])  # Sort according to the values of c
        representativesOrdered = representatives[order]
        fingerprintsOrdered = fingerprints[..., order]
    else:
        assert False, 'order representatives not implemented'
    return representativesOrdered, fingerprintsOrdered
def europe_fingerprinting(nrep, nClusters, maskname='full', landmask='all', type='SERZ'):
    ndim_par = 1  # If the parameter would me multidimensional we have to change the code.
    # namefile = pd.read_csv(DATA_PATH + str(year) + '_namefile.txt')
    masks = getTimeMask(maskname)

    # Path to the fits
    pathFits = DATANP_PATH

    # Path to save the intermediate data to. Since the fingerprints file is quite large we only save it once for every
    # number of representatives

    pathRepresentatives = RESULTS_PATH + landmask + '/R' + str(nrep) + '/'
    if not os.path.exists(pathRepresentatives):
        os.makedirs(pathRepresentatives)
    for mask in masks.keys():
        pathSavelabeledGridMasked = pathRepresentatives + mask + '/C' + str(nClusters) + '/'
        if not os.path.exists(pathSavelabeledGridMasked):
            os.makedirs(pathSavelabeledGridMasked)
        pathSaveFingerprintsRepresentativesMasked = pathRepresentatives + mask + '/'
        if not os.path.exists(pathSaveFingerprintsRepresentativesMasked):
            os.makedirs(pathSaveFingerprintsRepresentativesMasked)

    # Path where the figures are saved.
    pathFiguresRepresentatives = FIG_PATH + landmask + '/R' + str(nrep) + '/'
    if not os.path.exists(pathFiguresRepresentatives):
        os.makedirs(pathFiguresRepresentatives)
    for mask in masks.keys():
        pathFigureslabeledGridMasked = pathFiguresRepresentatives + mask + '/C' + str(nClusters) + '/'
        if not os.path.exists(pathFigureslabeledGridMasked):
            os.makedirs(pathFigureslabeledGridMasked)

    # Load the fitting data as a numpy array of shape (ntime, nlat, nlon, npar)
    def loadDataAsNumpy(landmask, data='SERZ'):
        if data=='SERZ':
            return loadSERZasNumpy(landmask, pathFits)
        elif data=='linear':
            return loadLinearasNumpy(landmask, pathFits)
    dataNp, shape_original, landmask = loadDataAsNumpy(landmask, type)

    # Compute 'the best' nrep representatives of the flattened data of shape (ntime*nlat*nlon, npar)
    def computeRepresentatives(dataNp, nrep):
        """
        Compute the nrep best representatives for the flattened dataNp.

        Save the representatives and return them.
        Save the classificaton algorithm as well to be able to compute the fingerprint of each pixel for new data.
        :param dataNp: data to compute the representatives of. (nd.array of shape (ntime, nlat, nlon, npar)).

        Since we are working with large datasets, we have included some print statements to keep track of the progress
        of this routine.
        :param nrep: Number of representatives (int, default = 10)
        :return representatives: (nd.array, shape (nrep, npar)) with the representative profiles.
        :return fingerprints: (nd.array, shape (nlat, nlon, nrep)) with the fingerprints of each pixel
        """

        shape = dataNp.shape
        print('Reshaping input ...')
        dataNp = dataNp.reshape(np.prod(shape[:-ndim_par]), shape[-ndim_par])
        # TODO: Make this more general so that we can take more than 1 parameter dimension

        # Preprocess data before clustering
        print('Scaling input ...')
        # scaler = StandardScaler().fit(dataNp)
        scaler = QuantileTransformer(n_quantiles=5000, subsample=50000, random_state=2022, output_distribution='normal').fit(dataNp)
        dataNp = scaler.transform(dataNp)

        print('Clustering ...')
        kmeans = MiniBatchKMeans(n_clusters=nrep, random_state=2022, verbose=2, max_iter=200, n_init=20,
                                 batch_size=256*50, max_no_improvement=None, tol=1e-4, reassignment_ratio=0.1)
        kmeans = kmeans.fit(dataNp)

        representatives = kmeans.cluster_centers_
        # Postprocessing representatives
        print('Transforming representatives ...')
        representatives = scaler.inverse_transform(representatives)

        # dataNp = scaler.inverse_transform(dataNp)
        # dataNp = dataNp.reshape((ntime, nlat, nlon, npar))
        return representatives, kmeans, scaler

    representatives, clustering, scaler = computeRepresentatives(dataNp, nrep=nrep)
    np.save(pathRepresentatives + 'representatives.npy', representatives)
    pickle.dump(clustering, open(pathRepresentatives + 'kmeans.p', 'wb'))
    pickle.dump(scaler, open(pathRepresentatives + 'scaler.p', 'wb'))

    # Compute nClusters clusters of pixels with the flattened fingerprint data of shape (nlat*nlon, nrep)
    def computeFingerprints(clustering, scaler,  dataNp):
        # Unflatten labels
        print('Reshaping labels ...')

        shape = dataNp.shape  # We assume the first dimension is the time dimension
        print('Reshaping input ...')
        dataNp = dataNp.reshape(np.prod(np.array(shape[:-ndim_par])), shape[-ndim_par])
        dataNp = scaler.transform(dataNp)
        # TODO: Make this more general so that ndim_par can be more than 1

        labels = clustering.predict(dataNp)
        labels = labels.reshape(*shape[:-ndim_par])

        # compute fingerprints
        print('Computing fingerprints ...')
        fingerprints = np.full(list(shape[1:-1])+[nrep], np.NaN, dtype=np.float32)
        for i in range(nrep):
            fingerprints[..., i] = np.sum(labels == i, axis=0)/shape[0]

        # Check if probabilities sum up to 1
        assert np.allclose(np.sum(fingerprints, axis=-1), np.ones(shape[1:-ndim_par])), 'Probabilities do not sum to 1'

        return fingerprints
    clustering = pickle.load(open(pathRepresentatives + 'kmeans.p', 'rb'))
    scaler = pickle.load(open(pathRepresentatives + 'scaler.p', 'rb'))
    for mask in masks.keys():
        fingerprints = computeFingerprints(clustering, scaler, dataNp[masks[mask]])
        fingerprints = reconstruct_data_from_masked_data(fingerprints, landmask,
                                                         (shape_original[1], shape_original[2], nrep),
                                                         time_axis=False)
        np.save(pathRepresentatives + mask + '/fingerprints.npy', fingerprints)

    def computeClustersFingerprints(fingerprints, nClusters):
        """
        Compute nClusters clusters of the fingerprints for the flattened fingerprints data. Save the grid of labels of
        each pixel and return it.
        :param fingerprints: (nd.array, shape (nlat, nlon, nrep)) with the fingerprints of each pixel
        :param nClusters: (int, default = 6) the number of clusters
        :return labeledGrid: (nd.array, shape (nlat, nlon)) with the labeled pixels
        """
        print('Computing clusters of fingerprints ...')
        aggClustering = AgglomerativeClustering(n_clusters=nClusters, linkage='ward', affinity='euclidean',
                                                compute_full_tree=True)

        shape = fingerprints.shape
        fingerprints = fingerprints.reshape(np.prod(shape[:-ndim_par]), nrep)

        aggClustering = aggClustering.fit(fingerprints)
        labels = aggClustering.labels_

        labeledGrid = labels.reshape(shape[:-ndim_par])
        return labeledGrid
    for mask in masks.keys():
        fingerprints = np.load(pathRepresentatives + mask + '/fingerprints.npy')
        fingerprints = limit_to_mask(fingerprints, landmask, time_axis=False)
        labeledGrid = computeClustersFingerprints(fingerprints, nClusters=nClusters)
        labeledGrid = reconstruct_data_from_masked_data(labeledGrid, landmask,
                                                        (shape_original[1], shape_original[2]),
                                                        time_axis=False)
        np.save(pathRepresentatives + mask + '/C' + str(nClusters) + '/labeledGrid.npy', labeledGrid)

    # visualize the result
    def visualize(representatives, fingerprints, labeledGrid, pathFiguresRepresentatives, pathFigureslabeledGridMasked, mask, type='SERZ'):
        """
        Visualise and save the results. Figures that are required:

        - map with cluster indices.
        - aggregated fingerprint for each cluster.
        :return:
        """
        print('visualizing ...')


        representativesOrdered, fingerprintsOrdered = orderRepresentatives(representatives, fingerprints, type=type)
        representativesFigures(representativesOrdered, pathFiguresRepresentatives, type)

        aggregatedFingerprints(fingerprintsOrdered, labeledGrid, pathFigureslabeledGridMasked, cmap_discrete)

        # mapClusters(labeledGrid, nClusters, mask, pathFigureslabeledGridMasked)
        fig = combinedClustersFingerprints(labeledGrid, fingerprintsOrdered, nClusters, mask, pathFigureslabeledGridMasked, cmap_discrete)
        fig.suptitle(maskname)
        fig.savefig(pathFigureslabeledGridMasked + 'clustering_fingerprints.png')
        plt.close('all')
    # representatives = np.load(pathRepresentatives + 'representatives.npy')
    # for mask in masks.keys():
        # fingerprints = np.load(pathRepresentatives + mask + '/fingerprints.npy')
        # labeledGrid = np.load(pathRepresentatives + mask + '/C' + str(nClusters) + '/labeledGrid.npy')
        # pathFigureslabeledGridMasked = pathFiguresRepresentatives + mask + '/C' + str(nClusters) + '/'
        # visualize(representatives, fingerprints, labeledGrid, pathFiguresRepresentatives, pathFigureslabeledGridMasked, mask, type)

#
#
nreps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
# landmasks = ['all', 'land', 'sea']
landmasks = ['all']
timemasks = ['full', 'seasonal', 'temporal']
timemasks = ['full']
for nrep in nreps:
    for landmask in landmasks:
        for timemask in timemasks:
            for nClusters in [6]:
                print(nrep, nClusters, landmask, timemask)
                europe_fingerprinting(nrep=nrep, nClusters=nClusters, maskname=timemask, landmask=landmask, type=type)
                gc.collect()
