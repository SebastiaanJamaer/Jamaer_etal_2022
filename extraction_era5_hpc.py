import netCDF4 as nc
import pandas as pd
import numpy as np
import sys
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.optimize import minimize, Bounds, least_squares
from RZse import RZSEfit, RZsemodel, RZfit
from Line_optimization import smart_slsqp_optimization
import time as time_module

###############################################
## Data settings                             ##
###############################################
IN_PATH  = 'data-path/'
OUT_PATH = 'data-out/'
# print('started script')

max_iter = 500
initialGuess = 'RZ'
conditioning = 'constraints'
solver = 'trust-constr'  # Keep this fixed if you're not absolutely sure of what you are doing.
jacobian = 'manual'  # Manual computation of jacobian or approximation, keep at manual if unsure, is more accurate and more efficient
hessian = 'manual'  # same as jacobian, though difference in performance (both accuracy and efficiency) negligible
xtol = 1e-6  # Tolerance for convergence of the optimization problem
gtol = 1e-6  # Tolerance for convergence of the optimization problem
verbose = 0  # verbose of the optimization, keep at zero if you're doing many pixels/timestamps!!!

def extraction_era5_hpc(infile, outfile, model='SERZ'):
    # print('Reading input...' + infile)
    ds_input = nc.Dataset(infile)
    # print('Done')
    # Create new nc file with output
    # print('Creating output...'+outfile)
    ds_output = nc.Dataset(outfile, mode='w',format='NETCDF4_CLASSIC')
    ds_output.createDimension('lat', len(ds_input['lat'][:]))
    ds_output.createDimension('lon', len(ds_input['lon'][:]))
    ds_output.createDimension('time', None)

    # Dimension variables
    lat = ds_output.createVariable('lat', np.float32, ('lat', ))
    lat.units = ds_input['lat'].units
    lat.long_name = 'latitude'
    lon = ds_output.createVariable('lon', np.float32, ('lon', ))
    lon.units = ds_input['lon'].units
    lon.long_name = 'longitude'
    time = ds_output.createVariable('time', np.float32, ('time', ))
    time.units = ds_input['time'].units
    time.long_name = 'time'
    dims = ('time', 'lat', 'lon')

    # Set coordinates
    lat[:] = ds_input['lat'][:]
    lon[:] = ds_input['lon'][:]
    time[:] = ds_input['time'][:]

    if model== 'SERZ':
        # Variables
        thm = ds_output.createVariable('thm', np.float32, dims)
        thm.units = 'Kelvin'
        thm.long_name = 'constant temperature'
        a = ds_output.createVariable('a', np.float32, dims)
        a.units = 'Kelvin'
        a.long_name = 'coefficient of f'
        b = ds_output.createVariable('b', np.float32, dims)
        b.units = 'Kelvin'
        b.long_name = 'coefficient of g'
        c = ds_output.createVariable('c', np.float32, dims)
        c.units = 'Kelvin'
        c.long_name = 'coefficient of h'
        l = ds_output.createVariable('l', np.float32, dims)
        l.units = 'meter'
        l.long_name = 'height of the CI'
        dh = ds_output.createVariable('dh', np.float32, dims)
        dh.units = 'meter'
        l.long_name = 'width of the CI'
        alpha = ds_output.createVariable('alpha', np.float32, dims)
        alpha.units = 'm/m'
        alpha.long_name = 'Fraction of the BL that is surface layer'
        MAE = ds_output.createVariable('MAE', np.float32, dims)
        MAE.units = 'Kelvin'
        MAE.long_name ='Mean absolute error of fit'
        MSE = ds_output.createVariable('MSE', np.float32, dims)
        MSE.units = 'Kelvin'
        MSE.long_name = 'Mean squared error of fit'
        status = ds_output.createVariable('status', np.float32, dims)
        status.units ='[]'
        status.long_name = 'Status of optimization'
        nfev = ds_output.createVariable('nfev', np.float32, dims)
        nfev.units = '[]'
        nfev.long_name = 'Number of function evaluations'
        success = ds_output.createVariable('success', np.float32, dims)
        success.units = '[]'
        success.long_name = 'Success boolean of the optimization'
        computation_time = ds_output.createVariable('computation_time', np.float32, dims)
        computation_time.units = 'seconds'
        computation_time.long_name = 'commputation time for fit'
        used_p0RZ = ds_output.createVariable('used_p0RZ', np.float32, dims)
        used_p0RZ.units = '[]'
        used_p0RZ.long_name = 'Bool if the p0RZ was used or not'

        results = {'a': np.zeros((len(lat[:]), len(lon[:]))), 'b': np.zeros((len(lat[:]), len(lon[:]))),
                   'c': np.zeros((len(lat[:]), len(lon[:]))), 'thm': np.zeros((len(lat[:]), len(lon[:]))),
                   'l': np.zeros((len(lat[:]), len(lon[:]))), 'dh': np.zeros((len(lat[:]), len(lon[:]))),
                   'alpha': np.zeros((len(lat[:]), len(lon[:]))), 'MAE': np.zeros((len(lat[:]), len(lon[:]))),
                   'MSE': np.zeros((len(lat[:]), len(lon[:]))),
                   'computation_time': np.zeros((len(lat[:]), len(lon[:]))),
                   'success': np.zeros((len(lat[:]), len(lon[:]))), 'status': np.zeros((len(lat[:]), len(lon[:]))),
                   'nfev': np.zeros((len(lat[:]), len(lon[:]))), 'used_p0RZ': np.zeros((len(lat[:]), len(lon[:])))}
    elif model== '3Lines':
        th0 = ds_output.createVariable('th0', np.float32, dims)
        th0.units = 'Kelvin'
        th0.long_name = 'temperature intersect'
        alpha1 = ds_output.createVariable('alpha1', np.float32, dims)
        alpha1.units = 'Kelvin/m'
        alpha1.long_name = 'gradient first line'
        zeta1 = ds_output.createVariable('zeta1', np.float32, dims)
        zeta1.units = 'm'
        zeta1.long_name = 'height of first node'
        alpha2 = ds_output.createVariable('alpha2', np.float32, dims)
        alpha2.units = 'Kelvin/m'
        alpha2.long_name = 'gradient second line'
        zeta2 = ds_output.createVariable('zeta2', np.float32, dims)
        zeta2.units = 'm'
        zeta2.long_name = 'height of second node'
        alpha3 = ds_output.createVariable('alpha3', np.float32, dims)
        alpha3.units = 'Kelvin/m'
        alpha3.long_name = 'gradient first line'

        MAE = ds_output.createVariable('MAE', np.float32, dims)
        MAE.units = 'Kelvin'
        MAE.long_name ='Mean absolute error of fit'
        MSE = ds_output.createVariable('MSE', np.float32, dims)
        MSE.units = 'Kelvin'
        MSE.long_name = 'Mean squared error of fit'
        status = ds_output.createVariable('status', np.float32, dims)
        status.units ='[]'

        computation_time = ds_output.createVariable('computation_time', np.float32, dims)
        computation_time.units = 's'
        computation_time_initial = ds_output.createVariable('computation_time_initial', np.float32, dims)
        computation_time_initial.units = 's'
        nfev = ds_output.createVariable('nfev', np.float32, dims)
        nfev.units = '[]'
        nfev.long_name = 'Number of function evaluations'
        success = ds_output.createVariable('success', np.float32, dims)
        success.units = '[]'
        success.long_name = 'Success boolean of the optimization'

        results = {'th0': np.zeros((len(lat[:]), len(lon[:]))), 'alpha1': np.zeros((len(lat[:]), len(lon[:]))),
                   'zeta1': np.zeros((len(lat[:]), len(lon[:]))), 'alpha2': np.zeros((len(lat[:]), len(lon[:]))),
                   'zeta2': np.zeros((len(lat[:]), len(lon[:]))), 'alpha3': np.zeros((len(lat[:]), len(lon[:]))),
                   'MAE': np.zeros((len(lat[:]), len(lon[:]))), 'MSE': np.zeros((len(lat[:]), len(lon[:]))),
                   'computation_time': np.zeros((len(lat[:]), len(lon[:]))), 'computation_time_initial': np.zeros((len(lat[:]), len(lon[:]))),
                   'success': np.zeros((len(lat[:]), len(lon[:]))), 'status': np.zeros((len(lat[:]), len(lon[:]))),
                   'nfev': np.zeros((len(lat[:]), len(lon[:])))}

    elif model=='RZ':
        thm = ds_output.createVariable('thm', np.float32, dims)
        thm.units = 'Kelvin'
        thm.long_name = 'constant temperature'
        a = ds_output.createVariable('a', np.float32, dims)
        a.units = 'Kelvin'
        a.long_name = 'coefficient of f'
        b = ds_output.createVariable('b', np.float32, dims)
        b.units = 'Kelvin'
        b.long_name = 'coefficient of g'
        l = ds_output.createVariable('l', np.float32, dims)
        l.units = 'meter'
        l.long_name = 'height of the CI'
        dh = ds_output.createVariable('dh', np.float32, dims)
        dh.units = 'meter'
        l.long_name = 'width of the CI'
        MAE = ds_output.createVariable('MAE', np.float32, dims)
        MAE.units = 'Kelvin'
        MAE.long_name ='Mean absolute error of fit'
        MSE = ds_output.createVariable('MSE', np.float32, dims)
        MSE.units = 'Kelvin'
        MSE.long_name = 'Mean squared error of fit'
        status = ds_output.createVariable('status', np.float32, dims)
        status.units ='[]'
        status.long_name = 'Status of optimization'
        nfev = ds_output.createVariable('nfev', np.float32, dims)
        nfev.units = '[]'
        nfev.long_name = 'Number of function evaluations'
        success = ds_output.createVariable('success', np.float32, dims)
        success.units = '[]'
        success.long_name = 'Success boolean of the optimization'
        computation_time = ds_output.createVariable('computation_time', np.float32, dims)
        computation_time.units = 'seconds'
        computation_time.long_name = 'commputation time for fit'

        results = {'a': np.zeros((len(lat[:]), len(lon[:]))), 'b': np.zeros((len(lat[:]), len(lon[:]))),
                   'thm': np.zeros((len(lat[:]), len(lon[:]))),
                   'l': np.zeros((len(lat[:]), len(lon[:]))), 'dh': np.zeros((len(lat[:]), len(lon[:]))),
                   'MAE': np.zeros((len(lat[:]), len(lon[:]))),
                   'MSE': np.zeros((len(lat[:]), len(lon[:]))),
                   'computation_time': np.zeros((len(lat[:]), len(lon[:]))),
                   'success': np.zeros((len(lat[:]), len(lon[:]))), 'status': np.zeros((len(lat[:]), len(lon[:]))),
                   'nfev': np.zeros((len(lat[:]), len(lon[:])))}
    else:
        assert False, 'given model is not implemented'

    # print('Done')

    # Set coordinates



    def calculategeoh(fis, sp, ts, qs, levels):
        '''
        Calculate geopotential and height (relative to terrain)
        '''
        heighttoreturn=np.full(ts.shape[0], -999, np.double)
        geotoreturn=np.full(ts.shape[0], -999, np.double)
        thetavtoreturn=np.full(ts.shape[0], -999, np.double)

        gravity = 9.80665    # [m s-2]
        P0 = 1.e5 # Reference pressure [Pa]
        Rd = 287.058 # Specific gas constant for dry air [J kg-1 K-1]
        Cp_air = 1005   # Specific heat of air [J kg-1 K-1]
        eps = 0.609133  # Rv/Rd-1

        z_h = 0

        # A and B parameters to calculate pressures for model levels,
        pv = [
          0.0000000000,    0.000000,
          2.0003650000,    0.000000,
          3.1022410000,    0.000000,
          4.6660840000,    0.000000,
          6.8279770000,    0.000000,
          9.7469660000,    0.000000,
          13.605424000,    0.000000,
          18.608931000,    0.000000,
          24.985718000,    0.000000,
          32.985710000,    0.000000,
          42.879242000,    0.000000,
          54.955463000,    0.000000,
          69.520576000,    0.000000,
          86.895882000,    0.000000,
          107.41574100,    0.000000,
          131.42550700,    0.000000,
          159.27940400,    0.000000,
          191.33856200,    0.000000,
          227.96894800,    0.000000,
          269.53958100,    0.000000,
          316.42074600,    0.000000,
          368.98236100,    0.000000,
          427.59249900,    0.000000,
          492.61602800,    0.000000,
          564.41345200,    0.000000,
          643.33990500,    0.000000,
          729.74414100,    0.000000,
          823.96783400,    0.000000,
          926.34491000,    0.000000,
          1037.2011720,    0.000000,
          1156.8536380,    0.000000,
          1285.6103520,    0.000000,
          1423.7701420,    0.000000,
          1571.6229250,    0.000000,
          1729.4489750,    0.000000,
          1897.5192870,    0.000000,
          2076.0959470,    0.000000,
          2265.4316410,    0.000000,
          2465.7705080,    0.000000,
          2677.3481450,    0.000000,
          2900.3913570,    0.000000,
          3135.1193850,    0.000000,
          3381.7436520,    0.000000,
          3640.4682620,    0.000000,
          3911.4904790,    0.000000,
          4194.9306640,    0.000000,
          4490.8173830,    0.000000,
          4799.1494140,    0.000000,
          5119.8950200,    0.000000,
          5452.9907230,    0.000000,
          5798.3447270,    0.000000,
          6156.0742190,    0.000000,
          6526.9467770,    0.000000,
          6911.8706050,    0.000000,
          7311.8691410,    0.000000,
          7727.4121090,    0.000007,
          8159.3540040,    0.000024,
          8608.5253910,    0.000059,
          9076.4003910,    0.000112,
          9562.6826170,    0.000199,
          10065.978516,    0.000340,
          10584.631836,    0.000562,
          11116.662109,    0.000890,
          11660.067383,    0.001353,
          12211.547852,    0.001992,
          12766.873047,    0.002857,
          13324.668945,    0.003971,
          13881.331055,    0.005378,
          14432.139648,    0.007133,
          14975.615234,    0.009261,
          15508.256836,    0.011806,
          16026.115234,    0.014816,
          16527.322266,    0.018318,
          17008.789063,    0.022355,
          17467.613281,    0.026964,
          17901.621094,    0.032176,
          18308.433594,    0.038026,
          18685.718750,    0.044548,
          19031.289063,    0.051773,
          19343.511719,    0.059728,
          19620.042969,    0.068448,
          19859.390625,    0.077958,
          20059.931641,    0.088286,
          20219.664063,    0.099462,
          20337.863281,    0.111505,
          20412.308594,    0.124448,
          20442.078125,    0.138313,
          20425.718750,    0.153125,
          20361.816406,    0.168910,
          20249.511719,    0.185689,
          20087.085938,    0.203491,
          19874.025391,    0.222333,
          19608.572266,    0.242244,
          19290.226563,    0.263242,
          18917.460938,    0.285354,
          18489.707031,    0.308598,
          18006.925781,    0.332939,
          17471.839844,    0.358254,
          16888.687500,    0.384363,
          16262.046875,    0.411125,
          15596.695313,    0.438391,
          14898.453125,    0.466003,
          14173.324219,    0.493800,
          13427.769531,    0.521619,
          12668.257813,    0.549301,
          11901.339844,    0.576692,
          11133.304688,    0.603648,
          10370.175781,    0.630036,
          9617.5156250,    0.655736,
          8880.4531250,    0.680643,
          8163.3750000,    0.704669,
          7470.3437500,    0.727739,
          6804.4218750,    0.749797,
          6168.5312500,    0.770798,
          5564.3828130,    0.790717,
          4993.7968750,    0.809536,
          4457.3750000,    0.827256,
          3955.9609380,    0.843881,
          3489.2343750,    0.859432,
          3057.2656250,    0.873929,
          2659.1406250,    0.887408,
          2294.2421880,    0.899900,
          1961.5000000,    0.911448,
          1659.4765630,    0.922096,
          1387.5468750,    0.931881,
          1143.2500000,    0.940860,
          926.50781300,    0.949064,
          734.99218800,    0.956550,
          568.06250000,    0.963352,
          424.41406300,    0.969513,
          302.47656300,    0.975078,
          202.48437500,    0.980072,
          122.10156300,    0.984542,
          62.781250000,    0.988500,
          22.835938000,    0.991984,
          3.7578130000,    0.995003,
          0.0000000000,    0.997630,
          0.0000000000,    1.000000]

        levelSize=137
        A = pv[0::2]  # Slice notation python: Elements start:stop:steps (0::2) all elements starting from 0 in steps of 2
        B = pv[1::2]

        Ph_levplusone = A[levelSize] + B[levelSize]*sp

        #Get a list of level numbers in reverse order
        reversedlevels=np.full(levels.shape[0], -999, np.int32)
        for iLev in list(reversed(range(levels.shape[0]))):
            reversedlevels[levels.shape[0] - 1 - iLev] = levels[iLev]

        #Integrate up into the atmosphere from lowest level
        for lev in reversedlevels:
            #lev is the level number 1-60, we need a corresponding index into ts and qs
            ilevel=np.where(levels==lev)[0]
            t_level=ts[ilevel]
            q_level=qs[ilevel]

            #compute moist temperature
            t_level = t_level * (1.+eps*q_level)

            #compute the pressures (on half-levels)
            Ph_lev = A[lev-1] + (B[lev-1] * sp)

            # We do not use the top levels
            if lev == 1:
                dlogP = np.log(Ph_levplusone/0.1)  # the 0.1 comes from the table
                alpha = np.log(2)  # this value can be found in the explanation of the vertical discretisation
            else:
                dlogP = np.log(Ph_levplusone/Ph_lev)
                dP    = Ph_levplusone-Ph_lev
                alpha = 1. - ((Ph_lev/dP)*dlogP)

            TRd = t_level*Rd
            
            # z_f is the geopotential of this full level
            # integrate from previous (lower) half-level z_h to the full level
            z_f = z_h + (TRd*alpha)

            #Convert geopotential to height
            heighttoreturn[ilevel] = z_f / gravity

            #Geopotential (add in surface geopotential)
            geotoreturn[ilevel] = z_f + fis

            #Virtual potential temperature
            thetavtoreturn[ilevel] = t_level * (2.*P0/(Ph_levplusone + Ph_lev))**(Rd/Cp_air)

            # z_h is the geopotential of 'half-levels'
            # integrate z_h to next half level
            z_h=z_h+(TRd*dlogP)

            Ph_levplusone = Ph_lev

        return geotoreturn, heighttoreturn, thetavtoreturn



    # print('Starting computation')
    # for k in range(1):
    for k in range( len(lat[:])):
    # for k in range(20):
        for l in range(len(lon[:])):
        # for l in range(20):
            # print(k, l)
            geopotential, geoheight, theta = calculategeoh(
                ds_input['FIS'][0, k, l],
                ds_input['PS'][0, k, l],
                ds_input['T'][0, :, k, l],
                ds_input['QS'][0, :, k, l],
                ds_input['level'][:]
            )
            theta = theta[geoheight<5000]
            geopotential = geopotential[geopotential<5000]
            geoheight = geoheight[geoheight<5000]
            if model=='SERZ':
                CIdata = RZSEfit(z=geoheight, th=theta, conditioning=conditioning, solver=solver, jacobian=jacobian,
                    hessian=hessian, options={'xtol': xtol, 'gtol': gtol, 'maxiter': max_iter, 'verbose': verbose},
                    initialGuess=initialGuess)

                results['a'][k, l] = CIdata['a']
                results['b'][k, l] = CIdata['b']
                results['c'][k, l] = CIdata['c']
                results['thm'][k, l] = CIdata['thm']
                results['l'][k, l] = CIdata['l']
                results['dh'][k, l] = CIdata['dh']
                results['alpha'][k, l] = CIdata['alpha']
                results['MAE'][k, l] = CIdata['MAE']
                results['MSE'][k, l] = CIdata['MSE']
                results['nfev'][k, l] = CIdata['nfev']
                results['status'][k, l] = CIdata['status']
                results['success'][k, l] = CIdata['success']
                results['computation_time'][k, l] = CIdata['time']
                # print('used_p0RZ', CIdata['used_p0RZ'])
                results['used_p0RZ'][k, l] = CIdata['used_p0RZ']

            elif model=='3Lines':
                CIdata = smart_slsqp_optimization(nLines=4, z=geoheight, th=theta, solver=solver, jacobian=jacobian,
                                                  options={'xtol': xtol, 'gtol': gtol, 'maxiter': max_iter, 'verbose': verbose})

                results['th0'][k, l] = CIdata['th0']
                results['alpha1'][k, l] = CIdata['alpha1']
                results['zeta1'][k, l] = CIdata['zeta1']
                results['alpha2'][k, l] = CIdata['alpha2']
                results['zeta2'][k, l] = CIdata['zeta2']
                results['alpha3'][k, l] = CIdata['alpha3']
                results['MAE'][k, l] = CIdata['MAE']
                results['MSE'][k, l] = CIdata['MSE']
                results['nfev'][k, l] = CIdata['nfev']
                results['status'][k, l] = CIdata['status']
                results['success'][k, l] = CIdata['success']
                results['computation_time'][k, l] = CIdata['time']
                results['computation_time_initial'][k, l] = CIdata['time_init']
                # import matplotlib.pyplot as plt
                # print(CIdata['th0'], CIdata['alpha1'], CIdata['zeta1'], CIdata['alpha2'], CIdata['zeta2'], CIdata['alpha3'], CIdata['time'], CIdata['time_init'])
                # print('used_p0RZ', CIdata['used_p0RZ'])
            elif model=='RZ':
                CIdata = RZfit(z=geoheight, th=theta, dh_max=4000, max_nfev=max_iter)

                results['a'][k, l] = CIdata['a']
                results['b'][k, l] = CIdata['b']
                results['thm'][k, l] = CIdata['thm']
                results['l'][k, l] = CIdata['l']
                results['dh'][k, l] = CIdata['dh']
                results['MAE'][k, l] = CIdata['MAE']
                results['MSE'][k, l] = CIdata['MSE']
                results['nfev'][k, l] = CIdata['nfev']
                results['status'][k, l] = CIdata['status']
                results['success'][k, l] = CIdata['success']
                results['computation_time'][k, l] = CIdata['time']
    ds_input.close()
    # print('Writing results to output')
    for par in results.keys():
        ds_output[par][0, :, :] = results[par]
    ds_output.close()
    # print('Done')

if __name__ == '__main__':
    extraction_era5_hpc(*sys.argv[1:])
    # extraction_era5_hpc('cas2020010101.nc', outfile='fit_linear2020010101.nc')
