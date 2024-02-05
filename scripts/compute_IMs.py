#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:14:29 2022

@author: vjs
"""
# %% (1) ALWAYS RUN THIS CELL
import numpy as np
import obspy as obs
import pandas as pd
import pyrotd
from mudpy import ruptfunctions as ruptfn
from obspy.core.utcdatetime import UTCDateTime


# %% (2) ALWAYS RUN THIS CELL

############# PARAMETERS #############
mseed_dir = '/Users/vjs/willamettevalley/event_october2022/data/mseed/'
fig_dir = '/Users/vjs/willamettevalley/event_october2022/figs/'
meta_dir = '/Users/vjs/willamettevalley/event_october2022/data/metadata/'

## figure info:
fig_size = (10,4)
station_markers = ['o','^','s','d']

## geographic region:
# minlat = 46.995
# maxlat = 51.097
# minlon = -131.4
# maxlon = -123.772

## OR central longitude/latitude and radius:
event_lon = -122.551
event_lat = 44.540
search_radius_min = 0 ## in degrees
search_radius_max = 3.0 ## in degrees


#### DOWNLOAD PARAMETERS

## Seconds before P wave to start download:
sec_dLbefore_Parr = 10


## ANALYSIS PARAMETERS
## Signal length for SNR in seconds:
signal_length_seconds = 8
## Rotation angle in DEGREES for rotated time series for PGA (rotdXX)
rot_angle = 50
## Damping for SAs in fraction (i.e., 5% is 0.05)
sa_osc_damping = 0.05
## SA Frequencies to compute:
sa_osc_freqs = np.array([0.1,0.2,0.5,1,2,3,5,7,10])


# %% (7) CAN RUN AS STANDALONE AFTER FIRST TWO
        
######## COMPUTE PGA and SAs ###########        

## get metadata for downloaded events:
## read in downloaded metadata:
metadata_df = pd.read_csv((meta_dir + 'metadata_dldata.csv'))

## GEt hypocentral distance, collection times, etc.
rhypo = metadata_df.rhypo.values 
sta_df = metadata_df.stations.values
ev_mag_df = metadata_df.mag.values
ev_origintime_df = metadata_df.origint.values
ev_collecttime = metadata_df.collecttime.values
ev_endtime = metadata_df.endtime.values
    
# Define empty PGA
PGA = np.full_like(metadata_df.rhypo.values,0)
SNR_E = np.full_like(metadata_df.rhypo.values,0)
SNR_N = np.full_like(metadata_df.rhypo.values,0)
SNR = np.full_like(metadata_df.rhypo.values,0)

## Make empty dataframe for SAs:
sa_df = pd.DataFrame(columns=sa_osc_freqs.astype('str'),
                  index=range(len(metadata_df)))



#### !!!!! NOTE !!!!! This loop wouldn't run as a script, had to be copy-pasted
###     into command line.

## For each row in the downloaded data dataframe:
##    (1) import file, (2) compute SNR, (3) compute rotated time series,
##      (4) Compute PGA, and (5) Compute SAs. Save.
for record_i in range(len(metadata_df)):
    ## Paramters for path to file are based off of what is in the dataframe:
    i_datetime = UTCDateTime(metadata_df.collecttime[record_i]).datetime
    i_time_for_path = str(i_datetime.year) + str(i_datetime.month)+ str(i_datetime.day)+ 'T'+ str(i_datetime.hour)+ str(i_datetime.minute)+ str(i_datetime.second)
    i_network = metadata_df.network[record_i]
    i_station = metadata_df.stations[record_i]
    i_mag = int(metadata_df.mag[record_i])
    i_distance = int(metadata_df.rhypo[record_i])
    i_instrumenttype_base = metadata_df.channels[record_i][0:2]
    
    i_mseedpath = f'{mseed_dir}{i_network}_{i_station}_{i_instrumenttype_base}_m{i_mag}_{i_distance}km_{i_time_for_path}.mseed' 
    print('working on %i, path %s' % (record_i,i_mseedpath))
    
    # i_datetime = UTCDateTime(metadata_df.collecttime.values[record_i]).datetime
    # i_time_for_path = str(i_datetime.year) + str(i_datetime.month)+ str(i_datetime.day)+ str(i_datetime.hour)+ str(i_datetime.minute)+ str(i_datetime.second)+ str(i_datetime.microsecond)
    # i_mseedpath = '%s%s_%s_%s_m%.0f%.0fkm_%s.mseed' % (mseed_dir,metadata_df.network[record_i],metadata_df.stations[record_i],j_instrumenttype_base,metadata_df.mag[record_i],metadata_df.distance[record_i],i_time_for_path)
    i_st = obs.read(i_mseedpath)   

    print('computing SNR')
    ## Get SNR:
    ## First get window to compute on:
    i_noise_start = 0
    # Collect time is sec_dLbefore_Parr seconds before P arrival according to tauP
    # So start signal (sec_dLbefore_Parr+2) seconds after record start time (collect time)
    # ... and end noise (sec_dLbefor_Parr - 2) seconds after record start time
    i_noise_end = sec_dLbefore_Parr - 2
    i_signal_start = sec_dLbefore_Parr + 2
    
    ## End signal in signal_length_seconds
    i_signal_end = i_signal_start + signal_length_seconds
    
    ## Get noise data, for each component:
    for j_component in range(len(i_st)):
        j_channel = i_st[j_component].stats.channel
        
        ## Get noise indices and data
        j_noise_indices = (i_st[j_component].times() < i_noise_end) & (i_st[j_component].times() >= i_noise_start)
        j_noise_data = i_st[j_component].data[j_noise_indices]
        
        ## Get signal indices and data:
        j_signal_indices = (i_st[j_component].times() >= i_signal_start) & (i_st[j_component].times() < i_signal_end)
        j_signal_data = i_st[j_component].data[j_signal_indices]
        
        ij_SNR = np.std(j_signal_data) / np.std(j_noise_data)
        if j_channel[2] == 'N':
            SNR_N[record_i] = ij_SNR
        elif j_channel[2] == 'E':
            SNR_E[record_i] = ij_SNR
            
    ## Get the overal SNR for this event/station pair:
    SNR[record_i] = min(SNR_N[record_i],SNR_E[record_i])

    print('rotating time series and getting PGA')
    ## Get the rotated time series:
    if (len(i_st[0].data) == len(i_st[1].data)):
        i_rotd50 = ruptfn.rotateTimeSeries(i_st[0].data, i_st[1].data, rot_angle)
         ## If it's strong-motion, get the PGA:
        if metadata_df.channels[record_i][1] == 'N':
            i_pga = np.max(np.abs(i_rotd50))
            PGA[record_i] = i_pga
        ## If it's a broadband, differentiate:
        elif metadata_df.channels[record_i][1] == 'H':
            i_rotd50_acc = np.diff(i_rotd50)/np.diff(i_st[0].times())
            i_pga = np.max(np.abs(i_rotd50_acc))
            PGA[record_i] = i_pga
        
    else:
        print('Time series are not the same length! Computing PGA with geom mean')
        i_pga = np.sqrt(np.max(np.abs(i_st[0].data))*np.max(np.abs(i_st[1].data)))
        
    print('computing SA')
    ## Compute rotated spectral accelrations:
     ## If it's strong-motion, get the PGA:
    if metadata_df.channels[record_i][1] == 'N':
        i_timestep = i_st[0].stats.delta
        i_comp1 = i_st[0].data
        i_comp2 = i_st[1].data
        i_SA = pyrotd.calc_rotated_spec_accels(i_timestep,i_comp1,i_comp2,sa_osc_freqs,sa_osc_damping)
    elif metadata_df.channels[record_i][1] == 'H':
        i_timestep = i_st[0].stats.delta
        i_comp1 = np.diff(i_st[0].data)/np.diff(i_st[0].times())
        i_comp2 = np.diff(i_st[1].data)/np.diff(i_st[1].times())
        i_SA = pyrotd.calc_rotated_spec_accels(i_timestep,i_comp1,i_comp2,sa_osc_freqs,sa_osc_damping)

    ## Figure out how to add back into dataframe... take the 50th percentile
    for i_sa_ind,i_sa in enumerate(sa_osc_freqs.astype('str')):
        sa_df[i_sa].loc[record_i] = i_SA[i_SA.percentile == 50].spec_accel[i_sa_ind]
    
## Make a new dataframe and add PGA to it:
flatfile_df = metadata_df.copy()
flatfile_df['PGA'] = PGA
flatfile_df['SNR_E'] = SNR_E
flatfile_df['SNR_N'] = SNR_N

## Concatenate SA dataframe:
flatfile_df = pd.concat([flatfile_df,sa_df], axis=1)

flatfile_df.to_csv((meta_dir + 'flatfile.csv'),index=False)

