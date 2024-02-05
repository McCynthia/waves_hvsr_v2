#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:14:14 2020

@author: vjs
"""

## DL events on offshore stations of interest near turbidity meters, compute 
## distances and PGA, and plot per station.

# %% (1) ALWAYS RUN THIS CELL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import obspy as obs
from obspy.clients.fdsn.client import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel
from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
import cartopy.crs as ccrs
#import gmpe as gm
#from openquake.hazardlib import imt

# %% (2) ALWAYS RUN THIS CELL

############# PARAMETERS #############
# mseed_dir = '/Users/vjs/turbidites/observational_2020/data/mseed4pga/'
mseed_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/data/mseed4pga/'
# fig_dir = '/Users/vjs/turbidites/observational_2020/figs/'
fig_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/figs/'
# meta_dir = '/Users/vjs/turbidites/observational_2020/data/metadata/'
meta_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/metadata/'

## figure info:
fig_size = (10,4)
station_markers = ['o','^','s','d']

## geographic region:
minlat = 46.995
maxlat = 51.097
minlon = -131.4
maxlon = -123.772

## OR central longitude/latitude and radius:
central_lon = -126.058197
central_lat = 48.314655
search_radius_min = 0 ## in degrees
search_radius_max = 2.5 ## in degrees

## time constraints for event search:
stime = UTCDateTime("2009-01-01T000000.000")
etime = UTCDateTime("2019-12-31T235959.000")

## min depth in km:
min_depth = 6

## stations of interest:
# netcode = 'NV'
netcode = 'UW,UO,TA,PB'
# all_stations = 'BACME,NCBC,CQS64,NC89'
## Changed original to get all stations
all_stations = '*'
channels = ['HN*'] # high gain accelerometers, could do high gain broadbands ('HH*',)
## NOTE: BACME only has HN (W1 loc code); NCBC has
## Changed channels to 'HH*' and 'BH*'
channels = ['HH*', 'BH*']

## CLIENTS
## Set up the client to be IRIS, for downloading:
client = Client('IRIS')
# taup:
model = TauPyModel(model="iasp91")

# %% (3) CAN RUN AS STANDALONE AFTER FIRST TWO

############# CATALOG DOWNLOADS ##############



## Find stations metadata for position:
sta_inventory = client.get_stations(network=netcode,station=all_stations,channel=channels[0])

## Find earthquakes 
#eq_catalog = client.get_events(starttime=stime, endtime=etime,
#                        minlatitude=minlat, maxlatitude=maxlat, 
#                        minlongitude=minlon, maxlongitude=maxlon,
#                        minmagnitude=3)
eq_catalog = client.get_events(starttime=stime, endtime=etime,
                        latitude=central_lat, longitude=central_lon, 
                        minradius=search_radius_min, maxradius=search_radius_max,
                        minmagnitude=3)

# %% (4) NEED TO RUN (1) - (3) FIRST

######## GET CATALOG AND METADATA ###########
## Extract the positions of the infrasound stations:
st_lons = []
st_lats = []
st_stas = []
st_start_year = []
for network in sta_inventory:
    for station in network:
        st_stas.append(station.code)
        st_lons.append(station.longitude)
        st_lats.append(station.latitude)
        st_start_year.append(station.start_date.year)
        
stdict = {'stName':st_stas, 'stlon':st_lons, 'stlats':st_lats}
stdf = pd.DataFrame(stdict)
stdf.to_csv(meta_dir + 'station_inventory.csv',index=False)
        
## Extract event information:
ev_lons = []
ev_lats = []
ev_depth = []
ev_origint = []
ev_mag = []
for event in eq_catalog:
    ev_lons.append(event.origins[0].longitude)
    ev_lats.append(event.origins[0].latitude)
    ev_depth.append(event.origins[0].depth)
    ev_origint.append(event.origins[0].time)
    ev_mag.append(event.magnitudes[0].mag)
    
## WRite out event metadata to file to use later:
evdict = {'evlon':ev_lons, 'evlats':ev_lats, 'evdepth':ev_depth,
          'evM':ev_mag, 'evorigint':ev_origint}
evdf = pd.DataFrame(evdict)
evdf.to_csv(meta_dir + 'event_catalog.csv',index=False)

# %% (5) NEED TO RUN (1) - (4) FIRST
######## get distances btwn stations and events ###########
## make dataframe with station repeated, use this info for dl-ing waveforms
distances_all = []
sta_df_all = []
ev_mag_df_all = []
ev_origintime_df_all = []
ev_collecttime_all = []
ev_endtime_all = []
evdf_lat_all = []
evdf_lon_all = []
evdf_depth_all = []



for i_station in range(len(st_stas)):
    i_stlon = st_lons[i_station]
    i_stlat = st_lats[i_station]
    for event in eq_catalog:    
        ## add station name to long array:
        sta_df_all.append(st_stas[i_station])
        ## append event info:
        evdf_lat_all.append(event.origins[0].latitude)
        evdf_lon_all.append(event.origins[0].longitude)
        evdf_depth_all.append(event.origins[0].depth)
        
        ## get great circle distance between this event, and station:
        ij_grcircle_distance = gps2dist_azimuth(i_stlat, i_stlon, event.origins[0].latitude, event.origins[0].longitude)
        ij_grcircle_distance_km = ij_grcircle_distance[0]/1000
        ij_degrees = kilometers2degrees(ij_grcircle_distance_km)
        #ij_ttime = IRISclient.traveltime(model='iasp91', phases=['p'], evdepth=event.origins[0].depth, distkm=[ij_grcircle_distance_km], traveltimeonly=True)
        ## For each distance, assume 5 km/s average p-wave speed to get p-wave travel time:
        #ij_pwv_ttime = ij_grcircle_distance_km / 5.
        ij_pwv_ttime_allP = model.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                  distance_in_degree=ij_degrees,phase_list=["p","P"])
        ij_pwv_ttime = ij_pwv_ttime_allP[0].time
        
        ## start collecting waveform data 10 seconds before predicted p-wave arrival
        ev_collecttime_all.append(event.origins[0].time + (ij_pwv_ttime - 10))
        ## collect a total of 45 seconds
        ev_endtime_all.append(event.origins[0].time + (ij_pwv_ttime - 10) + 120)
        
        ## append:
        distances_all.append(ij_grcircle_distance_km)
        ev_mag_df_all.append(event.magnitudes[0].mag)
        ev_origintime_df_all.append(event.origins[0].time)
        
## Save then into a dictionary/dataframe, and sdave to file:
metadata_all = {'distance':distances_all, 'stations':sta_df_all, 'mag':ev_mag_df_all, 
                'origint':ev_origintime_df_all, 'collecttime':ev_collecttime_all, 'endtime':ev_endtime_all,
                'evlat':evdf_lat_all, 'evlon':evdf_lon_all, 'evdepth':evdf_depth_all}

metadata_all_df = pd.DataFrame(metadata_all)
metadata_all_df.to_csv((meta_dir + 'metadata_all_ev2sta.csv'),index=False)

# %% (6)  CAN RUN AS STANDALONE AFTER FIRST TWO

######## DOWNLOAD AND SAVE PROCESSED WAVEFORMS ###########
all_metadata_df = pd.read_csv((meta_dir + 'metadata_all_ev2sta.csv'))
#all_metadata_df = pd.read_csv((meta_dir + 'metadata_dldata.csv'))



dlsuccess = [False] * len(all_metadata_df)
for recording_i in range(len(all_metadata_df)):
    try:
        i_st = client.get_waveforms(network=netcode,station=all_metadata_df.stations[recording_i],location="*",channel="HNE,HNN",starttime=all_metadata_df.collecttime[recording_i],endtime=all_metadata_df.endtime[recording_i],attach_response=True)
        dlsuccess[recording_i] = True
        print(np.str(recording_i) +  ' for station ' + np.str(all_metadata_df.stations[recording_i]) + ' evm ' + np.str(all_metadata_df.mag[recording_i]) + ' downloaded waveforms')
    except:
        dlsuccess[recording_i] = False
        print(np.str(recording_i) + ' has nothing')
    ## if there's data, keep going...
    if dlsuccess[recording_i]==True:
        print('moving on...')
        # remove response - get sensitivity for accelerometer since it's a flat gain:
        for j_channel in range(len(i_st)):
            ij_sensitivity = i_st[j_channel].stats.response.instrument_sensitivity.value
            i_st[j_channel].data = i_st[j_channel].data / ij_sensitivity
        
        ## Remove pre-event baseline mean:
        for j_channel in range(len(i_st)):
            ij_5sec_index = np.where(i_st[j_channel].times() <= 5)[0]
            ij_preeventmean = np.mean(i_st[j_channel].data[ij_5sec_index])
            i_st[j_channel].data = i_st[j_channel].data - ij_preeventmean
        
        ### save the data....
        i_datetime = UTCDateTime(all_metadata_df.collecttime[recording_i]).datetime
        i_time_for_path = np.str(i_datetime.year) + np.str(i_datetime.month)+ np.str(i_datetime.day)+ np.str(i_datetime.hour)+ np.str(i_datetime.minute)+ np.str(i_datetime.second)+ np.str(i_datetime.microsecond)
        i_mseedpath = mseed_dir + np.str(all_metadata_df.stations[recording_i]) + '_' + np.str(all_metadata_df.mag[recording_i]) + '_' + np.str(np.round(all_metadata_df.distance[recording_i],decimals=2)) + '_' + i_time_for_path + '.mseed'
        i_st.write(i_mseedpath,format='mseed')

## save metadata file with only dlsuccess:
dl_metadata_df = all_metadata_df[dlsuccess]
dl_metadata_df.to_csv((meta_dir + 'metadata_dldata.csv'),index=False)

# # %% (7) CAN RUN AS STANDALONE AFTER FIRST TWO
        
# ######## COMPUTE PGA ###########        

# ## get metadata for downloaded events:
# ## read in downloaded metadata:
# metadata_df = pd.read_csv((meta_dir + 'metadata_dldata.csv'))


# distances = metadata_df.distance.values 
# sta_df = metadata_df.stations.values
# ev_mag_df = metadata_df.mag.values
# ev_origintime_df = metadata_df.origint.values
# ev_collecttime = metadata_df.collecttime.values
# ev_endtime = metadata_df.endtime.values
    
# # Define empty PGA
# pga = []

# for record_i in range(len(metadata_df)):
#     i_datetime = UTCDateTime(metadata_df.collecttime.values[record_i]).datetime
#     i_time_for_path = np.str(i_datetime.year) + np.str(i_datetime.month)+ np.str(i_datetime.day)+ np.str(i_datetime.hour)+ np.str(i_datetime.minute)+ np.str(i_datetime.second)+ np.str(i_datetime.microsecond)
#     i_mseedpath = mseed_dir + sta_df[record_i] + '_' + np.str(ev_mag_df[record_i]) + '_' + np.str(np.round(distances[record_i],decimals=2)) + '_' + i_time_for_path + '.mseed'
#     i_st = obs.read(i_mseedpath)        
    
#     ## Get PGA...    
#     i_pga_all = []
#     for j_channel in range(len(i_st)):
#         ij_data = i_st[j_channel].data
#         ij_pga = np.max(np.abs(ij_data))
#         i_pga_all.append(ij_pga)
#     ## get geom. mean:
#     i_pga = np.max(i_pga_all)
#     pga.append(i_pga) 
    
# ## Make a new dataframe and add PGA to it:
# flatfile_df = metadata_df.copy()
# flatfile_df['pga'] = pga
# flatfile_df.to_csv((meta_dir + 'flatfile.csv'),index=False)



# # %% (8) CAN RUN AS STANDALONE AFTER FIRST THREE CELLS
# ######### GET BCHYDRO PREDICTIONS ########
# IMT = [imt.PGA()]
# rrup_logspace = np.logspace(np.log10(25),np.log10(400),num=50)
# rhypo_logspace = rrup_logspace.copy()

# rrup_pgaVM = np.array([50,150,300])
# rhypo_pgaVM = rrup_pgaVM.copy()

# M_pgaVM = np.linspace(2.8,7,num=50)
# M_pgaVdist = np.array([3,5,7])

# gmpe_type='central'
# event_type = 'interface'
# hypo_depth = np.median(ev_depth)/1000

# ## full arrays for computing various distances:, for the versus magnitude version:
# forebackarc = [False]
# vs30 = np.array([760])

# ## FOR PGA VS. DIST:
# ## Get BCHydro predictions:
# lmean_vdist, lmeanplus_vdist, lmeanmin_vdist, sd_vdist = [],[],[],[]
# ## units of PGA are in g, so miultiply by 9.81 when obtain m/s2.
# for i_M in range(len(M_pgaVdist)):
#     i_lmean,i_lmeanplus,i_lmeanmin,i_sd = gm.bchydro_fixeddist(IMT,rrup_logspace,rhypo_logspace,hypo_depth,M_pgaVdist[i_M],gmpe_type,event_type)
#     lmean_vdist.append(i_lmean[0]*9.81)
#     lmeanplus_vdist.append(i_lmeanplus[0]*9.81)
#     lmeanmin_vdist.append(i_lmeanmin[0]*9.81)
#     sd_vdist.append(i_sd[0]*9.81)
    
# ## FOR PGA VS. M:
# ## Get BCHydro predictions:
# lmean_vm = []
# ## units of PGA are in g, so miultiply by 9.81 when obtain m/s2.
# for i_dist in range(len(rrup_pgaVM)):
#     i_lmean_vm = []
#     for j_M in range(len(M_pgaVM)):
#         ij_lmean,ij_lmeanplus,ij_lmeanmin,i_sd = gm.bchydro(IMT,np.array([rrup_pgaVM[i_dist]]),np.array([rhypo_pgaVM[i_dist]]),hypo_depth,M_pgaVM[j_M],gmpe_type,vs30,forebackarc,event_type)
#         i_lmean_vm.extend(ij_lmean[0]*9.81)
#     lmean_vm.append(i_lmean_vm)


# # %% (9) CAN RUN AS STANDALONE AFTER (1), (2), and (8)

# ######## Make PGA plots ###########
# flatfile = pd.read_csv((meta_dir + 'flatfile.csv'))

# ## get markers for each station:
# sta_plt = all_stations.split(',')
# sta_ind = []

# for station_i in range(len(sta_plt)):
#     i_station = sta_plt[station_i]
#     i_ind = np.where(flatfile.stations.values == i_station)[0]
#     ## add this to the station ind:
#     sta_ind.append(i_ind)

# ## M colors for GMM:
# linestyle_M = ['dashdot','dotted','solid']
# linestyle_R = ['dashdot','dotted','solid']

# ## Initiate plot:
# pga_fig,axes = plt.subplots(nrows=1,ncols=2,figsize=fig_size)

# ## first plot is pga vs. distance.
# for station_i in range(len(sta_plt)):
#     i_plot_ind = sta_ind[station_i]
#     dist_ax = axes[0].scatter(flatfile.distance.values[i_plot_ind],flatfile.pga.values[i_plot_ind],marker=station_markers[station_i],c=flatfile.mag.values[i_plot_ind],vmin=3,vmax=7,edgecolor='k',label=sta_plt[station_i])
# # PLOT BCHydro for M 3, 5, 7
# for i_M in range(len(M_pgaVdist)):
#     axes[0].plot(rrup_logspace,lmean_vdist[i_M],linestyle=linestyle_M[i_M],linewidth=1.5,color='k',label='M'+np.str(M_pgaVdist[i_M]))
    
# axes[0].set_yscale('log')
# axes[0].set_xscale('log')
# axes[0].set_ylim([0.0001,0.5])
# axes[0].set_xlim([25,400]) 
# axes[0].set_xlabel('Distance (km)',labelpad=0)
# axes[0].set_ylabel('PGA (m/s/s)')
# axes[0].grid(True)
# dist_cb = pga_fig.colorbar(dist_ax,ax=axes[0])
# dist_cb.set_label('Magnitude')
# axes[0].legend(fontsize=9,loc='lower right',bbox_to_anchor=(0.2,0.82),ncol=2,handletextpad=0.1,columnspacing=0.03)

# ## then plot is pga vs. m.
# for station_i in range(len(sta_plt)):
#     i_plot_ind = sta_ind[station_i]
#     m_ax = axes[1].scatter(flatfile.mag.values[i_plot_ind],flatfile.pga.values[i_plot_ind],marker=station_markers[station_i],c=flatfile.distance.values[i_plot_ind],vmin=25,vmax=300,edgecolor='k',label=sta_plt[station_i])

# # PLOT BCHydro for M 3, 5, 7
# for i_r in range(len(rrup_pgaVM)):
#     axes[1].plot(M_pgaVM,lmean_vm[i_r],linestyle=linestyle_R[i_r],linewidth=1.5,color='k',label=np.str(rrup_pgaVM[i_r])+'km')

# ## vertical line to represent acceleration in dugan case:
# axes[1].vlines(5,40,140)

# axes[1].set_yscale('log')
# axes[1].set_ylim([0.00007,0.5])
# axes[1].set_xlim([2.8,7])
# axes[1].set_xlabel('Magnitude')
# axes[1].set_ylabel('PGA (m/s/s)')
# axes[1].grid(True)
# m_cb = pga_fig.colorbar(m_ax,ax=axes[1])
# m_cb.set_label('Distance (km)')
# axes[1].legend(fontsize=9,loc='lower right',bbox_to_anchor=(0.2,0.82),ncol=2,handletextpad=0.1,columnspacing=0.07)

# plt.subplots_adjust(wspace=0.4,bottom=0.15)


# pga_fig.savefig(fig_dir + 'pga_figure.png')
