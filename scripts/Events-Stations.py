#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:33:23 2023

@author: CynthiaMC
"""

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
import cartopy.feature as cf
import pygmt
from pyproj import Geod
#import gmpe as gm
#from openquake.hazardlib import imt

# %% (2) ALWAYS RUN THIS CELL

############# PARAMETERS #############
# mseed_dir = '/Users/vjs/turbidites/observational_2020/data/mseed4pga/'
mseed_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/data/'
# fig_dir = '/Users/vjs/turbidites/observational_2020/figs/'
fig_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/figs/'
# meta_dir = '/Users/vjs/turbidites/observational_2020/data/metadata/'
meta_dir = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/metadata/'
# willamette valley outline
wv_path = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/data/wv_ol_vs30_400.txt'

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

## max depth in km:
max_depth = 50000

## topo:
topo_data = '@earth_relief_03s' # 30 arc second global relief

## stations of interest:
# netcode = 'NV'
netcode = 'UW,UO,TA,PB'
# all_stations = 'BACME,NCBC,CQS64,NC89'
## Changed original to get all stations
all_stations = '*'
#channels = ['HN*'] # high gain accelerometers, could do high gain broadbands ('HH*',)
## NOTE: BACME only has HN (W1 loc code); NCBC has
## Changed channels to 'HH*' and 'BH*'
channels = ['HH*', 'BH*']

## Record length to download
record_length_seconds = 180
## Seconds before P-wave to start download
sec_dLbefore_Parr = 10

## CLIENTS
## Set up the client to be IRIS, for downloading:
client = Client('IRIS')
# taup:
model = TauPyModel(model="iasp91")

respfilt_bottom = [0.063,0.28]  # 0.063 just above 16 sec microsiesem, .28 just above 4 sec

# %% (3) CAN RUN AS STANDALONE AFTER FIRST TWO

############# CATALOG DOWNLOADS ##############



## Find stations metadata for position:
    # ---> Add code to plot only stations in Willamette Valley, create box with max lon, etc
#sta_inventory = client.get_stations(network=netcode,station=all_stations,channel=channels[0])
sta_inventory = client.get_stations(network=netcode,station=all_stations,channel='HH*',
                                    latitude=44, longitude=-122, 
                                    minradius=search_radius_min, maxradius=search_radius_max)
                                    #minlatitude=43, maxlatitude=46, minlongitude=-120,
                                    #maxlongitude=-125)

## Find earthquakes 
#eq_catalog = client.get_events(starttime=stime, endtime=etime,
#                        minlatitude=minlat, maxlatitude=maxlat, 
#                        minlongitude=minlon, maxlongitude=maxlon,
#                        minmagnitude=3)
eq_catalog = client.get_events(starttime=stime, endtime=etime,
                        latitude=44, longitude=-122, 
                        minradius=search_radius_min, maxradius=search_radius_max,
                        minmagnitude=2.5, mindepth=0.0)

# %% (4) NEED TO RUN (1) - (3) FIRST

######## GET CATALOG AND METADATA ###########
## Extract the positions of the infrasound stations:
st_lons = []
st_lats = []
st_stas = []
st_elvs = []
st_net = []
st_start_year = []
for network in sta_inventory:
    for station in network:
        st_stas.append(station.code)
        st_lons.append(station.longitude)
        st_lats.append(station.latitude)
        st_elvs.append(station.elevation)
        st_net.append(network.code)
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

## Extract longitude and latitude of Willamette Valley outline
column_names = ["WV_Lon", "WV_Lat"]
wv_df = pd.read_csv(wv_path, sep=" ", names = column_names)
#print(wv_df)
wvlons = wv_df['WV_Lon'].values
wvlats = wv_df['WV_Lat'].values

# %% (5) RUN (1) TO (3) FIRST

######## Get Distances Between Stations and Events ###########
gr_cir_distances_all = []
rhypo = []
repi = []

network_all = []
sta_df_all = []
st_lon_all = []
st_lat_all = []
st_elv_all = []

ev_mag_df_all = []
ev_origintime_df_all = []
ev_Parrtime_all = []
ev_collecttime_all = []
ev_endtime_all = []
evdf_lat_all = []
evdf_lon_all = []
evdf_depth_all = []

for i_station in range(len(st_stas)):
    i_stlon = st_lons[i_station]
    i_stlat = st_lats[i_station]
    for event in eq_catalog:
        ## Add station name and info into long arrays for permutations:
        sta_df_all.append(st_stas[i_station])
        network_all.append(st_net[i_station])
        st_lon_all.append(st_lons[i_station])
        st_lat_all.append(st_lats[i_station])
        st_elv_all.append(st_elvs[i_station])
        
        ## Append event info
        evdf_lat_all.append(event.origins[0].latitude)
        evdf_lon_all.append(event.origins[0].longitude)
        evdf_depth_all.append(event.origins[0].depth)

        ## Get great circle distance between this event and station:
        ij_grcircle_distance = gps2dist_azimuth(i_stlat, i_stlon,
                                                event.origins[0].latitude, event.origins[0].longitude)
        ij_grcircle_distance_km = ij_grcircle_distance[0]/1000
        ij_degrees = kilometers2degrees(ij_grcircle_distance_km)

        ## Get rhypo and repi for GMMs later
        # Get projection:
        g = Geod(ellps = 'WGS84')
        
        # Get distances:
        i_az, i_backaz, i_horizdist = g.inv(i_stlon, i_stlat, event.origins[0].longitude,
                                            event.origins[0].latitude)
        
        # Get overall:
        i_rhypo = np.sqrt(i_horizdist**2 + event.origins[0].depth**2)
        
        # Append in km:
        repi.append(i_horizdist/1000)
        rhypo.append(i_rhypo/1000)
        
        ## For each distance, assume 5km/s average p-wave speed to get p-wave travel time:
        ij_pwv_ttime_allP = model.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                                    distance_in_degree=ij_degrees, phase_list=["p", "P"])

        ij_pwv_ttime = ij_pwv_ttime_allP[0].time
        
        ## Append to list
        ev_Parrtime_all.append(ij_pwv_ttime)
        
        ## Start collecting waveform data 10 seconds before predicted p-wave arrival:
        ev_collecttime_all.append(event.origins[0].time + (ij_pwv_ttime - sec_dLbefore_Parr))
        ## Collect a total of 45 seconds
        ev_endtime_all.append(event.origins[0].time + (ij_pwv_ttime - sec_dLbefore_Parr) + record_length_seconds)
        
        ## Append:
        gr_cir_distances_all.append(ij_grcircle_distance_km)
        ev_mag_df_all.append(event.magnitudes[0].mag)
        ev_origintime_df_all.append(event.origins[0].time)
        
## Save then into a dictionary/dataframe and save to file:
metadata_all = {'network': network_all, 'stations': sta_df_all, 'st_lon': st_lon_all,
                'st_lat': st_lat_all, 'st_elv': st_elv_all,
                'gr_cir_distance': gr_cir_distances_all, 'rhypo': rhypo, 'repi': repi,
                'mag': ev_mag_df_all, 'origint': ev_origintime_df_all,
                'Parrtime': ev_Parrtime_all, 'collecttime': ev_collecttime_all,
                'endtime': ev_endtime_all, 'evlat': evdf_lat_all,
                'evlon': evdf_lon_all, 'evdepth': evdf_depth_all}

metadata_all_df = pd.DataFrame(metadata_all)
metadata_all_df.to_csv((meta_dir + 'metadata_all_ev2sta.csv'), index=False)


# =============================================================================
# # %% (6) RUN (1) TO (4) FIRST
# ############# SETTING UP THE PYGMT FOR PLOTTING #############
# 
# # Plot the map using pyGMT
# 
# plot_region = [-125, -121, 43.5, 46.5]
# 
# ## Create the map
# fig = pygmt.Figure()
# 
# pygmt.makecpt(cmap = 'geo',
#               series = '-8000/4000/25',
#               continuous=True)
# 
# # fig.basemap(region = plot_region, projection = "M15c", frame = ["WSne", "x5+lLatitude(°)", "y5+lLongitude(°)"])
# fig.grdimage(grid = topo_data, region = plot_region, 
#               projection = "M15c", shading = True, frame = True)
# fig.coast(shorelines = True, frame = "ag", resolution = "i")
# 
# pygmt.makecpt(cmap = "jet",
#               series = [evdf.evdepth.min(), max_depth])#evdf.evdepth.max()])
# 
# fig.plot(x=stdf.stlon, y=stdf.stlats,
#           #cmap = True,
#           style = "t0.7c",
#           fill = "yellow",
#           pen = "black",
#           transparency = 15)
# fig.text(x=stdf.stlon-0.05, y=stdf.stlats-0.05, text=stdf.stName,
#           angle=0, font='6p,Helvetica-Bold,Black', justify='LM')
# 
# fig.plot(x=evdf.evlon, y=evdf.evlats,
#           cmap = True,
#           size = 0.1*2**evdf.evM,
#           style = "cc",
#           fill = evdf.evdepth,
#           pen = "black",
#           transparency = 30)
# 
# fig.plot(x=wvlons, y=wvlats,
#           style="p0.05c",
#           pen="black")
# 
# 
# fig.colorbar(frame=["x+lDepth"])
# fig.show()
# fig.savefig(fig_dir + 'events-name_stations-HH_channel.pdf')
# =============================================================================


# %% (7)  CAN RUN AS STANDALONE AFTER FIRST TWO

######## DOWNLOAD AND SAVE PROCESSED WAVEFORMS ###########
all_metadata_df = pd.read_csv((meta_dir + 'metadata_all_ev2sta.csv'))
#all_metadata_df = pd.read_csv((meta_dir + 'metadata_dldata.csv'))

## Run for each type of instrument, and only download if both horizontal components exist
##   To do this, need to make a channels list for each instrument type

## Make an empty array for channels:
all_channels = np.full_like(all_metadata_df.network.values,'')

## Set the download success to false for all records first
dlsuccess = [False] * len(all_metadata_df)

## Loop through columns in dataframe and try to download data for each event/station pair
for recording_i in range(len(all_metadata_df)):
    ## and try it for each instrument type specified above
    for j_instrument_type in channels:
        ## Set a horizontal components counter equal to 0
        j_horiz_counter = 0
        j_instrumenttype_base = j_instrument_type[0:2]
        j_channel_list = [j_instrumenttype_base + 'E', j_instrumenttype_base + 'N', j_instrumenttype_base + 'Z']
        
        ## Set stream to empty:
        i_st = obs.core.stream.Stream()
        ## Try a download for each horizontal:
        for k_horizcomp in j_channel_list:
            try:
                i_st += client.get_waveforms(network=netcode,station=all_metadata_df.stations[recording_i],location="*",channel=k_horizcomp,starttime=UTCDateTime(all_metadata_df.collecttime[recording_i]),endtime=UTCDateTime(all_metadata_df.endtime[recording_i]),attach_response=True)
                print('%i for network/station %s/%s, channel %s, evm %.1f, downloaded waveforms' % (recording_i,all_metadata_df.network[recording_i],all_metadata_df.stations[recording_i],k_horizcomp,all_metadata_df.mag[recording_i]))
                #print(str(recording_i) +  ' for station ' + str(all_metadata_df.stations[recording_i]) + ' evm ' + str(all_metadata_df.mag[recording_i]) + ' downloaded waveforms')
                ## It worked, so add to the horizontal counter
                j_horiz_counter += 1
            except:
                dlsuccess[recording_i] = False
                print('%i for network/station %s/%s, channel %s, evm %.1f, has nothing' % (recording_i,all_metadata_df.network[recording_i],all_metadata_df.stations[recording_i],k_horizcomp,all_metadata_df.mag[recording_i]))
                # print(str(recording_i) + ' has nothing')
                
        ## if there's data on all three channels, keep going...
        if j_horiz_counter == 3:
            dlsuccess[recording_i] = True
            all_channels[recording_i] = j_instrumenttype_base + '*'
            print('data exists on both channels, removing response...')
            
            ## If it's an accelerometer:
            if j_instrument_type[1] == 'N':  
                print('Accelerometer, remove sensitivity')
                # remove response - get sensitivity for accelerometer since it's a flat gain:
                for j_channel in range(len(i_st)):
                    ij_sensitivity = i_st[j_channel].stats.response.instrument_sensitivity.value
                    i_st[j_channel].data = i_st[j_channel].data / ij_sensitivity
                    
            ## If it's a broadband:
            elif j_instrument_type[1] == 'H':
                print('broadband, remove response')
                ## Get the sampling rate:
                ij_samplingrate = i_st[0].stats.sampling_rate
                ij_nyquist = ij_samplingrate/2
                
                ## Make a prefilter that removes some microseism, and lowpasses at fN-5 to fN
                ij_pre_filt = [respfilt_bottom[0], respfilt_bottom[1],ij_nyquist-5,ij_nyquist]
                
                ## Then remove:
                print('running operation')

                for j_channel in range(len(i_st)):
                    ## Check if 'inventory' parameter works
                    try:
                        i_st[j_channel].remove_response(pre_filt=ij_pre_filt, output="VEL",plot=False)
                        
                        ## Remove pre-event baseline mean:
                        for j_channel in range(len(i_st)):
                            print('remove pre-event baseline mean in 5 seconds after collect time')
                            ij_5sec_index = np.where(i_st[j_channel].times() <= 5)[0]
                            ij_preeventmean = np.mean(i_st[j_channel].data[ij_5sec_index])
                            i_st[j_channel].data = i_st[j_channel].data - ij_preeventmean
                        
                        ### save the data....
                        print('saving data')
                        i_datetime = UTCDateTime(all_metadata_df.collecttime[recording_i]).datetime
                        i_time_for_path = str(i_datetime.year) + str(i_datetime.month)+ str(i_datetime.day)+ 'T'+ str(i_datetime.hour)+ str(i_datetime.minute)+ str(i_datetime.second)
                        i_network = all_metadata_df.network[recording_i]
                        i_station = all_metadata_df.stations[recording_i]
                        i_mag = int(all_metadata_df.mag[recording_i])
                        i_distance = int(all_metadata_df.rhypo[recording_i])
                        
                        i_mseedpath = f'{mseed_dir}{i_network}_{i_station}_{j_instrumenttype_base}_m{i_mag}_{i_distance}km_{i_time_for_path}.mseed' 

            #            i_mseedpath = '%s%s_%s_%s_m%.0f%.0fkm_%s.mseed' % (mseed_dir,all_metadata_df.network[recording_i], all_metadata_df.stations[recording_i],j_instrumenttype_base,all_metadata_df.mag[recording_i],all_metadata_df.distance[recording_i],i_time_for_path)
                        # i_mseedpath = mseed_dir + str(all_metadata_df.stations[recording_i]) + '_' + str(all_metadata_df.mag[recording_i]) + '_' + str(np.round(all_metadata_df.distance[recording_i],decimals=2)) + '_' + i_time_for_path + '.mseed'
                        i_st.write(i_mseedpath,format='mseed')
                        
                    except:
                        dlsuccess[recording_i] = False

            
        elif j_horiz_counter < 2:
            dlsuccess[recording_i] = False

## save metadata file with only dlsuccess, after adding channels:
## Add to dataframe:
all_metadata_df['channels'] = all_channels
dl_metadata_df = all_metadata_df[dlsuccess].reset_index(drop=True)
dl_metadata_df.to_csv((meta_dir + 'metadata_dldata.csv'),index=False)

# %% (8) CAN RUN AS STANDALONE AFTER FIRST TWO

################ PLOTTING STATIONS AND EARTHQUAKES ###################3


# Get stations from csv file
dl_data = '/Users/CynthiaMC/wv_hvsr/turbidities/observational_2020/metadata/metadata_dldata.csv'
dl_df = pd.read_csv(dl_data).dropna()

# Get unique station names and their indexes and storing them in arrays
unique_stations, index_list = np.unique(dl_df.stations, return_index=True)
print(unique_stations)
print(index_list)

# Create np array with zeros
sta_freq = np.zeros_like(unique_stations)

# Populate array with number of times unique_sta appears
for index, i_sta in enumerate(unique_stations):
    i_freq = len(np.where(dl_df.stations == i_sta)[0])
    sta_freq[index] = i_freq

print(sta_freq)

# Get lon and lat pf each unique station
unique_sta_lon = dl_df.st_lon[index_list]
unique_sta_lat = dl_df.st_lat[index_list]
 
# Plot Oregon map using pyGMT
 
plot_region = [-125, -121, 43.5, 46.5]
 
## Create the map
fig = pygmt.Figure()

pygmt.makecpt(cmap = 'geo',
              series = '-8000/4000/25',
              continuous=True)
 
fig.grdimage(grid = topo_data, region = plot_region, 
               projection = "M15c", shading = True, frame = True)
fig.coast(shorelines = True, frame = "ag", resolution = "i")

pygmt.makecpt(cmap = "jet",
               series = [min(sta_freq), max(sta_freq)])

fig.plot(x=unique_sta_lon, y=unique_sta_lat,
          cmap = True,
          style = "t0.7c",
          fill = sta_freq,
          pen = "black")

fig.plot(x=wvlons, y=wvlats,
          style="p0.05c",
          pen="black")


fig.colorbar(frame=["x+lEarthquakes Recorded"])
fig.show()
fig.savefig(fig_dir + 'station-magnitude.pdf')



