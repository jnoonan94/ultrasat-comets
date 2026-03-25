#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:25:20 2023
This script is used to a) generate the RA and Decs of the UltraSat field of view and b)loop through calls to 
the sbdb_ident_query functions to produce JSON objects with the objects in a given FOV

@author: johnnoonan
"""

from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,get_sun
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude 
from astropy import units as u
from astropy.time import Time
import numpy as np
from sbdb_ident_query import *
import datetime
import time
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

"""Hardcoding in some parameters for the Ultrasat"
"""
ultrasat_fov=[Angle(20*u.deg),Angle(10*u.deg)]
full_field = [Angle(180*u.deg),Angle(30*u.deg)]
dec_NGP = 27.13*np.pi/180
ra_NGP = 192.85*np.pi/180
l_NCP = 123*np.pi/180

def build_fov_arrays(ga_lon_range,ga_lat_range,N):
    """

    Parameters
    ----------
    ga_lat_range : TUPLE
        DESCRIPTION.
    ga_lon_range : TUPLE
        DESCRIPTION.
    N : INTEGER
        Number of FOV centers to produce

    Returns
    -------
    fov_array : TYPE
        DESCRIPTION.\

    """
    #How to convert a simple condition in galactic coordinates to a set of fixed FOVs in RA and Dec. 
    #Want to return a set of tuples that is the center location of the Ultrasat FOV for 40 of these that cover a swath of sky between GA -30 and 30 lat. 
    #Convert to radians
    ga_lat_range = np.array(ga_lat_range)*np.pi/180
    ga_lon_range = np.array(ga_lon_range)*np.pi/180
    #First we want to define the boundaries in RA and Dec as a function
    def dec(l,b):
        dec = np.arcsin(np.sin(dec_NGP)*np.sin(b)+np.cos(dec_NGP)*np.cos(b)*np.cos(l_NCP-l))
        return dec
    def ra(l,b,dec):
        ra = np.arcsin(np.cos(b)*np.sin(l_NCP-l)/np.cos(dec))+ra_NGP
        return ra
    l_mesh,b_mesh = np.meshgrid(np.linspace(ga_lon_range[0],ga_lon_range[1],N),np.linspace(ga_lat_range[0],ga_lat_range[1],N),sparse=False)
    l_col = l_mesh.flatten()
    b_col = b_mesh.flatten()
    ga_coord_centers = np.vstack((l_col,b_col))
    fov_array = []
    for i in range(0,len(ga_coord_centers[0])):
        l = ga_coord_centers[:,i][0]*180/np.pi
        b = ga_coord_centers[:,i][1]*180/np.pi
        coord_galactic = SkyCoord(l,b,frame=Galactic,unit=u.deg)
        coord_icrf = coord_galactic.icrs
        fov = coord_icrf.to_string()
        fov_array.append(fov)
    return fov_array 
def retrieve_fov_array(debug=False):
    '''
    

    Parameters
    ----------
    debug : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fov_array : TYPE
        DESCRIPTION.

    '''
    ultrasat_fields = pd.read_csv('LCS_nonoverlapping_grid.csv')
    coord_ult_lcs = SkyCoord(ultrasat_fields['RA'].values,ultrasat_fields['Dec'].values,frame=ICRS,unit='deg')
    fov_array = coord_ult_lcs.to_string()
    return fov_array
def build_date_range(start,stop,step):
    """
    

    Parameters
    ----------
    start : TYPE
        DESCRIPTION.
    stop : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.

    Returns
    -------
    date_array : TYPE
        DESCRIPTION.

    """
    N = step # Based on on the observing cadence of Ultrasat
    datelist = pd.date_range(start=start, end=stop, periods=N).tolist()
    return datelist

def api_query(ra_lim,dec_lim,date,v_lim,fov_center=False):
    """
    

    Parameters
    ----------
    ra_lim : TYPE
        DESCRIPTION.
    dec_lim : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    lat = 32.028140
    lon = 34.898320
    alt = 35785
    vmag = v_lim
    parameters = param_build(date, vmag, ra_lim, dec_lim,lat=lat,lon=lon,alt=alt,fov_center=fov_center)
    response = get_response(parameters)
    result = response.json()
    name_list = []
    coord_list = []
    #For the moment we are only going to return the number of objects, but we may want
    #to add more so we will return them as a dictionary
    #Get all object names
    try:
      n_results = result['n_second_pass']
      for i in range(0,result['n_second_pass']):
          name = result['data_second_pass'][i][0]
          ra = result['data_second_pass'][i][1]
          dec_init = result['data_second_pass'][i][2]
          dec = dec_init.replace("'"," ")
          dec = dec.replace('"',"")
          dec_corrected = dec.replace("\\","")
          try:
              coord = SkyCoord(ra=ra,dec=dec_corrected, unit=(u.hourangle,u.deg))
          except ValueError:
              print(ra, dec_init, dec_corrected)
          name_list.append(name)
          coord_list.append([coord.ra.deg,coord.dec.deg])
    except KeyError:
        try:
            print(response.json()['message'])
            if 'fov-dec-lim' in response.json()['message']:
                print('Issue is in the declination: '+dec_lim)
            if 'fov-ra-lim' in response.json()['message']:
                print('Issue is in the RA: '+ra_lim)
        except KeyError:
           if 'no matching records' in result['warning']:
               print('No matching records')
               n_results = 0
               name_list = []
               coord_list = []
           else:
               print(result['warning'])
   
    query = (n_results,name_list,coord_list)
    return query
def sbdb_ra_rewrite(ra_min,ra_max,ra_lim):
    
    if abs(ra_min) < 150:
        if abs(ra_min) < 15:
            if ra_min < 0:
                temp = list(ra_lim)
                temp= list('0')+temp[1:]
                ra_lim ="".join(temp)
            else:
                temp = list(ra_lim)
                temp = list('0')+temp
                ra_lim = "".join(temp)
        elif ra_min < 0:
            temp = list(ra_lim)
            temp[0] = '0'
            ra_lim ="".join(temp)
        else:
            temp = list(ra_lim)
            ra_lim_list = list('0')+temp
            ra_lim = "".join(ra_lim_list)
    if abs(ra_max) < 150:
        if abs(ra_max) < 15:
            if ra_max < 0:
                temp = list(ra_lim)
                idx = temp.index(',')
                temp[idx+1] = '0'
                ra_lim = "".join(temp)
            else:
                temp = list(ra_lim)
                idx = temp.index(',')
                ra_lim_list = temp[:idx+1]+list('0')+temp[idx+1:]
                ra_lim = "".join(ra_lim_list)
        elif ra_max < 0:
            temp = list(ra_lim)
            idx = temp.index(',')
            temp[idx+1] = '0'
            ra_lim = "".join(temp)
        else:
            temp = list(ra_lim)
            idx = temp.index(',')
            ra_lim_list = temp[:idx+1]+list('0')+temp[idx+1:]
            ra_lim = "".join(ra_lim_list)
    return ra_lim
def sbdb_dec_rewrite(dec_min,dec_max,dec_lim):
    """
    

    Parameters
    ----------
    dec_min : TYPE
        DESCRIPTION.
    dec_max : TYPE
        DESCRIPTION.
    dec_lim : TYPE
        DESCRIPTION.

    Returns
    -------
    dec_lim : TYPE
        DESCRIPTION.

    """
    
    if abs(dec_min) <10:
        if abs(dec_min) < 1:
            if dec_min < 0:
                print('Value should be less than -1 degree')
                temp = list(dec_lim)
                temp = dec_lim[1:]
                print(temp)
                dec_lim_list = 'M0'+temp
                dec_lim = "".join(dec_lim_list)
                print(dec_lim, ' from sbdb rewrite')
            else:
                temp = list(dec_lim)
                temp = dec_lim
                dec_lim_list = '0'+temp
                dec_lim = "".join(dec_lim_list)
        elif dec_min < 0:
            temp = list(dec_lim)
            temp = dec_lim[1:]
            dec_lim_list ='M0' + temp
            dec_lim = "".join(dec_lim_list)
        else:
            print('Value is less than 10 degrees')
            temp = list(dec_lim)
            dec_lim_list = list('0') + temp
            dec_lim = "".join(dec_lim_list)
    if abs(dec_min) > 10:
        if dec_min < 0:
            temp = list(dec_lim)
            temp[0] = 'M'
            dec_lim = "".join(temp)
    if abs(dec_max) <10:
        small_negative_flag = False
        if abs(dec_max) < 1:
            if dec_max < 0:
                print('Value should be less than -1 degree')
                temp = list(dec_lim)
                idx = temp.index(',')
                temp[idx+1] = 'M'
                dec_lim_list = temp[:idx+2]+list('0')+temp[idx+2:]
                dec_lim = "".join(dec_lim_list)
                print(dec_lim, ' from sbdb rewrite')
                small_negative_flag = True
            else:
                temp = list(dec_lim)
                idx = temp.index(',')
                dec_lim_list = temp[:idx+1]+list('0')+temp[idx+1:]
                dec_lim = "".join(dec_lim_list)
        elif dec_max < 0:
            if small_negative_flag == True:
                print('Almost reset set the Dec Max again')
                dec_lim = dec_lim
            else:
                temp = list(dec_lim)
                idx = temp.index(',')
                temp[idx+1] = '0'
                dec_lim_list = temp[:idx+1]+list('M')+temp[idx+1:]
                dec_lim = "".join(dec_lim_list)
        else:
            temp = list(dec_lim)
            idx = temp.index(',')
            dec_lim_list = temp[:idx+1]+list('0')+temp[idx+1:]
            dec_lim = "".join(dec_lim_list)
    if abs(dec_max) > 10:
        if dec_max < 0:
            temp = list(dec_lim)
            idx = temp.index(',')
            temp[idx+1]='M'
            dec_lim = "".join(temp)
    
    return dec_lim
        
def api_loop(fov_array,date_range,v_lim,debug=False,stare=False,single_field=False):
    """
    

    Parameters
    ----------
    ra_array : TYPE
        DESCRIPTION.
    dec_array : TYPE
        DESCRIPTION.
    date_array : TYPE
        DESCRIPTION.
    debug : Boolean, toggles verbose printouts.
    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    obs_dict =  {}
    for i,date in enumerate(date_range):
        date_str = datetime.datetime.strftime(date_range[i],'%Y-%m-%d_%H:%M:%S')
        date_dict =  {}
        if single_field == False:
            for j,fov in enumerate(fov_array):
                fov_list = fov.split(' ')
                #Calculate the new lims of RA and Dec for the FOV
                ra_cen = Angle(float(fov_list[0])*u.deg)
                dec_cen = Angle(float(fov_list[1])*u.deg)
                
                ra_lim = (ra_cen-ultrasat_fov[0]/2).to_string(unit=u.hour,sep='-')+','+(ra_cen+ultrasat_fov[0]/2).to_string(unit=u.hour,sep='-') #should be in decimal degrees, need to double check units with Astropy
                ra_min,ra_max = (ra_cen-ultrasat_fov[0]/2)/u.deg,(ra_cen+ultrasat_fov[0]/2)/u.deg
                ra_lim_api = sbdb_ra_rewrite(ra_min, ra_max, ra_lim)
                dec_min,dec_max = (dec_cen-ultrasat_fov[1]/2)/u.deg,(dec_cen+ultrasat_fov[1]/2)/u.deg
                dec_lim = (dec_cen-ultrasat_fov[1]/2).to_string(unit=u.deg,sep='-')+','+(dec_cen+ultrasat_fov[1]/2).to_string(unit=u.deg,sep='-') #should be in decimal degrees, need to double check units with Astropy
                dec_lim_api = sbdb_dec_rewrite(dec_min, dec_max, dec_lim)

                if debug == True:
                    #print(dec_lim)
                    print(dec_min,dec_max)
                    print(dec_lim_api)
                    print(ra_min,ra_max)
                    print(ra_lim_api)
                    print('----------')
                query = api_query(ra_lim_api,dec_lim_api,date_str,v_lim)
                #write necessary parts of result to obs_dictrun
                date_dict[j] = {'FOV':fov_list,'Number of Objects':query[0],'Names':query[1],"ICRS Coordinates":query[2]}
                #date_dict['Number of Objects'].append(query['Number of objects'])
                #date_dict['Names'].append(query['Names'])
        if single_field == True:
            ra_lim_api = ['00-00-00',180]
            dec_lim_api = ['00-00-00',90]
            fov_list = [0,0]
            query = api_query(ra_lim_api,dec_lim_api,date_str,v_lim,fov_center=True)
            #write necessary parts of result to obs_dictrun
            date_dict[0] = {'FOV':fov_list,'Number of Objects':query[0],'Names':query[1],"ICRS Coordinates":query[2]}
            #date_dict['Number of Objects'].append(query['Number of objects'])
            #date_dict['Names'].append(query['Names'])
        obs_dict[i] = {'Date':date_str,'Data':date_dict}
    return obs_dict #Should be a dictionary, one key contains number of objects as a function of date,
            # another that reports object RA and DEC

def run_observability_simulator(v_lim,start,stop,step,lat_range,lon_range,N,lcs_ingest = False,stare=False,debug=False,single_field=False):
    """
    

    Parameters
    ----------
    start : TYPE
        DESCRIPTION.
    stop : TYPE
        DESCRIPTION.

    Returns
    -------
    obs_dict : TYPE
        DESCRIPTION.

    """
    sim_start = time.time()
    date_range = build_date_range(start,stop,step)
    if stare == False:
        if single_field == False:
            print('Using general protocol...')
            fov_array = build_fov_arrays(lat_range,lon_range,N) #Make sure input lat and lon are in galactic coordinates
            if lcs_ingest == True:
                fov_array = retrieve_fov_array()
        elif single_field == True:
            print('Using single field protocol...')
            coord_galactic = SkyCoord(lon_range,lat_range,frame=Galactic,unit='deg')
            coord_icrf = coord_galactic.icrs
            fov_array = [coord_icrf.to_string(unit=u.deg,sep='-')]
            print(fov_array)
    else:
        print('Using stare protocol...')
        fov_array = [str(lat_range)+' '+str(lon_range)] #For the stares we know the RA and Dec, so as long as the RA and Dec are in API friendly terms this should work. 
    #print(fov_array)
    obs_dict = api_loop(fov_array,date_range,v_lim,debug=debug,stare=stare,single_field=single_field)
    sim_end = time.time()
    tot = sim_end-sim_start
    print(f"Simulation took {tot:2f} seconds")
    return obs_dict

def gen_total_observability(obs_dict):
    num_dates = len(obs_dict.keys())
    num_fov = len(obs_dict[0]['Data'].keys())
    ra_fov_array,dec_fov_array,n_objects_array,coord_obj,names,no_filt_coord,gal_filt = [],[],[],[],[],[],[]
    for i in range(0,num_dates):
        date = Time(obs_dict[i]['Date'].replace('_','T'))
        sun_coord = get_sun(date)
        moon_coord = get_sun(date)
        for j in range(0,num_fov):
            ra_fov_array.append(float(obs_dict[i]['Data'][j]['FOV'][0]))
            dec_fov_array.append(float(obs_dict[i]['Data'][j]['FOV'][1]))
            n_objects_array.append(obs_dict[i]['Data'][j]['Number of Objects'])
            n_obj = obs_dict[i]['Data'][j]['Number of Objects']
            toss =0
            for k in range(0,n_obj):
                coord = SkyCoord(obs_dict[i]['Data'][j]['ICRS Coordinates'][k][0],obs_dict[i]['Data'][j]['ICRS Coordinates'][k][1], unit = (u.deg,u.deg))
                no_filt_coord.append(obs_dict[i]['Data'][j]['ICRS Coordinates'][k])
                if abs(coord.galactic.b/u.deg) < 30:
                    gal_filt.append(obs_dict[i]['Data'][j]['ICRS Coordinates'][k])
                    if coord.separation(sun_coord)/u.deg > 70 and coord.separation(moon_coord)/u.deg > 30:
                        coord_obj.append(obs_dict[i]['Data'][j]['ICRS Coordinates'][k])
                        names.append(obs_dict[i]['Data'][j]['Names'][k])
                    else:
                        
                        toss += 1
                else:
                    toss += 1
                
    coord_array = np.array(coord_obj)
    no_filt_coord_array = np.array(no_filt_coord)
    gal_filt_coord_array =  np.array(gal_filt)
    print(str(toss) + ' entries rejected due to field limits')
    return ra_fov_array,dec_fov_array,n_objects_array,coord_array,names,no_filt_coord_array,gal_filt_coord_array

def plot_total_observability(ra_fov_array,dec_fov_array,n_objects_array,coord_array,no_filt_coord,gal_filt_coord,names):   
    fig1,ax1 = plt.subplots()
    ax1.grid(c='k',ls='solid')
    
    im1 = ax1.scatter(ra_fov_array,dec_fov_array, c=n_objects_array,vmin=0,vmax = 5)
    fig1.colorbar(im1,ax=ax1)
    ax1.set_xlabel('Right Ascension (J2000)')
    ax1.set_ylabel('Declination (J2000)')
    print(coord_array[0,0])
    com_coord= SkyCoord(coord_array[:,0],coord_array[:,1], unit = (u.deg,u.deg))
    com_coord_no_filt = SkyCoord(no_filt_coord[:,0],no_filt_coord[:,1], unit = (u.deg,u.deg))
    com_coord_gal_filt = SkyCoord(gal_filt_coord[:,0],gal_filt_coord[:,1], unit = (u.deg,u.deg))
    
    fig2,ax2 = plt.subplots()
    ax2.grid(c='k',ls='solid')
    ax2.scatter(com_coord_no_filt.ra.wrap_at('180d').deg, com_coord_no_filt.dec.deg,marker = '*',c='r',s=10)
    ax2.set_xlabel('Right Ascension (J2000)',fontsize=15)
    ax2.set_ylabel('Declination (J2000)',fontsize=15)
    
    fig3,ax3, = plt.subplots()
    ax3.grid()
    gal_coord = com_coord.galactic
    im3 = ax3.hexbin(gal_coord.l.wrap_at('180d').deg, gal_coord.b.deg,gridsize=50,cmap='cool',vmin=0,vmax=100)
    ax3.set_ylabel('Galactic Latitude ($^{\circ}$)',fontsize=20)
    ax3.set_xlabel('Galactic Longitude ($^{\circ}$)',fontsize=20)
    cbar3 =fig3.colorbar(im3,ax=ax3,fraction=0.03, pad=0.04)
    cbar3.set_label('Number of Detections in 2025', rotation=270,fontsize=10)
    #ax3.title.set_text('Cumulative Comet Detections Per Sky Bin for 2025: All Pointing Constraints, Galactic Frame')
    
    fig4,ax4 = plt.subplots()
    ax4.grid()
    im4 = ax4.hexbin(com_coord.ra.wrap_at('180d').deg, com_coord.dec.deg,gridsize=50,cmap='cool',vmin=0,vmax=100)
    ax4.set_xlabel('Right Ascension (J2000) ($^{\circ}$)',fontsize=20)
    ax4.set_ylabel('Declination (J2000) ($^{\circ}$)',fontsize=20)
    cbar4 = fig4.colorbar(im4,ax=ax4,fraction=0.03, pad=0.04)
    cbar4.set_label('Number of Detections in 2025', rotation=270,fontsize=10)
    #ax4.title.set_text('Cumulative Comet Detections Per Sky Bin for 2025 - All Pointing Constraints, ICRS Frame')
    
    fig5,ax5 = plt.subplots()
    ax5.grid()
    im5 = ax5.hexbin(com_coord_no_filt.ra.wrap_at('180d').deg, com_coord_no_filt.dec.deg,gridsize=50,cmap='cool',vmin=0,vmax=50)
    ax5.set_xlabel('Right Ascension (J2000) ($^{\circ}$)',fontsize=15)
    ax5.set_ylabel('Declination (J2000) ($^{\circ}$)',fontsize=15)
    fig5.colorbar(im5,ax=ax5,fraction=0.03, pad=0.04)
    ax5.title.set_text('Cumulative Comet Detections Per Sky Bin for 2025 - No Pointing Constraints')
    
    fig6,ax6 = plt.subplots()
    ax6.grid()
    im6 = ax6.hexbin(com_coord_gal_filt.ra.wrap_at('180d').deg, com_coord_gal_filt.dec.deg,gridsize=50,cmap='cool',vmin=0,vmax=50)
    ax6.set_xlabel('Right Ascension (J2000) ($^{\circ}$)',fontsize=15)
    ax6.set_ylabel('Declination (J2000) ($^{\circ}$)',fontsize=15)
    fig6.colorbar(im6,ax=ax6,fraction=0.03, pad=0.04)
    ax6.title.set_text('Cumulative Comet Detections Per Sky Bin for 2025 - $\pm$30$^{\circ}$ of Galactic Plane')
    
            

    
