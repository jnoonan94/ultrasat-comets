#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 08:58:19 2023

@author: johnnoonan

This script should be used to query the JPL Small Body Database Identification API for gathering large numbers of targets in a FOV. The AstroQuery package currently
only allows specific object searches. I think that Mike Kelley might have a smiliar code in sbpy, but I haven't found it yet so I'm writing this one. 

I'll note that I'm basing the structure off of the API query code that is available from AstroQuery for the SBDB, but tweaking the input parameters for the Identification API.
"""

import requests
import json
import time

def jprint(obj):
    """
    

    Parameters
    ----------
    obj : dict, produced from the requests.get() function
        This is a simple script to make the result of the request.get() function readable when printed. 

    Returns
    -------
    None.

    """
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)
def param_build(date,vmag,fov_ra,fov_dec,loc=0,lat=0,lon=0,alt=0,fov_center=False):
    """There are several required parameters for a successful query, see https://ssd-api.jpl.nasa.gov/doc/sb_ident.html for more
    loc: string,MPC observer location code
    date: [required] date/time of the observing night (formatted as YYYY-MM-DD[_hh:mm:ss] or JD, e.g., 2459555.5)
    vmag: float, visual magnitude threshold
    fov_ra: right-ascension values of the edges of the FOV (hh-mm-ss[.ss]). Fields separated by commas. For negative values, replace minus sign with M, e.g. M10-00-00.02
    fov_dec: declination values of the edges of the FOV (dd-mm-ss[.ss]). Fields separated by commas. For negative values, replace minus sign with M, e.g. M10-00-00.02
    """
    if loc != 0:
        if fov_center == False:
            parameters = {'mpc-code':loc
                  ,'obs-time':date
                  ,'vmag-lim':vmag
                  ,'sb-class':('JFc,ETc,HTC,PAR,HYP,COM')
                  ,'fov-ra-lim':fov_ra,'fov-dec-lim':fov_dec
                  ,'mag-required':'true'
                  ,'suppress-first-pass':'true'
                  ,'two-pass':'true'
                  }
        elif fov_center == True:
            parameters = {'mpc-code':loc
                  ,'obs-time':date
                  ,'vmag-lim':vmag
                  ,'sb-class':('JFc,ETc,HTC,PAR,HYP,COM')
                  ,'mag-required':'true'
                  ,'suppress-first-pass':'true'
                  ,'two-pass':'true'
                  ,'fov-ra-center':fov_ra[0]
                  ,'fov-dec-center':fov_dec[0]
                  ,'fov-ra-hwidth':fov_ra[1]
                  ,'fov-dec-hwidth':fov_dec[1]
                  }  
    if loc == 0:
        if fov_center == False:
            parameters = {'lat':lat
                  ,'lon':lon
                  ,'alt':alt
                  ,'obs-time':date
                  ,'vmag-lim':vmag
                  ,'mag-required':'true'
                  ,'suppress-first-pass':'true'
                  ,'two-pass':'true'
                  ,'sb-class':('JFc,ETc,HTC,PAR,HYP,COM')
                  ,'fov-ra-lim':fov_ra,'fov-dec-lim':fov_dec
                  }
   
        elif fov_center == True:
            parameters = {'lat':lat
                  ,'lon':lon
                  ,'alt':alt
                  ,'obs-time':date
                  ,'vmag-lim':vmag
                  ,'mag-required':'true'
                  ,'suppress-first-pass':'true'
                  ,'two-pass':'true'
                  ,'sb-class':('JFc,ETc,HTC,PAR,HYP,COM')
                  ,'fov-ra-center':fov_ra[0]
                  ,'fov-dec-center':fov_dec[0]
                  ,'fov-ra-hwidth':fov_ra[1]
                  ,'fov-dec-hwidth':fov_dec[1]
                  }
    return parameters

def get_response(parameters):
    """
    a

    Parameters
    ----------
    parameters : DICTIONARY
        Product of param_build function, used for the API query to the Small Body Database Identification API.

    Returns
    -------
    response : JSON
        JSON object with queried data from JPL SBDB

    """    
    response = requests.get("https://ssd-api.jpl.nasa.gov/sb_ident.api", params=parameters)
    return response
def test_query():
    """
    This is a simple test script to makes sure that a query for observable objects from McDonald Observatory works. Used for debugging. 

    Returns
    -------
    response: dict, API query result

    """
    start = time.time()
    #for RA and Dec defined limits
    parameters = param_build('2021-02-09' , 20, '10-10-00,10-20-00', '10-00-00,10-30-00',loc='568')
    #For RA and Dec Centeres and Half maxes
    #parameters = param_build('2019-01-01' , 20, ['14-40-00',10], ['66-00-00',5],loc='711',fov_center=True)
    response = get_response(parameters)
    end = time.time()
    print('Runtime= '+str(end-start)+' s')
    return response