'''
  Some helpers with geolocation

  Requires:
  pip3 install geopy

'''

import re
from geopy.geocoders import Nominatim


def dms_to_dec(_degrees, _minutes,_seconds, _ref):
  """Convert from DMS coordinates to DEC

  Credits/Adapted from here:
  https://gist.github.com/chrisjsimpson/076a82b51e8540a117e8aa5e793d06ec
  
  Args:
      _degrees ([type]): [description]
      _minutes ([type]): [description]
      _seconds ([type]): [description]
      _ref ([type]): [description]

  Returns:
      [type]: [description]
  """  
  sign = -1 if re.search('[swSW]', _ref) else 1
  return sign * (int(_degrees) + float(_minutes) / 60 + float(_seconds) / 3600)




def reverse_coordinates_geopy(latitude_dec: float, longitude_dec: float, user_agent='myGeocoder'):
  """Call geopy to resolve the location of given DEC coordinates.

  Args:
      latitude_dec (float): [description]
      longitude_dec (float): [description]
      user_agent (str, optional): [description]. Defaults to 'myGeocoder'.

  Returns:
      [type]: [description]
  """  

  geolocator = Nominatim(user_agent=user_agent)

  coordinates = f'''{latitude_dec}, {longitude_dec}'''
  location = geolocator.reverse(coordinates)

  raw_location = location.raw['address']

  return raw_location


def get_text_location(raw_location: dict):
  """Get a summary of the geopy location

  Args:
      raw_location (dict): Retrieved with reverse_coordinates_geopy

  Returns:
      [str]: location in plain text
  """  

  town = raw_location.get('town', None)
  city = raw_location.get('city', '')
  country = raw_location.get('country', '')

  text_location = ''
  if town:
    text_location += f'''{town}-{city}'''
  else:
    text_location += f'''{city}'''

  text_location += f''' ({country})'''

  return text_location