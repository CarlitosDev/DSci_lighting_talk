'''
  Some methods for images


  Requires:  
  
  brew install libffi libheif
  pip3 install pyheif

'''

import cv2
import matplotlib.pyplot as plt
import pyheif
from PIL import Image
import os
import exifread
import numpy as np
import re
import matplotlib
matplotlib.use('tkagg')

# CV2 is not working. No idea why?
# img_path = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750/_DSC9677.JPG'
#image = cv2.imread(img_path)
#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()





def load_image(img_path, useCV2=False):
  '''
    Load an image given the correct path.
    Both methods return a uint8 'numpy.ndarray' but the CV2 intensities are not 
    directly compatible with matplotlib.
  '''
  if useCV2:
    this_image = cv2.imread(img_path)
  else:
    this_image = plt.imread(img_path)

  return this_image


def save_image(image_data, img_path: str, useCV2=True):
  """Write image to disk using CV2

  Args:
      image_data ([type]): [description]
      img_path (str): [description]
      useCV2 (bool, optional): [description]. Defaults to True.
  """  
  cv2.imwrite(img_path, image_data)

def show_image(this_image):
  '''
    Show image using matplotlib
  '''
  fig, ax = plt.subplots()
  im = ax.imshow(this_image)
  ax.axis('off')
  plt.show()


def show_imagefile(img_path):
  show_image(load_image(img_path))




def load_heic_image(img_path):
  '''
    Read HEIC images
  '''

  with open(img_path, 'rb') as f:
      data = f.read()

  pyheif_img = pyheif.read_heif(data)
  img_data = Image.frombytes(mode=pyheif_img.mode, \
    size=pyheif_img.size, data=pyheif_img.data)

  return img_data


def load_heic_image_as_numpy(img_path: str):
  """Load a HEIC image as a numpy array

    Recommended to fiddle around with PyTorch

  Args:
      img_path (str): fullpath to the HEIC file

  Returns:
      [type]: nd-array
  """  

  return np.asarray(load_heic_image(img_path))


def from_heic_to_jpg(img_1_path, img_output=None):
  
  img_1_data = load_heic_image(img_1_path)

  if not img_output:
    img_output = img_1_path.replace('HEIC', 'jpg')

  img_1_data.save(img_output, format="JPEG")
  print(f'JPG file saved to {img_output}')




def extract_frames_from_video(video_path, num_frames_to_capture = 10):

  foldername, filename, ext = fu.fileparts(video_path)
  cap = cv2.VideoCapture(video_path)

  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration_seconds = frame_count/fps

  frame_interval = int(frame_count/num_frames_to_capture)

  frame_counter = 0

  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      if frame_counter%frame_interval == 0:
          frame_name = os.path.join(foldername, f'filename_frame_{frame_counter}.jpg')
          cv2.imwrite(frame_name, frame)
      frame_counter+=1
      
  cap.release()
  cv2.destroyAllWindows()



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


def get_GPS_EXIF_info_from_file(path_name):
  '''
    Get the EXIF info from a file
    It uses the library exifread
  '''
  from fractions import Fraction
  with open(path_name, 'rb') as f:
    tags = exifread.process_file(f)

  gps_tags = [itag for itag in tags.keys() if 'gps' in itag.lower()]

  lat_ref = str(tags['GPS GPSLatitudeRef'])
  lat_values = str(tags['GPS GPSLatitude']).replace('[','').replace(']','').split(',')

  lat_degrees = int(lat_values[0])
  lat_minutes = int(lat_values[1])
  frc = Fraction(lat_values[-1])
  lat_seconds = frc.numerator/frc.denominator
  latitude_tag = f'''{lat_degrees} {lat_minutes}\' {lat_seconds}\'\' {lat_ref}'''

  latitude_dec = dms_to_dec(lat_degrees, lat_minutes, lat_seconds, lat_ref)


  lon_ref = str(tags['GPS GPSLongitudeRef'])
  lon_values = str(tags['GPS GPSLongitude']).replace('[','').replace(']','').split(',')
  frc = Fraction(lon_values[-1])


  lon_degrees = int(lon_values[0])
  lon_minutes = int(lon_values[1])
  lon_seconds = frc.numerator/frc.denominator
  longitude_tag = f'''{lon_degrees} {lon_minutes}\' {lon_seconds}\'\' {lon_ref}'''

  longitude_dec = dms_to_dec(lon_degrees, lon_minutes, lon_seconds, lon_ref)

  datetime_tag = str(tags['Image DateTime'])


  exif_info  = {'latitude_tag': latitude_tag,
  'lat_ref':lat_ref,
  'lat_values':lat_values,
  'lat_degrees':lat_degrees,
  'lat_minutes':lat_minutes,
  'lat_seconds':lat_seconds,
  'longitude_tag': longitude_tag,
  'latitude_dec': latitude_dec,
  'lon_ref':lon_ref,
  'lon_values':lon_values,
  'lon_degrees':lon_degrees,
  'lon_minutes':lon_minutes,
  'lon_seconds':lon_seconds,
  'longitude_dec': longitude_dec,
  'datetime_tag':datetime_tag}

  return exif_info



def get_EXIF_info_from_file(path_name: 'path to a EXIF-tagged picture', \
  get_default_tags = True):
  """Extract EXIF information from (EXIF-tagged) pictures

  Args:
      path_name (path to a EXIF): [description]
      get_default_tags (bool, optional): [description]. Defaults to True.
  """  
  with open(path_name, 'rb') as f:
    tags = exifread.process_file(f)

  img_keyword = 'Image '
  exif_keyword = 'EXIF '

  if get_default_tags:

    image_tags_to_extract = ['Image Make', 'Image Model', 'Image Software', 
    'Image DateTime', 'Image Orientation', 'Image XResolution', 'Image YResolution']

    exif_tags_to_extract = ['EXIF ExposureTime','EXIF FNumber','EXIF ISOSpeedRatings',
    'EXIF ComponentsConfiguration','EXIF ShutterSpeedValue','EXIF ApertureValue',
    'EXIF Flash','EXIF FocalLength','EXIF ExifImageWidth','EXIF ExifImageLength',
    'EXIF ExposureMode','EXIF SceneType',
    'EXIF LensMake','EXIF LensModel']
  else:
    image_tags_to_extract = [k for k in tags.keys() if img_keyword in k]
    exif_tags_to_extract  = [k for k in tags.keys() if exif_keyword in k]

  # process image tags
  
  image_tags = {k.replace(img_keyword, ''):str(v) for k,v in tags.items() if k in image_tags_to_extract}
  EXIF_tags = {k.replace(exif_keyword, ''):str(v) for k,v in tags.items() if k in exif_tags_to_extract}
  return image_tags, EXIF_tags




def overlay_text(image: np.array, this_text: str, line_sep='\n', \
  x0=None, y0=None, dy=None):
  """Overlay text over a given image

  Args:
      image (np.array): [description]
      this_text (str): [description]
      line_sep (str, optional): [description]. Defaults to '\n'.
      y0 ([type], optional): [description]. Defaults to None.
      dy ([type], optional): [description]. Defaults to None.

  Returns:
      image: image with overlayed text
  """  
  
  (H, W) = image.shape[:2]

  # x -> horizontal
  if not x0:
    x0 = int(W*0.5)
  if not y0:
    y0 = int(H*0.80)
  if not dy:
    dy = int(W/30)

  lineType = 10
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale  = 5
  # Blue color in BGR
  fontColor = [0,0,255]

  for i, line in enumerate(this_text.split(line_sep)):
    y1 = y0 + i*dy
    cv2.putText(image, line, (x0, y1), font, fontScale, fontColor, lineType)
  
  return image