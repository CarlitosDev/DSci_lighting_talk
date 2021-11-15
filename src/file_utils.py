import pandas as pd
import os
import pickle
import subprocess, sys
from datetime import datetime
from shutil import copyfile, move
import json
import yaml
from pygments import highlight, lexers, formatters

import re
import datetime as dt
import uuid
import carlos_utils.utils_root as ur
from carlos_utils.aws_data_utils import read_json_from_s3

import glob
from pathlib import Path

def fullfile(foldername, filename):
    return os.path.join(foldername, filename)

def fileparts(thisPath):
    [fPath, fName] = os.path.split(thisPath)
    [file, ext] = os.path.splitext(fName)
    return fPath, file, ext

def makeFolder(thisPath):
    if not os.path.exists(thisPath):
        os.makedirs(thisPath)

def osOpenFile(filePath):
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, filePath])


def get_folder_contents_timesorted(folder):
  # return sorted(str(Path(folder).iterdir()), key=os.path.getmtime)
  return [str(iFile) for iFile in \
    sorted(Path(folder).iterdir(), key=os.path.getmtime)]


# Read/Write text files
def readTextFile(filePath):
    with open(filePath, 'r') as in_file:
        textData = in_file.read()
    return textData

def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)

# JSON helpers
def writeJSONFile(inputObj: 'Ideally a dictionary', thisFile):
  formatted_json = 'Error'
  if isinstance(inputObj, dict):
    formatted_json = beautifyJSON(inputObj)
  elif isinstance(inputObj, str):
    # This is quick but prints out ugly stuff
    try:
      formatted_json = beautifyJSON(json.loads(inputObj))
    except:
      formatted_json = inputObj
  with open(thisFile, 'w') as f:
      f.write(formatted_json)

def readJSONFile(thisFile):
    with open(thisFile, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def readYAMLFile(thisFile):
    with open(thisFile, 'r') as yaml_file:
        yaml_data = yaml.full_load(yaml_file)
    return yaml_data

# TODO: Keys are not sorted. Add a default parameter.
def beautifyJSON(jsonBlob: 'A dictionary'):
    return json.dumps(jsonBlob, indent=2, sort_keys=False, default=str)

def printJSON(inputObj: 'Ideally a dictionary'):
  formatted_json = 'Error'
  if isinstance(inputObj, dict):
    formatted_json = beautifyJSON(inputObj)
  elif isinstance(inputObj, str):
    # This is quick but prints out ugly stuff
    try:
      formatted_json = beautifyJSON(json.loads(inputObj))
    except:
      formatted_json = inputObj
  elif isinstance(inputObj, list):
    for this_item in inputObj:
      printJSON(this_item)
  print(highlight(formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()))

# Pickle helpers
def toPickleFile(data, filePath):
    print(f'Writing file {filePath}...')
    with open(filePath, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

# Binary data, not text
def readPickleFile(filePath):
    with open(filePath, 'rb') as fId:
        pickleData = pickle.load(fId)
    return pickleData

# Pandas dataframes
def dfToTabCSV(df, csvPath):
    df.to_csv(csvPath, sep='\t', index=False)

def dfToTabMAT(df, matPath):
    import scipy.io as sio
    dictFromDF = df.to_dict(orient='list');
    sio.savemat(matPath, {'dataFromPD': dictFromDF});

def dataFrameToPickle(df, currentFile):
    print(f'Saving DF in {currentFile}...')
    with open(currentFile, 'wb') as f:
        pickle.dump(df,f)

def dataFrameToXLS(df, xlsFile, sheetName = 'DF', \
        writeIndex = False, float_format = '%.2f'):
    if not df.empty:
        xlsWriter = pd.ExcelWriter(xlsFile)
        df.to_excel(xlsWriter, sheetName, \
                index = writeIndex, float_format = float_format)
        xlsWriter.save();

def dataFrameToXLSv2(df, xlsFile, sheetName = 'DF', \
        writeIndex = False, float_format = '%.2f', freezeTopRow = True, \
        remove_timezone=True):
    '''
    Write DF to Excel centering the columns and freezing the top row
    '''        
    if not df.empty:
        xlsWriter = pd.ExcelWriter(xlsFile, engine='xlsxwriter', \
                options={'remove_timezone': remove_timezone})
        
        if remove_timezone:
                for i_var in df.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist():
                        df[i_var] = df[i_var].astype(str)
        
        df.to_excel(xlsWriter, sheetName, \
                index = writeIndex, float_format = float_format)

        # Get the xlsxwriter workbook and worksheet objects.
        workbook  = xlsWriter.book
        worksheet = xlsWriter.sheets[sheetName]
        
        if freezeTopRow:
                worksheet.freeze_panes(1, 0)

        # set the format for the cells
        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')

        # set the col format (fake Autolimit)
        colNames_lenght = df.columns.str.len().values
        for col in range(0, df.shape[1]):
                maxWidth = 1 + max(colNames_lenght[col], \
                        df.iloc[:, col].astype(str).str.len().max())
                worksheet.set_column(col, col, maxWidth, cell_format)

        xlsWriter.save()




def dataFrameToXLSMultisheet(df_dictionary, xlsFile, \
  writeIndex = False, float_format = '%.2f', freezeTopRow = True, \
  remove_timezone=True):
  """Write as Excel tabs the keys of a dictionary of DFs

  Args:
      df_dictionary ([type]): [description]
      xlsFile ([type]): [description]
      float_format (str, optional): [description]. Defaults to '%.2f'.
      freezeTopRow (bool, optional): [description]. Defaults to True.
  """
  with pd.ExcelWriter(xlsFile) as xlsWriter:
    # set the format for the cells
    workbook  = xlsWriter.book
    cell_format = workbook.add_format()
    cell_format.set_align('center')
    cell_format.set_align('vcenter')
    workbook.set_properties({
        'title':    'Autogenerated spreadsheet',
        'author':   'Carlos Aguilar'})
    for sheetName, df in df_dictionary.items():
      df.to_excel(xlsWriter, sheetName, \
          index = writeIndex, float_format = float_format)
      # Get the xlsxwriter workbook and worksheet objects.
      worksheet = xlsWriter.sheets[sheetName]
      worksheet.freeze_panes(1, 0)

      # set the col format (fake Autolimit)
      colNames_lenght = df.columns.str.len().values
      for col in range(0, df.shape[1]):
        maxWidth = 1 + max(colNames_lenght[col], \
                df.iloc[:, col].astype(str).str.len().max())
        worksheet.set_column(col, col, maxWidth, cell_format)
    xlsWriter.save()


def sort_files_in_folder(thisPath):
    '''
    Sort files in a folder by sorted/year/month/extension
    '''
    files = []
    for f in os.listdir(thisPath):
        filePath = os.path.join(thisPath, f)
        if os.path.isfile(filePath) and '.DS_Store' not in filePath:
            files.append(f)
            file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
            _, current_extension = os.path.splitext(filePath)
            folderName = os.path.join(thisPath, 'sorted', str(file_ct.year), \
                    file_ct.strftime("%B"), current_extension.replace('.',''))
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            newFilePath = os.path.join(folderName, f)
            move(filePath, newFilePath)
            print(f'Moving {newFilePath}...')


def get_files_in_folder(thisPath):
        '''
        Get the names of the files within a folder
        '''
        full_path_to_files = []
        files = []
        for f in os.listdir(thisPath):
                filePath = os.path.join(thisPath, f)
                if os.path.isfile(filePath) and '.DS_Store' not in filePath:
                        full_path_to_files.append(filePath)
                        files.append(f)
        return full_path_to_files, files


def get_datefile_name(content_name, file_ext='.pickle'):
  '''
    De facto naming for files
  '''
  return dt.datetime.today().strftime('%d_%m_%Y_%HH') + '_' + content_name + file_ext


def find_latest_datefile(path_to_folder, content_name, file_ext = '.pickle'):
  '''
    Find the latest file given a content name, a folder and a extension
  '''
  latest_filepath = []
  
  # Add the extension
  content_name_ext = content_name + file_ext

  latest_date = dt.datetime(1, 1, 12, 15, 0)
  files_in_folder = os.listdir(path_to_folder)
  content_file_list = [iFile for iFile in files_in_folder if content_name_ext in iFile]

  if content_file_list != []:
    regex_def = r'(\d+)_(\d+)_(\d+)_(\d+)H_(\S+)' + file_ext
    date_regex  = re.compile(regex_def)
    getPart = lambda regex_match, idx: int(regex_match.group(idx))
    for thisFile in content_file_list:
      matched = date_regex.match(thisFile)
      if matched != None:
        content_date = dt.datetime(getPart(matched, 3), \
          getPart(matched, 2), getPart(matched, 1), getPart(matched, 4))
        if latest_date < content_date:
          latest_date = content_date

    latest_file = latest_date.strftime('%d_%m_%Y_%HH') + '_' + content_name_ext
    latest_filepath = os.path.join(path_to_folder, latest_file)

  return latest_filepath

def find_datefile_name(path_to_folder, content_name, file_ext = '.pickle'):
  '''
    Find the files given a content name and a folder
  '''
  # force to follow the convention
  content_name += file_ext

  file_list,_ = get_latest_files_folder(path_to_folder, file_ext = file_ext)
  thisFile = [os.path.join(path_to_folder, iFile) for iFile in file_list if content_name in iFile]
  return thisFile


def get_latest_files_folder(thisFolder, file_ext = '.pickle'):
  '''
    Parse de facto naming files
  '''
  #regex_def = r'(\d+)_(\d+)_(\d+)_(\d+)H_(\w+).pickle'
  regex_def = r'(\d+)_(\d+)_(\d+)_(\d+)H_(\S+)' + file_ext
  date_regex  = re.compile(regex_def)
  latest_files = {}
  list_of_files = []
  list_of_content_name = []

  getPart = lambda regex_match, idx: int(regex_match.group(idx))

  for _, _, files in os.walk(thisFolder):
      for thisFile in files:
        matched = date_regex.match(thisFile)

        if matched != None:
          content_date = dt.datetime(getPart(matched, 3), \
            getPart(matched, 2), getPart(matched, 1), getPart(matched, 4))
          content_name = matched.group(5)

          current_content = latest_files.get(content_name, [])
          if current_content == []:
            latest_files.update({content_name: content_date})
          elif current_content < content_date:
            latest_files[content_name] = content_date


  # Convert back to strings
  for content_name, content_date in latest_files.items():
    list_of_files.append(content_date.strftime('%d_%m_%Y_%HH') + '_' + content_name + file_ext)
    list_of_content_name.append(content_name + file_ext)

  return list_of_files, list_of_content_name


def to_random_excel_file(df_to_save, writeIndex=False):
  '''
    Only use with the TF analysis data
  '''
  datetimeVars = df_to_save.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist()

  for i_var in datetimeVars:
    df_to_save[i_var] = df_to_save[i_var].astype(str)

  outputFolder = os.path.join(ur.get_dsci_root(), 'data', 'xls_to_delete')
  makeFolder(outputFolder)
  xls_filepath = os.path.join(outputFolder, str(uuid.uuid4()) + '.xlsx')
  dataFrameToXLSv2(df_to_save, xls_filepath, writeIndex = False)
  osOpenFile(xls_filepath)



def filter_filenames_by_date(bucketContents: 'list of filenames', \
        min_date: 'dt.datetime', regex_def=r'(\d+)/(\d+)/(\d+)', \
        max_date = None):
  '''
    From a list of files where the creation time is implicit and the date format is known,
    filter the ones that are older than the min_date
    
  '''
  filtered_files = []
  date_regex = re.compile(regex_def)
  getPart = lambda regex_match, idx: int(regex_match.group(idx))

  skipMaxDate = max_date == None

  for thisFile in bucketContents:
    matched = date_regex.match(thisFile)
    content_date = dt.datetime(getPart(matched, 1), getPart(matched, 2), getPart(matched, 3))
    min_date_ok = min_date < content_date
    if skipMaxDate and min_date_ok:
        filtered_files.append(thisFile)
    elif min_date_ok and (max_date > content_date):
        filtered_files.append(thisFile)

  return filtered_files


def sort_files_in_folder_by_extension(thisPath):
    '''
    Sort files in a folder by extension
    '''
    files = []
    for f in os.listdir(thisPath):
        filePath = os.path.join(thisPath, f)
        if os.path.isfile(filePath) and '.DS_Store' not in filePath:
            files.append(f)
            file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
            _, current_extension = os.path.splitext(filePath)
            folderName = os.path.join(thisPath, 'sorted', current_extension.replace('.',''))
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            newFilePath = os.path.join(folderName, f)
            move(filePath, newFilePath)
            print(f'Moving {newFilePath}...')



## Some helpers
def get_json_file_contents(iFile):
  '''
    Read JSON from either local or AWS S3
  '''
  folderName, fileName = os.path.split(iFile)
  if 's3://' in folderName:
    this_bucket = folderName.replace('s3://', '')
    json_contents = read_json_from_s3(fileName, this_bucket)
  else:
    json_contents = readJSONFile(iFile)

  return json_contents


# read pickles into DF
def read_pickles_as_DF(dataFolder):
  '''
    Read a collection of serialised dictionaries in a folder
    into a DataFrame
  '''
  glob_pattern = os.path.join(dataFolder, '*.pickle')
  folder_files = glob.glob(glob_pattern)
  data_list = []
  for this_file in folder_files:
    this_info = readPickleFile(this_file)
    if isinstance(this_info, list):
      data_list.extend(this_info)
    else:
      data_list.append(this_info)
  df = pd.DataFrame(data_list)
  return df