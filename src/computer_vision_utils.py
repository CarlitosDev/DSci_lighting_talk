'''

  Collection of utilities related to Computer Vision

  source ~/.bash_profile && pip3 install mediapipe

  Updates:
    04.11.2021 - Add method for object detection using Yolo V5 models.
                 Add PANOPTIC segmentation with Detectron2.

'''

import os
import cv2
import torch
import torchvision

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# initialise Detectron2
setup_logger()
# to run detectron on single images
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'

# from cvzone.SelfiSegmentationModule import SelfiSegmentation


# set the configuration for mediapipe 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


default_style = False

if default_style:
  tesselation_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
  face_mesh_contours_style = mp_drawing_styles.get_default_face_mesh_contours_style()

  face_mesh_iris_connections_style = mp_drawing_styles.get_default_face_mesh_iris_connections_style()

else:

  mesh_colour = mp_drawing_styles._GREEN
  mesh_thickness = 1
  tesselation_style = mp_drawing.DrawingSpec(color=mesh_colour, thickness=mesh_thickness)

  from mediapipe.python.solutions import face_mesh_connections
  _THICKNESS_CONTOURS = 4
  _FACEMESH_CONTOURS_CONNECTION_STYLE = {
      face_mesh_connections.FACEMESH_LIPS:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._RED, thickness=_THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_LEFT_EYE:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._PURPLE, thickness=_THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_LEFT_EYEBROW:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._YELLOW, thickness=_THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_RIGHT_EYE:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._PURPLE, thickness=_THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._YELLOW, thickness=_THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_FACE_OVAL:
          mp_drawing.DrawingSpec(color=mp_drawing_styles._RED, thickness=_THICKNESS_CONTOURS)
  }
  face_mesh_contours_style = {}
  for k, v in _FACEMESH_CONTOURS_CONNECTION_STYLE.items():
    for connection in k:
      face_mesh_contours_style[connection] = v

  face_mesh_iris_connections_style = {}
  left_spec = mp_drawing.DrawingSpec(color=mp_drawing_styles._PEACH, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
    face_mesh_iris_connections_style[connection] = left_spec
  right_spec = mp_drawing.DrawingSpec(color=mp_drawing_styles._PEACH, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
    face_mesh_iris_connections_style[connection] = right_spec


rawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)





def get_facemesh_from_image(this_image, max_faces=1, min_detection_confidence=0.5):
  """Generate a facemesh using medidapipe's functions

  Args:
      this_image ([type]): [description]
      max_faces (int, optional): [description]. Defaults to 1.
      min_detection_confidence (float, optional): [description]. Defaults to 0.5.

  Returns:
      [type]: [description]
  """  
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, \
    max_num_faces=max_faces,refine_landmarks=True, \
      min_detection_confidence = min_detection_confidence)

  results = face_mesh.process(cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB))
  return results


def generate_mesh_image(input_image, mesh_results):
  annotated_image = input_image.copy()
  if mesh_results.multi_face_landmarks:
    for face_landmarks in mesh_results.multi_face_landmarks:
        # facemesh
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=tesselation_style
            )
        # face contours (includes lips and eyebrows)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_contours_style)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_iris_connections_style)
  return annotated_image



def object_detection_yolov5(this_image, yolo_model='yolov5m', show_detection=False):
  """Detect objects using YoloV5 (Ultralytics)

  Args:
      this_image (np.array): [description]
      yolo_model (str, optional): Defaults to 'yolov5m'. Available models in https://pytorch.org/hub/ultralytics_yolov5/
      show_detection (bool, optional): [description]. Defaults to False.

  Returns:
      [type]: [description]
  """  
  model = torch.hub.load('ultralytics/yolov5', yolo_model, pretrained=True)
  results = model(this_image)
  results.print()
  
  if show_detection:
    results.show()

  df_results = results.pandas().xyxy[0]
  _cols = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence']

  detected_objects = {}
  for _, iRow in df_results.iterrows():
    detected_objects.setdefault(iRow['name'], []).append(iRow[_cols].to_dict())

  return detected_objects



def segmentation_detectron2(this_image, detectron2_model='COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'):
  """[summary]

  Args:
      this_image (np.array): [description]
      detectron2_model (str, optional): [description]. Defaults to 'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'.

  Returns:
      segments_info: description of the segmented objects
      out_image (np.array): segmented image
  """
  cfg.merge_from_file(model_zoo.get_config_file(detectron2_model))
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2_model)
  predictor = DefaultPredictor(cfg)

  outputs = predictor(this_image)
  panoptic_seg, segments_info = outputs["panoptic_seg"]
  v = Visualizer(this_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

  out_image = out.get_image()[:, :, ::-1]

  # 'translate' the segments info
  total_area = float(this_image.shape[0]*this_image.shape[1])

  metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
  for sinfo in segments_info:
    category_idx = sinfo["category_id"]
    sinfo['percentage_area'] = sinfo['area']/total_area

    if sinfo['isthing']:
      text = metadata.thing_classes[category_idx]
    else:
      text = metadata.stuff_classes[category_idx]
    sinfo['object'] = text

  return segments_info, out_image
  