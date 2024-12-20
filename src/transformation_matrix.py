from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from helpers import create_folder
from camera_parameters import get_camera_serial_number, get_init_camera_paramaters, get_runtime_camera_parameters

def get_red_point_pixels(image, roi):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower_blue = np.array([100, 150, 70])  # Lower boundary of blue (Hue: 100-120, Saturation: High, Value: Medium-High)
  upper_blue = np.array([130, 255, 255])  # Upper boundary of blue
  blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
  contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  f_contours = []
  for contour in contours:
      M = cv2.moments(contour)
      if M["m00"] != 0:
          cx = int(M["m10"] / M["m00"])  # x coordinate
          cy = int(M["m01"] / M["m00"])  # y coordinate
          #print(f"Red dot detected at: ({cx}, {cy})")
          if roi[0][0]<cx<roi[1][0]:
             if roi[0][1]<cy<roi[1][1]:
                f_contours.append([cx,cy])
                #print(f'point found in the roi')
          else:
             pass
             #print('skipped..')
 
  return f_contours
def show_points(rgb, red_dots):
  plt.imshow(rgb)
  for i,dot in enumerate(red_dots):
    print(f'point {i} at {dot[0], dot[1]}')
    plt.plot(dot[0], dot[1], 'x')
    plt.text(dot[0], dot[1], i)
  plt.show()
  pass

def calculate_transformation_kabsch(P,Q):
    # Rotationsmatrix und Translationsvektor zwischen zwei Punktewolken nach Kabsch-Algorithmus berechnen
    P = P.T
    Q = Q.T
    k = P.shape[0]
    center_P = P.mean(axis=0)
    center_Q = Q.mean(axis=0)
    P_c = P - center_P
    Q_c = Q - center_Q
    H = np.matmul(P_c.T,Q_c)
    U, Sigma, V_T = np.linalg.svd(H)
    mat1 = np.array([[1,0,0],[0,1,0],[0,0,np.sign(np.linalg.det(np.matmul(V_T.T,U.T)))]])
    R = np.matmul(np.matmul(V_T.T,mat1),U.T)
    v_t = center_Q - np.matmul(center_P,R.T)
    rmsd = np.sqrt(((np.matmul(P_c,R.T)-Q_c)**2).sum()/k)
    return R,v_t,rmsd

def transform_new_point_mat(new_point, transformation_matrix):
   return np.matmul(transformation_matrix,np.concatenate([new_point,np.ones((1,new_point.shape[1]))],axis = 0))

def transfor_new_point_np(new_point, transformation_matrix):
   return np.matmul(transformation_matrix,new_point)

if __name__ == '__main__': 

  args_dict = {'camera_resolution': 'HD1080', 'coordinate_units': 'mm'}

  current_time = datetime.now()
  folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + '_transformation'
  _, s_path = create_folder(sub_name=folder_name, parent_name='project_aux')

  cameras,serial_number = get_camera_serial_number()

  init_params = get_init_camera_paramaters(args = args_dict, serial_number= serial_number, save_path = s_path)
  runtime_params = get_runtime_camera_parameters(args = args_dict, save_path=s_path)
  cam = Camera(initparameters= init_params, runtimeparameters=runtime_params, save_path= s_path, show_workspace= False)
  rgb, depth_cam, rgb_path, depth_path = cam.rgb, cam.depth.copy(), cam.rgb_path, cam.depth_path

  #roi = [top left corner, bottomright corner] in the image
  roi = [[828,267],[1633,941]]
  red_dots = get_red_point_pixels(rgb, roi)
  show_points(rgb, red_dots)
  cam_3d_coords = []
  for point in red_dots:
     xc,yc,zc =cam.get_xyz(point[0], point[1])
     cam_3d_coords.append([xc,yc,zc])

  cam.close_cam()
  # robot coordinates measured from robot to the detected points, [[X_r, Y_r, Z_r]]
  #test_robot_coords = [[0.01,0.6,0],[0,0.33,0],[-0.34,0.6,0],[-0.33,0.28,0]] #example coordinates in robot reference frame im meters
  test_robot_coords = [[10,600,0],[0,330,0],[-340,600,0],[-330,280,0]] #example coordinates in robot reference frame in mm

  #make arrays for the 3d points
  campoints_mat = np.array(cam_3d_coords, dtype=np.float32)
  robotpoints_mat = np.array(test_robot_coords, dtype=np.float32)
  print(campoints_mat, robotpoints_mat)

  
  rm, tv, rmsd_value = calculate_transformation_kabsch(campoints_mat.T,robotpoints_mat.T)
  M = np.concatenate([rm,tv.reshape(-1,1)],axis = 1)
  print(f'deviation: {rmsd_value}')
  print(f'transformation matrix: {M.shape}')

  #when new point comes from the camera:
  x,y,z = 0,0,1.2
  new_cam_point = np.array([0,0,0,1]) #test the new points coordinates in robot frame
  new_cam_point_mat = np.array([x,y,z]).reshape((3,-1))
  new_robot_point = transform_new_point_mat(new_cam_point_mat, M)
  print(type(new_robot_point))


  #save the transformation matrix as a binary file
  _, s_path = create_folder(sub_name='transformation')
  np.save(s_path+'/'+'transformation_matrix.npy', M)
  

  