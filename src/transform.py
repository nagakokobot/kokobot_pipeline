# contains the code to transform the grasp coordinates to original image resolution
# also contains the class to convert the camera coordinates to robot coordinates
import numpy as np

from ggcnn.utils.dataset_processing.grasp import GraspRectangle, Grasp





def transform_grasps(obj_loc:dict, grasps:list)-> Grasp:
  '''
  grasps: A list of grasps per object crop
  returns: A list of new grasps in the image resolution
  '''

  image_grasps = []

  for i , grasp in enumerate(grasps):
    rect_points = grasp.as_gr.points
    rect_points[:,0] = rect_points[:,0] * obj_loc['aspect_y'] 
    rect_points[:,1] = rect_points[:,1] * obj_loc['aspect_x'] 
    rect_points[:,0] = rect_points[:,0] + obj_loc['ymin']
    rect_points[:,1] = rect_points[:,1] + obj_loc['xmin']
    gr_rec = GraspRectangle(rect_points)
    image_gr = gr_rec.as_grasp
    image_grasps.append(image_gr)


  return image_grasps

class Robot_coordinates:
  '''
  A class to load the existing transformation matrix and get the new robot coordinates for the grasps 
  Note: The class load a pre existing transformation matrix from transformation/transformation_matrix.npy
        The transformation matrix is obtained from transformation_matrix.py. Perform the transformation again if the camera or robot has been moved
  '''
  def __init__(self, matrix_path:str = None):
    
    self.m_path = matrix_path
    if not self.m_path:
      self.m_path = 'transformation/transformation_matrix.npy'
    self.M = np.load(self.m_path)

  def transform_camera_to_robot(self, cam_vals:list):
    '''
    cam_vals: is X_c,Y_c,Z_c values in camera coordinates(can be obtained from point cloud)
    returns: the new robot coordinate X_r,Y_r,Z_r
    '''

    cam = np.array([cam_vals[0],cam_vals[1],cam_vals[2]]).reshape((3,-1))
    rob = np.matmul(self.M,np.concatenate([cam,np.ones((1,cam.shape[1]))],axis = 0))
    X_r, Y_r, Z_r = rob.flatten()
    return X_r, Y_r, Z_r



if __name__ == '__main__':
  pass