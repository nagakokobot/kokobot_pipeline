# contains the code to transform the grasp coordinates to original image resolution

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


if __name__ == '__main__':
  pass