## om namah shivay

## imports
from PIL import Image
import streamlit as st
import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,VideoTransformerBase, WebRtcMode
import streamlit_webrtc
import streamlit as st
import pyvista as pv
from pyvista import Plotter

import av

class VideoProcessor(VideoTransformerBase):
    def __init__(self,img_to_stack,output_img,mesh,main_dir):
        self.img_to_stack=img_to_stack
        self.output_img=output_img
        self.mesh=mesh
        self.main_dir=main_dir

    def recv(self, frame):
        frm=frame.to_ndarray(format="bgr24")
        image=cv2.flip(frm,1)
        image2=self.logic(image)
        return av.VideoFrame.from_ndarray(image2,format="bgr24")
        # return av.VideoFrame.from_ndarray(frm,format="bgr24")
    
    def logic(self,image):
        img_h,img_w,img_c=image.shape
        results=face_mesh.process(image)
        face_2d=[]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                idx=0
                ## indexes of 6 points used during training
                for idx in [1,199,263,33,291,61]:
                    lm= face_landmarks.landmark[idx]
                    x,y=int(lm.x*img_w),int(lm.y*img_h)
                    face_2d.append([x,y])
                    cv2.circle(image,(x,y),2,(0,255,0),thickness=-1)
             
                face_2d=np.array(face_2d,dtype=np.float64)
                data=face_2d.flatten()
                data=np.reshape(data,(1,12,))
                label=headpose_model.predict(data,verbose=0)[0]
                deg_x,deg_y,deg_z,t_x,t_y,t_z=label
                self.img_to_stack=self.rotate(deg_x,deg_y,deg_z) 
                # self.img_to_stack=np.zeros((480,640,3),dtype="uint8")
                self.output_img=np.vstack((image,self.img_to_stack)) 
                # print(self.img_to_stack.shape,self.output_img.shape,image.shape)

                self.output_img=cv2.resize(self.output_img,(640,480))
                self.output_img=self.output_img.astype("uint8")
        return self.output_img
        # return np.ones((480,640,3),dtype="uint8")*200

    def rotate(self,angle_x,angle_y,angle_z):
        pre_x,pre_y,pre_z=90,180,0
        mesh=self.mesh.rotate_x(angle_x)
        mesh=mesh.rotate_y(-angle_y)
        mesh=mesh.rotate_z(-angle_z)
        
        plotter = Plotter(off_screen=True)
        plotter.add_mesh(mesh, show_edges=True)
        plotter.camera_position = 'xy'

        # # Render the plot to an image and display it
        screenshot_path = 'screenshot.png'
        plotter.show(screenshot=screenshot_path)
        plotter.close()
        image_array=cv2.imread(os.path.join(self.main_dir,'screenshot.png'))
        image_array=cv2.resize(image_array,(640,480))
      
        return image_array



def main(output_img,img_to_stack,mesh,main_dir):
    st.title("Real-time Video Stream using WebRTC")
   
    webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VideoProcessor(img_to_stack,output_img,mesh,main_dir),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    
)


## function for rotating iron man mask model through specified angles
# the overall idea is to get back model to its original position by using identity
# then give absolute rotations in all three directions 


  

if __name__=="__main__":
    main_dir=os.path.dirname(__file__)
    img_to_stack=np.zeros((480,640,3),dtype="uint8")
    output_img=np.zeros((480,640,3),dtype="uint8")
    counter=0
    pre_x,pre_y,pre_z=90,180,0
    
    file_path = os.path.join(main_dir,"ImageToStl.com_mark_85.stl")
    mesh = pv.read(file_path)
    
    headpose_model_json_path=os.path.join(main_dir,"models","head_pose_model.json")
    headpose_model_weights_path=os.path.join(main_dir,"models","head_pose_model_weights.h5")

    ##loading models
    with open(headpose_model_json_path,"r") as file:
      headpose_model=file.read()
      headpose_model=model_from_json(headpose_model)
      headpose_model.load_weights(headpose_model_weights_path)

    mp_face_mesh=mp.solutions.face_mesh
    face_mesh=mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    mp_drawing=mp.solutions.drawing_utils
    drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

    main(output_img,img_to_stack,mesh,main_dir)


    
#     # Rotate the mesh
    
#     # Create a PyVista plotter



# import streamlit as st
# import numpy as np
# import pyvista as pv
# from pyvista import Plotter
# import cv2
# st.title("3D Mesh Viewer")

# # Fixed file path
# file_path = r"C:\Users\07032\github_projects\head-pose-estimator-and-renderer\ImageToStl.com_mark_85.stl"

# # Inputs for rotation angles
# x_angle = st.number_input("Enter the rotation angle around the X-axis (in degrees)", value=0, step=1)
# y_angle = st.number_input("Enter the rotation angle around the Y-axis (in degrees)", value=0, step=1)
# z_angle = st.number_input("Enter the rotation angle around the Z-axis (in degrees)", value=0, step=1)

# mesh = pv.read(file_path)
 
# # Rotate the mesh
# mesh.rotate_x(x_angle, inplace=True)
# mesh.rotate_y(y_angle, inplace=True)
# mesh.rotate_z(z_angle, inplace=True)

# # Create a PyVista plotter
# plotter = Plotter(off_screen=True)
# plotter.add_mesh(mesh, show_edges=True)
# plotter.camera_position = 'xy'

# # Render the plot to a NumPy array
# plotter.show(interactive=False, screenshot='screenshot.png')
# plotter.close()

# # Read the rendered image as a NumPy array
# image_array = cv2.imread('screenshot.png')

# # Display the image using Streamlit
# st.image(image_array, caption='3D Mesh Visualization', use_column_width=True)
# # Optionally, you can also display some mesh information

# # 60,0,135
