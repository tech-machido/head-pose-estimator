## om namah shivay

## imports
from vtk import vtkObject ,vtkWindowToImageFilter,vtkGLTFImporter,vtkRenderer,vtkRenderWindow,vtkTransform
from PIL import Image
import streamlit as st
import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,VideoTransformerBase, WebRtcMode
import streamlit_webrtc
import av
    

def rotate(angle_x,angle_y,angle_z):
        global transform,imported_actors,actor,render_window
        pre_x,pre_y,pre_z=90,180,0
        transform.Identity()
        transform.RotateX(-angle_x-pre_x)
        transform.RotateY(-angle_z-pre_z)
        transform.RotateZ(-angle_y-pre_y)
        imported_actors.InitTraversal()
        actor = imported_actors.GetNextActor()
        while actor:
            actor.SetUserTransform(transform)
            actor = imported_actors.GetNextActor()
        render_window.Render()
        window_to_image_filter = vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()
        # # Convert vtkImageData to numpy array
        vtk_image = window_to_image_filter.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        vtk_array.SetNumberOfComponents(3)  # Ensure RGB
        np_image = np.array(vtk_array)
        np_image=np.reshape(np_image,(480,640,3))
        # # # Convert RGB to BGR
        np_image = np_image[:, :, ::-1]
        return np_image

if __name__=="__main__":
    main_dir=os.path.dirname(__file__)
    pre_x,pre_y,pre_z=90,180,0
    importer = vtkGLTFImporter()
    importer.SetFileName(os.path.join(main_dir,"mark_85.glb"))
    importer.Read()

    # # Create a renderer
    renderer = vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Add the imported actors to the renderer
    imported_actors = importer.GetRenderer().GetActors()
    imported_actors.InitTraversal()
    actor = imported_actors.GetNextActor()
    while actor:
        renderer.AddActor(actor)
        actor = imported_actors.GetNextActor()

    # Create a render window
    render_window = vtkRenderWindow()
    render_window.SetSize(640, 480)
    render_window.OffScreenRenderingOn()
    render_window.AddRenderer(renderer)
    transform = vtkTransform()
    vtkObject.GlobalWarningDisplayOff() 
    
    degrees_x = st.sidebar.slider('X Angle', min_value=-180.0, max_value=180.0, value=0.0)
    degrees_y = st.sidebar.slider('Y Angle', min_value=-180.0, max_value=180.0, value=0.0)
    degrees_z = st.sidebar.slider('Z Angle', min_value=-180.0, max_value=180.0, value=0.0)

    st.image(rotate(degrees_x,degrees_y,degrees_z))
