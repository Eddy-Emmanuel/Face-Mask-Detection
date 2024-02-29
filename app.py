import cv2
import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model
from keras.applications import mobilenet_v2
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer,  VideoTransformerBase, RTCConfiguration, WebRtcMode

classifier = load_model("mobilenetv2.h5") # Step 1
face_detector = cv2.dnn.readNet(model="deploy.prototxt",
                                config="res10_300x300_ssd_iter_140000.caffemodel") # Step 2

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoDisplay(VideoTransformerBase):
      def CALLBACK(self, cam):
          frame = cam.to_ndarray(format="bgr24")
      
          h, w, _ = frame.shape
          blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123))
          face_detector.setInput(blob=blob)
          detected_faces = face_detector.forward()
      
          for i in range(detected_faces.shape[2]):
              confidence = detected_faces[0, 0, i, 2]
      
              if confidence > 0.5:
                  x1, y1, x2, y2 = detected_faces[0, 0, i, 3:7]
                  startX, endX = max(0, int(x1 * w)), min(w - 1, int(x2 * w))
                  startY, endY = max(0, int(y1 * h)), min(h - 1, int(y2 * h))
      
                  face = frame[startY:endY, startX:endX]
                  rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                  prep_face = mobilenet_v2.preprocess_input(cv2.resize(rgb_face, (224, 224)))
                      
                  with_facemask, without_facemask = classifier.predict(np.expand_dims(prep_face, axis=0))[0]
                  color = (0, 0, 255) if without_facemask > with_facemask else (0, 255, 0)
                  predicted_class = "Facemask on" if with_facemask > without_facemask else "Facemask off"
      
                  cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                  cv2.putText(frame, f"{predicted_class}: {max(with_facemask, without_facemask):.2f}", 
                              (startX, startY - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
      
          return frame



st.markdown("<h1><center>Face Mask Detection 😎<center/><h1/>", unsafe_allow_html=True)

options = st.sidebar.selectbox(label="Select Operation", options=["None", "Live Test", "HeadShot Test", "Full Body Pics Test"])

def main():
  if options != "None":
      if options == "Live Test":
          webrtc_streamer(key="sample", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=VideoDisplay)
    
  
      if options == "HeadShot Test":
          picture = st.file_uploader("Upload Picture", ["png ", "jpg"], accept_multiple_files=False)
          if picture is not None:
              image = Image.open(picture)
              converted_image_to_array = cv2.resize(cv2.cvtColor(img_to_array(image), cv2.COLOR_RGBA2RGB), (224, 224))
              preprocessed_image = mobilenet_v2.preprocess_input(converted_image_to_array)
              predictions = classifier.predict(np.expand_dims(preprocessed_image, axis=0))
              st.markdown(f"Prediction: {'FaceMask On' if predictions[0][0] > predictions[0][1] else 'FaceMask Off'}")
              st.image(image=image, caption="Model Ouput")
      
      if options == "Full Body Pics Test":
          picture = st.file_uploader("Upload Picture", ["png ", "jpg"], accept_multiple_files=False)
          if picture is not None:
              image = cv2.cvtColor(cv2.imdecode(np.frombuffer(picture.read(), np.uint8), 1), cv2.COLOR_BGR2RGB)
              h, w, _ = image.shape
              Blob = cv2.dnn.blobFromImage(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), scalefactor=1.3, size=(300, 300), mean=(104, 177, 123))
              face_detector.setInput(blob=Blob)
              detected_faces = face_detector.forward()
              
              
              for i in range(detected_faces.shape[2]):
                  confidence = detected_faces[0, 0, i, 2]
                  if confidence > 0.5:
                      x1, y1, x2, y2 = detected_faces[0, 0, i, 3:7]
                      startx, endx = int(x1*w), int(x2*w)
                      starty, endy = int(y1*h), int(y2*h)
  
                      startx, endx = max(0, startx), min(w-1, endx)
                      starty, endy = max(0, starty), min(h-1, endy)
  
                      frame = image[starty:endy, startx:endx] 
                      rgb_face = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                      prep_face = mobilenet_v2.preprocess_input(cv2.resize(rgb_face, (224, 224)))
                          
                      with_facemask, without_facemask = classifier.predict(np.expand_dims(prep_face, axis=0))[0]
                      color = (0, 0, 255) if without_facemask > with_facemask else (0, 255, 0)
                      predicted_class = "Facemask on" if with_facemask > without_facemask else "Facemask off"
  
                      cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)
                      cv2.putText(image, f"{predicted_class}", (startx, starty-40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
              
              st.image(image=image, caption="Output")

if __name__ == "__main__":
  main()



