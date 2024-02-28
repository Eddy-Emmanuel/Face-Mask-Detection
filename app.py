import cv2
import imutils
import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model
from keras.applications import mobilenet_v2
from keras.preprocessing.image import img_to_array
from imutils.video import VideoStream, FileVideoStream

# classifier = load_model("classifier\mobilenetv2.h5") # Step 1
# face_detector = cv2.dnn.readNet(model=r"face_detector_model\deploy.prototxt",
#                                 config=r"face_detector_model\res10_300x300_ssd_iter_140000.caffemodel") # Step 2

# def Run_Live_Test(source):
#     loop = True
#     if source != 0:
#         cam = VideoStream(src=source).start() # Step 3
#     else:
#         cam = FileVideoStream(path=source).start()

#     if cam is not None:
#         while loop:
#             frame = imutils.resize(cam.read(), width=1000, height=1000) # Step 4
#             h, w, _ = frame.shape # Step 5
#             blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123)) # # Step 6
#             face_detector.setInput(blob=blob) # Step 7
#             detected_faces = face_detector.forward() # Step 8 -> Detect Faces

#             preprocessed_face = []
#             face_loc = []

#             for i in range(detected_faces.shape[2]): # Step 9
#                 confidence = detected_faces[0, 0, i, 2] # Step 10

#                 if confidence > 0.5:
#                     x1, y1, x2, y2 = detected_faces[0, 0, i, 3:7] # Step 11
#                     # Step 12
#                     startX, endX = int(x1*w), int(x2*w)
#                     startY, endY = int(y1*h), int(y2*h)

#                     startX, endX = max(0, startX), min(w-1, endX)
#                     startY, endY = max(0, startY), min(h-1, endY)

#                     # Step 13
#                     face = frame[startY:endY, startX:endX]
#                     rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                     prep_face = mobilenet_v2.preprocess_input(cv2.resize(rgb_face, (224, 224)))
#                     print(prep_face.shape)
#                     preprocessed_face.append(prep_face)
#                     face_loc.append([startX, startY, endX, endY])

#                 else:
#                     continue
            

#             try:
#                 if len(face) != 0:
#                     predictions = classifier.predict(np.array(preprocessed_face))

#                 for (PREDICTIONS, PRED_LOCATIONS) in zip(predictions, face_loc):
#                     with_facemask, without_facemask = PREDICTIONS

#                     color = (0, 0, 255) if without_facemask > with_facemask else (0, 255, 0)
#                     predicted_class = "Facemask on" if with_facemask > without_facemask else "Facemask off"
#                     MODEL_CONFIDENCE = with_facemask if with_facemask > without_facemask else without_facemask

#                     cv2.rectangle(frame, (PRED_LOCATIONS[0], PRED_LOCATIONS[1]), (PRED_LOCATIONS[2], PRED_LOCATIONS[3]), color, 2)
#                     cv2.putText(frame, f"{predicted_class}: {MODEL_CONFIDENCE:.2f}", (PRED_LOCATIONS[0], PRED_LOCATIONS[1]-40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
                
#                 cv2.imshow("Camera", frame)

#                 if cv2.waitKey(1) == ord("q"):
#                     loop = False
#             except:
#                 continue

#         cam.stop()
#         cv2.destroyAllWindows()


# st.markdown("<h1><center>Face Mask Detection 😎<center/><h1/>", unsafe_allow_html=True)

# options = st.sidebar.selectbox(label="Select Operation", options=["None", "Live Test", "HeadShot Test", "Full Body Pics Test"])

# if options != "None":
#     if options == "Live Test":
#         Run_Live_Test(source=0)

#     if options == "HeadShot Test":
#         picture = st.file_uploader("Upload Picture", ["png ", "jpg"], accept_multiple_files=False)
#         if picture is not None:
#             image = Image.open(picture)
#             converted_image_to_array = cv2.resize(cv2.cvtColor(img_to_array(image), cv2.COLOR_RGBA2RGB), (224, 224))
#             preprocessed_image = mobilenet_v2.preprocess_input(converted_image_to_array)
#             predictions = classifier.predict(np.expand_dims(preprocessed_image, axis=0))
#             st.markdown(f"Prediction: {'FaceMask On' if predictions[0][0] > predictions[0][1] else 'FaceMask Off'}")
#             st.image(image=image, caption="Model Ouput")
    
#     if options == "Full Body Pics Test":
#         picture = st.file_uploader("Upload Picture", ["png ", "jpg"], accept_multiple_files=False)
#         if picture is not None:
#             image = cv2.cvtColor(cv2.imdecode(np.frombuffer(picture.read(), np.uint8), 1), cv2.COLOR_BGR2RGB)
#             h, w, _ = image.shape
#             Blob = cv2.dnn.blobFromImage(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), scalefactor=1.3, size=(300, 300), mean=(104, 177, 123))
#             face_detector.setInput(blob=Blob)
#             detected_faces = face_detector.forward()
            
            
#             for i in range(detected_faces.shape[2]):
#                 confidence = detected_faces[0, 0, i, 2]
#                 if confidence > 0.5:
#                     x1, y1, x2, y2 = detected_faces[0, 0, i, 3:7]
#                     startx, endx = int(x1*w), int(x2*w)
#                     starty, endy = int(y1*h), int(y2*h)

#                     startx, endx = max(0, startx), min(w-1, endx)
#                     starty, endy = max(0, starty), min(h-1, endy)

#                     frame = image[starty:endy, startx:endx] 
#                     rgb_face = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
#                     prep_face = mobilenet_v2.preprocess_input(cv2.resize(rgb_face, (224, 224)))
                        
#                     with_facemask, without_facemask = classifier.predict(np.expand_dims(prep_face, axis=0))[0]
#                     color = (0, 0, 255) if without_facemask > with_facemask else (0, 255, 0)
#                     predicted_class = "Facemask on" if with_facemask > without_facemask else "Facemask off"

#                     cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)
#                     cv2.putText(image, f"{predicted_class}", (startx, starty-40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            
#             st.image(image=image, caption="Output")




