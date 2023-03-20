# Core Pkgs
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import face_recognition
import dlib
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import tensorflow
from cvzone.FaceDetectionModule import FaceDetector
from PIL import Image, ImageDraw


def detect_faces(our_image):
	image_to_detect =  np.array(our_image.convert('RGB'))
	image_to_detect_gray = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
	face_detection_classifier = dlib.get_frontal_face_detector()
	all_face_locations = face_detection_classifier(image_to_detect,1)

	for index,current_face_location in enumerate(all_face_locations):
		#start and end co-ordinates
		left_x, left_y, right_x, right_y = current_face_location.left(),current_face_location.top(),current_face_location.right(),current_face_location.bottom()

		#slicing the current face from main image
		current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
		
		#showing the current face with dynamic title
		#cv2.imshow("Face no "+str(index+1),current_face_image)
		
		#draw bounding box around the faces
		cv2.rectangle(image_to_detect,(left_x,left_y),(right_x,right_y),(0,255,0),2)

	return image_to_detect,all_face_locations

def detect_emotion(our_image):
	image_to_detect =  np.array(our_image.convert('RGB'))

	#load the model and load the weights
	face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
	face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
	#declare the emotions label
	emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

	#detect all faces in the image
	#arguments are image,no_of_times_to_upsample, model
	all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')

	for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four position values of current face
		top_pos,right_pos,bottom_pos,left_pos = current_face_location
		#printing the location of current face
		# print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
		#slicing the current face from main image
		current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
		#draw rectangle around the face detected
		cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
		
		#preprocess input, convert it to an image like as the data in dataset
		#convert to grayscale
		current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
		#resize to 48x48 px size
		current_face_image = cv2.resize(current_face_image, (48, 48))
		#convert the PIL image into a 3d numpy array
		img_pixels = image.img_to_array(current_face_image)
		#expand the shape of an array into single row multiple columns
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		#pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
		img_pixels /= 255 
		
		#do prodiction using model, get the prediction values for all 7 expressions
		exp_predictions = face_exp_model.predict(img_pixels) 
		#find max indexed prediction value (0 till 7)
		max_index = np.argmax(exp_predictions[0])
		#get corresponding lable from emotions_label
		emotion_label = emotions_label[max_index]
		#display the name as text in the image
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(image_to_detect, emotion_label, (left_pos,bottom_pos), font, 1.5, (255,255,255),1)

	return image_to_detect

def detect_age(our_image):
	image_to_detect =  np.array(our_image.convert('RGB'))
	all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')

	for index,current_face_location in enumerate(all_face_locations):
		top_pos,right_pos,bottom_pos,left_pos = current_face_location
		#printing the location of current face
		print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
		#slicing the current face from main image
		current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
		
		#The ‘AGE_GENDER_MODEL_MEAN_VALUES’ calculated by using the numpy. mean()        
		AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
		#create blob of current flace slice
		#params image, scale, (size), (mean),RBSwap)
		current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
		
		# Predicting Gender
		#declaring the labels
		gender_label_list = ['Male', 'Female']
		#declaring the file paths
		gender_protext = "dataset/gender_deploy.prototxt"
		gender_caffemodel = "dataset/gender_net.caffemodel"
		#creating the model
		gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
		#giving input to the model
		gender_cov_net.setInput(current_face_image_blob)
		#get the predictions from the model
		gender_predictions = gender_cov_net.forward()
		#find the max value of predictions index
		#pass index to label array and get the label text
		gender = gender_label_list[gender_predictions[0].argmax()]
		
		# Predicting Age
		#declaring the labels
		age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
		#declaring the file paths
		age_protext = "dataset/age_deploy.prototxt"
		age_caffemodel = "dataset/age_net.caffemodel"
		#creating the model
		age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
		#giving input to the model
		age_cov_net.setInput(current_face_image_blob)
		#get the predictions from the model
		age_predictions = age_cov_net.forward()
		#find the max value of predictions index
		#pass index to label array and get the label text
		age = age_label_list[age_predictions[0].argmax()]
			
		#draw rectangle around the face detected
		cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(140,0,200),3)
			
		#display the name as text in the image
		font = cv2.FONT_HERSHEY_PLAIN
		cv2.putText(image_to_detect, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 2 , (0,255,0),1)

	return image_to_detect

def draw_landmark(our_image):
	face_image =  np.array(our_image.convert('RGB'))
	#get the face landmarks list
	face_landmarks_list =  face_recognition.face_landmarks(face_image)
	#convert the numpy array image into pil image object
	pil_image = Image.fromarray(face_image)
	#convert the pil image to draw object
	d = ImageDraw.Draw(pil_image)

	#loop through every face
	index=0
	while index < len(face_landmarks_list):
		# loop through face landmarks
		for face_landmarks in face_landmarks_list:
		
			#join each face landmark points
			d.line(face_landmarks['chin'],fill=(0,255,0), width=3)
			d.line(face_landmarks['left_eyebrow'],fill=(0,255,0), width=3)
			d.line(face_landmarks['right_eyebrow'],fill=(0,255,0), width=3)
			d.line(face_landmarks['nose_bridge'],fill=(0,255,0), width=3)
			d.line(face_landmarks['nose_tip'],fill=(0,255,0), width=3)
			d.line(face_landmarks['left_eye'],fill=(0,255,0), width=3)
			d.line(face_landmarks['right_eye'],fill=(0,255,0), width=3)
			d.line(face_landmarks['top_lip'],fill=(0,255,0), width=3)
			d.line(face_landmarks['bottom_lip'],fill=(0,255,0), width=3)

		index +=1

	#show the final image    
	return pil_image

def main():
	"""Face Detection App"""

	st.title("Face Detection App")
	st.text("Build with Streamlit and OpenCV")

	activities = ["Image Detection","Webcam Detection","Live Detection","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Image Detection':
		
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			# st.write(type(our_image))
			st.image(our_image)

		enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
		
		if enhance_type == 'Gray-Scale':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# st.write(new_img)
			st.image(gray)
			
		elif enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.5,3.5)
			enhancer = ImageEnhance.Contrast(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.5,3.5)
			enhancer = ImageEnhance.Brightness(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Blurring':
			new_img = np.array(our_image.convert('RGB'))
			blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
			img = cv2.cvtColor(new_img,1)
			blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
			st.image(blur_img)
    
		# elif enhance_type == 'Original':
		# 	st.image(our_image,width=300)
		# else:
		# 	st.image(our_image,width=300)



		# Face Detection
		task = ["Faces","Emotion","Age & gender prediction","Face landmark"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):

			if feature_choice == 'Faces':
				result_img,result_faces = detect_faces(our_image)
				st.image(result_img)
				st.success("Found {} faces".format(len(result_faces)))


			elif feature_choice == 'Emotion':
				result_img = detect_emotion(our_image)
				st.image(result_img)

			elif feature_choice == 'Age & gender prediction':
				result_img = detect_age(our_image)
				st.image(result_img)
			elif feature_choice == 'Face landmark':
				result_img = draw_landmark(our_image)
				st.image(result_img)


	elif choice == 'Webcam Detection':

		task = ["Faces","Emotion"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		image_w = st.camera_input("Take a picture")
		# image_web = Image.open(image_w)


		if st.button("Process"):
			image_web = Image.open(image_w)
			if feature_choice == 'Faces':
				result_img,result_faces = detect_faces(image_web)
				st.image(result_img)
				st.success("Found {} faces".format(len(result_faces)))


			elif feature_choice == 'Emotion':
				result_img = detect_emotion(image_web)
				st.image(result_img)
	
	
	elif choice == 'Live Detection':
		task = ["Faces","Emotion","Landmark","Recognition"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		cap = cv2.VideoCapture(0)
		frame_window = st.image([])

		if st.button("Run Webcam"):

			if feature_choice == 'Faces':
				while True:
        		    #get the current frame from the video stream as an image
					ret,current_frame = cap.read()
					
					#resize the current frame to 1/4 size to proces faster
					current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
					
					#detect all faces in the image
					#arguments are image,no_of_times_to_upsample, model
					all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
					
					#looping through the face locations
					for index,current_face_location in enumerate(all_face_locations):
						
						#splitting the tuple to get the four position values of current face
						top_pos,right_pos,bottom_pos,left_pos = current_face_location
						
						#change the position maginitude to fit the actual size video frame
						top_pos = top_pos*4
						right_pos = right_pos*4
						bottom_pos = bottom_pos*4
						left_pos = left_pos*4
						#draw rectangle around the face detected
						cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,255,0),2)
						
					#showing the current face with rectangle drawn
					current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
					frame_window.image(current_frame)


			elif feature_choice == 'Emotion':
				#load the model and load the weights
				face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r",encoding="utf-8").read())
				face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
				#declare the emotions label
				emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
				#initialize the array variable to hold all face locations in the frame
				all_face_locations = []
				while True:
    				#get the current frame from the video stream as an image
					ret,current_frame = cap.read()
					#resize the current frame to 1/4 size to proces faster
					current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
					#detect all faces in the image
					#arguments are image,no_of_times_to_upsample, model
					all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
					
					#looping through the face locations
					for index,current_face_location in enumerate(all_face_locations):
						#splitting the tuple to get the four position values of current face
						top_pos,right_pos,bottom_pos,left_pos = current_face_location
						#change the position maginitude to fit the actual size video frame
						top_pos = top_pos*4
						right_pos = right_pos*4
						bottom_pos = bottom_pos*4
						left_pos = left_pos*4

						#Extract the face from the frame, blur it, paste it back to the frame
						#slicing the current face from main image
						current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
						
						#draw rectangle around the face detected
						cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
						
						#preprocess input, convert it to an image like as the data in dataset
						#convert to grayscale
						current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
						#resize to 48x48 px size
						current_face_image = cv2.resize(current_face_image, (48, 48))
						#convert the PIL image into a 3d numpy array
						img_pixels = image.img_to_array(current_face_image)
						#expand the shape of an array into single row multiple columns
						img_pixels = np.expand_dims(img_pixels, axis = 0)
						#pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
						img_pixels /= 255 
						
						#do prodiction using model, get the prediction values for all 7 expressions
						exp_predictions = face_exp_model.predict(img_pixels) 
						#find max indexed prediction value (0 till 7)
						max_index = np.argmax(exp_predictions[0])
						#get corresponding lable from emotions_label
						emotion_label = emotions_label[max_index]
						
						#display the name as text in the image
						font = cv2.FONT_HERSHEY_DUPLEX
						cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
						
					#showing the current face with rectangle drawn
					current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
					frame_window.image(current_frame)

			elif feature_choice == 'Landmark':
				#loop through every frame in the video
				while True:
					#get the current frame from the video stream as an image
					ret,current_frame = cap.read()

					#get the face landmarks list
					face_landmarks_list =  face_recognition.face_landmarks(current_frame)
					
					#print the face landmarks list
					#print(len(face_landmarks_list))
					
					#convert the numpy array image into pil image object
					pil_image = Image.fromarray(current_frame)
					#convert the pil image to draw object
					d = ImageDraw.Draw(pil_image)
					
					#loop through every face
					index=0
					while index < len(face_landmarks_list):
						# loop through face landmarks
						for face_landmarks in face_landmarks_list:
						
							
							#join each face landmark points
							d.line(face_landmarks['chin'],fill=(0,255,0), width=3)
							d.line(face_landmarks['left_eyebrow'],fill=(0,255,0), width=3)
							d.line(face_landmarks['right_eyebrow'],fill=(0,255,0), width=3)
							d.line(face_landmarks['nose_bridge'],fill=(0,255,0), width=3)
							d.line(face_landmarks['nose_tip'],fill=(0,255,0), width=3)
							d.line(face_landmarks['left_eye'],fill=(0,255,0), width=3)
							d.line(face_landmarks['right_eye'],fill=(0,255,0), width=3)
							d.line(face_landmarks['top_lip'],fill=(0,255,0), width=3)
							d.line(face_landmarks['bottom_lip'],fill=(0,255,0), width=3)
					
						index +=1
					
					#convert PIL image to RGB to show in opencv window    
					rgb_image = pil_image.convert('RGB') 
					rgb_open_cv_image = np.array(pil_image)
					
					# Convert RGB to BGR 
					bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
					bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()

					current_frame = cv2.cvtColor(bgr_open_cv_image, cv2.COLOR_BGR2RGB)
					#showing the current face with rectangle drawn
					frame_window.image(current_frame)

			elif feature_choice == 'Recognition':
				#load the sample images and get the 128 face embeddings from them
				modi_image = face_recognition.load_image_file('images/modi.jpg')
				modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

				trump_image = face_recognition.load_image_file('images/trump.jpg')
				trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

				anu_image = face_recognition.load_image_file('images/me.jpg')
				anu_face_encodings = face_recognition.face_encodings(anu_image)[0]

				ank_image = face_recognition.load_image_file('images/ankush.jpg')
				ank_face_encodings = face_recognition.face_encodings(ank_image)[0]

				#save the encodings and the corresponding labels in seperate arrays in the same order
				known_face_encodings = [modi_face_encodings, trump_face_encodings, anu_face_encodings, ank_face_encodings]
				known_face_names = ["Narendra Modi", "Donald Trump", "Anubrata","Ankush"]


				#initialize the array variable to hold all face locations, encodings and names 
				all_face_locations = []
				all_face_encodings = []
				all_face_names = []

				#loop through every frame in the video
				while True:
					#get the current frame from the video stream as an image
					ret,current_frame = cap.read()
					#resize the current frame to 1/4 size to proces faster
					current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
					#detect all faces in the image
					#arguments are image,no_of_times_to_upsample, model
					all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
					
					#detect face encodings for all the faces detected
					all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)


					#looping through the face locations and the face embeddings
					for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
						#splitting the tuple to get the four position values of current face
						top_pos,right_pos,bottom_pos,left_pos = current_face_location
						
						#change the position maginitude to fit the actual size video frame
						top_pos = top_pos*4
						right_pos = right_pos*4
						bottom_pos = bottom_pos*4
						left_pos = left_pos*4
						
						#find all the matches and get the list of matches
						all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
					
						#string to hold the label
						name_of_person = 'Unknown face'
						
						#check if the all_matches have at least one item
						#if yes, get the index number of face that is located in the first index of all_matches
						#get the name corresponding to the index number and save it in name_of_person
						if True in all_matches:
							first_match_index = all_matches.index(True)
							name_of_person = known_face_names[first_match_index]
						
						#draw rectangle around the face    
						cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
						
						#display the name as text in the image
						font = cv2.FONT_HERSHEY_DUPLEX
						cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
						
					current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
					#display the video
					frame_window.image(current_frame)

	elif choice == 'About':
		st.subheader("About Face Detection App")
		st.markdown("Built with Streamlit by Anubrata Das)")



if __name__ == '__main__':
		main()	