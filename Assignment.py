import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2 as cv
import os

from sklearn.datasets import fetch_olivetti_faces

# SVM for classifier training:
from sklearn.svm import SVC

# Process each face image put into the images dataset:
def process_face_image(image):
 # Load the Haar Cascade Classifier for face detection
 face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
 
 # Convert the image to grayscale
 gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
 
 # Detect the faces in the grayscale image
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
 
 # Check if at least one face is detected
 if len(faces) == 0:
  print("no face detected ... skipping for now.")
  return None
 
 # Get the coordinates of the first detected face
 (x, y, w, h) = faces[0]
 
 # Crop the face region found to 64x64
 face_image = cv.resize(gray[y:y+h, x:x+w], (64, 64))
 
 # Normalize the image:
 normalized_face_image = cv.equalizeHist(face_image)
 
 # Ravel the face image
 face_image = np.ravel(normalized_face_image)
 
 # Return the ravelled face image
 return face_image

# Heads Dataset taken from AI1703 heads dataset, used for the project:
def import_from_dataset(dataset_folder="Heads Dataset"):
 # Define the image size after scaling down
 img_size = (64, 64)

 # Initialize lists for storing images and labels
 images = []
 labels = []

 # Loop through each subfolder in the dataset folder
 for label in os.listdir(dataset_folder):
  # Get the subfolder path
  label_folder = os.path.join(dataset_folder, label)

  # Loop through each image in the subfolder
  for image_file in os.listdir(label_folder):
   print(os.path.join(label_folder, image_file))
   # Get the image file path
   image_path = os.path.join(label_folder, image_file)
   
   # Load the image in color
   image = cv.imread(image_path)
   
   new_image=process_face_image(image)
   if new_image is not None:
    # Add the image and label to the lists
    images.append(new_image)
    labels.append(label)
         
 # Convert the images and labels lists to numpy arrays
 images = np.array(images)
 labels = np.array(labels)
 
 return images,labels

# Load the face dataset: (Using olivetti dataset)
def load_faces():
 dataset = fetch_olivetti_faces()
 faces = dataset.data
 target = dataset.target
 return faces, target

def train_test_split(faces,target,test_size=0.25):

 # In theory, each face has a target.
 test_size_num=int(len(faces)*test_size)

 # Splits into train and test data based on test size:
 train_faces,test_faces = faces[:test_size_num],faces[test_size_num:]
 train_target,test_target = target[:test_size_num],target[test_size_num:]

 return train_faces,train_target,test_faces,test_target

# Apply PCA to the training set to obtain the eigenfaces:
def get_eigenfaces(train_faces,components=100):
 pca = PCA(n_components=components)
 pca.fit(train_faces)

 # reshaping each row vector into a 2D image of size 64x64, to get eigenfaces
 eigenfaces = pca.components_.reshape((components, 64, 64))

 return pca,eigenfaces

# Train SVM classifier based on training faces:
def train_model(train_faces,train_target):
 # Project training and testing features onto eigenspace:
 train_features = pca.transform(train_faces)

 # trains a SVM classifier using the training features and labels.
 svm = SVC(random_state=420)
 svm.fit(train_features, train_target)
 
 return svm

# Shows all eigenfaces:
def show_eigenfaces(eigenfaces):
 plt.figure(figsize=(16, 8))

 for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.imshow(eigenfaces[i], cmap='gray')
  plt.title(f"Eigenface #{i}")
  plt.axis('off')

 plt.show()

# Test the algorithm on any image:
def test(image):
 face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

 gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

 face_images = []
 for (x, y, w, h) in faces:
  face = gray[y:y+h, x:x+w]
  face_images.append(face)

 normalized_faces = []
 for face in face_images:
  resized_face = cv.resize(face, (64, 64))
  normalized_face = cv.equalizeHist(resized_face)
  # Normalized face must be ravelled to match trained face data.
  normalized_face=np.ravel(normalized_face)
  normalized_faces.append(normalized_face)

 test_features = pca.transform(np.array(normalized_faces))

 predicted_labels = svm.predict(test_features)

 for i, (x, y, w, h) in enumerate(faces):
  cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv.putText(image, f"{predicted_labels[i]}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

 cv.imshow("Detected Faces", image)
 cv.waitKey(0)
 cv.destroyAllWindows()

# Execution:

# Load the faces dataset and split it into training and testing sets
# (old unused code)
#faces, target = load_faces()

# Only run import_from_dataset if there's new data to add.
#faces, target = import_from_dataset()
#np.savez("savedfaces.npz",faces=faces,target=target)

# Gets data from the last processed iteration of import_from_dataset():
data=np.load('savedfaces.npz')
faces,target=data['faces'],data['target']

pca,eigenfaces=get_eigenfaces(faces,100)

svm=train_model(faces,target)

test(cv.imread("anh_1.png"))