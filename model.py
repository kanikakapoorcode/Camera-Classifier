from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL

class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=1000)
        self.trained = False  

    def train_model(self, counters):
        img_list = []  
        class_list = []  
        if not os.path.exists("1") or not os.path.exists("2"):
            print("Training directories do not exist.")
            return

        # Load images for class 1  
        for i in range(1, counters[0]):  
            img = cv.imread(f'1/frame{i}.jpg')  
            if img is not None:  
                img = img[:, :, 0]  
                img = img.reshape(-1)  
                img_list.append(img)  
                class_list.append(1)  

        # Load images for class 2  
        for i in range(1, counters[1]):  
            img = cv.imread(f'2/frame{i}.jpg')  
            if img is not None:  
                img = img[:, :, 0]  
                img = img.reshape(-1)  
                img_list.append(img)  
                class_list.append(2)  
        if len(img_list) == 0 or len(class_list) == 0:
            print("No images found for training.")
            return

        # Convert lists to numpy arrays
        img_list = np.array(img_list)
        class_list = np.array(class_list)

        if img_list.size == 0 or class_list.size == 0:
            print("No images found for training.")
            return

        self.model.fit(img_list, class_list)  
        self.trained = True  
        print("Model successfully trained!")
        self.trained = True  
        print("Model successfully trained!")

    def predict(self, frame):
        if not self.trained:
            print("Model is not trained yet. Please train the model before prediction.")
            return None
        
        if frame is None:
            print("Invalid frame received for prediction: frame is None")
            return None
        
        if not isinstance(frame, np.ndarray):  
            print(f"Invalid frame Type: {type(frame)}. Expected numpy.ndarray.")  
            return None  
        gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  
        img = gray_frame.reshape(-1)  
        prediction = self.model.predict([img])  
        
        try:
            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            cv.imwrite("frame.jpg", gray_frame)  
            img = PIL.Image.open("frame.jpg")
            img.thumbnail((150, 150), PIL.Image.LANCZOS)
            img.save("frame.jpg")

            img = cv.imread('frame.jpg')
            if img is None:
                print("Error reading the image for prediction.")
                return None

            img = img[:, :, 0]  
            img = img.reshape(-1)  
            prediction = self.model.predict([img])

            return prediction[0]
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None
