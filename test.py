from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
import tensorflow as tf
import numpy as np
import keras
import json
import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sklearn.model_selection import train_test_split
def model_rest(): 
    # create the base pre-trained model
    base_model = ResNet50(include_top=False, weights='imagenet' )
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    # and a logistic layer 
    predictions = Dense(1, activation='sigmoid')(x)
    
    # this is the model we will train
    base_model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = Adam(lr=0.001)
    base_model.compile(loss='binary_crossentropy', 
              optimizer=optimizer, 
              metrics=["accuracy"])
    
    return base_model
def get_panda_input(image):
    #image = efn.center_crop_and_resize(image, input_shape[1])
    image=cv2.resize(image,(224,224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, 0)
    image=image[0]
    #print(image.shape)
    return image
def create_dataset(path_folder):
   
    img_data_array=[]
    class_name=[]
    train_path = open("train.txt","w")
    test_path = open("test.txt","w")
    count_0=0
    count_1=0
    for path_json in os.listdir(path_folder):
        path_folder1=os.path.join(path_folder,path_json)
        #print("Folder 1", path_folder1)
        string = ""
        
        for path_json1 in os.listdir(path_folder1):
            if len(path_json1.split("."))>1:
                if path_json1=='croped_circle.jpg':
                    nn=os.path.join(path_folder1,path_json1)
                    #print("path_json1:", path_json1)
                    img=cv2.imread(os.path.join(path_folder1,path_json1))
                    x_input = get_panda_input(img)
                    img_data_array.append(x_input)
                    
                if path_json1.split(".")[1] == 'txt':
                    #print("path_json1:", path_json1)
                    f=open(os.path.join(path_folder1,path_json1),"r")
                    x_in = f.readlines()
                    if "khong-co-benh" in x_in:
                        class_name.append(0)
                        count_0 = count_0+1
                        mm=0
                    else: 
                        class_name.append(1)
                        mm=1
                        count_1=count_1+1
        string = nn+"\t" + str(mm)+"\n"
        if mm == 1:
            if count_0 <= 81:
                train_path.write(string)
            else:
                test_path.write(string)
        if mm == 0:
            if count_0 <= 94:
                train_path.write(string)
            else:
                test_path.write(string)
        
        #file_path.write(string)
    #print(class_name)
    print(count_0)
    print(count_1)
    img_data_array=np.array(img_data_array,np.float32)
    class_name=np.array(class_name)
    return img_data_array,class_name

# model_save_path = WEIGHTS_DIRECTORY + 'resnet50_pretrained_weights.h5'
# print("Loading weights from: {}".format(model_save_path))
# model.load_weights(model_save_path)
model=model_rest()
filepath='weights/resnet50.hdf5'
model_checkpoint_callback =tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')
# print(model.input[1])
X_train,y_train=create_dataset("New_RawAC")

print("load data train good")
X_train,X_test,y_train,y_test=train_test_split(X_train, y_train,stratify = y_train, test_size=0.2)
print(X_train.shape,y_train.shape)
print("load data test good")
history = model.fit(X_train,y_train,batch_size=16, validation_data=(X_test, y_test), epochs=200, verbose=1,callbacks=[model_checkpoint_callback])

