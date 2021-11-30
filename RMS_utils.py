#Class for general operations used in this project
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.math import confusion_matrix as tensorflow_confusion_matrix
import pandas as pd
import seaborn as sns

#Prepares the directory and extracts class weights from the total number of images per class
def prep_dir(stain_dir):
    train_dir = os.path.join(stain_dir, 'train')
    train_dir_arms = os.path.join(train_dir, 'ARMS')
    train_dir_erms = os.path.join(train_dir, 'ERMS')

    validation_dir = os.path.join(stain_dir, 'validation')
    validation_dir_arms = os.path.join(validation_dir, 'ARMS')
    validation_dir_erms = os.path.join(validation_dir, 'ERMS')

    train_arms_fnames = os.listdir(train_dir_arms)
    train_erms_fnames = os.listdir(train_dir_erms)
    validation_arms_fnames = os.listdir(validation_dir_arms)
    validation_erms_fnames = os.listdir(validation_dir_erms)

    print("No. ARMS Train: " + str(len(train_arms_fnames)))
    print("No. ERMS Train: " + str(len(train_erms_fnames)))
    print("No. ARMS Validation: " + str(len(validation_arms_fnames)))
    print("No. ERMS Validation: " + str(len(validation_erms_fnames)))
    
    class_weight = {0: 1.0, 1: len(train_arms_fnames) / len(train_erms_fnames) }
    
    return train_dir, validation_dir, class_weight

#Prepares the data generators and sets image size
def prep_datagen(train_dir, validation_dir, train_datagen, batch_size, image_size):
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = batch_size,
                                                    class_mode = 'binary', 
                                                    target_size = (image_size, image_size))   
    
    validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = batch_size,
                                                          class_mode  = 'binary', 
                                                          target_size = (image_size, image_size),  shuffle=False)

    return train_generator, validation_generator

#Output AUC and Loss results of fitting, option for saving
def show_results(history, save):
    
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Retrieves results on training and validation sets
    auc      = history.history[     'auc' ]
    val_auc  = history.history[ 'val_auc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    epochs   = range(len(auc)) # No. of epochs

    # Plots training and validation accuracy per epoch
    plt.plot  ( epochs,     auc )
    plt.plot  ( epochs, val_auc )
    plt.title ('Training and Validation AUC')
    
    if (save):
        plt.savefig("AUC.png", dpi=500, bbox_inches='tight') 
    
    plt.figure()
    
    # Plot training and validation loss per epoch
    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and Validation loss')
    
    if (save):
        plt.savefig("Loss.png", dpi=500, bbox_inches='tight') 

#Outputs confusion matrix and classification report
def show_preds(model, validation_generator):
    prediction = model.predict(validation_generator)
    y_pred = np.matrix.flatten(prediction)
    Y_pred = np.around(y_pred, 0)

    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, Y_pred))

    target_names = ['ARMS', 'ERMS']
    print(classification_report(validation_generator.classes, Y_pred, target_names=target_names))

#Outputs basic Heatmap for classification report. Option for saving 
def show_heatmap(model, validation_generator, save):
    prediction = model.predict(validation_generator)
    y_pred = np.matrix.flatten(prediction)
    Y_pred = np.around(y_pred, 0)
    
    truncValues = []
    for value in Y_pred:
        truncValues.append(int(value))
    
    con_mat = tensorflow_confusion_matrix(labels=validation_generator.classes, predictions=truncValues).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = ["ARMS", "ERMS"]
    con_mat_df = pd.DataFrame(con_mat_norm,
                         index = classes, 
                         columns = classes)
    
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True ,cbar=False,cmap=plt.cm.YlOrBr)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if (save):
        plt.savefig("Heatmap.png", dpi=500, bbox_inches='tight') 

    plt.show()
    