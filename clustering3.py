# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:49:16 2020

@author: William
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from PIL import Image
from sklearn.cluster import DBSCAN

class Data:
    
    def __init__(self, name, angle = 25):
        
        '''
        name : nom de l'image
        '''
        self.path_scripts = "C:\\Users\\William\\Documents\\BIRSE\\3A\\Projet\\scripts"
        self.path_images = "C:\\Users\\William\\Documents\\BIRSE\\3A\\Projet\decoupage_DATASET\images_champs1"
        self.name = name
        self.image_array =  np.array(self.open_imagePIL())
        self.convertir_ExG()
        self.create_maskExG()
        self.rotation(angle)
        self.dict_object = {'image': self.image_array, 'ExG' : self.image_ExG, 'mask' : self.mask, 'mask_oriente': self.rang_aligne}
        
    def open_imagePIL(self):
        path = self.path_images
        os.chdir(path)
        image = Image.open(self.name)
        return image

    def convertir_ExG(self):
        '''
        converti image en ExG
        '''
        image = self.image_array
        image = np.dot(image[...,:3], [-1, 2, -1])
        self.image_ExG = image


    def create_maskExG(self):
        '''
        créer le mask : filtre valeur tel que ExG < (ExG_max - 100) 
        '''
        # masque de detection du vert
        seuil_min = np.mean(self.image_ExG) + 2*np.std(self.image_ExG)
        mask = np.zeros(np.shape(self.image_ExG))
        for i in range(0, np.shape(mask)[0]):
            for j in range(0, np.shape(mask)[1]):
                if self.image_ExG[i][j] >= seuil_min:
                    mask[i][j] = 255
        self.mask = mask
    
    def display(self, object_name):
        '''
        object_name : image, ExG, mask
        '''
        plt.imshow(self.dict_object[object_name])
        
    def rotation(self, angle):
        
        image = Image.fromarray(np.uint8(cm.gist_earth(self.mask)*255))
        image = image.rotate(angle, expand = 1, fillcolor='black')
        image = np.array(image)
        image = np.dot(image[...,:3], [1, 1, 1])
        self.rang_aligne = image 
        
        
class Clustering:
    
    def __init__(self):
        self.data = None
        self.name = None
        
    def load_data(self, name):
        print('Loading data...')
        self.data = Data(name) 
        self.name = name
        self.load_coord()
        print('OK')
    
    
    def load_coord(self):
        
        data = self.data
        #creation de sample : liste des coordonnées des points pixels non nuls
        sample = data.mask
        sample = np.where(sample!=0)
        sample = list(zip(sample[0], sample[1]))
        for pos in range(0, len(sample)):
            sample[pos] = list(sample[pos])
        
        self.sample = np.asarray(sample, order = 'C')
        
    def selection_rang(self, ecart_rang, decalage = None):
        
        data = self.data
        
        if decalage != None:
            y_start = int(np.shape(data.rang_aligne)[1]/2) + decalage
        else:
            y_start = int(np.shape(data.rang_aligne)[1]/2)
        
        print(y_start)
        image = data.rang_aligne
        # Creation figure et ses axes
        fig, ax = plt.subplots(1)
        # Affichage de l'image
        ax.imshow(image)
    
        for pas in range(y_start, np.shape(data.rang_aligne)[1], ecart_rang):
            #Creation des rectangles en passe avant +x
            x = pas - int(ecart_rang/4)
            y = 0
            width = ecart_rang
            height = np.shape(data.rang_aligne)[0] - 1
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            #ajout du rectangle
            ax.add_patch(rect)
            
        for pas in range(y_start, 0, -ecart_rang):
            #Creation des rectangles en passe arrière -x
            x = pas - int(ecart_rang/4)
            y = 0
            width = ecart_rang
            height = np.shape(data.rang_aligne)[0] - 1
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            #ajout du rectangle
            ax.add_patch(rect)

        plt.show()
        
        
    def DBSCAN_clustering(self, eps, min_samples):
        
        print('Please wait a few minutes ...')
        db = DBSCAN(eps = eps, min_samples = min_samples, metric = 'euclidean')
        db.fit(self.sample)
        db.fit_predict(self.sample)
        self.result_DBSCAN = db.labels_
        #on compte le nombre de cluster différent trouvés (en enlevant le bruit)
        self.n_clusters_DBSCAN = len(set(self.result_DBSCAN)) - (1 if -1 in self.result_DBSCAN else 0)
        print('Le nombre de tournesol trouvé est: ' + str(self.n_clusters_DBSCAN))
        self.core_samples_mask_DBSCAN = np.zeros_like(db.labels_, dtype=bool)
        self.core_samples_mask_DBSCAN[db.core_sample_indices_] = True
        
        
    def DBSCAN_display_cluster(self):
        
        try:
            self.result_DBSCAN
        except NameError:
            var_exists = False
        else:
            var_exists = True
        
        if var_exists == False:
            print('Aucun clustering fait')
            
        else:
            
            unique_labels = set(self.result_DBSCAN)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
    
                class_member_mask = (self.result_DBSCAN == k)
    
                xy = self.sample[class_member_mask & self.core_samples_mask_DBSCAN]
                plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col))
    
                #noise 
                #xy = self.sample[class_member_mask & ~self.core_samples_mask_DBSCAN]
                #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k')
            
            self.data.display('image')
            plt.title('Nombre de cluster estimé ' + str(self.n_clusters_DBSCAN))
            plt.show()
        
        

        