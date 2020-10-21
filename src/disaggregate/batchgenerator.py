import tensorflow as tf
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from transformations.mtf import MarkovTransitionField
import numpy as np 

class ImagingMethodError(Exception):
    pass

class BatchGenerator(tf.keras.utils.Sequence) :
    
  
    def __init__(self, x_sequence,y_sequence, batch_size,img_method = 'gasf',img_size = 28) :
        """
        Args:
            x_sequence (np.array[nb_samples, sequence_len ]): the aggregate power
            y_sequence (np.array[nb_samples, sequence_len ]): the target appliance consumption
            batch_size (int): the batch size
            img_method (str, optional): the type of image transform. Defaults to 'gasf'.
            img_size (int, optional): the output image size. Defaults to 28.
        """
        super.__init__(BatchGenerator,)
        self.x_sequence = x_sequence
        self.y_sequence = y_sequence
        self.batch_size = batch_size
        self.img_method = img_method
        self.img_size = img_size
    
    
    def __len__(self) :
        return (np.ceil(len(self.x_sequence) / float(self.batch_size))).astype(np.int)

    def ts_imaging(self, data):
        """
        Calcultes the image representation for each batch
        Args:
            data (tf.tensor(batch_size, sequence_length)): a batch of the aggregate power sequences

        Raises:
            ImagingMethodError: Error raised in case a wrong image transform is provided

        Returns:
            tensor(batch_size, img_size, img_size, 1): the image representation of the of the input data
        """
        if self.img_method == 'gasf':
            transformer = GramianAngularField(image_size=self.img_size,method='summation')
            tsi= transformer.fit_transform(data)
        elif self.img_method == 'gadf':
            transformer = GramianAngularField(image_size=self.img_size, method='difference')
            tsi= transformer.fit_transform(data)
        elif self.img_method == 'mtf':
            transformer = MarkovTransitionField(image_size=self.img_size)
            tsi= transformer.fit_transform(data)
        elif self.img_method == 'rp':
            transformer = RecurrencePlot(threshold='point' ,percentage=20)
            tsi= transformer.fit_transform(data)
        elif self.img_size == 'all':
            RP =  RecurrencePlot(threshold='point' ,percentage=20).fit_transform(data)
            GASF = GramianAngularField(image_size=self.img_size,method='summation').fit_transform(data)
            MTF = MarkovTransitionField(image_size=self.img_size).fit_transform(data)
            tsi = np.stack([RP,GASF,MTF],axis=3)
            
        else :
          raise ImagingMethodError()

        return tsi


    def __getitem__(self, idx) :
        """Generates the image representation for each sample

        Args:
            idx (int): the index of first element in the batch

        Returns:
            (batch_x, batch_y): the augmented data for the current batch
        """
        batch_x = self.x_sequence[idx * self.batch_size : (idx+1) * self.batch_size,:]
        batch_x = self.ts_imaging(batch_x).reshape(-1,self.img_size,self.img_size,1)
        if self.y_sequence is not None:
            batch_y = self.y_sequence[idx * self.batch_size : (idx+1) * self.batch_size,:]
        else:
            batch_y = None

        return batch_x, batch_y
