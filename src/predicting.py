import tensorflow as tf
from utils import predictimages

classnames = [str(i) for i in range(10)]
model = tf.keras.models.load_model('/model/savedmodel.keras')

#list your custom images file paths
images = []
predictimages(model,images,classnames)
     

