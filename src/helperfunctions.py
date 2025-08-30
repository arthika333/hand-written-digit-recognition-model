import matplotlib.pyplot as plt

def plotlosscurves(history):
  plt.figure(figsize=(10,4))
  a=history.history['loss']
  c=history.history['accuracy']
  epochs=range(len(a))
  plt.subplot(1,2,1)
  plt.plot(a,epochs,label='training loss')
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(c,epochs,label='training accuracy')
  plt.legend()
  plt.show()
  
def predandplot(model,features,labels,noofpreds,classnames):
  import random
  plt.figure(figsize=(12,4))
  for i in range(noofpreds):
    randomindex=random.randint(0,len(features))
    xtrue=features[randomindex]
    ytrue=classnames[labels[randomindex]]
    pred=model.predict(tf.expand_dims(xtrue,axis=0))
    predclass=classnames[pred.argmax()]
    plt.subplot(2,5,i+1)
    plt.imshow(xtrue,cmap='grey')
    plt.axis(False)
    if predclass==ytrue:
      c='green'
    else:
      c='red'
    plt.title(f'pred:{predclass} true label:{ytrue}',color=c)
  plt.show()

images=[] #include the filepaths of all custom images
def predictimages(model,images,classnames):
  import tensorflow
  i=0
  plt.figure(figsize=(12,8))
  for imagepath in images:
    image=tensorflow.io.read_file(imagepath)
    image=tensorflow.image.decode_png(image,channels=1)
    image=tensorflow.image.resize(image,(28,28))
    image=255-image
    image=image/255.
    image=tf.squeeze(image,axis=-1)
    pred=model.predict(tf.expand_dims(image,axis=0),verbose=0)
    predlabel=tf.argmax(pred[0]).numpy()
    plt.subplot(4,4,i+1)
    plt.imshow(image,cmap='grey')
    plt.axis(False)
    plt.title(f'predicted label : {classnames[predlabel]}')
    i=i+1
  plt.show()


     
