import tensorflow as tf
import numpy as np
def makeModel():
    import sys

    mnist = tf.keras.datasets.mnist
    imdb = tf.keras.datasets.imdb

    x_train,y_train,x_test,y_test = makeMnistData(mnist)
    dataShape = x_train[1].shape

    #showImage(x_train,y_train,True)

    layerData=[
        tf.keras.layers.Flatten(input_shape=dataShape),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]

    layer=makeLayer(layerData)

    opt = tf.keras.optimizers.SGD()
    
    model = makeModelShape(layerList=layer,optimizer=opt)

    trainedModel = trainModel(model,x_train,y_train,x_test,y_test)


def showImage(x_train,y_train,flag):
    if not flag:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()
    

def makeMnistData(mnist,slush=255):
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / slush, x_test / slush
    return x_train,y_train,x_test,y_test


def makeImdbData(imdb,dataLength=255):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    return train_data,train_labels,test_data,test_labels


def makeLayer(layerData):
    layer =layerData
    return layer


def makeModelShape(layerList=[],
                   optimizer='adam',
                   loss="sparse_categorical_crossentropy",
                   metrics=['accuracy']
                   ):

    model = tf.keras.models.Sequential(layerList)
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    return model


def trainModel(model,x_train,y_train,x_test,y_test,epochs=5):
    trainData = model.fit(x_train
                        , y_train
                        , epochs=epochs
                        ,validation_data=(x_test,y_test)
                        )
    testScore = model.evaluate(x_test, y_test)
    compare_TV(trainData)
    print(testScore)
    return model


def compare_TV(history):
    import matplotlib.pyplot as plt

        # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

        # 1) Accracy Plt
    plt.plot(epochs, acc, 'bo' ,label = 'training acc')
    plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
    plt.title('Training and Validation acc')
    plt.legend()

    plt.figure()

        # 2) Loss Plt
    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()


if __name__ =='__main__':
    makeModel()
