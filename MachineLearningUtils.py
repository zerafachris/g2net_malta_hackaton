from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def error_rate(y_test, y_pred):
    return np.sum(y_pred != y_test) / y_test.size

def plot_confustion_matrix(y_test, y_predicted, df, prob=False):
    if prob:
        cms = confusion_matrix(y_test.argmax(1), y_predicted.argmax(1))
    else:
        cms = confusion_matrix(y_test, y_predicted)
    test_score = np.trace(cms) / np.sum(cms)
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.transpose(cms), interpolation="nearest", cmap="cool")
    rows = cms.shape[0]
    cols = cms.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            value = int(cms[x, y])
            ax.text(x, y, value, color="black", ha="center", va="center")
    plt.title("Real vs Predicted data : Accuracy: {}".format(test_score))
    plt.colorbar(im)
    
    df_labels = df[['target','label']].drop_duplicates().sort_values(by='target')
    classes_values = [tt for tt in df_labels['target']]
    classes_labels = [tt for tt in df_labels['label']]

    plt.xticks(classes_values, classes_labels)
    plt.yticks(classes_values, classes_labels)
    plt.xlabel("Real data")
    plt.ylabel("Predicted data")
    b, t = plt.ylim()
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update ylim(bottom, top) 
    plt.show()



def plot_network_learning_graphs(cnn_net):
    fig,ax=plt.subplots(ncols=2,nrows=1, figsize=(20,5))
    epoch_number = len(cnn_net.history['loss'])
    for key in cnn_net.history.keys():
        if (key != 'lr') & (key[:3]!='val') :
            ax[0].plot(range(epoch_number), cnn_net.history[key], label=key)
        elif (key != 'lr') & (key[:3]=='val') :
            ax[1].plot(range(epoch_number), cnn_net.history[key], label=key)    
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_yscale('log')
    ax[0].set_title('Training')

    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_yscale('log')
    ax[1].set_title('Validation')
    plt.show();
         
