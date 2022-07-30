from cProfile import label
from turtle import color
from matplotlib.pyplot import hist
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

def disp_acc(result, metrics):
    plt.figure()
    for i in range(len(metrics)):
        metric = metrics[i]
        
        plt_train = result.history[metric]
        plt_test = result.history['val_' + metric]

        plt.subplot(1, 2, i+1)
        plt.title(metric)
        plt.plot(plt_train, label='training')
        plt.plot(plt_test, label='test')
        plt.legend()
    plt.show()

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')
                
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()