# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import keras
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense, Concatenate, Activation
# import pickle
# import wfdb
# from sklearn.utils import class_weight
# from sklearn.model_selection import train_test_split

# # Configure GPU
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         tf.config.set_visible_devices(gpus[0], 'GPU')
#     except RuntimeError as e:
#         print(e)

# # Hyper-parameters
# sequence_length = 240
# epochs = 1000  # int(input('Enter Number of Epochs (or enter default 1000): '))
# FS = 100.0

# def z_norm(result):
#     result_mean = np.mean(result)
#     result_std = np.std(result)
#     result = (result - result_mean) / result_std
#     return result

# def split_data(X):
#     X1 = []
#     X2 = []
#     for index in range(len(X)):
#         X1.append([X[index][0], X[index][1]])
#         X2.append([X[index][2], X[index][3]])

#     return np.array(X1).astype('float64'), np.array(X2).astype('float64')

# def get_data():
#     with open('train_input.pickle','rb') as f: 
#         X_train = pickle.load(f)
#     with open('train_label.pickle','rb') as f: 
#         y_train = np.asarray(pickle.load(f))
#     with open('val_input.pickle','rb') as f: 
#         X_val = pickle.load(f)
#     with open('val_label.pickle','rb') as f: 
#         y_val = np.asarray(pickle.load(f))
#     with open('test_input.pickle','rb') as f: 
#         X_test = pickle.load(f)
#     with open('test_label.pickle','rb') as f: 
#         y_test = np.asarray(pickle.load(f))

#     X_train1, X_train2 = split_data(X_train)
#     X_val1, X_val2 = split_data(X_val)
#     X_test1, X_test2 = split_data(X_test)

#     X_train1 = np.transpose(X_train1, (0, 2, 1))
#     X_test1 = np.transpose(X_test1, (0, 2, 1))
#     X_val1 = np.transpose(X_val1, (0, 2, 1))
#     return X_train1, X_train2, y_train, X_val1, X_val2, y_val, X_test1, X_test2, y_test

# def build_model():
#     with tf.device('/GPU:0'):
#         layers = {'input': 2, 'hidden1': 256, 'hidden2': 256, 'hidden3': 256, 'output': 1}
        
#         # First input branch
#         x1 = Input(shape=(sequence_length, layers['input']))
#         m1 = LSTM(layers['hidden1'], recurrent_dropout=0.5, return_sequences=True)(x1)
#         m1 = LSTM(layers['hidden2'], recurrent_dropout=0.5, return_sequences=True)(m1)
#         m1 = LSTM(layers['hidden3'], recurrent_dropout=0.5, return_sequences=False)(m1)

#         # Second input branch
#         x2 = Input(shape=(2,))
#         m2 = Dense(32)(x2)

#         # Merge branches
#         merged = Concatenate(axis=1)([m1, m2])

#         # Output layers
#         out = Dense(8)(merged)
#         out = Dense(layers['output'], kernel_initializer='normal')(out)
#         out = Activation("sigmoid")(out)
        
#         model = Model(inputs=[x1, x2], outputs=[out])

#         start = time.time()
#         model.compile(
#             loss="binary_crossentropy",
#             optimizer="adam",
#             metrics=['accuracy']
#         )
#         print("Compilation Time : ", time.time() - start)

#         model.summary()
#         return model

# def run_network(model=None, data=None):
#     global_start_time = time.time()

#     print('\nData Loaded. Compiling...\n')
#     print('Loading data... ')
#     X_train1, X_train2, y_train, X_val1, X_val2, y_val, X_test1, X_test2, y_test = get_data()

#     class_w = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     print(class_w)

#     if model is None:
#         model = build_model()

#     try:
#         print("Training")
#         class_w = {i: class_w[i] for i in range(2)}
        
#         callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#         history = model.fit(
#             [X_train1, X_train2],
#             y_train,
#             validation_data=([X_val1, X_val2], y_val),
#             callbacks=[callback],
#             epochs=epochs,
#             batch_size=256,
#             class_weight=class_w
#         )

#         plt.plot(history.history['loss'])
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train'], loc='upper left')
#         plt.show()

#         # Evaluate Model
#         y_pred = model.predict([X_test1, X_test2])
#         y_pred = (y_pred > 0.5).astype(int)

#         accuracy = np.mean(y_pred == y_test)
#         print(f"Test Accuracy: {accuracy}")

#     except KeyboardInterrupt:
#         print('Training duration (s) : ', time.time() - global_start_time)
#         return model

#     print('Training duration (s) : ', time.time() - global_start_time)
#     return model

# if __name__ == '__main__':
#     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#     run_network()