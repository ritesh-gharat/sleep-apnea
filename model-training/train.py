import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import gc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Activation
import pickle
from sklearn.utils import class_weight

# Configure GPU memory growth - improved for limited VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # More aggressive memory settings for 6GB VRAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Optional: set memory limit explicitly
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=5120)]  # 5GB limit to leave some for system
            )
        # Use only first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Hyper-parameters - adjusted for 6GB VRAM
sequence_length = 240
epochs = 100
FS = 100.0
BATCH_SIZE = 32  # Reduced from 64 to 32 for 6GB VRAM
GRAD_ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate larger batch

# Enable mixed precision for better performance on limited VRAM
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def z_norm(result):
    result_mean = np.mean(result)
    result_std = np.std(result)
    result = (result - result_mean) / result_std
    return result

def split_data(X):
    X1 = []
    X2 = []
    for index in range(len(X)):
        X1.append([X[index][0], X[index][1]])
        X2.append([X[index][2], X[index][3]])
    return np.array(X1, dtype='float32'), np.array(X2, dtype='float32')

@tf.function
def train_step(model, inputs, labels, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def get_data():
    # Load data using tf.data.Dataset for better performance
    with open('train_input.pickle','rb') as f: 
        X_train = pickle.load(f)
    with open('train_label.pickle','rb') as f: 
        y_train = np.asarray(pickle.load(f), dtype='float32')
    with open('val_input.pickle','rb') as f: 
        X_val = pickle.load(f)
    with open('val_label.pickle','rb') as f: 
        y_val = np.asarray(pickle.load(f), dtype='float32')
    with open('test_input.pickle','rb') as f: 
        X_test = pickle.load(f)
    with open('test_label.pickle','rb') as f: 
        y_test = np.asarray(pickle.load(f), dtype='float32')

    X_train1, X_train2 = split_data(X_train)
    X_val1, X_val2 = split_data(X_val)
    X_test1, X_test2 = split_data(X_test)

    X_train1 = np.transpose(X_train1, (0, 2, 1))
    X_test1 = np.transpose(X_test1, (0, 2, 1))
    X_val1 = np.transpose(X_val1, (0, 2, 1))

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(((X_train1, X_train2), y_train))\
        .shuffle(buffer_size=1024)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(((X_val1, X_val2), y_val))\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(((X_test1, X_test2), y_test))\
        .batch(BATCH_SIZE)

    return train_dataset, val_dataset, test_dataset

def build_model():
    # Reduced model size for 6GB VRAM
    layers = {'input': 2, 'hidden1': 128, 'hidden2': 128, 'hidden3': 128, 'output': 1}
    
    # First input branch
    x1 = Input(shape=(sequence_length, layers['input']))
    m1 = LSTM(layers['hidden1'], dropout=0.5, return_sequences=True)(x1)
    m1 = LSTM(layers['hidden2'], dropout=0.5, return_sequences=True)(m1)
    m1 = LSTM(layers['hidden3'], dropout=0.5, return_sequences=False)(m1)

    # Second input branch
    x2 = Input(shape=(2,))
    m2 = Dense(16)(x2)  # Reduced from 32 to 16

    # Merge branches
    merged = Concatenate(axis=1)([m1, m2])

    # Output layers
    out = Dense(8)(merged)
    out = Dense(layers['output'])(out)
    out = Activation("sigmoid")(out)
    
    model = Model(inputs=[x1, x2], outputs=[out])
    return model

# Modified training step with gradient accumulation
def train_with_grad_accumulation(model, optimizer, loss_fn, dataset):
    # Initialize the accumulated gradients
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
    num_samples = 0
    total_loss = 0
    
    # Iterate over the dataset
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)
            scaled_loss = loss_value / GRAD_ACCUMULATION_STEPS
        
        # Calculate gradients and accumulate them
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]
        
        # Update metrics
        num_samples += len(y_batch)
        total_loss += loss_value
        
        # Apply accumulated gradients after specified number of steps
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            # Reset accumulated gradients
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            
            if (step + 1) % 200 == 0:
                print(f"Training loss at step {step+1}: {float(total_loss/num_samples):.4f}")
                
    # Apply any remaining gradients
    if accumulated_gradients[0] is not tf.zeros_like(model.trainable_variables[0]):
        optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
    
    return total_loss / num_samples

def run_network(model=None):
    global_start_time = time.time()
    
    print('\nLoading data...')
    train_dataset, val_dataset, test_dataset = get_data()

    if model is None:
        model = build_model()
    
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Metrics
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    print("Training...")
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        
        # Training with gradient accumulation
        avg_loss = train_with_grad_accumulation(model, optimizer, loss_fn, train_dataset)
        print(f"Average loss over epoch: {float(avg_loss):.4f}")
        
        # Calculate training accuracy
        for x_batch_train, y_batch_train in train_dataset:
            train_logits = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_logits)
        train_acc = train_acc_metric.result()
        print(f"Training acc over epoch: {float(train_acc):.4f}")
        train_acc_metric.reset_states()
        
        # Run validation
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print(f"Validation acc: {float(val_acc):.4f}")
        
        # Clear memory between epochs
        gc.collect()
        if gpus:
            tf.keras.backend.clear_session()

    # Compile the model explicitly before saving to ensure metrics are built
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Save the model with `save_format='tf'` to avoid warnings
    model.save('sleep_apnea_model', save_format='tf')
    
    print('Training duration (s) : ', time.time() - global_start_time)

    # Test phase: Evaluate the model on the test dataset
    print("\nTesting the model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model

if __name__ == '__main__':
    # Enable XLA optimization
    tf.config.optimizer.set_jit(True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)
    run_network()