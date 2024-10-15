import os
import random
import numpy as np
import tensorflow as tf
import cv2
import time
from tensorflow.keras import Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Dropout, RandomFlip, RandomRotation, \
    BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from keras_tuner.tuners import GridSearch
from sklearn.model_selection import KFold
import sys
import pandas as pd
from tensorflow.keras.layers import Concatenate
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Setup TensorBoard logging directory
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


def read_parameters_from_file(file_name):
    parameters = {}
    with open(file_name, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            parameters[key] = value  # Convert to the appropriate data type if needed
    return parameters


import sys

parameter_file = sys.argv[1]
parameters = read_parameters_from_file(parameter_file)

Splitting_test_and_hold = float(parameters["Splitting_test_and_hold"])
Drop = float(parameters["Drop"])
batch = int(parameters["batch"])
learning_rate = float(parameters["LR"])
numberfeat = int(parameters["numberfeat"])
number_ft_layers = int(parameters['number_ft_layers'])
number_ft_layers_2 = int(parameters['number_ft_layers_2'])
LR_FT = float(parameters["LR_FT"])
feat = str(parameters["feature"])
textmod_hug = str(parameters["text_mod"])
textinfo = str(parameters["textinfo"])
max_text_length = int(parameters["max_text_length"])
epochs1 = int(parameters["epochs1"])
epochs2 = int(parameters["epochs2"])
epochs3 = int(parameters["epochs3"])
seed = int(parameters["seed"])
expname = 'results/' + str(parameters["expname"])

os.environ['PYTHONHASHSEED'] = str(seed)  # 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed)  # 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed)  # 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed)  # 5. For layers that introduce randomness like dropout, make sure to set seed values

print(Splitting_test_and_hold, Drop, batch, learning_rate, numberfeat, number_ft_layers, number_ft_layers_2, feat)

LR = [learning_rate]

MOD = 'InceptionV3'  # ---- !!!!! --- CHANGE IT IN CODE (4 lines below) and SH also

print("")
print(
    f"You are running the {MOD} model, with Splitting_test_and_hold={Splitting_test_and_hold}, Drop={Drop}, batch={batch}, LR={learning_rate}.")
print("")

# Step 1: Load model with pre-trained weights

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(186, 186, 3))
base_model.trainable = False

# Step 2: Prepare the data

print('')
print('---------------')
print('DATA PREPARATION')
print('---------------')
print('')

# Load dataframes
properties_df = pd.read_csv("./data/data_cleaning/Newdata_normalized_zscore_interval/filtered_FCC_properties.csv",
                            index_col=0)

images = []

props = [feat]

targets = properties_df[props].values.astype(np.float32)
scaler = StandardScaler()
normalized_targets = scaler.fit_transform(targets.reshape(targets.size, 1)).reshape(targets.shape)

# normalized_targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))

idss = properties_df.index

# Iterate over the rows of the DataFrame and reshape the flattened images
for id in idss:
    img_read = cv2.imread("./data/data_cleaning/Newdata_normalized_zscore_interval/normalized_data_" + str(
        id) + ".png")  # Reading png file format
    dim = (186, 186)  # original = 146*392 #Reducing the dimension of each image for easy processing
    resized_img = cv2.resize(img_read, dim, interpolation=cv2.INTER_AREA)
    images.append(resized_img)  # Appending the images in an empty array

images = np.array(images) / 255.0  # Reformatting the image data to array form and normalizing

shape_of_images_data_all = np.shape(images)

# Split the data into training and a temporary set containing both test and holdout samples
X_train, X_holdout_test, y_train, y_holdout_test = train_test_split(images, normalized_targets,
                                                                    test_size=Splitting_test_and_hold,
                                                                    random_state=seed)

# Split the temporary set into the final test and holdout sets
X_val, X_holdout, y_val, y_holdout = train_test_split(X_holdout_test, y_holdout_test, test_size=0.50, random_state=seed)

print(
    f'The length of training, testing, and holdout dataset is: {len(X_train)} {len(X_val)} {len(X_holdout)} respectively')
print(shape_of_images_data_all)
# -----------------------------------------------------------------


data_augmentation = Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.2),
])


def build_model(hp):
    inputs = Input(shape=(186, 186, 3))

    # inputs_augm = data_augmentation(inputs)

    x = base_model(inputs, training=False)

    # Hyperparameter grid search
    learning_rate = hp.Choice('learning_rate', values=LR)

    x = GlobalAveragePooling2D()(x)

    # Adding intermediate Dense layers
    x = Dense(1024, activation='relu', name='dense_1_image')(x)
    x = BatchNormalization(name='bn_1_image')(x)
    x = Dropout(Drop, seed=seed, name='dropout_1_image')(x)  # Dropout rate can be a hyperparameter too

    x = Dense(512, activation='relu', name='dense_2_image')(x)
    x = BatchNormalization(name='bn_2_image')(x)
    x = Dropout(Drop, seed=seed, name='dropout_2_image')(x)

    x = Dense(256, activation='relu', name='dense_3_image')(x)
    x = BatchNormalization(name='bn_3_image')(x)
    x = Dropout(Drop, seed=seed, name='dropout_3_image')(x)

    outputs = Dense(numberfeat, name='dense_4_image')(x)  # One output neuron for regression task

    model = Model(inputs, outputs, name="my_image_model")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    print(learning_rate)
    print(model.summary())  # Printing the model summary

    return model


# Combine all the data and labels to perform K-fold cross-validation
X_data = np.concatenate((X_train, X_val, X_holdout))
y_data = np.concatenate((y_train, y_val, y_holdout))

# Define the number of folds (K)
num_folds = 5

# Initialize KFold cross-validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Initialize a list to store the performance results for each fold
fold_losses = []

# Initialize variables to keep track of the best model and its performance
best_model = None
best_loss = float('inf')

import pickle

# Perform K-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(kfold.split(X_data, y_data)):
    print(f"Training fold {fold + 1}...")

    # Split data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_data[train_indices], X_data[val_indices]
    y_train_fold, y_val_fold = y_data[train_indices], y_data[val_indices]

    # Set up the tuner for this fold
    tuner = GridSearch(build_model,
                       objective='val_loss',
                       max_trials=10,  # Number of hyperparameter combinations to try
                       executions_per_trial=1,
                       directory=f'{expname}/image_only/tuner',
                       project_name=f'KFOLD_{fold}'
                       )

    model_name = f'{expname}/image_only/saved_models/KFold/fold_{fold}'
    mc = ModelCheckpoint(str(model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)  # Saving the best model only
    history = History()

    ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Search for the best hyperparameter configuration for this fold
    tuner.search(X_train_fold, y_train_fold,
                 validation_data=(X_val_fold, y_val_fold),
                 epochs=epochs1,
                 batch_size=batch,
                 callbacks=[ES, history])

    # Get the best hyperparameters from the search for this fold
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_learning_rate = best_hyperparameters.get('learning_rate')

    print("Best Learning Rate for fold", fold + 1, ":", best_learning_rate)

    # Build the final model with the best hyperparameters for this fold
    final_model = tuner.hypermodel.build(best_hyperparameters)

    print("FINAL MODEL:", final_model.summary())

    # Train the final model on the combined training and validation data for this fold
    F_fit = final_model.fit(X_train_fold, y_train_fold,
                            epochs=epochs2,
                            batch_size=batch,
                            validation_data=(X_val_fold, y_val_fold),
                            callbacks=[mc, ES, history, tensorboard_callback])

    data_to_save = {'history': F_fit.history, 'epoch': F_fit.epoch}
    os.makedirs(f'{expname}/image_only/histories', exist_ok=True)
    os.makedirs(f'{expname}/image_only/figs', exist_ok=True)
    with open(
            f'{expname}/image_only/histories/fold_{fold}.pkl',
            'wb') as file:
        pickle.dump(data_to_save, file)

    # Evaluate the performance of the final model on the holdout set for this fold
    final_model = load_model(mc.filepath)
    os.remove(mc.filepath)
    fold_loss = final_model.evaluate(X_holdout, y_holdout, batch_size=batch)
    print("Fold", fold + 1, "Loss:", fold_loss)

    # Store the performance result for this fold
    fold_losses.append(fold_loss)

    # Check if the current model has the best performance so far
    if fold_loss < best_loss:
        best_loss = fold_loss
        best_model = final_model

    break

# Save the best model with the lowest validation loss across all folds
if best_model is not None:
    best_model_path = f'{expname}/image_only/saved_models/best_model.h5'
    best_model.save(
        f'{expname}/image_only/saved_models/best_model.h5')

# Calculate the average performance across all folds
average_loss = np.mean(fold_losses)
print("Average Loss:", average_loss)

"""#Performance of final model"""

print("")
print("PERFORMANCES of the pretrained model, no fine-tuning")
print("")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Step 7: Evaluate the model (optional, after training)
loss = final_model.evaluate(X_val, y_val)
# Make predictions on the test data
y_pred = final_model.predict(X_val)

# Compute additional evaluation metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Test Loss (MSE):", loss)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Plotting the predicted values from the model
train_prediction = final_model.predict(X_train)
test_prediction = final_model.predict(X_val)
hold_prediction = final_model.predict(X_holdout)

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Pre-trained model, no fine-tuning yet. Cross validation with {num_folds} folds, batch-size {batch}')
plt.savefig(
    f'{expname}/image_only/figs/best_model.png')
plt.close()

"""#Gradual finetuning"""

from tensorflow.keras.models import load_model

# Load the Keras model from the .h5 file
#model_path = f'{expname}/image_only/saved_models/best_model_KFOLD_{num_folds}_LR_{learning_rate}_Drop_{Drop}_InceptionV3_batch_{batch}_SPL_{Splitting_test_and_hold}_LR_FT_{LR_FT}.h5'
loaded_model = load_model(best_model_path) #using the old definition of this path to not hardcode it twice

# FINETUNING
print("")
print("____________")
print("FINE-TUNING")
print("____________")
print("")

ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

"""##stage 1:
unfreezing 50/311 layers of the base model
"""

for layer in base_model.layers[-number_ft_layers:]:
    layer.trainable = True
loaded_model.compile(optimizer=Adam(learning_rate=LR_FT), loss='mean_squared_error')
ft_model_name = f'{expname}/image_only/saved_models/finetuned'
ft_mc = ModelCheckpoint(str(ft_model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)  # Saving the best model only
first_stage_FT = loaded_model.fit(X_train, y_train, epochs=epochs3, batch_size=batch, callbacks=[ES, ft_mc],
                                  validation_data=(X_val, y_val))

"""###Evaluating performances"""

import plotly.graph_objects as go
import plotly.io as pio

fig_stage1 = go.Figure()
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['loss'],
                                mode='lines',
                                name='Training Loss'))
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['val_loss'],
                                mode='lines',
                                name='Validation Loss'))
fig_stage1.update_layout(title='Fine-Tuning Loss - STAGE 1 ',
                         xaxis_title='Epoch',
                         yaxis_title='Loss')

# Save the plot as an HTML file
html_file_path = f'{expname}/image_only/figs/finetuned_performance.html'
pio.write_html(fig_stage1, file=html_file_path, auto_open=True)

data_to_save = {'history': first_stage_FT.history, 'epoch': first_stage_FT.epoch}
with open(
        f'{expname}/image_only/histories/finetuning.pkl',
        'wb') as file:
    pickle.dump(data_to_save, file)

# EVALUATE FINETUNED
loaded_model = load_model(ft_mc.filepath)
print('!!!THE FINAL RESULTS FOR IMAGE ONLY:')

# Plotting the predicted values from the model
train_prediction = loaded_model.predict(X_train)
test_prediction = loaded_model.predict(X_val)
hold_prediction = loaded_model.predict(X_holdout)

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Fine-tuned model.')
plt.savefig(
    f'{expname}/image_only/figs/finetuned_performance.png')
plt.close()

inps = loaded_model.inputs
outs = loaded_model.layers[-3].output
saved_image_model = Model(inputs=inps, outputs=outs, name='my_image_model_headless')
saved_image_model.trainable = False  # it's been frozen :)


# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////


def filter_sentences(df):
    new_df = pd.DataFrame(columns=['MPID', 'Description'])

    def filter_sentence(sentence):
        words = sentence.split()
        result = [words[0] + '.']

        for i in range(1, len(words)):
            if words[i] == 'bond' and i < len(words) - 1 and words[i + 1] == 'lengths':
                result.append(
                    words[i - 2] + ' ' + words[i - 1] + ' ' + words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' +
                    words[i + 3] + ' ' + words[i + 4])
        return ' '.join(result)

    # Apply the filter to each row in the original DataFrame
    for index, row in df.iterrows():
        modified_sentence = filter_sentence(row['Description'])
        new_row = pd.DataFrame({'MPID': row.name, 'Description': modified_sentence}, index=[0])
        new_df = pd.concat([new_df, new_row])
    return new_df.set_index('MPID')


print('')
print('---------------')
print('ADDING TEXT ON TOP')
print('---------------')
print('')

from tensorflow.keras.layers import Input, AveragePooling1D, Flatten

if textmod_hug == 'roberta-base':
    from transformers import RobertaTokenizer, TFRobertaModel

    tokenizer = RobertaTokenizer.from_pretrained(textmod_hug)
    text_model = TFRobertaModel.from_pretrained(textmod_hug)
    custom_objects = {'TFRobertaModel': TFRobertaModel}

elif textmod_hug == 'gpt2-medium':
    from transformers import GPT2Tokenizer, TFGPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(textmod_hug)
    text_model = TFGPT2Model.from_pretrained(textmod_hug)
    custom_objects = {'TFGPT2Model': TFGPT2Model}

elif textmod_hug == 'gpt2-large':
    from transformers import GPT2Tokenizer, TFGPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(textmod_hug)
    text_model = TFGPT2Model.from_pretrained(textmod_hug)
    custom_objects = {'TFGPT2Model': TFGPT2Model}

elif textmod_hug == 'albert-base-v2':
    from transformers import AlbertTokenizer, TFAlbertModel

    tokenizer = AlbertTokenizer.from_pretrained(textmod_hug)
    text_model = TFAlbertModel.from_pretrained(textmod_hug)
    custom_objects = {'TFAlbertModel': TFAlbertModel}

# Freeze the RoBERTa model
text_model.trainable = False

text_df = pd.read_csv("./data/data_cleaning/Newdata_normalized_zscore_interval/filtered_FCC_descriptions.csv",
                      index_col=0)

# --------------------------
# Extract text descriptions from the DataFrame

if textinfo == '1_word':

    text_descriptions = text_df['Description'].values.tolist()
    # Truncate each sentence to keep only the first word
    text_descriptions = [" ".join(sentence.split()[:1]) for sentence in text_descriptions]
    print(text_descriptions)
    tokenizer.pad_token = tokenizer.eos_token
    text_data = tokenizer(text_descriptions, return_tensors='tf', padding=True)
    X_text_input_ids = text_data['input_ids']
    X_text_attention_mask = text_data['attention_mask']


elif textinfo == 'KEY':

    filtered_df = filter_sentences(text_df)
    text_descriptions = filtered_df['Description'].values.tolist()
    tokenizer.pad_token = tokenizer.eos_token
    text_data = [tokenizer(sentence, padding='max_length', truncation=True, max_length=100, return_tensors="tf") for
                 sentence in text_descriptions]
    input_ids_list = [sentence['input_ids'] for sentence in text_data]
    attention_mask_list = [sentence['attention_mask'] for sentence in text_data]
    X_text_input_ids = tf.concat(input_ids_list, axis=0)
    X_text_attention_mask = tf.concat(attention_mask_list, axis=0)

elif textinfo == 'most':

    text_descriptions = text_df['Description'].values.tolist()
    # Truncate each sentence to keep only the first word
    # text_descriptions = [" ".join(sentence.split()[:1]) for sentence in text_descriptions]
    # print(text_descriptions)
    tokenizer.pad_token = tokenizer.eos_token
    text_data = [tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="tf") for
                 sentence in text_descriptions]
    input_ids_list = [sentence['input_ids'] for sentence in text_data]
    attention_mask_list = [sentence['attention_mask'] for sentence in text_data]
    X_text_input_ids = tf.concat(input_ids_list, axis=0)
    X_text_attention_mask = tf.concat(attention_mask_list, axis=0)

X_text_input_ids = np.array(X_text_input_ids)
X_text_attention_mask = np.array(X_text_attention_mask)
# print()
# print(len(text_data))
# print(text_data['input_ids'].shape)
# print(text_data['attention_mask'].shape)

# images = np.array(images)

normalized_targets = np.array(normalized_targets)

# Split the data
(  # X_images_train, X_images_holdout_test,
    X_text_input_ids_train, X_text_input_ids_holdout_test,
    X_text_attention_mask_train, X_text_attention_mask_holdout_test,
    y_train, y_holdout_test) = train_test_split(
    # image,
    X_text_input_ids, X_text_attention_mask, normalized_targets, test_size=Splitting_test_and_hold, random_state=seed
)

# Split the temporary set into the final test and holdout sets for both images and text data
(  # X_images_val, X_images_holdout,
    X_text_input_ids_val, X_text_input_ids_holdout,
    X_text_attention_mask_val, X_text_attention_mask_holdout,
    y_val, y_holdout) = train_test_split(
    # X_images_holdout_test,
    X_text_input_ids_holdout_test, X_text_attention_mask_holdout_test, y_holdout_test,
    test_size=0.50, random_state=seed
)
print(
    f'The length of training, testing, and holdout dataset is:\n images, {len(X_train)} {len(X_val)} {len(X_holdout)} \n text, {len(X_text_input_ids_train)} {len(X_text_attention_mask_val)} {len(X_text_input_ids_holdout)} respectively')

from tensorflow.keras.models import clone_model


# base_model = clone_model(loaded_model.layers[-3:]) #256
# base_model.trainable = False

def build_multimodal_model(hp):
    image_inputs = Input(shape=(186, 186, 3))

    # text_input = Input(shape=(max_text_length,))
    text_input_ids = Input(shape=(max_text_length,), name='input_ids', dtype='int32')  # Input for text input_ids
    text_attention_mask = Input(shape=(max_text_length,), name='attention_mask',
                                dtype='int32')  # Input for text attention_mask

    # x = data_augmentation(image_inputs)

    # x = headless_model(image_inputs, training=False)

    # Hyperparameter grid search
    learning_rate = hp.Choice('learning_rate', values=LR)

    # Text processing branch for RoBERTa embeddings
    text_embeddings = text_model(input_ids=text_input_ids,
                                 attention_mask=text_attention_mask)  # Use the RoBERTa model to obtain embeddings

    # Average pooling along the sequence dimension (axis=1) (or last_hidden_state)
    text_embeddings = AveragePooling1D(pool_size=max_text_length)(text_embeddings['last_hidden_state'])

    # Flatten the averaged embeddings to make them compatible with the rest of your model
    y = Flatten()(text_embeddings)  # 768
    # y = Dense(8, activation='relu')(lstm_layer)

    # Combine image and text features
    y = Dense(512, activation='relu', name='dense_1_text')(y)  # 1024, 512, 256, 1
    y = BatchNormalization(name='bn_1_text')(y)
    y = Dropout(0.1, seed=seed, name='dropout_1_text')(y)
    y = Dense(512, activation='relu', name='dense_2_text')(y)  # 512
    y = BatchNormalization(name='bn_2_text')(y)
    y = Dropout(0.1, seed=seed, name='dropout_2_text')(y)
    y = Dense(256, activation='relu', name='dense_3_text')(y)  # 1
    y = BatchNormalization(name='bn_3_text')(y)
    y = Dropout(0.1, seed=seed, name='dropout_3_text')(y)
    outputs = Dense(numberfeat, activation="linear", name='dense_4_text')(y)  # Drop

    # Create the model
    model = Model(inputs=[image_inputs, text_input_ids, text_attention_mask], outputs=outputs, name="my_text_model")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    print(learning_rate)
    print(model.summary())  # Printing the model summary

    return model


# Define the number of folds (K)
num_folds = 5

# Initialize KFold cross-validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Initialize a list to store the performance results for each fold
fold_losses = []

# Initialize variables to keep track of the best model and its performance
best_model = None
best_loss = float('inf')

for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")

    # Split data into training and validation sets for this fold
    X_train_fold = X_train[train_indices]
    X_text_input_ids_train_fold = X_text_input_ids_train[train_indices]
    X_text_attention_mask_train_fold = X_text_attention_mask_train[train_indices]
    y_train_fold = y_train[train_indices]

    X_val_fold = X_train[val_indices]
    X_text_input_ids_val_fold = X_text_input_ids_train[val_indices]
    X_text_attention_mask_val_fold = X_text_attention_mask_train[val_indices]
    y_val_fold = y_train[val_indices]

    # Set up the tuner for this fold
    tuner = GridSearch(build_multimodal_model,
                       objective='val_loss',
                       max_trials=10,  # Number of hyperparameter combinations to try
                       executions_per_trial=1,
                       directory=f'{expname}/text_only/tuner',
                       project_name=f'KFOLD_{fold}'
                       )

    model_name = f'{expname}/text_only/saved_models/KFold/fold_{fold}'
    mc = ModelCheckpoint(str(model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)  # Saving the best model only
    history = History()

    ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Search for the best hyperparameter configuration for this fold
    tuner.search([X_train_fold, X_text_input_ids_train_fold, X_text_attention_mask_train_fold], y_train_fold,
                 validation_data=([X_val_fold, X_text_input_ids_val_fold, X_text_attention_mask_val_fold], y_val_fold),
                 epochs=epochs1,
                 batch_size=batch,
                 callbacks=[ES, history])

    # Get the best hyperparameters from the search for this fold
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_learning_rate = best_hyperparameters.get('learning_rate')

    print("Best Learning Rate for fold", fold + 1, ":", best_learning_rate)

    # Build the final model with the best hyperparameters for this fold
    final_model = tuner.hypermodel.build(best_hyperparameters)

    print("FINAL MODEL:", final_model.summary())

    # Train the final model on the combined training and validation data for this fold
    F_it_fit = final_model.fit([X_train_fold, X_text_input_ids_train_fold, X_text_attention_mask_train_fold],
                               y_train_fold,
                               epochs=epochs2,
                               batch_size=batch,
                               validation_data=(
                               [X_val_fold, X_text_input_ids_val_fold, X_text_attention_mask_val_fold], y_val_fold),
                               callbacks=[mc, ES, history])

    data_to_save = {'history': F_it_fit.history, 'epoch': F_it_fit.epoch}
    os.makedirs(f'{expname}/text_only/histories', exist_ok=True)
    os.makedirs(f'{expname}/text_only/figs', exist_ok=True)
    with open(
            f'{expname}/text_only/histories/fold_{fold}.pkl',
            'wb') as file:
        pickle.dump(data_to_save, file)

    # Evaluate the performance of the final model on the holdout set for this fold
    final_model = load_model(mc.filepath, custom_objects=custom_objects)
    os.remove(mc.filepath)
    fold_loss = final_model.evaluate([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout], y_holdout,
                                     batch_size=batch)
    print("Fold", fold + 1, "Loss:", fold_loss)

    # Store the performance result for this fold
    fold_losses.append(fold_loss)

    # Check if the current model has the best performance so far
    if fold_loss < best_loss:
        best_loss = fold_loss
        best_model = final_model

    break

# Save the best model with the lowest validation loss across all folds
if best_model is not None:
    best_model.save(
        f'{expname}/text_only/saved_models/best_model.h5')

# Calculate the average performance across all folds
average_loss = np.mean(fold_losses)
print("Average Loss:", average_loss)

"""#Performance of final model"""

print("")
print("PERFORMANCES of the pretrained model, no fine-tuning")
print("")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Step 7: Evaluate the model (optional, after training)
loss = final_model.evaluate([X_val, X_text_input_ids_val, X_text_attention_mask_val], y_val)
# Make predictions on the test data
y_pred = final_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])

# Compute additional evaluation metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Test Loss (MSE):", loss)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Plotting the predicted values from the model
train_prediction = final_model.predict([X_train, X_text_input_ids_train, X_text_attention_mask_train])
test_prediction = final_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])
hold_prediction = final_model.predict([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout])

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Pre-trained model, no fine-tuning yet. Cross validation with {num_folds} folds, batch-size {batch}')
plt.savefig(
    f'{expname}/text_only/figs/best_model.png')
plt.close()

"""#Gradual finetuning"""

from tensorflow.keras.models import load_model

# Load the model with custom_objects

# Load the Keras model from the .h5 file
model_path = f'{expname}/text_only/saved_models/best_model.h5' # here I prefer hardcoding that saving the path again under the same variable ;)
# loaded_model = load_model(model_path)
loaded_model = load_model(model_path, custom_objects=custom_objects)

# FINETUNING
print("")
print("____________")
print("FINE-TUNING")
print("____________")
print("")

ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

"""##stage 1:
unfreezing 50/311 layers of the base model
"""

for layer in loaded_model.layers[-number_ft_layers:]:
    layer.trainable = True
loaded_model.compile(optimizer=Adam(learning_rate=LR_FT), loss='mean_squared_error')
ft_model_name = f'{expname}/text_only/saved_models/finetuned'
ft_mc = ModelCheckpoint(str(ft_model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)  # Saving the best model only
first_stage_FT = loaded_model.fit([X_train, X_text_input_ids_train, X_text_attention_mask_train], y_train,
                                  epochs=epochs3, batch_size=batch, callbacks=[ES, ft_mc],
                                  validation_data=([X_val, X_text_input_ids_val, X_text_attention_mask_val], y_val))

"""###Evaluating performances"""

import plotly.graph_objects as go
import plotly.io as pio

fig_stage1 = go.Figure()
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['loss'],
                                mode='lines',
                                name='Training Loss'))
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['val_loss'],
                                mode='lines',
                                name='Validation Loss'))
fig_stage1.update_layout(title='Fine-Tuning Loss - STAGE 1 ',
                         xaxis_title='Epoch',
                         yaxis_title='Loss')

# Save the plot as an HTML file
html_file_path = f'{expname}/text_only/figs/finetuned_performance.html'
pio.write_html(fig_stage1, file=html_file_path, auto_open=True)

data_to_save = {'history': first_stage_FT.history, 'epoch': first_stage_FT.epoch}
with open(
        f'{expname}/text_only/histories/finetuning.pkl',
        'wb') as file:
    pickle.dump(data_to_save, file)

# EVALUATE FINETUNED
loaded_model = load_model(ft_mc.filepath, custom_objects=custom_objects)
print('!!!THE FINAL RESULTS FOR TEXT ONLY:')

# Plotting the predicted values from the model
train_prediction = loaded_model.predict([X_train, X_text_input_ids_train, X_text_attention_mask_train])
test_prediction = loaded_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])
hold_prediction = loaded_model.predict([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout])

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Fine-tuned model.')
plt.savefig(
    f'{expname}/text_only/figs/finetuned_performance.png')
plt.close()

inps = loaded_model.inputs
outs = loaded_model.layers[-3].output
saved_text_model = Model(inputs=inps, outputs=outs, name='my_text_model_headless')
saved_text_model.trainable = False


# /////////////////////////////////////////////////////
# //////////////////////////////////////////////////////
# /////////////////////////////////////////////////////
# //////////////////////////////////////////////////////
# //////////////////////////////////////////////////////
# /////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////


def filter_sentences(df):
    new_df = pd.DataFrame(columns=['MPID', 'Description'])

    def filter_sentence(sentence):
        words = sentence.split()
        result = [words[0] + '.']

        for i in range(1, len(words)):
            if words[i] == 'bond' and i < len(words) - 1 and words[i + 1] == 'lengths':
                result.append(
                    words[i - 2] + ' ' + words[i - 1] + ' ' + words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' +
                    words[i + 3] + ' ' + words[i + 4])
        return ' '.join(result)

    # Apply the filter to each row in the original DataFrame
    for index, row in df.iterrows():
        modified_sentence = filter_sentence(row['Description'])
        new_row = pd.DataFrame({'MPID': row.name, 'Description': modified_sentence}, index=[0])
        new_df = pd.concat([new_df, new_row])
    return new_df.set_index('MPID')


print('')
print('---------------')
print('ADDING TEXT ON TOP')
print('---------------')
print('')

from tensorflow.keras.layers import Input, AveragePooling1D, Flatten

if textmod_hug == 'roberta-base':
    from transformers import RobertaTokenizer, TFRobertaModel

    tokenizer = RobertaTokenizer.from_pretrained(textmod_hug)
    text_model = TFRobertaModel.from_pretrained(textmod_hug)
    custom_objects = {'TFRobertaModel': TFRobertaModel}

elif textmod_hug == 'gpt2-medium':
    from transformers import GPT2Tokenizer, TFGPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(textmod_hug)
    text_model = TFGPT2Model.from_pretrained(textmod_hug)
    custom_objects = {'TFGPT2Model': TFGPT2Model}

elif textmod_hug == 'gpt2-large':
    from transformers import GPT2Tokenizer, TFGPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(textmod_hug)
    text_model = TFGPT2Model.from_pretrained(textmod_hug)
    custom_objects = {'TFGPT2Model': TFGPT2Model}

elif textmod_hug == 'albert-base-v2':
    from transformers import AlbertTokenizer, TFAlbertModel

    tokenizer = AlbertTokenizer.from_pretrained(textmod_hug)
    text_model = TFAlbertModel.from_pretrained(textmod_hug)
    custom_objects = {'TFAlbertModel': TFAlbertModel}

# Freeze the RoBERTa model
text_model.trainable = False

text_df = pd.read_csv("./data/data_cleaning/Newdata_normalized_zscore_interval/filtered_FCC_descriptions.csv",
                      index_col=0)

# --------------------------
# Extract text descriptions from the DataFrame

if textinfo == '1_word':

    text_descriptions = text_df['Description'].values.tolist()
    # Truncate each sentence to keep only the first word
    text_descriptions = [" ".join(sentence.split()[:1]) for sentence in text_descriptions]
    print(text_descriptions)
    tokenizer.pad_token = tokenizer.eos_token
    text_data = tokenizer(text_descriptions, return_tensors='tf', padding=True)
    X_text_input_ids = text_data['input_ids']
    X_text_attention_mask = text_data['attention_mask']


elif textinfo == 'KEY':

    filtered_df = filter_sentences(text_df)
    text_descriptions = filtered_df['Description'].values.tolist()
    tokenizer.pad_token = tokenizer.eos_token
    text_data = [tokenizer(sentence, padding='max_length', truncation=True, max_length=100, return_tensors="tf") for
                 sentence in text_descriptions]
    input_ids_list = [sentence['input_ids'] for sentence in text_data]
    attention_mask_list = [sentence['attention_mask'] for sentence in text_data]
    X_text_input_ids = tf.concat(input_ids_list, axis=0)
    X_text_attention_mask = tf.concat(attention_mask_list, axis=0)

elif textinfo == 'most':

    text_descriptions = text_df['Description'].values.tolist()
    # Truncate each sentence to keep only the first word
    # text_descriptions = [" ".join(sentence.split()[:1]) for sentence in text_descriptions]
    # print(text_descriptions)
    tokenizer.pad_token = tokenizer.eos_token
    text_data = [tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="tf") for
                 sentence in text_descriptions]
    input_ids_list = [sentence['input_ids'] for sentence in text_data]
    attention_mask_list = [sentence['attention_mask'] for sentence in text_data]
    X_text_input_ids = tf.concat(input_ids_list, axis=0)
    X_text_attention_mask = tf.concat(attention_mask_list, axis=0)

X_text_input_ids = np.array(X_text_input_ids)
X_text_attention_mask = np.array(X_text_attention_mask)
# print()
# print(len(text_data))
# print(text_data['input_ids'].shape)
# print(text_data['attention_mask'].shape)

# images = np.array(images)

normalized_targets = np.array(normalized_targets)

# Split the data
(  # X_images_train, X_images_holdout_test,
    X_text_input_ids_train, X_text_input_ids_holdout_test,
    X_text_attention_mask_train, X_text_attention_mask_holdout_test,
    y_train, y_holdout_test) = train_test_split(
    # image,
    X_text_input_ids, X_text_attention_mask, normalized_targets, test_size=Splitting_test_and_hold, random_state=seed
)

# Split the temporary set into the final test and holdout sets for both images and text data
(  # X_images_val, X_images_holdout,
    X_text_input_ids_val, X_text_input_ids_holdout,
    X_text_attention_mask_val, X_text_attention_mask_holdout,
    y_val, y_holdout) = train_test_split(
    # X_images_holdout_test,
    X_text_input_ids_holdout_test, X_text_attention_mask_holdout_test, y_holdout_test,
    test_size=0.50, random_state=seed
)
print(
    f'The length of training, testing, and holdout dataset is:\n images, {len(X_train)} {len(X_val)} {len(X_holdout)} \n text, {len(X_text_input_ids_train)} {len(X_text_attention_mask_val)} {len(X_text_input_ids_holdout)} respectively')

from tensorflow.keras.models import clone_model


# base_model = clone_model(loaded_model.layers[-3:]) #256
# base_model.trainable = False


def build_multimodal_model(hp):
    image_inputs = Input(shape=(186, 186, 3))

    # text_input = Input(shape=(max_text_length,))
    text_input_ids = Input(shape=(max_text_length,), name='input_ids', dtype='int32')  # Input for text input_ids
    text_attention_mask = Input(shape=(max_text_length,), name='attention_mask',
                                dtype='int32')  # Input for text attention_mask

    # x = data_augmentation(image_inputs)

    x = saved_image_model(image_inputs, training=False)

    # Hyperparameter grid search
    learning_rate = hp.Choice('learning_rate', values=LR)

    # Text processing branch for RoBERTa embeddings
    y = saved_text_model([image_inputs, text_input_ids, text_attention_mask],
                         training=False)  # Use the RoBERTa model to obtain embeddings

    # Flatten the averaged embeddings to make them compatible with the rest of your model
    # y = Dense(8, activation='relu')(lstm_layer)

    # Combine image and text features

    combined_features = Concatenate(name='concat_both')([x, y])  # Concatenate the image and text features
    z = Dense(512, activation='relu', name='dense_1_both')(combined_features)  # 512
    z = BatchNormalization(name='bn_1_both')(z)
    z = Dropout(0.1, seed=seed, name='dropout_1_both')(z)
    z = Dense(256, activation='relu', name='dense_2_both')(z)  # 1
    z = BatchNormalization(name='bn_2_both')(z)
    z = Dropout(0.1, seed=seed, name='dropout_2_both')(z)
    outputs = Dense(numberfeat, activation="linear", name='dense_3_both')(z)  # Drop

    # Create the model
    model = Model(inputs=[image_inputs, text_input_ids, text_attention_mask], outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    print(learning_rate)
    print(model.summary())  # Printing the model summary
    return model


# Define the number of folds (K)
num_folds = 5

# Initialize KFold cross-validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Initialize a list to store the performance results for each fold
fold_losses = []

# Initialize variables to keep track of the best model and its performance
best_model = None
best_loss = float('inf')

# Perform K-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")

    # Split data into training and validation sets for this fold
    X_train_fold = X_train[train_indices]
    X_text_input_ids_train_fold = X_text_input_ids_train[train_indices]
    X_text_attention_mask_train_fold = X_text_attention_mask_train[train_indices]
    y_train_fold = y_train[train_indices]

    X_val_fold = X_train[val_indices]
    X_text_input_ids_val_fold = X_text_input_ids_train[val_indices]
    X_text_attention_mask_val_fold = X_text_attention_mask_train[val_indices]
    y_val_fold = y_train[val_indices]

    # Set up the tuner for this fold
    tuner = GridSearch(build_multimodal_model,
                       objective='val_loss',
                       max_trials=10,  # Number of hyperparameter combinations to try
                       executions_per_trial=1,
                       directory=f'{expname}/both/tuner',
                       project_name=f'KFOLD_{fold}'
                       )


    model_name = f'{expname}/both/saved_models/KFold/fold_{fold}'
    mc = ModelCheckpoint(str(model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)  # Saving the best model only
    history = History()

    ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Search for the best hyperparameter configuration for this fold
    tuner.search([X_train_fold, X_text_input_ids_train_fold, X_text_attention_mask_train_fold], y_train_fold,
                 validation_data=([X_val_fold, X_text_input_ids_val_fold, X_text_attention_mask_val_fold], y_val_fold),
                 epochs=epochs1,
                 batch_size=batch,
                 callbacks=[ES, history])


    # Get the best hyperparameters from the search for this fold
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_learning_rate = best_hyperparameters.get('learning_rate')

    print("Best Learning Rate for fold", fold + 1, ":", best_learning_rate)

    # Build the final model with the best hyperparameters for this fold
    final_model = tuner.hypermodel.build(best_hyperparameters)

    print("FINAL MODEL:", final_model.summary())

    # Train the final model on the combined training and validation data for this fold
    F_it_fit = final_model.fit([X_train_fold, X_text_input_ids_train_fold, X_text_attention_mask_train_fold],
                               y_train_fold,
                               epochs=epochs2,
                               batch_size=batch,
                               validation_data=(
                               [X_val_fold, X_text_input_ids_val_fold, X_text_attention_mask_val_fold], y_val_fold),
                               callbacks=[mc, ES, history])

    data_to_save = {'history': F_it_fit.history, 'epoch': F_it_fit.epoch}
    os.makedirs(f'{expname}/both/histories', exist_ok=True)
    os.makedirs(f'{expname}/both/figs', exist_ok=True)
    with open(
            f'{expname}/both/histories/fold_{fold}.pkl',
            'wb') as file:
        pickle.dump(data_to_save, file)

    # Evaluate the performance of the final model on the holdout set for this fold
    final_model = load_model(mc.filepath, custom_objects=custom_objects)
    os.remove(mc.filepath)
    fold_loss = final_model.evaluate([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout], y_holdout,
                                     batch_size=batch)
    print("Fold", fold + 1, "Loss:", fold_loss)

    # Store the performance result for this fold
    fold_losses.append(fold_loss)

    # Check if the current model has the best performance so far
    if fold_loss < best_loss:
        best_loss = fold_loss
        best_model = final_model

    break

# Save the best model with the lowest validation loss across all folds
if best_model is not None:
    best_model.save(
        f'{expname}/both/saved_models/best_model.h5')

# Calculate the average performance across all folds
average_loss = np.mean(fold_losses)
print("Average Loss:", average_loss)

"""#Performance of final model"""

print("")
print("PERFORMANCES of the pretrained model, no fine-tuning")
print("")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Step 7: Evaluate the model (optional, after training)
loss = final_model.evaluate([X_val, X_text_input_ids_val, X_text_attention_mask_val], y_val)
# Make predictions on the test data
y_pred = final_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])

# Compute additional evaluation metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Test Loss (MSE):", loss)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Plotting the predicted values from the model
train_prediction = final_model.predict([X_train, X_text_input_ids_train, X_text_attention_mask_train])
test_prediction = final_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])
hold_prediction = final_model.predict([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout])

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'!! Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Pre-trained model, no fine-tuning yet. Cross validation with {num_folds} folds, batch-size {batch}')
plt.savefig(
    f'{expname}/both/figs/best_model.png')
plt.close()

"""#Gradual finetuning"""

from tensorflow.keras.models import load_model

# Load the model with custom_objects

# Load the Keras model from the .h5 file
model_path = f'{expname}/both/saved_models/best_model.h5'
# loaded_model = load_model(model_path)
loaded_model = load_model(model_path, custom_objects=custom_objects)

# FINETUNING
print("")
print("____________")
print("FINE-TUNING")
print("____________")
print("")

ES = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

"""##stage 1:
unfreezing 50/311 layers of the base model
"""

# loaded model is input, inception, batchnorm, dropout, linear
# we unfreeze last 50 of layers, and we have 7 layers

for layer in loaded_model.layers[-number_ft_layers:]:
    layer.trainable = True
loaded_model.compile(optimizer=Adam(learning_rate=LR_FT), loss='mean_squared_error')
ft_model_name = f'{expname}/both/saved_models/finetuned'
ft_mc = ModelCheckpoint(str(ft_model_name) + '.hdf5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)  # Saving the best model only
first_stage_FT = loaded_model.fit([X_train, X_text_input_ids_train, X_text_attention_mask_train], y_train,
                                  epochs=epochs3, batch_size=batch, callbacks=[ES, ft_mc],
                                  validation_data=([X_val, X_text_input_ids_val, X_text_attention_mask_val], y_val))

"""###Evaluating performances"""

import plotly.graph_objects as go
import plotly.io as pio

fig_stage1 = go.Figure()
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['loss'],
                                mode='lines',
                                name='Training Loss'))
fig_stage1.add_trace(go.Scatter(x=first_stage_FT.epoch,
                                y=first_stage_FT.history['val_loss'],
                                mode='lines',
                                name='Validation Loss'))
fig_stage1.update_layout(title='Fine-Tuning Loss - STAGE 1 ',
                         xaxis_title='Epoch',
                         yaxis_title='Loss')

# Save the plot as an HTML file
html_file_path = f'{expname}/both/figs/finetuned_performance.html'
pio.write_html(fig_stage1, file=html_file_path, auto_open=True)

data_to_save = {'history': first_stage_FT.history, 'epoch': first_stage_FT.epoch}
with open(
        f'{expname}/both/histories/finetuning.pkl',
        'wb') as file:
    pickle.dump(data_to_save, file)

# EVALUATE FINETUNED
del loaded_model
loaded_model = load_model(ft_mc.filepath, custom_objects=custom_objects)
print('!!!THE FINAL RESULTS FOR IMAGE AND TEXT AFTER FT :')

# Plotting the predicted values from the model
train_prediction = loaded_model.predict([X_train, X_text_input_ids_train, X_text_attention_mask_train])
test_prediction = loaded_model.predict([X_val, X_text_input_ids_val, X_text_attention_mask_val])
hold_prediction = loaded_model.predict([X_holdout, X_text_input_ids_holdout, X_text_attention_mask_holdout])

# RMSE and R2
train_rmse = round(math.sqrt(mean_squared_error(y_train, train_prediction)), 2)
test_rmse = round(math.sqrt(mean_squared_error(y_val, test_prediction)), 2)
hold_rmse = round(math.sqrt(mean_squared_error(y_holdout, hold_prediction)), 2)

train_r2 = round(r2_score(y_train, train_prediction), 2)
test_r2 = round(r2_score(y_val, test_prediction), 2)
hold_r2 = round(r2_score(y_holdout, hold_prediction), 2)

print(
    f'Training rmse: {train_rmse}  and Testing rmse: {test_rmse} and Hold rmse: {hold_rmse}\nTraining R2: {train_r2}  and Testing R2: {test_r2} and Hold R2: {hold_r2}')

# Making scatter plots
import matplotlib.pyplot as plt

plt.plot([np.min(normalized_targets), np.max(normalized_targets)],
         [np.min(normalized_targets), np.max(normalized_targets)], '--')
plt.scatter(y_train, train_prediction, label=f'Training data: rmse = {train_rmse}, R2 = {train_r2}')
plt.scatter(y_val, test_prediction, label=f'Testing data: rmse = {test_rmse}, R2 = {test_r2} ')
plt.scatter(y_holdout, hold_prediction, label=f'Holdout data: rmse = {hold_rmse}, R2 = {hold_r2} ')
plt.xlabel('True ' + feat)
plt.ylabel('Pedicted ' + feat)
plt.legend(loc=2)
plt.xlim(np.min(normalized_targets), np.max(normalized_targets))
plt.ylim(np.min(normalized_targets), np.max(normalized_targets))
# plt.title(f'Fine-tuned model.')
plt.savefig(
    f'{expname}/both/figs/finetuned_performance.png')
plt.close()