import matplotlib
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['figure.figsize'] = [12, 8]
np.set_printoptions(precision=3, suppress=True)

# Accuracy achieved 92.57
# Setting global parameters of the code and hyperparameters of the mlp model
seed = 22
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
hidden_neurons = 64
number_of_hidden_layers = 1
batch_size = 128
learning_rate = 0.00001
max_epochs = 100
train_val_split = 0.5  # percentage of data the total data separated for training. This parameter means that 50% of the data will be reserved for training, 30% for validation and 20% for testing
activation_func = 'relu'
padding_token = "<pad>"


# Function to plot the results
def plot_result(item):
    if item == 'loss':
        item_pt_br = 'perda'
        plt.plot(history.history[item], label=item_pt_br)
        plt.plot(history.history["val_" + item], label=item_pt_br + " de validação")
    elif item == 'binary_accuracy':
        item_pt_br = 'acurácia binária'
        plt.plot(history.history[item], label=item_pt_br)
        plt.plot(history.history["val_" + item], label="validação de " + item_pt_br)
    plt.xlabel("Épocas")
    plt.ylabel(item_pt_br)
    plt.title("{} da Validação e Treino através das épocas".format(item_pt_br), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


# Function that counts and returns the genres in the dataset
def get_number_of_genres(data):
    number_genres = 0
    list_of_genres = []
    for i in range(len(data)):
        string = ''.join(data.iloc[i])
        new_string = string.replace('[', '')
        new_string = new_string.replace(']', '')
        new_string = new_string.replace("'", '')
        new_string = new_string.replace("'", '')
        new_string = new_string.split(", ")
        for j in range(len(new_string)):
            # if new_string[j] != '':
            if new_string[j] not in list_of_genres:
                list_of_genres.append(new_string[j])
                number_genres += 1
    return np.array(list_of_genres), number_genres


# Function that preprocess the target data in a format that is better suited to be added in a tensorflow dataset
def preprocess_target_data(target):
    targets = []
    for i in range(len(target)):
        string = ''.join(target.iloc[i])
        new_strings = string.replace('[', '')
        new_strings = new_strings.replace(']', '')
        new_strings = new_strings.replace("'", '')
        new_strings = new_strings.replace("'", '')
        new_strings = new_strings.split(", ")
        targets.append(new_strings)
    return np.array(targets, dtype=object)


# Function that creates a tensorflow dataset with the given data and target
def make_dataset(data, target, is_train=True):
    labels = tf.ragged.constant(target)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices((data, label_binarized))
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


# Function to vectorize the data so that the mlp model can understand it
def vectorize_text(vector_text, vector_label):
    vector_text = tf.expand_dims(vector_text, -1)
    return vectorize_layer(vector_text), vector_label


# Function to normalize the data
def normalize(vector_data, vector_label):
    return normalize_layer(vector_data), vector_label


# Collecting the data
descriptions = pd.read_csv('descriptions2.csv')
genres = pd.read_csv('genres2.csv')
_ = genres.pop('Unnamed: 0')
_ = descriptions.pop('Unnamed: 0')
df = pd.concat([descriptions, genres], axis=1)
column_indices = {name: i for i, name in enumerate(df.columns)}

# Splitting data into train, validation and testing dataframes
n = len(df)
train_df = df[0:int(n * train_val_split)]
val_df = df[int(n * train_val_split):int(n * 0.8)]
test_df = df[int(n * 0.8):]

# Creating tensorflow datasets from training, validation and testing
# Separate the data from the targets
target_train_data = train_df.pop('genres')
train_data = train_df.pop('description')
train_data = train_data.astype(str)
val_target = val_df.pop('genres')
val_data = val_df.pop('description')
val_data = val_data.astype(str)
target_test_data = test_df.pop('genres')
test_data = test_df.pop('description')
test_data = test_data.astype(str)

# Convert the data into tensors
tensor_train_data = tf.convert_to_tensor(train_data)
tensor_val_data = tf.convert_to_tensor(val_data)
tensor_test_data = tf.convert_to_tensor(test_data)

# Get the number of classes and their names
list_genres, number_of_genres = get_number_of_genres(genres)

print('1: Train new model\n2: Load trained model')
choice = int(input())

if choice == 1:
    # Preprocessing the target data to later on turn them into ragged tensors (tensors with different shapes) since each data point can be part of more than one class
    target_train_data = preprocess_target_data(target_train_data)
    val_target = preprocess_target_data(val_target)
    target_test_data = preprocess_target_data(target_test_data)

    arrays_train = [np.array(x) for x in target_train_data]
    arrays_val = [np.array(x) for x in val_target]
    arrays_test = [np.array(x) for x in target_test_data]

    # Preparing a vocabulary so that te model can output in a multi one hot encode form
    terms = tf.ragged.constant(list_genres)
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(terms)

    # Turning the data and targets into single datasets
    raw_train_ds = make_dataset(tensor_train_data, arrays_train, is_train=True)
    raw_val_ds = make_dataset(tensor_val_data, arrays_val, is_train=False)
    raw_test_ds = make_dataset(tensor_test_data, arrays_test, is_train=False)

    # Setting a separate vocabulary for the input so that we can vectorize the data. In other words, process the data in a way that our mlp model can understand
    with open('words.txt', 'r') as file:
        words = file.read()
    df = pd.DataFrame([x.split() for x in words.split('\n')], columns=["words"])
    vocab = df.pop('words')
    vocab = vocab.astype(str)
    vocabulary = set()
    vocab.str.lower().str.split().apply(vocabulary.update)
    vocabulary_size = len(vocabulary)

    # Creating and training the model
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        ngrams=2,
        output_mode='tf_idf'
    )
    with tf.device("/CPU:0"):
        vectorize_layer.adapt(raw_train_ds.map(lambda text, label: text))

    normalize_layer = tf.keras.layers.Normalization(axis=None)
    with tf.device("/CPU:0"):
        normalize_layer.adapt(vectorize_layer(tensor_train_data))

    callback0 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=5)

    model = tf.keras.Sequential([])
    model.add(vectorize_layer)
    model.add(normalize_layer)
    model.add(tf.keras.Input(shape=(vocabulary_size,)))

    for layer in range(number_of_hidden_layers):
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(hidden_neurons, activation=activation_func,
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(0.001),
                                        bias_regularizer=tf.keras.regularizers.l1_l2(0.001)))

    model.add(tf.keras.layers.Dense(lookup.vocabulary_size(), activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=tf.keras.metrics.binary_accuracy)
    history = model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=max_epochs, batch_size=batch_size, callbacks=[callback0, callback1], use_multiprocessing=True, verbose=1)

    model.save('DescriptionClassifier')

    # Plotting the training results
    plot_result("loss")
    plot_result("binary_accuracy")

    # Printing the accuracy on the test set
    _, binary_acc = model.evaluate(raw_test_ds)
    print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")

else:
    model = tf.keras.models.load_model('DescriptionClassifier')
    while True:
        print("Insert the description you wish to test:")
        description = str(input())
        description_tensor = tf.convert_to_tensor(description)
        result = model.predict_step(description_tensor)
        result = result.numpy()
        indexes = []
        for g in range(result[0].__len__()):
            if result[0][g] > 0.5:
                indexes.append(g)
        print(list_genres[indexes])
        indexes.clear()
