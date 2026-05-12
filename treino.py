import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAÇÕES
# ==========================

img_size = (150, 150)
batch_size = 32

# ==========================
# CARREGAMENTO DO DATASET
# ==========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# ==========================
# NOMES DAS CLASSES
# ==========================

class_names = train_dataset.class_names

print("\nClasses encontradas:")
print(class_names)

# ==========================
# IGNORAR IMAGENS CORROMPIDAS
# ==========================

train_dataset = train_dataset.ignore_errors()

val_dataset = val_dataset.ignore_errors()

# ==========================
# NORMALIZAÇÃO
# ==========================

normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

val_dataset = val_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# ==========================
# OTIMIZAÇÃO
# ==========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ==========================
# MODELO CNN
# ==========================

model = models.Sequential([

    layers.Input(shape=(150, 150, 3)),

    layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation='sigmoid'
    )
])

# ==========================
# COMPILAÇÃO
# ==========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================
# RESUMO DO MODELO
# ==========================

print("\nResumo do modelo:")
model.summary()

# ==========================
# TREINAMENTO
# ==========================

print("\nIniciando treinamento...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# ==========================
# SALVAR MODELO
# ==========================

model.save("modelo_gatos_cachorros.h5")

print("\nModelo salvo com sucesso!")

# ==========================
# GRÁFICO
# ==========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Acurácia do Modelo')

plt.ylabel('Acurácia')

plt.xlabel('Época')

plt.legend(['Treino', 'Validação'])

plt.show()