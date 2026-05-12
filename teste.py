import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CARREGAR MODELO
# ==========================

model = tf.keras.models.load_model(
    "modelo_gatos_cachorros.h5"
)

# ==========================
# CAMINHO DA IMAGEM
# ==========================

img_path = "teste.jpeg"

# ==========================
# MOSTRAR IMAGEM
# ==========================

img_show = plt.imread(img_path)

plt.imshow(img_show)

plt.title("Imagem enviada")

plt.axis("off")

plt.show()

# ==========================
# PROCESSAMENTO
# ==========================

img = image.load_img(
    img_path,
    target_size=(150,150)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0

# ==========================
# PREVISÃO
# ==========================

prediction = model.predict(img_array)

probabilidade = prediction[0][0]

# ==========================
# RESULTADO
# ==========================

print("\n====================")

if probabilidade > 0.5:

    porcentagem = probabilidade * 100

    print(f"Resultado: Cachorro")
    print(f"Confiança: {porcentagem:.2f}%")

else:

    porcentagem = (1 - probabilidade) * 100

    print(f"Resultado: Gato")
    print(f"Confiança: {porcentagem:.2f}%")

print("====================")