import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import random

# Загрузка модели
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Загрузка тестовых данных
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255
x_test = x_test.reshape(-1, 28, 28)

# Названия классов
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Функция предсказания и отображения изображения
def predict_image(index):
    img = x_test[index]          # Берем изображение по индексу
    img_expanded = img.reshape(1, 28, 28)  # Добавляем размерность для модели
    predictions = model.predict(img_expanded)  # Делаем предсказание
    predicted_class = predictions[0].argmax()  # Получаем индекс класса

    plt.imshow(img, cmap='gray')
    plt.title(f"Предсказано: {class_names[predicted_class]}\n"
              f"Истинный класс: {class_names[y_test[index]]}")
    plt.axis("off")
    plt.show()

# Пример предсказания для случайного изображения
random_index = random.randint(0, len(x_test) - 1)
predict_image(random_index)

