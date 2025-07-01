import torch

# Задание 1: Создание и манипуляции с тензорами

def create_tensors():
    # 1.1 Создание тензоров

    tensor1 = torch.rand(3, 4)
    tensor2 = torch.zeros(2, 3, 4)
    tensor3 = torch.ones(5, 5)
    tensor4 = torch.arange(16).reshape(4, 4)

    print("1.1 Создание тензоров")
    print("Тензор 3x4 (случайные числа 0-1):\n", tensor1)
    print("Тензор 2x3x4 (0):\n", tensor2)
    print("Тензор 5x5 (1):\n", tensor3)
    print("Тензор 4x4 (от 0 до 15)+ reshape:\n", tensor4)



def tensor_operations():
    # 1.2 Операции с тензорами

    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    A_T = A.T                          # Транспонирование тензора A
    matmul_result = torch.matmul(A, B) # Матричное умножение A и B
    B_T = B.T
    elementwise = A * B_T              # Поэлементное умножение A и транспонированного B
    total_sum = A.sum()                # Сумма всех элементов тензора A

    print("1.2 Операции с тензорами")
    print("A:\n", A)
    print("B:\n", B)
    print("A^T:\n", A_T)
    print("A @ B:\n", matmul_result)
    print("A * B^T (поэлементно):\n", elementwise)
    print("Сумма элементов A:", total_sum.item())



def tensor_indexing():
    # 1.3 Индексация и срезы

    t = torch.arange(125).reshape(5, 5, 5)

    first_row = t[0, 0, :]
    all_first_row = t[:, 0, :]
    last_column = t[:, :, -1]
    center_submatrix = t[2, 1:3, 1:3]
    even_elements = t[::2, ::2, ::2]

    print("\n1.3 Индексация и срезы")
    print("Изначальный тензор:\n", t)
    print("Первая строка первого 'слоя':\n", first_row)
    print("Все первые строки:\n", all_first_row)
    print("Последний столбец:\n", last_column)
    print("Подматрица 2x2 из центра:\n", center_submatrix)
    print("Элементы с чётными индексами:\n", even_elements)



def tensor_reshaping():
    # 1.4 Работа с формами
    
    t = torch.arange(24)

    shape_2x12 = t.reshape(2, 12)
    shape_3x8 = t.reshape(3, 8)
    shape_4x6 = t.reshape(4, 6)
    shape_2x3x4 = t.reshape(2, 3, 4)
    shape_2x2x2x3 = t.reshape(2, 2, 2, 3)

    print("\n1.4 Работа с формами")
    print("Исходный тензор:\n", t)
    print("Форма 2x12:\n", shape_2x12)
    print("Форма 3x8:\n", shape_3x8)
    print("Форма 4x6:\n", shape_4x6)
    print("Форма 2x3x4:\n", shape_2x3x4)
    print("Форма 2x2x2x3:\n", shape_2x2x2x3)



if __name__ == "__main__":
    create_tensors()
    tensor_operations()
    tensor_indexing()
    tensor_reshaping()
