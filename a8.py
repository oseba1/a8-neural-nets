from neural import *

xor_training_data = [
    ([1,1], [0]),
    ([0,1], [1]),
    ([0,1], [1]),
    ([0,0], [0])
]


xornet = NeuralNet(2, 1, 1)

xornet.train(xor_training_data)

print(xornet.test_with_expected(xor_training_data))
print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
print("\n\nTraining XOR\n\n")



xor_training_data = [
    ([1, 1], [0]), 
    ([1, 0], [1]), 
    ([0, 1], [1]), 
    ([0, 0], [0])
    ]

xorn = NeuralNet(2, 2, 1)

xorn.train(xor_training_data, learning_rate = 0.7, iters =10000, print_interval = 1000)
print(xorn.test_with_expected(xor_training_data))



print("\n\nTraining XOR\n\n")
xor_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1],[1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4],[1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3],[1]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0])
]
xorn = NeuralNet(5,2,1)

xorn.train(xor_training_data, iters=1000)
print(xorn.test_with_expected(xor_training_data))