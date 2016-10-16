A Multi-Layer Perceptron implementation with customizable hidden layer & units, error limit, learning rate, and training data.
It consist of a backpropagation algorithm implementation with linear combinator and non-linear sigmoid activation function.

<h1>How To Use :</h1>
<b>1.Customizing the training data</b>
```
	static double [][][] trainingData = {
            {{in1,in2, ...}, {out1,out2,out3,out4,out5,out6,out7,out8,         ...}},
            {{in3,in4, ...}, {out9,out10,out11,out12,out13,out14,out15,out16 , ...}},
            {{in5,in6, ...}, {out17,out18,out19,out20,out21,out22,out23,out24, ...}},
            ...
    };
```
Array of first column refers to input and array of second column output.
Creation of new units and separated by commas.
Output range can only be from 0-1. To output a range larger than 1 or lower than 0, change the activation function along with its derivative in the lines of codes
or multiply each output by the desired power of 10.

<b>2.Modifying amount of hidden layers & hidden layer units</b>
```
static int[] hiddenLayerUnits = {18,15};
```
The example above refers there are 2 hidden layers each with 18 neurons and 15 neurons respectively.
To create a new layer and its number of hidden units, type values of integer types and separate by commas.

<b>2.Modifying amount of learning rate and error limit</b>
```
	static double learningRate= 0.1;
	static double errorLimit = 0.01;
```




