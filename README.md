# SenValEva

A repository for various SENsor VALue EVAluation. To achieve that Machine Learning techniques are used.
The applications want to provide basic implementations on how to use Machine Learning to extract information out of simple sensor values.

Used Frameworks:
* Keras for Deep Neural Networks
* Scikit-learn for other types of ML like Random Forest, Linear Regression, SVM, etc.

Train Data:
* Currently the train and test values, which are used, have been generated from a capacitive sensor, which is used for human-robot-collaboration.
* The Train Data has 5 rows
* The first 2 rows are sensor values (inputs) and the following 3 are distance values of an object relative to the sensors (outputs)
* There is one data-set for training and one for testing