# 446-project
captioning
Our code is not neat and we can't upload all the data files so figuring out how to run our project is hard.

The following instruction is only about how to use MCCOCO dataset.
To run the code, you have to 

1.Clone neural talk2, follow the instruction to prepare COCO dataset.

2.clone https://github.com/woodrush/neural-art-tf.git and use his code to get the binary file

3.change the keys of the dic read from binary file, add vgg_ in front of each of them.

4.Change json_path and h5_path in preprocess_data.ipynb to the right routine

5.Run preprocess_data.ipynb (Due to bad management, in preprocess_data.ipynb, many code are my experiments 
to explore tensorflow, even I don't know which part of code is truely useful)

5.change json_path and h5_path in main.ipynb to the right routine

6.run main.ipynb
 
