# Walking, Running or Nordic Walking - this is the question
<h1 align="center">
  <br>
  <img src="https://cdn2.vectorstock.com/i/1000x1000/53/36/nordic-walking-and-jogging-vector-925336.jpg" alt="Walk" width="1000" height="780">
</h1>
My goal in this project is to classify if a given 4 second sample represent Walking, Running or Nordic Walking

## Install
Clone repository, then:
```
$ pip install -r requirements.txt
```
I'm working with python 3.9.7. </br>
The list of requirements.txt is long for some reason. You can use my Conda env as well, using the "environment.yml" file.
Clone repository, then:
```
$ conda env create -f environment.yml
```

## Usage
Predict function from solution excpect to get a dataframe with the same structure from here: [Data](https://drive.google.com/file/d/1rvLrLS0W4LqBtbJgMtyfFsSEEeEEEoDa/view). it will return a [#num_samples, #classes] numpy array with probabilites to each class.

## Report
My full report can be found here: [Report](https://colab.research.google.com/drive/1PqlzmYjh5yLe6AExqtwQBd1vPus9SbJP?usp=sharing). The report includs: installation, imports, eda, preprocess, model training and results evaluation.

## Important note
Along the report there are some insights I wrote. All the consideration I took (like data cleaning, etc.) was chosen due to time consideration, however many more could be done. </br>
Final note: my code works a little bit slow, i'm sure that there are faster ways, but i chose the more easy one so my programming will be quick.

## Help
For help, feel free to contact me at: yardenrotem2@gmail.com

