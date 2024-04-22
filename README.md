## Dataset

1. Download raw dataset from [https://github.com/xing-hu/DeepCom]
2. Parse them with parser.jar



## tree dataset processing
Run `java -jar parser.jar -f [filename] -d [dirname]`.

#### example
`java -jar parser.jar -f valid.json -d valid`


## Usage
1. Prepare tree-structured data with `dataset.py`
    - Run `$ python dataset.py [dir]`
2. Train and evaluate model with `train.py`
    - Run `$ python main.py`

## requirement
Java 1.8,
pytorch 1.11.0,
tensorflow-gpu==1.10.1
