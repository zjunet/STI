# Social-Aware Time Series Imputation

This project implements the social-aware time series imputation method proposed in [1], which is an imputation algorithm for time series data in the social network.

## Testing

This project is implemented in Python 3.6

### Dependency: 

- Python 3.6. Version 3.6.4 has been tested.
- PyTorch. Version 0.4.0 has been tested. Note that the GPU support is encouraged as it greatly boosts training efficiency.
- Other Python modules. Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip ```pip install -r requirements.txt```

### Testing the Project:

``` 
python impute.py -f data/data.npy -n data/network.pkl
```

## Usage

Given an array of users' time series data and the social relationship among these users, this program can be used to replace the missing value in these time series data with reasonable values.

### Input Format

The input files are expected to be two parts: 

(1) data file: a numpy array (.npy) file which contains users' data shaped N * L * D, where N denotes the number of users, L denotes the sequence length and D denotes the number of channels at each time stamp. **The missing data should be marked as -1.**

(2) social network file: a pickle file which contains the social network information formated as the adjacent list:
```
[[node0 's neighbors], [node1's neighbors],..., nodeN's neighbors]
e.g. [[1], [0,2,3], [1,3,4], [2]]
```
each node index is corresponding to the index of the row in the data array in (1)
See an example in the ```data``` directory.
### Output Format
The program outputs to a file named ```imputed_data.npy``` which contains the data after imputation, i.e., the missing elements are replaced by reasonable values.
### Main Script
The help of main script can be obtained by excuting command:
```
usage: impute.py [-h] [-f DATA_FILE] [-n SOCIAL_NETWORK] [-o OUTPUT_FILE]
                 [-m MISSING_LABEL] [-s HIDDEN_SIZE] [-b BATCH_SIZE]
                 [-k DIM_MEMORY] [-e NUM_EPOCH] [-l LEARNING_RATE]
                 [-w WEIGHT_DECAY] [-d DROPOUT] [-r DECODER_LEARNING_RATIO]
                 [--log]

optional arguments:
  -h, --help            show this help message and exit
  -f DATA_FILE, --data_file DATA_FILE
                        path of input file
  -n SOCIAL_NETWORK, --social_network SOCIAL_NETWORK
                        path of network file
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        path of output file
  -m MISSING_LABEL, --missing_label MISSING_LABEL
  -s HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        size of hidden feature in LSTM
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -k DIM_MEMORY, --dim_memory DIM_MEMORY
                        dimension of memory matrix
  -e NUM_EPOCH, --num_epoch NUM_EPOCH
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  -d DROPOUT, --dropout DROPOUT
  -r DECODER_LEARNING_RATIO, --decoder_learning_ratio DECODER_LEARNING_RATIO
  --log                 output log information
```
## Reference
[1] Zongtao, L; Yang, Y; Wei, H; Zhongyi, T; Ning, L and Fei, W, 2019, How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation, In WWW, 2019 
```
 @inproceedings{liu2019imputation, 
    title={How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation},
    author={Zongtao Liu and Yang Yang and Wei Huang and Zhongyi Tang and Ning Li and Fei Wu},
    booktitle={Proceedings of WWW},
    year={2019}
    }
```

