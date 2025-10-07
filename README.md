# transformers_tvm


This repository implemented TVM auto-scheduler to optimize transformer encoder and decoder models.
It allows user to search for optimal scheduler for both linear (GEMM) and non-linear (GELU, LayerNorm, Softmax) operations in transformers.


### Search and validate the best GEMM scheduler
To search for the best scheduler for GEMM operations in transformers, run:

`python auto_scheduler_GEMM.py 1 5 8 2 5 gpu`
`python auto_scheduler_GEMM.py 1 5 8 0 5 gpu`

These two run menas:
1 threads, model 5, data type 8 (int32, int32, int8), 2 iterations of autoscheduler search, casting method 5 and run in GPU
1 threads, model 5, data type 8 (int32, int32, int8), 0 iterations of autoscheduler search (just validate), casting method 5 and run in GPU



### Search and validate the best non-linear scheduler
To search and validate the best schedule for non-linear operations in transformers (GELU, LayerNorm, Softmax), run:  

`python nonLinner.py`

You can configure the model and host hardware in the script.
You can find the log file of this run in `logs/` and searched configurations in `json/`.



