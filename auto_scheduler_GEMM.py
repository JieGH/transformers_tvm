import tvm
from tvm import te
import numpy as np
import time
import logging
from tvm import te, auto_scheduler
import os
import datetime
import sys
import traceback

# define a generic gemm function with auto scheduler decorator
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def gemm_native_mac_autoS(M, N, K, typeA, typeB, typeC, cast_method):
    a = te.placeholder((M, K), name="a", dtype=typeA)
    b = te.placeholder((K, N), name="b", dtype=typeB)
    k = te.reduce_axis((0, K), "k")

    if typeA == typeB == typeC:
        c = te.compute(
            (M, N),
            lambda i, j: te.sum(a[i, k] * b[k, j], axis=k),
            name="C"
        )
    elif cast_method == 5:
        c_b4cast = te.compute(
            (M, N),
            lambda i, j: te.sum(
                a[i, k] * b[k, j], axis=k
            ),
            name="c_b4cast"
        )
        c = te.compute(
            (M, N),
            lambda i, j: c_b4cast[i, j].astype(typeC),
            name="C"
        )
    return [a, b, c]

def autoS_valid(M, N, K, Gtarget, dev, tvm_json, cast_method, gpu_or_cpu,
                A_np, B_np, return_matrix, dataType, name, list_gflops_autoS):     # validate run from the best schedule
    task = tvm.auto_scheduler.SearchTask(func=gemm_native_mac_autoS, args=(M, N, K, typeA, typeB, typeC, cast_method), target=Gtarget)
    sch, args = task.apply_best(tvm_json)
    func_autoS = tvm.build(sch, args, Gtarget)

    a_tvm = tvm.nd.array(A_np, device=dev)
    b_tvm = tvm.nd.array(B_np, device=dev)
    c_tvm = tvm.nd.empty(return_matrix.shape, device=dev, dtype=typeC)
 
    func_autoS(a_tvm, b_tvm, c_tvm)

    evaluator = func_autoS.time_evaluator(func_autoS.entry_name, dev, min_repeat_ms=10000)
    tt = np.median(evaluator(a_tvm, b_tvm, c_tvm).results)
    gflops = ((2.0 * M * N * K) / (1e9 * 1.0)) / tt
    print("autoS_valid: M,N,K:", M, N, K)
    print(f"Auto scheduler valid Time taken: {tt} s, GFLOPS: {gflops:.3f}")

    list_gflops_autoS.append(gflops)
    print(" End of search %f", gflops)

def autoS_search(M, N, K, Gtarget, dev, tvm_json, tvm_iter, cast_method, typeA, typeB, typeC, gpu_or_cpu,
                 A_np, B_np, return_matrix, dataType, name):     # search run

    task = tvm.auto_scheduler.SearchTask(func=gemm_native_mac_autoS, args=(M, N, K, typeA, typeB, typeC, cast_method), target=Gtarget)

    builder = auto_scheduler.LocalBuilder(timeout=60)   # allow up to 60s for compilation
    runner = auto_scheduler.LocalRunner(
        repeat=5,
        min_repeat_ms=3000,
        timeout=60    # allow up to 120s for execution
    )

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tvm_iter,
        measure_callbacks=[auto_scheduler.RecordToFile(tvm_json)],
        verbose=2,
        builder=builder,
        runner=runner
    )
        
    task.tune(tune_option)
    sch, args = task.apply_best(tvm_json)

    func_autoS = tvm.build(sch, args, Gtarget)

    a_tvm = tvm.nd.array(A_np, device=dev)
    b_tvm = tvm.nd.array(B_np, device=dev)
    c_tvm = tvm.nd.empty(return_matrix.shape, device=dev, dtype = typeC )
    
    func_autoS(a_tvm, b_tvm, c_tvm)

    evaluator = func_autoS.time_evaluator(func_autoS.entry_name, dev, min_repeat_ms=10000)

    tt = np.median(evaluator(a_tvm, b_tvm, c_tvm).results)
    gflops = ((2.0 * M * N * K) / (1e9 * 1.0)) / tt
    print(f"autoScheduler,Time taken: {tt} ms, GFLOPS: {gflops:.3f}")
    print("autoS: M,N,K:", M, N, K)
    print("autoS time: ", tt)

    list_gflops_autoS.append(gflops)

def exit_sequence():
    tb = traceback.format_stack()
    logging.info("Traceback (most recent call last):\n" + "".join(tb))
    logging.info("      == Warning, exit triggered, check log ==")
    os.system(f"cat {filenameS}")
    exit(0)

def set_hardware_target(gpu_or_cpu, Gtarget=None, dev=None):
    if gpu_or_cpu == "gpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        Gtarget = tvm.target.cuda(model='a100', arch="sm_80")
        dev = tvm.device(Gtarget.kind.name, 0)
    elif gpu_or_cpu == "cpu": 
        dev = tvm.cpu()
        Gtarget = tvm.target.Target("llvm -mcpu=znver2")
    return Gtarget, dev

def run_precheck( parallel, model, configs, typeA, typeB, typeC, cast_method, dataType, tvm_iter, gpu_or_cpu):
    logging.info(" --------------  Precheck of this execution ------- ")
    if parallel >= 1:
        logging.info("      Number of threads: %s", parallel)
        if isinstance(parallel, int) and parallel > 0:
            os.environ["TVM_NUM_THREADS"] = str(parallel)
            os.environ["OMP_NUM_THREADS"] = "1"
    else:
        logging.info("      Thread set incorrectly, exiting")
        exit_sequence()

    if model >= 0:
        logging.info("      Model chosen: %s", MODEL_configs[model]["name"])
        logging.info("      There are %d configs inside", len(configs))
        for config in configs:
            logging.info("      Configs are: %s", config)
    else:
        logging.info("      Model set incorrectly, exiting")
        exit_sequence()

    if typeA==typeB==typeC and cast_method > 0:
        logging.info("      Since typeABC are the same, casting method will not take effect, incase of confusion, exiting")
        exit_sequence()
    if dataType >= 1:
        logging.info("      Data type chosen: %s %s %s", typeA, typeB, typeC)
    else:
        logging.info("      Data type set incorrectly, exiting")
        exit_sequence()

    for config in configs:
        name = config["name"]
        if  tvm_iter == 0:
            logging.info("      You enabled the TVM auto scheduler search, but set tvm_iter to 0, auto scheduler will validate the best schedule, if exists")
            for config in configs:
                M = config["M"]
                N = config["N"]
                K = config["K"]
                name = config["name"]
                if gpu_or_cpu == 'gpu':
                    tvm_json = "autoS_json/tvm_gpu_" + str(name) + "_" + str(M) + "_" + str(N) + "_" + str(K) + "_" + str(dataType) +  "_" + str(cast_method) + "_" + str(parallel) + ".json"
                elif gpu_or_cpu == 'cpu':
                    tvm_json = "autoS_json/tvm_cpu_" + str(name) + "_" + str(M) + "_" + str(N) + "_" + str(K) + "_" + str(dataType) +  "_" + str(cast_method) + "_" + str(parallel) + ".json"
                if os.path.isfile(tvm_json):
                    logging.info("     json File %s", name)
                else:
                    logging.info("      == Warning json file unable to be found for config name: %s : ==", name)
                    exit_sequence()
        elif tvm_iter >= 0:
            logging.info("      You enabled the TVM auto scheduler search, and set tvm_iter to %d, auto scheduler will search", tvm_iter)
        else:
            exit_sequence()
    logging.info(" -------- End of run precheck -------------- ")

np.set_printoptions(suppress=True, precision=3)

filenameS = f"logs/log_tvm{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=filenameS,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d-%m %H:%M"  # Show only up to minutes
)

logging.info(" ")
logging.info(" ###########   START  auto_scheduler_GEMM  ############## ")

# # print intro msg explain and input argument user can set 
print(" \nExplain of the input argument user can set: ")
print(" 1. parallel: set the number of threads to run the search")
print(" 2. model: set the model index to run the search")
print(" 3. data type, 1 means fp32 for A, B, C; 8 means int32 for A, B, int8 for C; ")
print(" 4. tvm_iter: set the number of tvm auto scheduler search iteration, if 0, only validate, if dis_autoS set to 1, not run any autoS related")
print(" 5. cast_method: set the cast method, 0: no cast, 5: cast after accumulate")
print(" 6. gpu_or_cpu: set to run with gpu or cpu, input string gpu or cpu")
print(" auto scheduler example: python3 auto_scheduler_GEMM.py 1 2 1 1000 0 gpu, means run auto scheduler search with 1 thread, model index 2: Bert-Large, fp32 types, and 1000 TVM auto scheduler search iteration, no cast, and run with CUDA")
print(" validate auto scheduler example: python3 auto_scheduler_GEMM.py 1 2 1 0 0 gpu, means run auto scheduler validate with 1 thread, model index 2: Bert-Large, fp32 types, and 0 TVM auto scheduler search iteration, only validate results, no cast, and run with CUDA")
print (" Int 32 Example python auto_scheduler_GEMM.py 1 5 8 2 5 cpu , means run auto scheduler search with 1 thread, model index 5: GPT2-large, int8 for A and B, int32 for C, and 2 TVM auto scheduler search iteration, cast after accumulate, and run with CPU")

if len(sys.argv) > 1:
    parallel = int(sys.argv[1])
    model = int(sys.argv[2])
    dataType = int(sys.argv[3])
    if dataType == 1:
        typeA="float32"
        typeB="float32"
        typeC="float32"
        test = 1
    elif dataType == 2:
        typeA="float16"
        typeB="float16"
        typeC="float32"
        test = 2
    elif dataType == 3:
        typeA="int8"
        typeB="int8"
        typeC="float32"
        test = 3
    elif dataType == 4:
        typeA="int8"
        typeB="int8"
        typeC="float16"
        test = 4
    elif dataType == 5:
        typeA="int8"
        typeB="int8"
        typeC="int32"
        test = 5
    elif dataType == 6:
        typeA="int8"
        typeB="int8"
        typeC="int16"
        test = 6
    elif dataType == 8:
        typeA="int32"
        typeB="int32"
        typeC="int8"
        test = 8
    else:
        typeA="float16"
        typeB="float16"
        typeC="float16"
        test = 7

    tvm_iter = int(sys.argv[4])
    cast_method = int(sys.argv[5])

    gpu_or_cpu = sys.argv[6]
    if gpu_or_cpu == "gpu":
        print(" Your input set to run with CUDA")
    elif gpu_or_cpu == "cpu":
        print(" Your input set to run with CPU")
else:
    logging.info(" Need to set input argument, exiting")
    print(" Need to set input argument, exiting")
    exit(0)

MODEL_configs = [
    {"name": "BERT-small", "d": 512, "h": 8, "f": 2048},
    {"name": "BERT-Base", "d": 768, "h": 12, "f": 3072},
    {"name": "BERT-Large", "d": 1024, "h": 16, "f": 4096},
    {"name": "GPT-2-small", "d": 768, "h": 12, "f": 3072},
    {"name": "GPT-2-medium", "d": 1024, "h": 16, "f": 4096},
    {"name": "GPT-2-large", "d": 1280, "h": 20, "f": 5120},
    {"name": "GPT-2-xl", "d": 1600, "h": 25, "f": 6400},
]

l = 384
one = 1
batch = 1
t = 3
MODEL_configs_idx = model
d = MODEL_configs[MODEL_configs_idx]["d"]
h = MODEL_configs[MODEL_configs_idx]["h"]
f = MODEL_configs[MODEL_configs_idx]["f"]
lb = l * batch  
b= batch
d_head = d // h  
dh = int( d / h )

list_name = []
list_gflops_autoS = []
gflops = 0.0

if model == 2:
    configs = [
        {"name": "l_BERT_enc_M123", "M": d, "N": lb, "K": d     },  
        {"name": "l_BERT_enc_M5", "M": l, "N": l, "K": int(d / h) },
        {"name": "l_BERT_enc_M7", "M": int(d/h), "N": l, "K": l   },
        {"name": "l_BERT_enc_F11", "M": f, "N": lb, "K": d        },
        {"name": "l_BERT_enc_F13", "M": d, "N": lb, "K": f        },
    ]

elif model == 5:
    configs = [
        {"name": "l_GPT2_dec_M123",   "M": d, "N": l, "K": d },  
        {"name": "l_GPT2_dec_M5",     "M": t,"N": one, "K": dh },
        {"name": "l_GPT2_dec_M7",     "M": dh,"N": one, "K": t }, 
        {"name": "l_GPT2_dec_M9",     "M": d, "N": one, "K": d },  
        {"name": "l_GPT2_dec_F11",    "M": f, "N": one, "K": d },
        {"name": "l_GPT2_dec_F13",    "M": d, "N": one, "K": f },
    ]

Gtarget, dev = set_hardware_target(gpu_or_cpu)
run_precheck( parallel, model, configs, typeA, typeB, typeC, cast_method, dataType, tvm_iter, gpu_or_cpu)
 
for config in configs:
    M = config["M"]
    N = config["N"]
    K = config["K"]
    name = config["name"]

    MNK = (M, N, K)
    logging.info("   You are at this config: %s", config)
    print("\nIn config: ", config)

    list_name.append(config["name"]) 

    A_np = np.random.uniform(size=(M, K)).astype(typeA)
    B_np = np.random.uniform(size=(K, N)).astype(typeB)
 
    return_matrix = np.zeros((M, N), dtype="float32")

    logging.info(" --------------  Auto Scheduler ------- ")
    start_time_autoS = time.time()

    if gpu_or_cpu == 'gpu':
        tvm_json = "autoS_json/tvm_gpu_" + str(name) + "_" + str(M) + "_" + str(N) + "_" + str(K) + "_" + str(dataType) +  "_" + str(cast_method) + "_" + str(parallel) + ".json"
    elif gpu_or_cpu == 'cpu':
        tvm_json = "autoS_json/tvm_cpu_" + str(name) + "_" + str(M) + "_" + str(N) + "_" + str(K) + "_" + str(dataType) +  "_" + str(cast_method) + "_" + str(parallel) + ".json"

    if tvm_iter != 0:
        logging.info(" --------------  You are starting a auto Scheduler search ------- ")
        autoS_search(M, N, K, Gtarget, dev, tvm_json, tvm_iter, cast_method, typeA, typeB, typeC, gpu_or_cpu,
            A_np, B_np, return_matrix, dataType, name)
    elif tvm_iter == 0:
        logging.info(" --------------  You are starting a auto Scheduler validation run if you have the right json file  ------- ")
        autoS_valid(M, N, K, Gtarget, dev, tvm_json, cast_method, gpu_or_cpu,
        A_np, B_np, return_matrix, dataType, name, list_gflops_autoS)
    else:
        exit_sequence()

    # log time spend
    end_time_autoS = time.time()
    elapsed_time_autoS = end_time_autoS - start_time_autoS
    logging.info("                 AutoS Time spent: %.2f seconds ", elapsed_time_autoS)
    logging.info(" --------------  End of Auto Scheduler  ------- ")

for i in range(0, len(list_gflops_autoS)):
    logging.info("      Config: %s, GFLOPS: %.3f", list_name[i], list_gflops_autoS[i])
 
logging.info(" Input argument you set: ")
logging.info("      Number of threads: %s", parallel)
logging.info("      Model chosen: %s", MODEL_configs[model]["name"])
logging.info("      There are %d configs inside", len(configs))
for config in configs:
    logging.info("      Configs are: %s", config)
logging.info("      Data type chosen: %s %s %s", typeA, typeB, typeC)
logging.info("      TVM auto scheduler search iteration: %s", tvm_iter)
logging.info("      Cast method chosen: %s", cast_method)
logging.info("      CUDA or CPU: %s", gpu_or_cpu)

logging.info(" ###########   END   ############## ")
os.system(f"cat {filenameS}")

