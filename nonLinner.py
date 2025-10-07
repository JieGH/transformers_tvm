import numpy as np
import tvm
from scipy.linalg import cdf2rdf
from tvm import te, topi, auto_scheduler
import logging
import sys
import numpy as np
from tvm.auto_scheduler import RecordReader
from tvm import IRModule, relax
from tvm.relax.frontend import nn
from tvm.relax.expr_functor import PyExprMutator, mutator
from scipy.special import erf  # From SciPy's special functions module
import os

def set_hardware_target(gpu_or_cpu, Gtarget=None, dev=None):
    if gpu_or_cpu == "gpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        Gtarget = tvm.target.cuda(model='a100', arch="sm_80")
        dev = tvm.device(Gtarget.kind.name, 0)
    elif gpu_or_cpu == "cpu": 
        dev = tvm.cpu()
        Gtarget = tvm.target.Target("llvm -mcpu=znver2")
    return Gtarget, dev

def create_task(func, args, target):
    task = tvm.auto_scheduler.SearchTask(func=func, args=args, target=target)
    return task


@auto_scheduler.register_workload  # this called decorator
def te_GELU_generic(M, N, dtype):
    A = te.placeholder((M, N), name="A", dtype=G_dtype)
    c_GELU = te.compute((M, N),
                        lambda i, j: A[i, j] * 0.5 * (1 + te.erf(A[i, j] / te.sqrt(te.const(2, dtype=G_dtype)))),
                        name="c_GELU"
                        )
    return [A, c_GELU]

@auto_scheduler.register_workload  # this called decorator
def te_Lnorm_generic(M, N, G_dtype):
    gamma = te.placeholder((1,), name="gamma", dtype=G_dtype)
    beta = te.placeholder((1,), name="beta", dtype=G_dtype)
    eps = te.placeholder((1,), name="eps", dtype=G_dtype)
    A = te.placeholder((M, N), name="A", dtype=G_dtype)
    B = te.placeholder((M, N), name="A", dtype=G_dtype)

    c_m_add = te.compute((M, N),
                        lambda i, j: A[i, j] + B[i, j],
                        name="c_m_add"
                        )
    # calculate mean
    r_axis_sum = te.reduce_axis((0, N), name="r_axis_sum")
    c_sum = te.compute((M, 1),
                    lambda i, _: te.sum(c_m_add[i, r_axis_sum], axis=r_axis_sum),
                    name="c_sum"
                    )
    c_mean = te.compute((M, 1),
                        lambda i, _: c_sum[i, 0] / N,
                        name="c_mean"
                        )
    c_sub_power = te.compute((M, N),
                            lambda i, j: (c_m_add[i, j] - c_mean[i, 0]) * (c_m_add[i, j] - c_mean[i, 0]),
                            name="c_sub_power"
                            )
    r_axis_var = te.reduce_axis((0, N), name="r_axis_var")
    c_var_sum = te.compute((M, 1),
                        lambda i, _: te.sum(c_sub_power[i, r_axis_var], axis=r_axis_var),
                        name="c_var_sum"
                        )
    c_var = te.compute((M, 1),
                    lambda i, _: c_var_sum[i, 0] / N,
                    name="c_var"
                    )
    c_norm = te.compute(
        (M, N),
        lambda i, j: (c_m_add[i, j] - c_mean[i, 0]) / te.sqrt(c_var[i, 0] + eps[0]),
        name="c_norm"
    )

    c_Lnorm = te.compute(
        (M, N),
        lambda i, j: c_norm[i, j] * gamma[0] + beta[0],
        name="c_Lnorm"
    )
    return [gamma, beta, eps, A, B, c_Lnorm]


logging.basicConfig(
    filename="logs/nonLinner.log",  # Log file name
    level=logging.DEBUG,  # Log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
)


gpu_or_cpu = "cpu"  
# gpu_or_cpu = "gpu"  
G_dtype = "float32"
validate_or_search = "search"
# validate_or_search = "validate"
iteration   = 20

Gtarget, dev = set_hardware_target(gpu_or_cpu)


configs = [
    {"name": "Bert_L_M6", "operation": "Softmax", "M": 384 , "N":  384 },
    {"name": "Bert_L_F12", "operation": "GELU", "M": 4096, "N": 384},
    {"name": "Bert_L_M10_F14", "operation": "Lnorm", "M": 1024 , "N": 384 },
    {"name": "GPT_L_F12",  "operation": "GELU", "M": 5120, "N": 1},
    {"name": "GPT_L_M10_F14", "operation": "Lnorm", "M": 1280 , "N":  1 },
]

def load_from_json_autoS(log_file, M, N):
    sch, args = task.apply_best(log_file)
    best_record = tvm.auto_scheduler.load_best_record(log_file)
    return sch, args

def tune_autoS(log_file, M, N):
    print("Search...")
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=iteration, 
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
        runner=auto_scheduler.LocalRunner(repeat=1, min_repeat_ms=300, timeout=24000)
    )
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    return sch, args

for config in configs:
    logging.info(" ")
    logging.info("Entry of configs:" + config["name"])

    M = config["M"]
    N = config["N"]

    log_file = "autoS_json/" + config["operation"] + str(gpu_or_cpu) + str(G_dtype) + config["name"] + "_MN_" + str(M) + '_' + str(N) + ".json"


    if validate_or_search == "search" and config["operation"] == "GELU":
        task = create_task(te_GELU_generic, (M, N, G_dtype), Gtarget)
        sch, args = tune_autoS(log_file, M, N)
    elif validate_or_search == "validate" and config["operation"] == "GELU": 
        task = create_task(te_GELU_generic, (M, N, G_dtype), Gtarget) 
        sch, args = load_from_json_autoS(log_file, M, N)
    elif validate_or_search == "search" and config["operation"] == "Lnorm":
        task = create_task(te_Lnorm_generic, (M, N, G_dtype), Gtarget)
        sch, args = tune_autoS(log_file, M, N)
    elif validate_or_search == "validate" and config["operation"] == "Lnorm":  
        task = create_task(te_Lnorm_generic, (M, N, G_dtype), Gtarget)
        sch, args = load_from_json_autoS(log_file, M, N)
    elif config["operation"] == "Softmax":
        A = te.placeholder((M, N), name="A", dtype=G_dtype)
        B = topi.nn.softmax(A, axis=1)
        sch = te.create_schedule(B.op)
        args = [A, B]
        
    
    # elif validate_or_search == "search" and config["operation"] == "Softmax":
    #     task = create_task(te_softmax_generic, (M, N, G_dtype), Gtarget)
    #     sch = te.create_schedule (task.compute_dag.outputs[0].op)
    # elif validate_or_search == "validate" and config["operation"] == "Softmax":
    #     task = create_task(te_softmax_generic, (M, N, G_dtype), Gtarget)
    #     sch, args = load_from_json_autoS(log_file, M, N)

    
    func = tvm.build(sch, args, Gtarget) # now apply the results of best shecheler into the func
    
    
    if config["operation"] == "Lnorm":
        a_np = np.random.uniform(size=(M, N)).astype(G_dtype)
        b_np = np.random.uniform(size=(M, N)).astype(G_dtype)
        c_np = np.zeros((M, N), dtype=G_dtype)
        gamma_np = np.random.uniform(size=(1,)).astype(G_dtype)
        beta_np = np.random.uniform(size=(1,)).astype(G_dtype)
        eps_np = np.array([1e-5], dtype=G_dtype)

        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        gamma_tvm = tvm.nd.array(gamma_np, device=dev)
        beta_tvm = tvm.nd.array(beta_np, device=dev)
        eps_tvm = tvm.nd.array(eps_np, device=dev)
        func(gamma_tvm, beta_tvm, eps_tvm, a_tvm, b_tvm, c_tvm)
        
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        
        # print the name of config, operations,  M, N, dtype, cup or gpu, time cost in s 
        t = evaluator(gamma_tvm, beta_tvm, eps_tvm, a_tvm, b_tvm, c_tvm).mean
        print(f"{config['name']}, {config['operation']}, M={M}, N={N}, dtype={G_dtype}, {gpu_or_cpu}, {t:.10f} s") 
        logging.info(f"{config['name']}, {config['operation']}, M={M}, N={N}, dtype={G_dtype}, {gpu_or_cpu}, {t:.10f}")
        continue

    elif config["operation"] == "GELU" or config["operation"] == "Softmax":
        a_np = np.random.uniform(size=(M, N)).astype(G_dtype)
        c_np = np.zeros((M, N), dtype=G_dtype)

        a_tvm = tvm.nd.array(a_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        func(a_tvm, c_tvm)
        
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        
        # print the name of config, operations,  M, N, dtype, cup or gpu, time cost in s 
        t = evaluator(a_tvm, c_tvm).mean
        print(f"{config['name']}, {config['operation']}, M={M}, N={N}, dtype={G_dtype}, {gpu_or_cpu}, {t:.10f} s") 
        logging.info(f"{config['name']}, {config['operation']}, M={M}, N={N}, dtype={G_dtype}, {gpu_or_cpu}, {t:.10f}")

