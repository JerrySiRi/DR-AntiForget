
# 3 mode for running: all, eval, infer
# 2 version for getting parameters; simple mode (command line), configuration file (json)

import json
import os
import subprocess
from functools import partial


#* Returns a list of visible GPU indices without importing heavyweight libraries (e.g., torch).
#* Priority:
#* - If CUDA_VISIBLE_DEVICES is set: respect it.
#* - Else: attempt to query GPUs via `nvidia-smi`.
#* - Fallback: return an empty list if not available.
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        #* Example: "0,1,2" -> [0, 1, 2]
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        #* `nvidia-smi --list-gpus | wc -l` -> number of GPUs
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        #* No NVIDIA driver / command not found / running on CPU-only environment.
        return []


# --- Distributed runtime metadata (single-process defaults). --- #
#* RANK/WORLD_SIZE are usually set by torchrun/slurm.
"""
world size = 分布式作业中有多少进程（2 台机器，每台 8 张 GPU，总共 16 个进程）
rank = 当前这个进程在全部进程里的 编号（全局唯一 ID）。作用是：
- 打日志（避免每个进程都打印一遍）
- 下载/准备数据（避免重复下载）
- 汇总评测结果、写最终输出
local = 这台机器上的数据
"""
RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 1))

GPU_LIST = get_gpu_list()
#* If this node runs multiple local processes, slice GPU visibility per process.
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    #* Enforce per-process GPU binding via CUDA_VISIBLE_DEVICES.
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )


#* scieval provides:
#* - model registry/config shortcuts (`supported_VLM`)
#* - dataset builders and dataset registries
#* - inference runners (text/image/video/multi-turn)
#* - misc utilities (logging, filesystem, time, git hash, etc.) via `scieval.smp`
from scieval.config import supported_VLM, supported_LLM
from scieval.dataset.video_dataset_config import supported_video_datasets
from scieval.dataset import build_dataset
from scieval.inference import infer_data_job
from scieval.inference_video import infer_data_job_video
from scieval.inference_mt import infer_data_job_mt
from scieval.smp import *
from scieval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


# --- Build a model object using a config dict entry. --- #
#* Note: this temporarily hides WORLD_SIZE so model constructors don't think we're in distributed mode.
"""
意义：支持多种模型的eval，需要把每个模型处理不同的地方分开处理

原因：因为看到了 WORLD_SIZE 就误以为自己要进入分布式模式。
很多库/模型封装在初始化时会做类似判断：
“如果发现 WORLD_SIZE > 1，那我是不是该初始化 torch.distributed？”
“我是不是该自动切分模型、分配设备、开启并行推理？”
“我是不是该把一些参数（batch size、device map、进程间通信）按分布式处理？”
但在这个脚本里，分布式的初始化是 在 main() 里显式做的

fail_fast 遇到不可恢复错误时是否立刻抛异常停止（fail fast），而不是记录失败继续跑后续样本

verbose 是否输出更详细日志（例如请求/响应状态、进度、异常栈等）。

ignore_patterns 把某些错误信息关键字“视为可忽略”，从而将某些失败输出当作有效输出（常见于 API 返回里包含一些可接受的错误样式时）。

stream API 调用是否使用 streaming（流式输出）。

"""
def build_model_from_config(cfg, model_name, use_vllm=False, args=None):
    import scieval.api
    import scieval.vlm
    #* Some model code may read WORLD_SIZE; remove it during construction to avoid side effects.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    config = cp.deepcopy(cfg[model_name])
    if args is not None:
        if 'retry' not in config:
            if hasattr(args, 'retry') and args.retry is not None:
                config['retry'] = args.retry
        if 'fail_fast' not in config:
            if hasattr(args, 'fail_fast') and args.fail_fast:
                config['fail_fast'] = True
        if 'verbose' not in config:
            if hasattr(args, 'verbose') and args.verbose:
                config['verbose'] = True
        if 'ignore_patterns' not in config:
            if hasattr(args, 'ignore_patterns') and args.ignore_patterns:
                config['ignore_patterns'] = args.ignore_patterns
        if 'stream' not in config:
            if hasattr(args, 'stream') and args.stream:
                config['stream'] = True

    if use_vllm:
        config['use_vllm'] = use_vllm
    if 'class' not in config:
        if model_name in supported_VLM:
            return supported_VLM[model_name](**config)
        elif model_name in supported_LLM:
            return supported_LLM[model_name](**config)
        else:
            raise ValueError(f"Model {model_name} not found")
    cls_name = config.pop('class')
    if hasattr(scieval.api, cls_name):
        model = getattr(scieval.api, cls_name)(**config)
    elif hasattr(scieval.vlm, cls_name):
        model = getattr(scieval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `scieval.api` or `scieval.vlm`')

    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model


#* Build a dataset instance from a config dict entry.
#! This enables more control than `build_dataset(name)` (e.g., video fps/nframe, subtitles, etc.).
def build_dataset_from_config(cfg, dataset_name):
    import scieval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])

    #* Shortcut: empty config means "use default video dataset settings".
    if config == {}:
        return supported_video_datasets[dataset_name]()

    #* Otherwise, config must specify the dataset class name.
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(scieval.dataset, cls_name):
        cls = getattr(scieval.dataset, cls_name)

        #* Filter config keys to only those accepted by the dataset class __init__.
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}

        #* Basic validation for video datasets.
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')

        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `scieval.dataset`')


#* CLI argument parsing.
#* Two modes:
#* - "simple mode": pass `--data` and `--model` lists directly.
#* - "config mode": pass a JSON file via `--config` to define model/data sections.
def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `scieval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have scieval installed).
    To find all supported dataset names, please refer to the `scieval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from scieval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `scieval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `scieval.vlm` or `scieval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `scieval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `scieval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `scieval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=int, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument(
        '--no-use-vllm', dest='use_vllm', action='store_false', help='disable vllm for generation')
    parser.set_defaults(use_vllm=True)
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')
    parser.add_argument('--fail-fast', action='store_true', help='If set, the program will raise an exception and stop upon an unrecoverable API error '
             'after all retries are exhausted. If not set, it will record a failure message and continue. Specifically in generate_inner method in gpt.py, it should be fixed in future versions')
    parser.add_argument('--ignore-patterns', type=str, nargs='+',
                        default=None,
                        help='Keywords in error messages to ignore and treat as valid output')
    parser.add_argument('--stream', action='store_true', help='Use streaming mode for API calls. Default is False.')
    args = parser.parse_args()
    return args


#! Main entry point for end-to-end evaluation.
#* High-level pipeline:
#* 1) Parse args (either --data/--model lists OR a JSON --config).
#* 2) (Optional) Initialize distributed.
#* 3) For each model x dataset:
#*    - Build model/dataset
#*    - Run inference (unless mode == eval)
#*    - Run evaluation (unless mode == infer)
#*    - Write outputs under work_dir/model_name/eval_id

def main():
    logger = get_logger('RUN')
    args = parse_args()

    #*`use_config=True` means args.model/args.data are derived from the JSON config.
    # -- version 1: simple mode, getting parameter from command line
    # -- version 2: config mode, getting parameter from JSON file
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        #* In "simple mode", you must pass at least one dataset name via --data.
        assert len(args.data), '--data should be a list of data files'

    #* Resume behavior hint in logs.
    #! default setting
    #! In order to avoid interleaved and redundant information, only the process with rank id=0 has the access to print logs.
    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    #* Allow overriding output root via environment variable.
    #! set MMEVAL_ROOT to amend output directory from root to MMEVAL_ROOT
    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    #* support VLM in simple mode
    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v
            if args.fail_fast:
                v.keywords['fail_fast'] = True
            if args.ignore_patterns:
                v.keywords['ignore_patterns'] = args.ignore_patterns
            if args.stream:
                v.keywords['stream'] = True

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from scieval.config import api_models as supported_APIs
            from scieval.api import GPT4V
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')
    
    #! Communication in Distributed Computing 
    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group( # for sync
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm, args=args)
        for _, dataset_name in enumerate(args.data):
            #* Output layout:
            #* - pred_root (for this run): <work_dir>/<model>/<dataset>/<eval_id>/
            #* - pred_root_meta (for reuse/symlinks): <work_dir>/<model>/<dataset>/
            pred_root = osp.join(args.work_dir, dataset_name, model_name, eval_id)
            pred_root_meta = osp.join(args.work_dir, dataset_name, model_name)
            os.makedirs(pred_root, exist_ok=True)

            prev_pred_roots = ls(pred_root_meta, mode='dir')
            #* `prepare_reuse_files` expects a <model>-level meta root. We pass <work_dir>/<model>
            #* so it can find and copy previous prediction files across eval_ids.
            pred_root_meta_model = osp.join(args.work_dir, model_name)
            if len(prev_pred_roots) and args.reuse:
                prev_pred_roots.sort()

            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                pred_format = get_pred_file_format()
                result_file_base = f'{model_name}_{dataset_name}.{pred_format}'

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                
                logger.info("INFO - Dataset Ready")
                
                #* Handling Multi-Turn Dataset -- same model & dataset for several times
                result_file = osp.join(pred_root, result_file_base)
                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(
                        pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                        dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                    )

                if WORLD_SIZE > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                if args.mode != "eval": # need inference
                    #! Perform the Inference
                    if dataset.MODALITY == 'VIDEO':
                        model = infer_data_job_video(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            result_file_name=result_file_base,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            use_vllm=args.use_vllm)
                    elif dataset.TYPE == 'MT': # multi-turn messages (role/content)
                        model = infer_data_job_mt(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            ignore_failed=args.ignore,
                            use_vllm=args.use_vllm)
                    else: # orginal task: Q&A, VQA, instruction, etc.
                        model = infer_data_job(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            ignore_failed=args.ignore,
                            use_vllm=args.use_vllm,
                        )
                    logger.info("INFO - Inference finished")
                    

                # Set the judge kwargs first before evaluation or dumping
                 #! Configuration below is only used when LLM-as-judge
                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    # 'max_retries': args.max_retries,
                    # 'fail_fast': args.fail_fast,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }
                # Pass work_dir to dataset.evaluate so paths are constructed correctly
                judge_kwargs['work_dir'] = args.work_dir
                # Pass the current model name to dataset.evaluate so it can build proper output dirs
                judge_kwargs['eval_model_name'] = model_name

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    print(dataset_name)
                    
                    #! Not all dataset need LLM-as-judge, para "model" just for preparation
                    #! e.g. dataset with rule-based with ground truth (could use exact matching) does not need LLM-as-judge
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
                        ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
                    ):
                        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-mini'
                        elif listinstr(['VisuLogic'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        else:
                            judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                        if listinstr(['LLaVABench_KO'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-0806'
                        else:
                            judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(['VGRPBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['OlympiadBench'], dataset_name):
                        use_api_judger = judge_kwargs.get("olympiad_use_api_judger", False)
                        if use_api_judger:
                            judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['ChartMimic'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['VDC'], dataset_name):
                        judge_kwargs['model'] = 'llama31-8b'
                    elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
                        judge_kwargs['model'] = 'qwen-72b'
                    elif listinstr(['MMVMBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1'
                    elif listinstr(['M4Bench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['AyaVisionBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1'
                    elif listinstr(['MaScQA'], dataset_name):
                        judge_kwargs['model'] = 'o3'

                #! double evaluation
                if args.use_verifier:
                    judge_kwargs['use_verifier'] = True
                #! local model + support vllm
                if args.use_vllm:
                    judge_kwargs['use_vllm'] = True

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()
                    

                #! Only RANK 0 handles the evaluation part
                if RANK == 0:
                    #* Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    #* Handling Special dataset -> Convert format and submit to official evaluation
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue
                    # Setup the proxy for the evaluation (LLM-as-judge)
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy :
                        proxy_set(eval_proxy)

                    #* Convert evaluation api to normal when evaluating
                    #* support backup
                    env_backup = {}
                    new_keys_added = []
                    for key, value in list(os.environ.items()):
                        if key.endswith('_EVAL'):
                            if not value or value.strip() == "":
                                continue
                            target_key = key[:-5]
                            if target_key in os.environ:
                                env_backup[target_key] = os.environ[target_key]
                            else:
                                new_keys_added.append(target_key)
                            os.environ[target_key] = value
                            logger.info(f"[Eval Env] Overriding {target_key} using {key}")
                    try:
                        #! Perform the Evaluation
                        #! Each dataset has unique evaluation function
                        eval_results = dataset.evaluate(result_file, **judge_kwargs)
                        # Display Evaluation Results in Terminal
                        if eval_results is not None:
                            assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                            logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                            logger.info('Evaluation Results:')
                            
                            #* Two format to display evaluation results
                            if isinstance(eval_results, dict):
                                logger.info('\n' + json.dumps(eval_results, indent=4))
                            elif isinstance(eval_results, pd.DataFrame):
                                if len(eval_results) < len(eval_results.columns):
                                    eval_results = eval_results.T
                                logger.info('\n' + tabulate(eval_results))
                    except Exception as e:
                        raise(e)

                    #! Need backup env to avoid influencing following model & dataset
                    finally:
                        for key, value in env_backup.items():
                            os.environ[key] = value
                        for key in new_keys_added:
                            if key in os.environ:
                                del os.environ[key]
                        if eval_proxy is not None:
                            proxy_set(old_proxy)
                        
                        logger.info("INFO - Evaluation finished")


                    #* Create the symbolic links for the prediction files
                    #* TODO: put files into pred_root directory by using symbolic links
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
