# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import re
import os
import sys
import shutil
import time
import types
import importlib
import functools
import traceback
from typing import List, Optional, Callable, Any
import logging
import json
import multiprocessing
from packaging.version import parse, Version

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/utils/")
from cpp_extension import _jit_compile, get_hip_version
from file_baton import FileBaton
from chip_info import get_gfx

AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))


def mp_lock(
    lockPath: str,
    MainFunc: callable,
    FinalFunc: callable = None,
    WaitFunc: callable = None,
):
    """
    Using FileBaton for multiprocessing.
    """
    baton = FileBaton(lockPath)
    if baton.try_acquire():
        try:
            ret = MainFunc()
        finally:
            if FinalFunc is not None:
                FinalFunc()
            baton.release()
    else:
        baton.wait()
        if WaitFunc is not None:
            ret = WaitFunc()
        ret = None
    return ret


PREBUILD_KERNELS = False
if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/aiter_.so"):
    aiter_ = importlib.import_module(f"{__package__}.aiter_")
    PREBUILD_KERNELS = True
logger = logging.getLogger("aiter")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))

AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
AITER_LOG_MORE = int(os.getenv("AITER_LOG_MORE", 0))

find_aiter = importlib.util.find_spec("aiter")
if find_aiter is not None:
    if find_aiter.submodule_search_locations:
        package_path = find_aiter.submodule_search_locations[0]
    elif find_aiter.origin:
        package_path = find_aiter.origin
    package_path = os.path.dirname(package_path)
    package_parent_path = os.path.dirname(package_path)
    import site

    site_packages_dirs = site.getsitepackages()
    # develop mode
    isDevelopMode = (package_path not in site_packages_dirs) and (
        package_parent_path not in site_packages_dirs
    )
    if isDevelopMode:
        AITER_META_DIR = AITER_ROOT_DIR
    # install mode
    else:
        AITER_META_DIR = os.path.abspath(f"{AITER_ROOT_DIR}/aiter_meta/")
else:
    AITER_META_DIR = AITER_ROOT_DIR
    logger.warning("aiter is not installed.")
sys.path.insert(0, AITER_META_DIR)
AITER_CSRC_DIR = f"{AITER_META_DIR}/csrc"
AITER_GRADLIB_DIR = f"{AITER_META_DIR}/gradlib"
gfx = get_gfx()
AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/{gfx}/"
os.environ["AITER_ASM_DIR"] = AITER_ASM_DIR
CK_3RDPARTY_DIR = os.environ.get(
    "CK_DIR", f"{AITER_META_DIR}/3rdparty/composable_kernel"
)
CK_HELPER_DIR = f"{AITER_META_DIR}/3rdparty/ck_helper"


@functools.lru_cache(maxsize=1)
def get_asm_dir():
    return AITER_ASM_DIR


@functools.lru_cache(maxsize=1)
def get_user_jit_dir():
    if "JIT_WORKSPACE_DIR" in os.environ:
        path = os.getenv("JIT_WORKSPACE_DIR")
        os.makedirs(path, exist_ok=True)
        return path
    else:
        if os.access(this_dir, os.W_OK):
            return this_dir
    home_jit_dir = f"{os.path.expanduser('~')}/.aiter/{os.path.basename(this_dir)}"
    if not os.path.exists(home_jit_dir):
        shutil.copytree(this_dir, home_jit_dir)
    return home_jit_dir


bd_dir = f"{get_user_jit_dir()}/build"
# copy ck to build, thus hippify under bd_dir
if multiprocessing.current_process().name == "MainProcess":
    os.makedirs(bd_dir, exist_ok=True)
    # if os.path.exists(f"{bd_dir}/ck/library"):
    #     shutil.rmtree(f"{bd_dir}/ck/library")
CK_DIR = f"{bd_dir}/ck"


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    archs = [arch.strip() for arch in archs]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1100",
        "gfx950",
    ]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    return archs


@functools.lru_cache()
def hip_flag_checker(flag_hip: str):
    ret = os.system(f"hipcc {flag_hip} -x hip -c /dev/null -o /dev/null")
    if ret == 0:
        return [flag_hip]
    else:
        logger.warning(f"{flag_hip} is not supported by hipcc.")
        return []


def check_and_set_ninja_worker():
    max_num_jobs_cores = int(max(1, os.cpu_count() * 0.8))
    if int(os.environ.get("MAX_JOBS", "1")) < max_num_jobs_cores:
        import psutil

        # calculate the maximum allowed NUM_JOBS based on free memory
        free_memory_gb = psutil.virtual_memory().available / (
            1024**3
        )  # free memory in GB
        max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job

        # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
        max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
        max_jobs = str(max_jobs)
        os.environ["MAX_JOBS"] = max_jobs


def rename_cpp_to_cu(els, dst, recurisve=False):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if name.endswith(".cpp") or name.endswith(".cu"):
            newName = name.replace(".cpp", ".cu")
            ret.append(f"{dst}/{newName}")
        shutil.copy(f"{src}/{name}", f"{dst}/{newName}")

    ret = []
    for el in els:
        if not os.path.exists(el):
            logger.warning(f"---> {el} not exists!!!!!!")
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f"{el}/{entry}"):
                    if recurisve:
                        ret += rename_cpp_to_cu([f"{el}/{entry}"], dst, recurisve)
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el), os.path.dirname(el), dst, ret)
    return ret


@functools.lru_cache()
def check_numa():
    numa_balance_set = os.popen("cat /proc/sys/kernel/numa_balancing").read().strip()
    if numa_balance_set == "1":
        logger.warning(
            "WARNING: NUMA balancing is enabled, which may cause errors. "
            "It is recommended to disable NUMA balancing by running \"sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'\" "
            "for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing"
        )


__mds = {}


@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    check_numa()
    if md_name not in __mds:
        __mds[md_name] = importlib.import_module(f"{__package__}.{md_name}")
    return __mds[md_name]


rebuilded_list = ["module_aiter_enum"]


def rm_module(md_name):
    os.system(f"rm -rf {get_user_jit_dir()}/{md_name}.so")


@functools.lru_cache()
def recopy_ck():
    if os.path.exists(CK_DIR):
        os.system(f"rm -rf {CK_DIR}")
    shutil.copytree(CK_3RDPARTY_DIR, CK_DIR, dirs_exist_ok=True)
    shutil.copy(f"{CK_HELPER_DIR}/config.h", f"{CK_DIR}/include/ck/config.h")


def clear_build(md_name):
    os.system(f"rm -rf {bd_dir}/{md_name}")


def build_module(
    md_name,
    srcs,
    flags_extra_cc,
    flags_extra_hip,
    blob_gen_cmd,
    extra_include,
    extra_ldflags,
    verbose,
    is_python_module,
    is_standalone,
    torch_exclude,
    hipify=True,
):
    lock_path = f"{bd_dir}/lock_{md_name}"
    startTS = time.perf_counter()
    target_name = f"{md_name}.so" if not is_standalone else md_name

    def MainFunc():
        recopy_ck()
        if AITER_REBUILD == 1:
            rm_module(md_name)
            clear_build(md_name)
        elif AITER_REBUILD >= 2:
            rm_module(md_name)
        op_dir = f"{bd_dir}/{md_name}"
        logger.info(f"start build [{md_name}] under {op_dir}")

        opbd_dir = f"{op_dir}/build"
        src_dir = f"{op_dir}/build/srcs"
        os.makedirs(src_dir, exist_ok=True)
        if os.path.exists(f"{get_user_jit_dir()}/{target_name}"):
            os.remove(f"{get_user_jit_dir()}/{target_name}")

        sources = rename_cpp_to_cu(srcs, src_dir)

        flags_cc = ["-O3", "-std=c++17"]
        flags_hip = [
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm --amdgpu-kernarg-preload-count=16",
            # "-v --save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-Wno-macro-redefined",
            "-Wno-missing-template-arg-list-after-template-kw",
            "-fgpu-flush-denormals-to-zero",
        ]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = parse(get_hip_version().split()[-1].rstrip("-").replace("-", "+"))
        if hip_version <= Version("6.3.42132"):
            flags_hip += ["-mllvm --amdgpu-enable-max-ilp-scheduling-strategy=1"]
        if hip_version > Version("5.5.00000"):
            flags_hip += ["-mllvm --lsr-drop-solution=1"]
        if hip_version > Version("5.7.23302"):
            flags_hip += ["-fno-offload-uniform-block"]
        if hip_version > Version("6.1.40090"):
            flags_hip += ["-mllvm -enable-post-misched=0"]
        if hip_version > Version("6.2.41132"):
            flags_hip += [
                "-mllvm -amdgpu-early-inline-all=true",
                "-mllvm -amdgpu-function-calls=false",
            ]
        if hip_version > Version("6.2.41133"):
            flags_hip += ["-mllvm -amdgpu-coerce-illegal-types=1"]
        if get_gfx() == "gfx950" and int(os.getenv("AITER_FP4x2", "1")) > 0:
            flags_hip += ["-D__Float4_e2m1fn_x2"]
        flags_cc += flags_extra_cc
        flags_hip += flags_extra_hip
        archs = validate_and_update_archs()
        flags_hip += [f"--offload-arch={arch}" for arch in archs]
        flags_hip = list(set(flags_hip))  # remove same flags
        flags_hip = [el for el in flags_hip if hip_flag_checker(el)]
        check_and_set_ninja_worker()

        def exec_blob(blob_gen_cmd, op_dir, src_dir, sources):
            if blob_gen_cmd:
                blob_dir = f"{op_dir}/blob"
                os.makedirs(blob_dir, exist_ok=True)
                if AITER_LOG_MORE:
                    logger.info(f"exec_blob ---> {PY} {blob_gen_cmd.format(blob_dir)}")
                os.system(f"{PY} {blob_gen_cmd.format(blob_dir)}")
                sources += rename_cpp_to_cu([blob_dir], src_dir, recurisve=True)
            return sources

        if isinstance(blob_gen_cmd, list):
            for s_blob_gen_cmd in blob_gen_cmd:
                sources = exec_blob(s_blob_gen_cmd, op_dir, src_dir, sources)
        else:
            sources = exec_blob(blob_gen_cmd, op_dir, src_dir, sources)

        # TODO: Move all torch api into torch folder
        old_bd_include_dir = f"{op_dir}/build/include"
        os.makedirs(old_bd_include_dir, exist_ok=True)
        rename_cpp_to_cu(
            [f"{AITER_CSRC_DIR}/include"] + extra_include, old_bd_include_dir
        )

        if not is_standalone:
            bd_include_dir = f"{op_dir}/build/include/torch"
            os.makedirs(bd_include_dir, exist_ok=True)
            rename_cpp_to_cu(
                [f"{AITER_CSRC_DIR}/include/torch"] + extra_include, bd_include_dir
            )

        extra_include_paths = [
            f"{CK_DIR}/include",
            f"{CK_DIR}/library/include",
            f"{old_bd_include_dir}",
        ]

        try:
            _jit_compile(
                md_name,
                sources,
                extra_cflags=flags_cc,
                extra_cuda_cflags=flags_hip,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=opbd_dir,
                verbose=verbose or AITER_LOG_MORE > 0,
                with_cuda=True,
                is_python_module=is_python_module,
                is_standalone=is_standalone,
                torch_exclude=torch_exclude,
                hipify=hipify,
            )
            if is_python_module and not is_standalone:
                shutil.copy(f"{opbd_dir}/{target_name}", f"{get_user_jit_dir()}")
            else:
                shutil.copy(
                    f"{opbd_dir}/{target_name}", f"{AITER_ROOT_DIR}/op_tests/cpp/mha"
                )
        except:
            tag = f"\033[31mfailed build jit [{md_name}]\033[0m"
            logger.error(
                f"{tag}\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\n-->[History]: {{}}{tag}\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191".format(
                    re.sub(
                        "error:",
                        "\033[31merror:\033[0m",
                        "-->".join(traceback.format_exception(*sys.exc_info())),
                        flags=re.I,
                    ),
                )
            )
            raise

    def FinalFunc():
        logger.info(
            f"finish build [{md_name}], cost {time.perf_counter()-startTS:.8f}s"
        )

    mp_lock(lockPath=lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)


def get_args_of_build(ops_name: str, exclude=[]):
    d_opt_build_args = {
        "srcs": [],
        "md_name": "",
        "flags_extra_cc": [],
        "flags_extra_hip": [],
        "extra_ldflags": None,
        "extra_include": [],
        "verbose": False,
        "is_python_module": True,
        "is_standalone": False,
        "torch_exclude": False,
        "hip_clang_path": None,
        "blob_gen_cmd": "",
    }

    def convert(d_ops: dict):
        for k, val in d_ops.items():
            if isinstance(val, list):
                for idx, el in enumerate(val):
                    if isinstance(el, str):
                        if "torch" in el:
                            import torch as torch
                        val[idx] = eval(el)
                d_ops[k] = val
            elif isinstance(val, str):
                d_ops[k] = eval(val)
            else:
                pass

        # undefined compile features will be replaced with default value
        d_opt_build_args.update(d_ops)
        return d_opt_build_args

    with open(this_dir + "/optCompilerConfig.json", "r") as file:
        data = json.load(file)
        if isinstance(data, dict):
            # parse all ops
            if ops_name == "all":
                d_all_ops = {
                    "srcs": [],
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "extra_include": [],
                    "blob_gen_cmd": [],
                }
                # traverse opts
                for ops_name, d_ops in data.items():
                    # Cannot contain tune ops
                    if ops_name.endswith("tune"):
                        continue
                    # exclude
                    if ops_name in exclude:
                        continue
                    single_ops = convert(d_ops)
                    for k in d_all_ops.keys():
                        if isinstance(single_ops[k], list):
                            d_all_ops[k] += single_ops[k]
                        elif isinstance(single_ops[k], str) and single_ops[k] != "":
                            d_all_ops[k].append(single_ops[k])

                return d_all_ops
            # no find opt_name in json.
            elif data.get(ops_name) == None:
                logger.warning(
                    "Not found this operator ("
                    + ops_name
                    + ") in 'optCompilerConfig.json'. "
                )
                return d_opt_build_args
            # parser single opt
            else:
                compile_ops_ = data.get(ops_name)
                return convert(compile_ops_)
        else:
            logger.warning(
                "ERROR: pls use dict_format to write 'optCompilerConfig.json'! "
            )


def compile_ops(
    _md_name: str,
    fc_name: Optional[str] = None,
    gen_func: Optional[Callable[..., dict[str, Any]]] = None,
):
    def decorator(func):
        func.arg_checked = False

        @functools.wraps(func)
        def wrapper(*args, custom_build_args={}, **kwargs):
            loadName = fc_name
            md_name = _md_name
            if fc_name is None:
                loadName = func.__name__
            try:
                module = None
                if gen_func is not None:
                    custom_build_args.update(gen_func(*args, **kwargs))
                if PREBUILD_KERNELS:
                    if hasattr(aiter_, loadName):
                        module = aiter_
                elif AITER_REBUILD and md_name not in rebuilded_list:
                    rebuilded_list.append(md_name)
                    raise ModuleNotFoundError("")
                if module is None:
                    md = custom_build_args.get("md_name", md_name)
                    module = get_module(md)
            except ModuleNotFoundError:
                d_args = get_args_of_build(md_name)
                d_args.update(custom_build_args)

                # update module if we have coustom build
                md_name = custom_build_args.get("md_name", md_name)

                srcs = d_args["srcs"]
                flags_extra_cc = d_args["flags_extra_cc"]
                flags_extra_hip = d_args["flags_extra_hip"]
                blob_gen_cmd = d_args["blob_gen_cmd"]
                extra_include = d_args["extra_include"]
                extra_ldflags = d_args["extra_ldflags"]
                verbose = d_args["verbose"]
                is_python_module = d_args["is_python_module"]
                is_standalone = d_args["is_standalone"]
                torch_exclude = d_args["torch_exclude"]
                hipify = d_args.get("hipify", True)
                hip_clang_path = d_args.get("hip_clang_path", None)
                prev_hip_clang_path = None
                if hip_clang_path is not None and os.path.exists(hip_clang_path):
                    prev_hip_clang_path = os.environ.get("HIP_CLANG_PATH", None)
                    os.environ["HIP_CLANG_PATH"] = hip_clang_path

                build_module(
                    md_name,
                    srcs,
                    flags_extra_cc,
                    flags_extra_hip,
                    blob_gen_cmd,
                    extra_include,
                    extra_ldflags,
                    verbose,
                    is_python_module,
                    is_standalone,
                    torch_exclude,
                    hipify,
                )

                if hip_clang_path is not None:
                    if prev_hip_clang_path is not None:
                        os.environ["HIP_CLANG_PATH"] = prev_hip_clang_path
                    else:
                        os.environ.pop("HIP_CLANG_PATH", None)

                if is_python_module:
                    module = get_module(md_name)
                if md_name not in __mds:
                    __mds[md_name] = module

            if isinstance(module, types.ModuleType):
                op = getattr(module, loadName)
            else:
                return None

            def check_args():
                get_asm_dir()
                import inspect
                import typing
                import re
                import torch

                if not op.__doc__.startswith("Members:"):
                    doc_str = op.__doc__.split("\n")[0]
                    doc_str = re.sub(r"<(.*?)\:.*?>", r"\g<1>", doc_str)
                    namespace = {
                        "List": List,
                        "Optional": Optional,
                        "torch": torch,
                    }
                    exec(f"from aiter import*\ndef {doc_str}: pass", namespace)
                    foo = namespace[doc_str.split("(")[0]]
                    sig = inspect.signature(foo)
                    func.__signature__ = sig
                    ann = {k: v.annotation for k, v in sig.parameters.items()}
                    ann["return"] = sig.return_annotation

                    callargs = inspect.getcallargs(func, *args, **kwargs)
                    for el, arg in callargs.items():
                        expected_type = ann[el]
                        origin = typing.get_origin(expected_type)
                        sub_t = typing.get_args(expected_type)

                        if origin is None:
                            if not isinstance(arg, expected_type):
                                raise TypeError(
                                    f"{el} needs to be {expected_type} but got {type(arg)}"
                                )
                        elif origin is list:
                            if (
                                not isinstance(arg, list)
                                # or not all(isinstance(i, sub_t) for i in arg)
                            ):
                                raise TypeError(
                                    f"{el} needs to be List[{sub_t}] but got {arg}"
                                )
                        elif origin is typing.Union:
                            if arg is not None and not isinstance(arg, sub_t):
                                raise TypeError(
                                    f"{el} needs to be Optional[{sub_t}] but got {arg}"
                                )
                        else:
                            raise TypeError(f"Unsupported type: {expected_type}")

                    func_hints = typing.get_type_hints(func)
                    if ann["return"] is None:
                        func_hints["return"] = None
                    if ann != func_hints:
                        logger.warning(
                            f"type hints mismatch, override to --> {doc_str}"
                        )
                return True

            if not func.arg_checked:
                # func.arg_checked = check_args()
                func.arg_checked = True

            if AITER_LOG_MORE == 2:
                from ..test_common import log_args

                log_args(func, *args, **kwargs)

            return op(*args, **kwargs)

        return wrapper

    return decorator
