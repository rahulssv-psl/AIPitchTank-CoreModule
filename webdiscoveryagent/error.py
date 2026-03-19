VLLM_ERROR = """ERROR 04-28 07:56:28 [registry.py:354] Error in inspecting model architecture 'GraniteForCausalLM'
ERROR 04-28 07:56:28 [registry.py:354] Traceback (most recent call last):
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 586, in _run_in_subprocess
ERROR 04-28 07:56:28 [registry.py:354]     returned.check_returncode()
ERROR 04-28 07:56:28 [registry.py:354]   File "/usr/lib64/python3.12/subprocess.py", line 502, in check_returncode
ERROR 04-28 07:56:28 [registry.py:354]     raise CalledProcessError(self.returncode, self.args, self.stdout,
ERROR 04-28 07:56:28 [registry.py:354] subprocess.CalledProcessError: Command '['/opt/vllm/bin/python', '-m', 'vllm.model_executor.models.registry']' returned non-zero exit status 1.
ERROR 04-28 07:56:28 [registry.py:354]
ERROR 04-28 07:56:28 [registry.py:354] The above exception was the direct cause of the following exception:
ERROR 04-28 07:56:28 [registry.py:354]
ERROR 04-28 07:56:28 [registry.py:354] Traceback (most recent call last):
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 352, in _try_inspect_model_cls
ERROR 04-28 07:56:28 [registry.py:354]     return model.inspect_model_cls()
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 323, in inspect_model_cls
ERROR 04-28 07:56:28 [registry.py:354]     return _run_in_subprocess(
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 589, in _run_in_subprocess
ERROR 04-28 07:56:28 [registry.py:354]     raise RuntimeError(f"Error raised in subprocess:\n"
ERROR 04-28 07:56:28 [registry.py:354] RuntimeError: Error raised in subprocess:
ERROR 04-28 07:56:28 [registry.py:354] <frozen runpy>:128: RuntimeWarning: 'vllm.model_executor.models.registry' found in sys.modules after import of package 'vllm.model_executor.models', but prior to execution of 'vllm.model_executor.models.registry'; this may result in unpredictable behaviour
ERROR 04-28 07:56:28 [registry.py:354] Traceback (most recent call last):
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen runpy>", line 198, in _run_module_as_main
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen runpy>", line 88, in _run_code
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 610, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     _run()
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 603, in _run
ERROR 04-28 07:56:28 [registry.py:354]     result = fn()
ERROR 04-28 07:56:28 [registry.py:354]              ^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 324, in <lambda>
ERROR 04-28 07:56:28 [registry.py:354]     lambda: _ModelInfo.from_model_cls(self.load_model_cls()))
ERROR 04-28 07:56:28 [registry.py:354]                                       ^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 327, in load_model_cls
ERROR 04-28 07:56:28 [registry.py:354]     mod = importlib.import_module(self.module_name)
ERROR 04-28 07:56:28 [registry.py:354]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/usr/lib64/python3.12/importlib/__init__.py", line 90, in import_module
ERROR 04-28 07:56:28 [registry.py:354]     return _bootstrap._gcd_import(name[level:], package, level)
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap_external>", line 995, in exec_module
ERROR 04-28 07:56:28 [registry.py:354]   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/granite.py", line 40, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from vllm.model_executor.layers.logits_processor import LogitsProcessor
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py", line 13, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from vllm.model_executor.layers.vocab_parallel_embedding import (
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py", line 139, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
ERROR 04-28 07:56:28 [registry.py:354]      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/__init__.py", line 2536, in fn
ERROR 04-28 07:56:28 [registry.py:354]     return compile(
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/__init__.py", line 2565, in compile
ERROR 04-28 07:56:28 [registry.py:354]     return torch._dynamo.optimize(
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 842, in optimize
ERROR 04-28 07:56:28 [registry.py:354]     return _optimize(rebuild_ctx, *args, **kwargs)
ERROR 04-28 07:56:28 [registry.py:354]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 896, in _optimize
ERROR 04-28 07:56:28 [registry.py:354]     backend = get_compiler_fn(backend)
ERROR 04-28 07:56:28 [registry.py:354]               ^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 783, in get_compiler_fn
ERROR 04-28 07:56:28 [registry.py:354]     from .repro.after_dynamo import wrap_backend_debug
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/repro/after_dynamo.py", line 16, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._dynamo.debug_utils import (
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/debug_utils.py", line 25, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._dynamo.testing import rand_strided
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/testing.py", line 27, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._dynamo.backends.debugging import aot_eager
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_dynamo/backends/debugging.py", line 10, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from functorch.compile import min_cut_rematerialization_partition
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/functorch/compile/__init__.py", line 2, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._functorch.aot_autograd import (
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 36, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._inductor.output_code import OutputCode
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_inductor/output_code.py", line 47, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._inductor.cudagraph_utils import (
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_inductor/cudagraph_utils.py", line 10, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._inductor.utils import InputType
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_inductor/utils.py", line 50, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from torch._inductor.runtime.hints import DeviceProperties
ERROR 04-28 07:56:28 [registry.py:354]   File "/opt/vllm/lib64/python3.12/site-packages/torch/_inductor/runtime/hints.py", line 67, in <module>
ERROR 04-28 07:56:28 [registry.py:354]     from triton.compiler.compiler import AttrsDescriptor
ERROR 04-28 07:56:28 [registry.py:354] ModuleNotFoundError: No module named 'triton.compiler'; 'triton' is not a package
ERROR 04-28 07:56:28 [registry.py:354]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 1130, in <module>
    uvloop.run(run_server(args))
  File "/opt/vllm/lib64/python3.12/site-packages/uvloop/__init__.py", line 109, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/usr/lib64/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
  File "/opt/vllm/lib64/python3.12/site-packages/uvloop/__init__.py", line 61, in wrapper
    return await main
           ^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 1078, in run_server
    async with build_async_engine_client(args) as engine_client:
  File "/usr/lib64/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 146, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
  File "/usr/lib64/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 166, in build_async_engine_client_from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/engine/arg_utils.py", line 1112, in create_engine_config
    model_config = self.create_model_config()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/engine/arg_utils.py", line 1000, in create_model_config
    return ModelConfig(
           ^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/config.py", line 516, in __init__
    self.multimodal_config = self._init_multimodal_config(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/config.py", line 585, in _init_multimodal_config
    if self.registry.is_multimodal_model(self.architectures):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 504, in is_multimodal_model
    model_cls, _ = self.inspect_model_cls(architectures)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 464, in inspect_model_cls
    return self._raise_for_unsupported(architectures)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/model_executor/models/registry.py", line 414, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['GraniteForCausalLM'] failed to be inspected. Please check the logs for more details"""

PYTORCH_ERROR = ""
JSOUP_ERROR = ""
GRADLE_ERRORS = ""
CARGO_ERROR=""

PODMAN_ERROR = """$ podman ps
Cannot connect to Podman. Please verify your connection to the Linux system using `podman system connection list`, or try `podman machine init` and `podman machine start` to manage a new Linux VM
Error: unable to connect to Podman. failed to create sshClient: dial unix /private/tmp/com.apple.launchd.WAh1QMSoLg/Listeners: connect: no such file or directory"""

REACT_ERROR = """./src/App.js
Module not found: Can't resolve './src/components/header/header' in '/home/wiseman/Desktop/React_Components/github-portfolio/src'"""