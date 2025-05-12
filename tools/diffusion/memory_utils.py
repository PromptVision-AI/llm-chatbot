# memory_utils.py
import gc, ctypes, sys, torch
from accelerate.hooks import remove_hook_from_module       # <-- no “submodules”

def _free_vram(pipe):
    """
    Remove Accelerate off-loading hooks from every nn.Module sitting
    inside a Diffusers pipeline, move weights to 'meta', flush GC/CUDA,
    and trim the glibc heap so RSS drops.
    """
    if pipe is None:
        return

    #iterate over every attribute of the pipeline;
    #keep only the real torch.nn.Module instances
    for name, maybe_mod in pipe.__dict__.items():
        if isinstance(maybe_mod, torch.nn.Module):
            try:
                # undo AlignDevicesHook / CpuOffloadHook, etc.
                remove_hook_from_module(maybe_mod, recurse=True)
            except Exception:
                pass

            # put all weights on a 0-byte storage
            try:
                maybe_mod.to("meta")
            except Exception:
                pass

    #drop python reference & run collectors
    del pipe
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    #release freed CPU pages back to the OS (Linux/glibc)
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
