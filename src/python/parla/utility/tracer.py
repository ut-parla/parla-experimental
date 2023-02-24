class NVTXTracer:
    """
    This is a wrapper class of Python nvtx.
    (https://docs.nvidia.com/nsight-visual-studio-edition/2020.1/nvtx/index.html)
    If the nvtx package is not installed, convert nvtx calls to no-ops.
    """

    nvtx = None

    def __init__(self):
        if NVTXTracer.nvtx is None:
            try:
                import nvtx as nvtx_module
                NVTXTracer.nvtx = nvtx_module
                # print("NVTX is enabled", flush=True)
            except ImportError:
                # print("NVTX is disabled", flush=True)
                NVTXTracer.nvtx = None

    def __getattr__(self, name):
        nvtx = NVTXTracer.nvtx
        if nvtx is not None:
            return getattr(nvtx, name)
        else:
            return "NVTX is not installed."

    @staticmethod
    def initialize():
        NVTXTracer()

    @staticmethod
    def push_range(message=None, color="blue", domain=None):
        nvtx = NVTXTracer.nvtx
        if nvtx is not None:
            nvtx.push_range(message, color, domain)

    @staticmethod
    def pop_range(domain=None):
        nvtx = NVTXTracer.nvtx
        if nvtx is not None:
            nvtx.pop_range(domain)
