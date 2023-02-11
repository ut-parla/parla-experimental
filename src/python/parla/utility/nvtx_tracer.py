class nvtx_tracer:

    def __init__(self):
        try:
            import nvtx
            self.nvtx = nvtx
            print("NVTX is enabled", flush=True)
        except ImportError:
            print("NVTX is disabled", flush=True)
            self.nvtx = None

    def __getattr__(self, name):
        if self.nvtx is not None:
            return getattr(self.nvtx, name)
        else:
            return "NVTX is not installed."

    def push_range(self, message=None, color="blue", domain=None):
        if self.nvtx is not None:
            self.nvtx.push_range(message, color, domain)

    def pop_range(self, domain=None):
        if self.nvtx is not None:
            self.nvtx.pop_range(domain)
