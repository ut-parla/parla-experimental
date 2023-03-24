import functools 
from  parla.common.globals import _Locals as Locals 

class VariantDefinitionError(ValueError):
    """
    A function variant definition is invalid.
    :see: `specialized`
    """
    pass

class _VariantFunction(object):
    """
    A function that can be specialized to different targets.
    We only support specialization on the architecture level, not on the level of specific devices.
    """

    def __init__(self, func):
        self._default = func 
        self._variants = {}
        functools.update_wrapper(self, func)

    def variant(self, spec_list, override=False):

        if not isinstance(spec_list, list):
            spec_list = [spec_list]
        
        if any(t in self._variants for t in spec_list) and not override:
            raise VariantDefinitionError("Variant already exists for one of the given specialization targets.")

        def variant(f):
            for t in spec_list:
                self._variants[t] = f
            return f
        
        variant.__name__ = "{}.variant".format(self._default.__name__)
        return variant

    def get_variant(self, spec_key):
        return self._variants.get(spec_key, self._default)
    
    def __repr__(self):
        return "{f} specialized to {targets}>".format(f=repr(self._default)[:-1], targets=tuple(self._variants.keys()))

    def __call__(self, *args, **kwargs):
        local_context = Locals.context 
        local_devices = local_context.get_all_devices()

        if len(local_devices) == 0:
            return self._default(*args, **kwargs)

        #Construct a architecture specialization key from the local devices.
        spec_key = tuple([d.architecture for d in local_devices])

        #Get the variant for the current specialization key.
        variant_f = self.get_variant(spec_key)

        #Call the variant.
        return variant_f(*args, **kwargs)


def specialize(func):
    """
    A decorator to declare that this function has specialized variants for specific architectures.
    The decorated function is the default implemention, used when no specialized implementation is available.
    The default can just be `raise NotImplementedError()` in cases where no default implementation is possible.
    To provide a specialized variant use the `variant` member of the main function:
    .. testsetup::
        from parla.function_decorators import *
    >>> @specialized
    ... def f():
    ...     raise NotImplementedError()
    >>> @f.variant(architecture)
    ... def f_gpu():
    ...     ...
    `architecture` above will often by something like `cpu` or `gpu`, but is extensible.
    Multiple architectures can be specified as separate parameters to use the same implementation on multiple architectures: `@f.variant(CPU, FPGA)`.
    Each architecture can only be used once on a given function.
    Architecture specialized functions are called just like any other function, but the implementation which is called is selected based on where the code executes.
    The compiler will make the choice when it is compiling for a specific target.
    """
    return _VariantFunction(func)


        

        



