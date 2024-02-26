import libgptvm
from types import FunctionType, MethodType


def _extract_wrapped(decorated):
    closure = decorated.__closure__
    if closure:
        for cell in closure:
            if (
                isinstance(cell.cell_contents, FunctionType)
                and cell.cell_contents.__closure__ is None
            ):
                return cell.cell_contents
            elif isinstance(cell.cell_contents, FunctionType):
                return _extract_wrapped(cell.cell_contents)
    return decorated


def register_function_injector(orig_func):
    def register(func):
        libgptvm.register_injector(orig_func.__code__, func)
        return func

    return register


def register_method_injector(cls, fn_name):
    method = _extract_wrapped(getattr(cls, fn_name))
    func = method.__func__ if isinstance(method, MethodType) else method
    if not hasattr(func, "_vtable"):
        func._vtable = dict()
    vtable = func._vtable

    def dispatch(self, *args, **kwargs):
        ty = self if isinstance(self, type) else type(self)
        if ty in vtable:
            return vtable[ty](self, *args, **kwargs)

        return func(self, *args, **kwargs)

    def register(func):
        vtable[cls] = func
        libgptvm.register_injector(method.__code__, dispatch)
        return func

    return register
