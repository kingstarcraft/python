def instance(instance, *kargs, **kwargs):
    return instance(*kargs, **kwargs) if type(instance).__name__ in ('type', 'function') else instance
