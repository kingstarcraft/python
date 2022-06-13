class Argument(dict):
    def __init__(self, func, *args, **kwargs):
        super(Argument, self).__init__()

        names = func.__code__.co_varnames[0:func.__code__.co_argcount]
        if func.__defaults__ is None:
            core = {}
        else:
            core = dict(zip(names[::-1], func.__defaults__[::-1]))

        for id, name in enumerate(names):
            if id < len(args):
                core[name] = args[id]
            elif name in kwargs:
                core[name] = kwargs.get(name)

        for name in names:
            try:
                self.__setitem__(name, core[name])
            except:
                pass


if __name__ == '__main__':
    def func(a, b, c, d=1, e=2, f=4):
        aa = 1
        bb = 2
        return aa, bb


    test = Argument(func, 1, c=3, d=None, e=3, b=-2)
    print(test)
