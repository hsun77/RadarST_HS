import hashlib
from types import SimpleNamespace as NS
from importlib import import_module

def _load(name, **deps):
    module = import_module("radarst._formal." + name)
    module.__dict__.update(deps)
    return NS(**module.__dict__)

def functions():
    b67 = _load('b67')
    b2 = _load('b2')
    b71 = _load('b71', B67=b67)
    helper = _load('helper', B67=b67, B71=b71)
    i = _load('i', B67=b67)
    d = _load('d')
    t = _load('t', D=d, B67=b67)
    u = _load('u', T=t)
    d22 = _load('d22', T=t, U=u, B67=b67)
    e = _load('e', D22=d22, T=t, B67=b67)
    a = _load('a', D=d)
    g = _load('g', A=a)
    k = _load('k', G=g)
    signed = _load('signed', D=d)
    r = _load('r', E=e, T=t, B67=b67)
    sel = _load('sel', R=r)
    sel.exact_token_assignment.__globals__['N_SEARCH_ATTEMPTS'] = 96
    c = _load('c')
    m = _load('m')
    internal = _load('internal')
    ext = _load('ext', hashlib=hashlib)
    expr = _load('expr')
    return NS(**locals())

def training_functions():
    f = functions()
    d = _load('training_d')
    a = _load('training_sample', B67=f.b67)
    t = _load('teacher_fit', D=d)
    signed = _load('signed_fit', D=d)
    harm = _load('harm_fit', A=f.a)
    rank = _load('rank_fit', G=harm)
    keys = _load('teacher_targets', B67=f.b67)
    teacher_generation = _load('teacher_generation', B67=f.b67)
    return NS(d=d, a=a, t=t, signed=signed, harm=harm, rank=rank, keys=keys, teacher_generation=teacher_generation, inference=f)
