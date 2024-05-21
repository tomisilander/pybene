#!/usr/bin/env python

def no_comments(filename):
    for l in open(filename):
        sl = l.strip()
        if len(sl)==0 or sl.startswith('#'):
            continue
        yield l

def fields(filename):
    for l in no_comments(filename):
        yield l.split('\t')

def fn2varnames(filename):
    return tuple(f[0] for f in fields(filename))

def fn2valnames(filename):
    return tuple(tuple(f[1:]) for f in fields(filename))

def gen_valcs(filename):
    for f in fields(filename):
        yield len(f)-1

def fn2valcs(filename):
    return tuple(gen_valcs(filename))

def load(filename):
    return (fn2varnames(filename), fn2valnames(filename))

def save(varnames, valnames, filename):
    vdf = open(filename,"w")
    for vrn, vlns in zip(varnames, valnames):
        line = '\t'.join((vrn,)+tuple(valnames))
        print(line, file=vdf)
    vdf.close()
