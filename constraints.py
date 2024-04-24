#!/isr/bin/env python

def get_t_src_dstn(cstrfile:str):
    for i,l in enumerate(open(cstrfile)):
        l = l.strip()
        if len(l) == 0 or l.startswith('#'):
            continue
        try:
            t, src, dst = l.split()
            if t not in '+-':
                raise RuntimeError('constraint type not + or - in {cstrfile} line {i+1}') 
            # should check that src and dst are ints or *
            yield (t,src,dst)
        except:
            print('Error: cannot parse line:', l)
            raise

def expand_sets(triplets):
    for t, src, dst in triplets:
        pass

def get_musts_and_bans(triplets, do_parents=True):
    """for each node with constraint give set of must or banned parents"""
    musts, bans = set(), set()
    for t, src, dst in triplets:
    
        assert t in '+-', 'Constraint type needs to be + or -' 
    
        if t == '+':
            ins, outs = musts, bans
        else:
            ins, outs = bans, musts
    
        arc = (int(src), int(dst))
        ins.add(arc)
        if arc in outs:
            outs.remove(arc)

    if do_parents:
        musts = parentize(musts)
        bans  = parentize(bans)

    return musts, bans

def parentize(arcs):
    parents = {}
    for (src,dst) in arcs:
        parents.setdefault(dst,set()).add(src)
    return parents

def file2musts_n_bans(filename:str):
    return get_musts_and_bans(get_t_src_dstn(filename))

if __name__ == '__main__':
    import sys
    musts, bans = file2musts_n_bans(sys.argv[1])

    print('MUSTS:')
    for src,dst in musts:
        print(src,dst)
    print()
    print('BANS:')
    for src,dst in bans:
        print(src,dst)