def pipe(inp, fns):
    out = inp
    for fn in fns:
        out=fn(out)
    return out