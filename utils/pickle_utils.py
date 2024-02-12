def p_save(self, obj, path):
    '''Compress and write any object to file.'''
    with bz2.BZ2File(f'{path}.pbz2', 'w') as f:
        cPickle.dump(obj, f)

def p_load(self, path):
    '''Read and unzip and .pbz2 object.'''
    data = bz2.BZ2File(f'{path}.pbz2', 'rb')
    data = cPickle.load(data)
    return data