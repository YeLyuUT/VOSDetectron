from easydict import EasyDict

__C = EasyDict()
cfg = __C

__C.IDA.LEVEL0.out_dims = [256,512,1024,2048]
__C.IDA.LEVEL1.out_dims = [32,32,32]
__C.IDA.LEVEL2.out_dims = [32,32]
__C.IDA.LEVEL3.out_dims = [32]
__C.IDA.SCALES = [1./32.,1./16.,1./8.,1./4.]
