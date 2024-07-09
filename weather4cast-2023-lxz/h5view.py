import h5py

h5_filename = "./data/2019/HRIT/roxi_0004.test.reflbt0.ns.h5"
f = h5py.File(h5_filename,'r')

for item in f.keys():
    print(item + ":", f[item])

print(f.require_group('params').keys())