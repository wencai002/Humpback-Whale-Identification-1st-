#######################################################
### load the cropping result
#######################################################

with open('../input/humpback-whale-identification-model-files/bounding-box.pickle', 'rb') as f:
    p2bb = pickle.load(f)
list(p2bb.items())[:5]