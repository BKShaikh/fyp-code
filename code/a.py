import progressbar
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
b=['a','b','c','d']


for a in enumerate(b):
    d = 0;
    c = 155;
    print(a)
    progress.min_value=0
    progress.max_value=155
    for slice_ix in progress(range(d, c)):
        print(slice_ix)

