import gzip
import json
import urllib
import json
from PIL import Image
import os
import numpy

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))
def load(path):
	with open(path,'r') as f:
		return json.load(f)

data = parse('./meta_Electronics.json.gz')

price_list = []
cat_list = []
count = 0
items = 0
with open('./price_list.txt', 'w') as f:
    f.write('')
with open('./cat_list.txt', 'w') as f:
    f.write('')
with open('./log_list.txt', 'w') as f:
    f.write('')

urls = set()

for i in data:
        items += 1
	a = json.loads(i)
        if count > 2000:
            break
	try:
	    if a['price'] < 1.0:
		continue
            price_line = "%d.jpg "%count + str(float(a['price'])) +"\n"
            log_line = "%d.jpg "%count + str(float(numpy.log10(float(a['price'])))) +"\n"
            log_price = numpy.log10(float(a['price']))
            if log_price < 1.01:
                cat = 0
            elif log_price < 1.53:
                cat = 1
            else:
                cat = 2
            cat_line = "%d.jpg "%count + str(cat)+"\n"
            try:
                img_path = "./images/%d.jpg"%count
		if img_path not in urls:
            		urllib.urlretrieve(a['imUrl'], img_path)
			urls.update(img_path)
		else:
			print ('Duplicate')
			continue
		
            except KeyError:
                #print('keyError URL')
                continue
            if not os.stat(img_path).st_size == 0:
                im = Image.open(img_path)
                im_a = numpy.array(im)
                try:
                    im_a.shape[2]
                    if (im.size[0] > 96) and (im.size[1] > 96):
                        count = count + 1
                        with open('./price_list.txt', 'a') as f:
                            f.write(price_line)
			with open('./log_list.txt', 'a') as f:
			    f.write(log_line)
                        with open('./cat_list.txt', 'a') as f:
                            f.write(cat_line)
                except IndexError:
                    print('Single Channel')


	except KeyError:
            #print('keyError Price')
            continue

