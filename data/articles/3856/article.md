# OCR技术浅探：9\. 代码共享(完)

> 作者：苏剑林 · 科学空间 · 2016-06-26
>
> 原文：<https://spaces.ac.cn/archives/3856>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

文件说明：

> 1\. image\.py——图像处理函数，主要是特征提取；
> 
> 
> 
> 2\. model\_training\.py——训练CNN单字识别模型（需要较高性能的服务器，最好有GPU加速，否则真是慢得要死）；
> 
> 
> 
> 3\. ocr\.py——识别函数，包括单字分割、前面训练好的模型进行单字识别、动态规划提升效果；
> 
> 
> 
> 4\. main\.py——主文件，用来调用1、3两个文件。
> 
> 
> 
> 5、[我们的模型中包含的字\.txt](<https://spaces.ac.cn/usr/uploads/2016/06/4225300126.txt>)(UTF\-8编码)

文件1:image\.py



<pre><code class="language-python"># -*- coding:utf-8 -*-

import numpy as np
from scipy import misc,ndimage
from scipy.stats import gaussian_kde as kde
from tqdm import *

def myread(filename): #读取图像，放大两倍，做平方变换
	print u'读取图片中...'
	pic = misc.imread(filename, flatten = True)
	pic = ndimage.zoom(pic, 2)
	pic = pic**2
	pic = ((pic-pic.min())/(pic.max()-pic.min())*255).round()
	print u'读取完成.'
	return pic

def decompose(pic): #核密度聚类，给出极大值、极小值点、背景颜色、聚类图层
	print u'图层聚类分解中...'
	d0 = kde(pic.reshape(-1), bw_method=0.2)(range(256)) #核密度估计
	d = np.diff(d0)
	d1 = np.where((d[:-1]&lt;0)*(d[1:]&gt;0))[0] #极小值
	d1 = [0]+list(d1)+[256]
	d2 = np.where((d[:-1]&gt;0)*(d[1:]&lt;0))[0] #极大值
	if d1[1] &lt; d2[0]:
		d2 = [0]+list(d2)
	if d1[len(d1)-2] &gt; d2[len(d2)-1]:
		d2 = list(d2)+[255]
	dc = sum(map(lambda i: d2[i]*(pic &gt;= d1[i])*(pic &lt; d1[i+1]), range(len(d2))))
	print u'分解完成. 共%s个图层'%len(d2)
	return dc

def erosion_test(dc): #抗腐蚀能力测试
	print u'抗腐蚀能力测试中...'
	layers = []
	#bg = np.argmax(np.bincount(dc.reshape(-1)))
	#d = [i for i in np.unique(dc) if i != bg]
	d = np.unique(dc)
	for k in d:
		f = dc==k
		label_im, nb_labels = ndimage.label(f, structure=np.ones((3,3))) #划分连通区域
		ff = ndimage.binary_erosion(f) #腐蚀操作
		def test_one(i):
			index = label_im==i
			if (1.0*ff[index].sum()/f[index].sum() &gt; 0.9) or (1.0*ff[index].sum()/f[index].sum() &lt; 0.1):
				f[index] = False
		ff = map(test_one, trange(1, nb_labels+1))
		layers.append(f)
	print u'抗腐蚀能力检测完毕.'
	return layers

def pooling(layers): #以模仿池化的形式整合特征
	print u'整合分解的特征中...'
	result = sum(layers)
	label_im, nb_labels = ndimage.label(result, structure=np.ones((3,3)))
	def pool_one(i):
		index = label_im==i
		k = np.argmax([1.0*layers[j][index].sum()/result[index].sum() for j in range(len(layers))])
		result[index] = layers[k][index]
	t = map(pool_one, trange(1, nb_labels+1))
	print u'特征整合成功.'
	return result

def post_do(pic):
	label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
	print u'图像的后期去噪中...'
	def post_do_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		ss = 1.0 * len(pic.reshape(-1))/len(pic[index2].reshape(-1))**2
		#先判断是否低/高密度区，然后再判断是否孤立区。
		if (index.sum()*ss &lt; 16) or ((1+len(pic[index2].reshape(-1))-index.sum())*ss &lt; 16):
			pic[index] = False
		else:
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			index3 = (slice(max(0, 2*a-b),min(pic.shape[0], 2*b-a)), slice(max(0, 2*c-d),min(pic.shape[1], 2*d-c)))
			if (pic[index3].sum() == index.sum()) and (1.0*index.sum()/(b-a)/(d-c) &gt; 0.75):
				pic[index2] = False	
	t = map(post_do_one, trange(1, nb_labels+1))
	print u'后期去噪完成.'
	return pic

def areas(pic): #圈出候选区域
	print u'正在生成候选区域...'
	pic_ = pic.copy()
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def areas_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		pic_[index2] = True
	t = map(areas_one, trange(1, nb_labels+1))
	return pic_

#定义距离函数，返回值是距离和方向
#注意distance(o1, o2)与distance(o2, o1)的结果是不一致的
def distance(o1, o2): 
	delta = np.array(o2[0])-np.array(o1[0])
	d = np.abs(delta)-np.array([(o1[1]+o2[1])/2.0, (o1[2]+o2[2])/2.0])
	d = np.sum(((d &gt;= 0)*d)**2)
	theta = np.angle(delta[0]+delta[1]*1j)
	k = 1
	if np.abs(theta) &lt;= np.pi/4:
		k = 4
	elif np.abs(theta) &gt;= np.pi*3/4:
		k = 2
	elif np.pi/4 &lt; theta &lt; np.pi*3/4:
		k = 1
	else:
		k = 3
	return d, k

def integrate(pic, k=0): #k=0是全向膨胀，k=1仅仅水平膨胀
	label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
	def integrate_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
		cc = ((a+b)/2.0,(c+d)/2.0)
		return (cc, b-a, d-c)
	print u'正在确定区域属性...'
	A = map(integrate_one, trange(1, nb_labels+1))
	print u'区域属性已经确定，正在整合邻近区域...'
	aa,bb = pic.shape
	pic_ = pic.copy()
	def areas_one(i):
		dist = [distance(A[i-1], A[j-1]) for j in range(1, nb_labels+1) if i != j]
		dist = np.array(dist)
		ext = dist[np.argsort(dist[:,0])[0]] #通过排序找最小，得到最邻近区域
		if ext[0] &lt;= (min(A[i-1][1],A[i-1][2])/4)**2:
			ext = int(ext[1])
			index = label_im==i
			index2 = ndimage.find_objects(index)[0]
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			if ext == 1: #根据方向来膨胀
				pic_[a:b, c:min(d+(d-c)/4,bb)] = True
			elif ext == 3:
				pic_[a:b, max(c-(d-c)/4,0):d] = True
			elif ext == 4 and k == 0:
				pic_[a:min(b+(b-a)/6,aa), c:d] = True #基于横向排版假设，横向膨胀要大于竖向膨胀
			elif k == 0:
				pic_[max(a-(b-a)/6,0):b, c:d] = True
	t = map(areas_one, trange(1, nb_labels+1))
	print u'整合完成.'
	return pic_

def cut_blank(pic): #切除图片周围的白边，返回范围
	try:
		q = pic.sum(axis=1)
		ii,jj = np.where(q!= 0)[0][[0,-1]]
		xi = (ii, jj+1)
		q = pic.sum(axis=0)
		ii,jj = np.where(q!= 0)[0][[0,-1]]
		yi = (ii, jj+1)
		return [xi, yi]
	except:
		return [(0,1),(0,1)]

def trim(pic, pic_, prange=5): #剪除白边，删除太小的区域
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def trim_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		box = (pic*index)[index2]
		[(a1,b1), (c1,d1)] = cut_blank(box)
		pic_[index] = False
		if (b1-a1 &lt; prange) or (d1-c1 &lt; prange) or ((b1-a1)*(d1-c1) &lt; prange**2): #删除小区域
			pass
		else: #恢复剪除白边后的区域
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			pic_[a+a1:a+b1,c+c1:c+d1] = True
	t = map(trim_one, trange(1, nb_labels+1))
	return pic_

def bound(m):
	frange = (slice(m.shape[0]-1), slice(m.shape[1]-1))
	f0 = np.abs(np.diff(m, axis=0))
	f1 = np.abs(np.diff(m, axis=1))
	f2 = np.abs(m[frange]-m[1:,1:])
	f3 = f0[frange]+f1[frange]+f2[frange] != 0
	return f3

def trim_bound(pic, pic_): #剪除白边，删除太小的区域
	pic_ = pic_.copy()
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def trim_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		box = pic[index2]
		if 1.0 * bound(box).sum()/box.sum() &lt; 0.15:
			pic_[index] = False
	t = map(trim_one, trange(1, nb_labels+1))
	return pic_
</code></pre>



文件2:model\_training\.py



<pre><code class="language-python"># -*- coding:utf-8 -*-

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import glob

#包含的汉字列表（太长，仅仅截取了一部分）
hanzi = u'0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事'

#生成文字矩阵
def gen_img(text, size=(48,48), fontname='simhei.ttf', fontsize=48):
	im = Image.new('1', size, 1)
	dr = ImageDraw.Draw(im)
	font = ImageFont.truetype(fontname, fontsize)
	dr.text((0, 0), text, font=font)
	return (((np.array(im.getdata()).reshape(size)==0)+(np.random.random(size)&lt;0.05)) != 0).astype(float)

#生成训练样本
data = pd.DataFrame()
fonts = glob.glob('./*.[tT][tT]*')
for fontname in fonts:
	print fontname
	for i in range(-2,3):
		m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
		m['label'] = range(3062)
		data = data.append(m, ignore_index=True)
		m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
		m['label'] = range(3062)
		data = data.append(m, ignore_index=True)

x = np.array(list(data[0])).astype(float)
np.save('x', x) #保存训练数据

dic=dict(zip(range(3062),list(hanzi))) #构建字表

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 1024
nb_classes = 3062
nb_epoch = 30

img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

x = np.load('x.npy')
y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
weight = ((3062-np.arange(3062))/3062.0+1)**3
weight = dict(zip(range(3063),weight/weight.mean())) #调整权重，高频字优先

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    class_weight=weight)

score = model.evaluate(x,y)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('model.model')
</code></pre>



文件3:ocr\.py



<pre><code class="language-python"># -*- coding:utf-8 -*-

import numpy as np
from scipy import misc
from images import cut_blank

#包含的汉字列表（太长了，仅截取了一部分）
hanzi = u'0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事'

dic=dict(zip(range(3062),list(hanzi))) #构建字表

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128
nb_classes = 3062
img_rows, img_cols = 48, 48
nb_filters = 64
nb_pool = 2
nb_conv = 4

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('ocr.model')

import pandas as pd
zy = pd.read_csv('zhuanyi.csv', encoding='utf-8', header=None)
zy.set_index(0, inplace=True)
zy = zy[1]

def viterbi(nodes):
	paths = nodes[0]
	for l in range(1,len(nodes)):
		paths_ = paths.copy()
		paths = {}
		for i in nodes[l].keys():
			nows = {}
			for j in paths_.keys():
				try:
					nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+i]
				except:
					nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+'XX']
			k = np.argmax(nows.values())
			paths[nows.keys()[k]] = nows.values()[k]
	return paths.keys()[np.argmax(paths.values())]

# mode为direact和search
#前者直接给出识别结果，后者给出3个字及其概率（用来动态规划）
def ocr_one(m, mode='direact'):
	m = m[[slice(*i) for i in cut_blank(m)]]
	if m.shape[0] &gt;= m.shape[1]:
		p = np.zeros((m.shape[0],m.shape[0]))
		p[:,:m.shape[1]] = m
	else:
		p = np.zeros((m.shape[1],m.shape[1]))
		x = (m.shape[1]-m.shape[0])/2
		p[:m.shape[0],:] = m
	m = misc.imresize(p,(46,46), interp='nearest') #这步和接下来几步，归一化图像为48x48
	p = np.zeros((48, 48))
	p[1:47,1:47] = m 
	m = p
	m = 1.0 * m / m.max()
	k = model.predict(np.array([[m]]), verbose=0)[0]
	ks = k.argsort()
	if mode == 'direact':
		if k[ks[-1]] &gt; 0.5:
			return dic[ks[-1]]
		else:
			return ''
	elif mode == 'search':
		return {dic[ks[-1]]:k[ks[-1]],dic[ks[-2]]:k[ks[-2]],dic[ks[-3]]:k[ks[-3]]}

'''
#直接调用Tesseract
import os
def ocr_one(m):
	misc.imsave('tmp.png', m)
	os.system('tesseract tmp.png tmp -l chi_sim -psm 10')
	s = open('tmp.txt').read()
	os.system('rm tmp.txt \n rm tmp.png')
	return s.strip()
'''

def cut_line(pl): #mode为direact或viterbi
	pl = pl[[slice(*i) for i in cut_blank(pl)]]
	pl0 = pl.sum(axis=0)
	pl0 = np.where(pl0==0)[0]
	if len(pl0) &gt; 0:
		pl1=[pl0[0]]
		t=[pl0[0]]
		for i in pl0[1:]:
			if i-pl1[-1] == 1:
				t.append(i)
				pl1[-1]=i
			else:
				pl1[-1] = sum(t)/len(t)
				t = [i]
				pl1.append(i)
		pl1[-1] = sum(t)/len(t)
		pl1 = [0] + pl1 + [pl.shape[1]-1]
		cut_position = [1.0*(pl1[i+1]-pl1[i-1])/pl.shape[0] &gt; 1.2 for i in range(1,len(pl1)-1)]
		cut_position=[pl1[1:-1][i] for i in range(len(pl1)-2) if cut_position[i]] #简单的切割算法
		cut_position = [0] + cut_position + [pl.shape[1]-1]
	else:
		cut_position = [0, pl.shape[1]-1]
	l = len(cut_position)
	for i in range(1, l):
		j = int(round(1.0*(cut_position[i]-cut_position[i-1])/pl.shape[0]))
		ab = (cut_position[i]-cut_position[i-1])/max(j,1)
		cut_position = cut_position + [k*ab+cut_position[i-1] for k in range(1, j)]
	cut_position.sort()
	return pl, cut_position

def ocr_line(pl, mode='viterbi'): #mode为direact或viterbi
	pl, cut_position = cut_line(pl)
	if mode == 'viterbi':
		text = map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1], mode='search'), range(len(cut_position)-1))
		return viterbi(text)
	elif mode == 'direact':
		text = map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1]), range(len(cut_position)-1))
		''.join(text)
</code></pre>



文件4:main\.py



<pre><code class="language-python"># -*- coding:utf-8 -*-

from scipy import ndimage
print u'加载图片工具中...'
from images import *
print u'加载OCR模型中...'
from ocr import *
print u'加载完毕.'

if __name__ == '__main__':
	filename = '../cn.jpg'
	p = myread(filename)
	dc = decompose(p)
	layers = erosion_test(dc)
	result = pooling(layers)
	result = post_do(result)
	result_ = areas(result)
	result_ = integrate(result_, 1)
	result_ = trim(result, result_)
	result_ = integrate(result_, 1)
	result_ = trim(result, result_, 10)
	result_ = trim_bound(result, result_)
	label_im, nb_labels = ndimage.label(result_, structure=np.ones((3,3)))
	for i in range(1, nb_labels+1):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		print ocr_line(result[index2])
</code></pre>
