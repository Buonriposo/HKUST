# 第三章：内建数据结构、函数及文件

## 数据结构和序列

### 元组

- 元组是一种不可变的Python对象，创建元组最简单的方法就是用逗号分隔序列值
```python
tup = 4, 5, 6
```

- 使用tuple函数可以将任意序列或迭代器转换为元组
```python
tuple([4, 5, 6])
(4, 5, 6)
tuple('string')
('s','t','r','i','n','g')
```

- 元组中的元素可以通过中括号[ ]来获取，其索引从0开始

- 如果元组中一个对象是可变的，则可以在其内部进行修改
```python
tup = tuple(['foo', [1, 2], True])
tup[1].append(3)
tup
('foo', [1, 2, 3], True)
```

- 可以使用+号连接元组
```python
(4, None, 'foo') + (6, 0) + ('bar',)
(4, None, 'foo', 6, 0, 'bar')
```
    - 'bar' 后的','是必需的，要保证是个元组

#### 元组拆包

- 如果想将元组型的表达式赋值给变量，python会对元组进行拆包
```python
tup = (4, 5, 6)
a, b, c = tup
a = 4, b = 5, c = 6
```

- 也就是说，python可以实现同时赋值多变量
```python
a, b, c = 1, 2, 3
```
    - 此时等号右边即为一个元组
    

- 拆包的一个常用场景就是遍历元组或列表组成的序列：
```python
seq = [(1, 2, 3),(4, 5, 6),(7, 8, 9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c))
a = 1, b = 2, c = 3
a = 4, b = 5, c = 6
a = 7, b = 8, c = 9
```

#### 元组方法

- 元组的实例方法很少，一个常见的是count，用于计算某个数值在元组中出现的次数
```python
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)
4
```

### 列表

#### 增加和移除元素

- 使用insert方法可以将元素插入指定的列表位置：
```python
list1 = ['foo', 'peekaboo', 'baz', 'dwarf']
list1.insert(1, 'red')
list1
['foo', 'red', 'peekaboo', 'baz', 'dwarf']
```

- 使用pop根据索引移除元素，使用remove根据值移除第一个符合的元素
```python
list1.pop(2)
list1
['foo', 'peekaboo', 'dwarf']
list1.append('foo')
list1.remove('foo')
list1
['peekaboo', 'dwarf', 'foo']
```

#### 连接和联合列表

- 用 + 连接两个列表

- 用extend将新列表的元素添加至原列表
```python
list1 = ['foo', 'peekaboo', 'baz', 'dwarf']
list2 = ['foo', 'peekaboo', 'dwarf']
list1.extend(list2)
list1
['foo', 'peekaboo', 'baz', 'dwarf', 'foo', 'peekaboo', 'dwarf']
```

#### 排序

- 用sort进行排序，可使用参数key来进行
```python
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key = len)
b
['He', 'saw', 'six', 'small', 'foxes']
```
    - 此处是根据字符串长度排序
    
#### 二分搜索和已排序列表的维护

- bisect.bisect会找到元素应该被插入的位置，并保持序列排序，bisect.insort将元素插入到相应位置
```python
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 5)
6
bisect.insort(c, 6)
c
[1, 2, 2, 2, 3, 4, 6, 7]
```

#### 切片

- 步进值step可以在第二个冒号后面使用，意思是每多少个数取一个数
```python
c = [1, 2, 2, 2, 3, 4, 7]
c[::2]
[1, 2, 3, 7]
```

- 当需要对列表或者元组进行翻转时，一种做法是向步进传值-1
```python
c[::-1]
[7, 4, 3, 2, 2, 2, 1]
```

#### 列表生成式

- 根据if函数的位置决定是否需要else
```python
[x if x % 2 == 0 else -x for x in range(1, 11)] #if在for循环之前，需要else
[-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]
[x for x in range(1, 11) if x % 2 == 0] #if在for循环之后，不需要else
[2, 4, 6, 8, 10]
```


### 内建序列函数

#### enumerate

- enumerate函数返回值和其所在的索引
```python
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i
mapping
{'foo': 0, 'bar': 1, 'baz': 2}
```

#### sorted

- sorted函数返回一个根据任意序列中的元素新建的已排序列表，使用参数key可以传递排序依据，使用参数reverse选择是否反向排序
```python
sorted([7, 1, 2, 6, 0, 3, 2])
[0, 1, 2, 2, 3, 6, 7]
sorted([36, 5, -12, 9, -21], key = abs)
[5, 9, -12, -21, 36]
```

#### zip

- zip将列表、元组或其他序列的元素配对，新建一个元组构成的列表
```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped)
[('foo', 'one'), ('bar', 'two'), ('baz', 'three')]
```

- zip可以处理任意长度的序列，它生成列表长度由最短的序列决定
```python
seq3 = ['False', 'True']
list(zip(seq1, seq2, seq3))
[('foo', 'one', 'False'), ('bar', 'two', 'True')]
```

#### reversed

- reversed函数将序列的元素倒序排列
```python
list(reversed(range(10)))
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```
    - reversed是一个生成器，因此如果没有实例化（例如用list函数或者进行for循环）的时候，它并不会产生一个倒序的列表

### 字典

- 使用keys、values返回字典的键和值的迭代器
```python
d1 = {'a': 'some value', 'b': [1, 2, 3, 4], 7: 'an integer'}
list(d1.keys())
['a', 'b', 7]
list(d1.values())
['some value', [1, 2, 3, 4], 'an integer']
```

- 使用update将两个字典合并
```python
d1.update({'b': 'foo', 'c': 12})
d1
{'a': 'some value', 'b': 'foo', 7: 'an integer', 'c': 12}
```
    - 对于任何原字典中已经存在的键，如果传给update方法的数据也含有相同的键，则它的值将会被覆盖

#### 有效的字典键类型

- 字典的键必须是不可变的对象。可以使用哈希化，比如将列表转换为元组，而元组只要它内部元素都可以哈希化，则它自己也可以哈希化
```python
d = {}
d[tuple([1, 2, 3])] = 5
d
{(1, 2, 3): 5}
```

### 集合

- 创建集合可以通过set函数或是直接用大括号
```python
set([2, 2, 2, 1, 3, 3])
{1, 2, 3}
{2, 2, 2, 1, 3, 3}
{1, 2, 3}
```

#### 集合的操作

```python
a.add(x) ##将元素x加入集合a
a.clear() ##清除a的所有元素
a.remove(x) ##从集合a中移除元素x
a.pop() ##移除任意元素
a.union(b) ##返回集合a和b的并集
a.update(b) ##将集合a的的内容设置为a和b的并集
a.intersection(b) ##返回集合a和b的交集
a.intersection_update(b) ##将集合a的内容设置为a和b的交集
a.difference(b) ##返回集合a和b的差集
```


## 函数

### 匿名（Lambda）函数

- Lambda函数是一种通过单个语句生成函数的方法
```python
function = lambda x: x + 2
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key = lambda x: len(set(list(x))))
strings
['aaaa', 'foo', 'abab', 'bar', 'card']
```


### 函数的可变参数

- 通常定义函数时我们需要确定输入的参数，但我们可以通过指针将函数的参数变为可变参数
```python
def calc(*numbers):
    sum = 0
    for i in numbers:
        sum = sum + i*i
    return sum
calc(1,2,3)
14
calc()
0
```

- 如果已经有了一个list或者tuple，我们可以在其前面加一个*，将其元素变为可变参数
```python
nums = [1, 2, 3]
calc(*nums)
14
```


```python
def calc(*numbers):
    sum = 0
    for i in numbers:
        sum = sum + i*i
    return sum
nums = [1, 2, 3]
calc(*nums)
```




    14



### 生成器

- 迭代器是一种用于在上下文中（如for循环）向Python解释器生成对象的对象。大部分以列表或列表型对象为参数的方法都可以接收任意的迭代器对象。包括内建方法比如min、max和sum，以及类型构造函数比如list和tuple
```python
some_dict = {'a': 1, 'b': 2, 'c': 3}
dict_iterator = iter(some_dict)
list(dict_iterator)
['a', 'b', 'c']
```

#### 生成器表达式

- 创建生成器表达式，只需将列表推导式中的中括号替换为小括号即可：
```python
gen = (x ** 2 for x in range(100))
gen
<generator object <genexpr> at 0x00000241399C2348>
sum(gen)
328350
dict((i, i ** 2) for i in range(5))
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

#### itertools模块

- itertools模块是适用于大多数数据算法的生成器集合，例如groupby可以根据任意的序列和一个函数，通过函数的返回值对序列中连续的元素进行分组
```python
import itertools
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))
A ['Alan', 'Adam']
W ['Wes', 'Will']
A ['Albert']
S ['Steven']  
```

### 高阶函数

- map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回
```python
list(map(str, [1,2,3,4,5,6,7,8,9]))
['1', '2', '3', '4', '5', '6', '7', '8', '9']
```

- ruduce把结果不断地和序列的下一个元素做累积计算
```python
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
from functools import reduce
def fn(x, y):
    return 10 * x + y
reduce(fn, [1, 3, 5, 7, 9])
13579
```

- filter函数把传入的函数依次作用于每个元素，根据返回值是True还是False决定保留还是丢弃该元素
```python
def is_odd(n):
    return n % 2 == 1
list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
[1, 5, 9, 15]
#用filter求素数
n = 101
list0 = list(range(2, n + 1))
list1 = []
while list0:
    list1.append(list0[0])
    list0 = list(filter(lambda x: x % list0[0] > 0, list0))
```

## 文件与操作系统

- 打开文件的模式
    - r 只读模式
    - w 只写模式，创建新文件（清除同路径下的同名文件）
    - x 只写模式，创建新文件，但如果存在同名路径时会失败
    - r+ 读写模式
    
- 文件方法或属性
```python
read([size]) #将文件数据作为字符串返回，可选参数size控制读取的字节数
readlines([size]) #返回文件中行内容的列表，size参数可选
write(str) #将字符串写入文件
writelines(strings) #将字符串序列写入文件
```

# 第四章：Numpy基础：数组与向量化计算

## ndarray：多维数组对象

- ndarray是一个通用的多维同类数据容器，它包含的每一个元素均为相同类型。每一个数组都有一个shape属性，用来表征数组每一维度的数量；每一个数组都有一个dtype属性，用来描述数组的数据类型

```python
data = np.random.randn(2,3)
data.shape
(2, 3)
data.dtype
dtype('float64')
```

### 生成ndarray

- 用array函数可以将任意的序列型对象转换成一个Numpy数组
```python
data1 = [1, 2, 3, 4, 5]
np.array(data1)
array([1, 2, 3, 4, 5])
```
- 通过其它函数创建新数组
```python
np.zeros([shape]) #创建一个指定shape的全为0的数组
np.ones([shape]) #创建一个指定shape的全为1的数组
np.empty([shape]) #创建一个指定shape的空数组
np.full([shape],8) #创建一个指定shape的全为8的数组
np.arange(start, end, step) #创建一个从start到end的步长为step的数组
```


### ndarray的数据类型

- 使用astype方法显式地转换数组的数据类型
```python
arr = np.arange(10)
arr.dtype
dtype('int32')
float_arr = arr.astype(np.float64)
float_arr.dtype
dtype('float64')
```

- 也可以直接使用另一个数组的dtype属性
```python
arr1 = np.arange(10)
arr2 = np.array([0.11, 0.22, 0.33, 0.44])
arr1.astype(arr2.dtype)
array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
```

### Numpy数组算术

- Numpy在任意两个等尺寸数组之间的算术操作都应用了逐元素操作的方法
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr * arr
array([[ 1,  4,  9],
       [16, 25, 36]])
arr - arr
array([[0, 0, 0],
       [0, 0, 0]])
```

- 带有标量计算的算术操作，会把计算参数传递给数组的每一个元素
```python
1/arr
array([[1.        , 0.5       , 0.33333333],
       [0.25      , 0.2       , 0.16666667]])
arr ** 0.5
array([[1.        , 1.41421356, 1.73205081],
       [2.        , 2.23606798, 2.44948974]])
```

- 同尺寸数组时间的比较，会产生一个布尔值数组
```python
arr2 = np.array([[0, 4, 1], [7, 2, 12]])
arr2 > arr
array([[False,  True, False],
       [ True, False,  True]])
```


### 基础索引和切片

- 数组的切片是原数组的视图，如果将一个数值传递给切片，那么会修改原数组
```python
arr = np.arange(10)
arr[5:8] = 12
arr
array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])
arr_slice = arr[5:8]
arr_slice[1] = 123
arr
array([  0,   1,   2,   3,   4,  12, 123,  12,   8,   9])
```
    - 如果想要一份数组的拷贝而不是视图的话，必须显式地复制这个数组，例如arr[5:8].copy()

### 布尔索引

- 数组的比较操作可以产生一个布尔值数组，可以用其进行索引
```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7,4)
names == 'Bob'
array([ True, False, False, True, False, False, False])
data[names == 'Bob']
array([[-0.0401981 , -1.08999763, -0.23705872,  0.37544509],
       [-1.85427752,  1.39672162, -2.15039323,  1.34270594]])
```

- ~符号可以在想要对一个通用条件取反时使用
```python
data[~(names == 'Bob')]
array([[ 0.45418094,  1.2335795 , -1.55876532,  1.67808921],
       [ 0.51727714, -1.37704571,  0.45227577, -0.43464613],
       [ 0.99806139, -0.47847124,  0.85280745, -0.22771248],
       [ 0.59745235, -0.81065149, -0.6870047 ,  0.66622497],
       [ 1.16322907,  0.42810727, -2.09921891,  3.08241719]])
```
    - 使用布尔值索引选择数据时，总是生成数据的拷贝，即返回的数组并没有任何变化

### 神奇索引

- 使用整数数组进行数据索引
```python
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr
array([[0., 0., 0., 0.],
       [1., 1., 1., 1.],
       [2., 2., 2., 2.],
       [3., 3., 3., 3.],
       [4., 4., 4., 4.],
       [5., 5., 5., 5.],
       [6., 6., 6., 6.],
       [7., 7., 7., 7.]])
arr[[4, 3, 0, 6]]
array([[4., 4., 4., 4.],
       [3., 3., 3., 3.],
       [0., 0., 0., 0.],
       [6., 6., 6., 6.]])
arr1 = np.arange(32).reshape((8,4))
arr1
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
arr1[[1, 5, 7, 2], [0, 1, 2, 3]]
array([ 4, 21, 30, 11])
```

### 数组转置和换轴

- 根据T属性进行转置
```python
arr = np.arange(15).reshape((3,5))
arr
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
arr.T
array([[ 0,  5, 10],
       [ 1,  6, 11],
       [ 2,  7, 12],
       [ 3,  8, 13],
       [ 4,  9, 14]])
```

- 使用np.dot计算矩阵内积
```python
np.dot(arr.T, arr)
array([[125, 140, 155, 170, 185],
       [140, 158, 176, 194, 212],
       [155, 176, 197, 218, 239],
       [170, 194, 218, 242, 266],
       [185, 212, 239, 266, 293]])
np.dot(arr, arr.T)
array([[ 30,  80, 130],
       [ 80, 255, 430],
       [130, 430, 730]])
```

- 对于高纬度的数组，transpose方法可以用来置换轴
```python
arr = np.arange(16).reshape((2, 2, 4))
arr
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]]])
arr.transpose((1, 0, 2))
array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],
       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
```

## 通用函数：快速的逐元素数组函数

- 一元通用函数（一元指的是函数的对象为一个，即作用在一个数组上）
```python
np.abs #逐元素计算绝对值
np.sqrt #计算每个元素的平方根
np.exp #计算每个元素的自然指数值
np.isnan #返回数组中的元素是否是一个NaN，形式为布尔值数组
np.modf #分别将数组的小数部分和整数部分按数组形式返回
```
- 二元通用函数（作用在两个数组上）
```python
np.add #将数组的元素对应相加
np.subtract #第一个数组减去第二个数组对应的元素
np.multiply #数组对应元素相乘
np.divide #第一个数组的元素除以第二个数组对应的元素
np.power #第二个数组的元素作为第一个数组元素对应的幂次方
np.maximum, fmax #逐个元素计算两个数组中的最大值，fmax忽略NaN
np.mod #按元素的求模计算
greater, greater_equal, less, less_equal, equal, not_equal #逐个比较，返回布尔值
```

## 面向数组编程

### 将条件逻辑转为数组操作

- 使用np.where实现 x if condition else y
```python
xarr = np.array([1,1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
np.where(cond, xarr, yarr)
array([1.1, 2.2, 1.3, 1.4, 2.5])
arr = np.random.randn(4,4)
arr
[[-0.15883135  0.01060549  0.30512857 -0.70296048]
 [ 1.31770942  0.17347226  0.39723278  0.11603516]
 [-0.53199555 -1.22350962  0.59025332 -1.17206661]
 [-0.97625151 -0.39596333  0.14898891 -0.87923312]]
np.where(arr > 0, 2, -2)
[[-2  2  2 -2]
 [ 2  2  2  2]
 [-2 -2  2 -2]
 [-2 -2  2 -2]]
```



### 数学和统计方法

- mean、sum等函数可以接收一个可选参数axis，用于计算给定轴上的统计值，形成一个下降一维度的数组
```python
arr = np.arange(32).reshape((8,4))
arr.mean(axis = 1) #沿列计算，返回每一行的平均值
array([ 1.5,  5.5,  9.5, 13.5, 17.5, 21.5, 25.5, 29.5])
arr.sum(axis = 0) #沿行计算，返回每一列的和
array([112, 120, 128, 136])
```

- 基础数组统计方法
```python
argmin, argmax #最小值和最大值的位置
cumsum #从0开始元素累积和
cumprod #从1开始元素累积积
```


### 布尔值数组的方法

- any检查数组中是否至少有一个True，all检查是否每一个都是True
```python
bools = np.array([False, False, True, False])
bools.any()
True
bools.all()
False
```


### 唯一值与其他集合逻辑

- 使用np.unique返回数组中不重复的元素
```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
array(['Bob', 'Joe', 'Will'], dtype='<U4')
```

### 线性代数

- 常用的numpy.linalg函数
```python
np.diag(arr, [k = 0]) #将一个方阵的对角（或非对角，取k不为0即可）作为一维数组返回，即使不是方阵也还是按斜45度取数，直到最后一列；或者将一维数组转换成一个方阵，并在非对角线上有零点
arr1.dot(arr2) #等价于np.dot(arr1, arr2)，矩阵点乘
from numpy.linalg import * #此后的操作需要先导入包
trace(arr) #等价于arr.trace(),计算对角元素和，对角的含义与diag函数相同
det(arr) #计算矩阵的特征值
eig(arr) #计算矩阵的特征值和特征向量
inv(arr) #计算矩阵的逆矩阵
qr(arr) #计算QR分解
svd(arr) #计算奇异值分解（SVD）
sovle(A, b) #求解x的线性方程Ax = b，其中A是方阵
```

### 伪随机数生成

- numpy.random模块
```python
np.random.normal(loc = 0.0, scale = 1.0, size = None) #生成一个正态分布，均值为0，方差为1，size为矩阵的形状
np.random.seed() #设置随机数种子
np.random.randn() #从均值0方差1的正态分布中抽取样本
np.random.randint(low, high = None, size = None) #根据给定的范围抽取随机整数
```

# 第五章：pandas入门

## pandas数据结构介绍

### Series

- 通过values属性和index属性获得Series对象的值和索引
```python
obj = pd.Series([4, 7, -5, 3])
obj.values
array([ 4,  7, -5,  3], dtype=int64)
obj.index
RangeIndex(start=0, stop=4, step=1)
```

### DataFrame

- 用包含等长度列表或Numpy数组的字典来形成DataFrame
```python
data = {'state': ['Ohio', 'Nevada'],
        'year': [2000, 2001],
        'pop': [1.5, 1.7]}
pd.DataFrame(data)
       state    year    pop
0	Ohio	2000    1.5
1	Nevada	2001	1.7
```

## 基本功能

### 重建索引

- 使用reindex，将数据按照新的索引进行排列，如果某个索引值并不存在，则会引入缺失值
```python
obj = pd.Series([1, 2, 3, 4], index = ['d','b','a','c'])
obj.reindex(['a','b','c','d','e'])
a    3.0
b    2.0
c    4.0
d    1.0
e    NaN
dtype: float64
```

- 对于顺序数据，例如时间序列，在重建索引时可能会需要进行插值，method可选参数允许我们使用诸如ffill等方法在重建索引时插值；ffill方法会将值向前填充，bfill方法会将值向后填充
```python
obj = pd.Series(['blue','purple','yellow'], index = [0, 2, 4])
obj.reindex(range(6), method = 'ffill')
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
```

- 在DataFrame中，reindex可以改变行索引、列索引，也可以同时改变两者。当仅传入一个序列时，结果中的行会重建索引，而列可以使用columns关键字重建索引
```python
frame = pd.DataFrame(np.arange(9).reshape((3,3)),
                     index = ['a','c','d'],
                     columns = ['Ohio','Texas','California'])
frame.reindex(['a','b','c','d'])
    Ohio Texas California
a   0.0	  1.0	2.0
b   NaN	  NaN	NaN
c   3.0	  4.0	5.0
d   6.0	  7.0	8.0
states = ['Texas','Utah','California']
frame.reindex(columns = states)
    Texas  Utah  California
a     1	   NaN	    2
c     4	   NaN	    5
d     7	   NaN      8
```

### 轴向上删除条目

- 使用drop方法删除索引，加入关键字inplace选择是否替换原对象
```python
obj = pd.Series(np.arange(5), index = ['a','b','c','d','e'])
obj.drop(['c','d'])
a    0
b    1
e    4
dtype: int32
```

- 如果是DataFrame，则可使用关键字axis选择删除行还是列，注意默认axis = 0为行
```python
frame = pd.DataFrame(np.arange(9).reshape((3,3)),
                     index = ['a','c','d'],
                     columns = ['Ohio','Texas','California'])
frame.drop(['a','d'])
    Ohio  Texas  California
c	3	4	5
frame.drop(['Ohio','Texas'], axis = 1)
California
a	2
c	5
d	8
```

### 索引、选择与过滤

- 普通的python索引是不包含尾部的，但Series的切片不同，在非整数索引的Series中切片包含尾部；此外，使用loc进行索引时切片包含尾部，使用iloc时不包含尾部

### 算术和数据对齐

- 如果两个Series或DaTaFrame中有不相同的列或行，则进行算术运算时会用产生NaN

#### 使用填充值的算术方法

- 在一个对象上使用add方法，并使用参数fill_value
```python
df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns = list('abcd'))
df2 = pd.DataFrame(np.arange(20).reshape((4,5)),columns = list('abcde'))
df1 + df2
        a    b    c     d   e
0	0.0  2.0  4.0  6.0  NaN
1	9.0  11.0 13.0 15.0 NaN
2	18.0 20.0 22.0 24.0 NaN
3	NaN  NaN  NaN  NaN  NaN
df1.add(df2, fill_value = 0)
         a    b	   c	d    e
0	0.0  2.0  4.0  6.0  4.0
1	9.0  11.0 13.0 15.0 9.0
2	18.0 20.0 22.0 24.0 14.0
3	15.0 16.0 17.0 18.0 19.0
```

- 灵活算术方法，这里的方法前的r表示参数是翻转的，即将括号内的换在前面，前面的对象放进括号内
```python
add, radd #加法
sub, rsub #减法
div, rdiv #除法
floordiv, rfloordiv #整除
mul, rmul #乘法
pow, rpow #幂次方
```


### 函数应用和映射

- Numpy的通用函数对pandas对象也有效
- 使用apply方法可以将函数应用到一行或者一列的一维数组上，使用axis参数可以切换行和列
```python
frame = pd.DataFrame(np.arange(12).reshape((4,3)),
                   index = ['a','b','c','d'],
                   columns = ['Ohio','Texas','California'])
f = lambda x:x.max() - x.min()
frame.apply(f)
Ohio          9
Texas         9
California    9
dtype: int64
frame.apply(f, axis = 'columns') #切换为行
a    2
b    2
c    2
d    2
dtype: int64  
```


### 排序和排名

- 使用sort_index方法对索引进行字典型排序，使用参数axis来选择行和列，使用参数ascending = False选择降序（默认升序）
```python
obj = pd.Series(range(4), index = list('dabc'))
obj.sort_index()
a    1
b    2
c    3
d    0
dtype: int64
frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                     index = ['three','one'],
                     columns = list('dabc'))
frame.sort_index()
      d a b c
one   4 5 6 7
three 0 1 2 3
frame.sort_index(axis = 1)
      a b c d
three 1 2 3 0
one   5 6 7 4
```

- 使用sort_values方法来对值进行排序，NaN会被排至尾部
```python
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
2   -3
3    2
0    4
1    7
dtype: int64
```

- 对DataFrame排序时，使用b关键字by来选择排序依据的列
```python
frame = pd.DataFrame({'b':[4, 7, -3, 2], 'a':[0, 1, 0, 1]})
frame.sort_values(by = 'b')
	b	a
2      -3       0
3	2	1
0	4	0
1	7	1
frame.sort_values(by = ['a','b']) #先排序a再排序b
        b       a
2      -3       0
0	4	0
3	2	1
1	7	1
```

- 使用rank方法来返回每个值的排名
```python
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank() #将平均排名分配到每个组来打破平级关系，就是说把7分成两个6.5
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5
dtype: float64
obj.rank(method = 'first') #根据两者的索引顺序排序
0    6.0
1    1.0
2    7.0
3    4.0
4    3.0
5    2.0
6    5.0
dtype: float64
obj.rank(ascending = False, method = 'first') #改为降序
0    1.0
1    7.0
2    2.0
3    3.0
4    5.0
5    6.0
6    4.0
dtype: float64
```

- 对于DataFrame，可以使用关键字axis来选择行和列
```python
frame = pd.DataFrame({'b':[4.3, 7, -3, 2],
                      'a':[0, 1, 0, 1],
                      'c':[-2, 5, 8, -2.5]})
frame.rank()
         b   a   c
0	3.0 1.5 2.0
1	4.0 3.5 3.0
2	1.0 1.5 4.0
3	2.0 3.5 1.0
frame.rank(axis = 'columns')

         b   a   c
0	3.0 2.0 1.0
1	3.0 1.0 2.0
2	1.0 2.0 3.0
3	3.0 2.0 1.0
```

- 排名中的平级关系打破方法
```python
'average' #默认：每个组中分配平均排名
'min' #对整个组使用最小排名
'max' #对整个组使用最大排名
'first' #按照值在数据中出现的次序分配排名
'dense' #类似于method='min'，但组间排名总是增加1，而不是一个组中的相等元素的数量 
                      

## 描述性统计的概述和计算

- 使用DataFrame的sum方法返回列或行的和
```python
df = pd.DataFrame({'one': [1.4, 7.1, np.NaN, 0.75], 'two': [np.NaN, -4.5, np.NaN, -1.3]}, index = list('abcd'))
df.sum()
one    9.25
two   -5.80
dtype: float64
df.mean(axis = 1, skipna = True) #axis=1为沿列计算，返回行的和；skipna=True表示如果计算中有但不全是NaN则忽略NaN进行计算，如果都是NaN则结果还是NaN
a    1.400
b    1.300
c      NaN
d   -0.275
dtype: float64
```

- 使用describe方法返回统计值
```python
df = pd.DataFrame({'one': [1.4, 7.1, np.NaN, 0.75], 'two': [np.NaN, -4.5, np.NaN, -1.3]}, index = list('abcd'))
df.describe()
          one              two
count   3.000000	 2.000000
mean    3.083333	-2.900000
std	3.493685	 2.262742
min	0.750000	-4.500000
25%	1.075000	-3.700000
50%	1.400000	-2.900000
75%	4.250000	-2.100000
max	7.100000	-1.300000
```

- 描述性统计和汇总统计
```python
count #非NA值的个数
argmin, argmax #计算最小值，最大值所在的索引位置（整数）
idxmin, idxmax #计算最小值，最大值所在的索引位置
quantile #计算样本的从0到1间的分位数
prod #所有值的积
var #值的样本方差
std #值的样本标准差
skew #样本偏度（第三时刻）值
kurt #样本偏度（第四时刻）值
cumsum #累计值
cummin, cummax #累计值的最小值或最大值
cumprod #值的累计积
diff #计算相邻的值的差
pct_change #计算百分比变化
```

### 相关性和协方差

- 使用corr和cov方法计算相关性和协方差
```python
import pandas_datareader.data as web
import datetime as dt
start = dt.datetime(2013, 1, 1)
end = dt.datetime(2016, 12, 31)
df = web.DataReader(['AAPL','IBM','MSFT','GOOG'], 'yahoo', start, end)
data = df.Close
rt = data.pct_change().dropna()
rt['MSFT'].corr(rt['IBM'])
rt.cov()
Symbols   AAPL             IBM             MSFT            GOOG			
AAPL	0.000252	0.000056	0.000080	0.000070
IBM	0.000056	0.000147	0.000073	0.000062
MSFT	0.000080	0.000073	0.000227	0.000106
GOOG	0.000070	0.000062	0.000106	0.000218
```

### 唯一值、计数和成员属性

- 使用unique函数返回序列中的不重复值，使用value_counts函数计算Series包含的每个不重复值的个数
```python
obj = pd.Series(list('cadaabbcc'))
obj.unique()
array(['c', 'a', 'd', 'b'], dtype=object)
obj.value_counts()
a    3
c    3
b    2
d    1
dtype: int64
```

- 使用isin函数计算Series中每个值是否包含于传入序列的布尔值数组
```python
obj = pd.Series(list('cadaabbcc'))
obj.isin(['b','c'])
0     True
1    False
2    False
3    False
4    False
5     True
6     True
7     True
8     True
dtype: bool
```

# 第六章：数据载入、存储及文件格式

## 文本格式数据的读写

- pandas读取文件的解析函数
```python
read_csv #从文件中读取分隔好的数据，逗号是默认分隔符
read_table #从文件中读取分割好的数据，制表符（'\t'）是默认分隔符，当数据是由多种不同数量的空格分开时，使用（'\s+'）作为分隔符
read_excel, read_html, read_json, read_sql #从不同类型的文件中读取数据
```
- 解析函数read_csv、read_table的函数参数
```python
path #文件位置的字符串、URL或文件型对象
sep #分隔符
header #用作列名的行号，默认为0，如果没有则应改为None
index_col #用作索引的列号
names #生成DataFrame的列名列表，当header = None时使用
skiprows #从文件开头起，需要跳过的行号列表；使用这个关键字时，先跳过，再将第一行作为header
na_values #需要用NA替换的值序列
nrows #从文件开头处读入的行数
skip_footer #忽略文件尾部的行数
encoding #Unicode文本编码
```

### 分块读入文本文件

- 在处理大文件之前，可以先对pandas的显示设置进行调整，使之更为紧凑
```python
pd.options.display.max_rows = 10
```

- 为了分块读入文件，可以指定chunksize作为每一块的行数
```python
chunker = pd.read_csv(path, chunksize = 10000)
```

### 将数据写入文本格式

- to_csv的参数
```python
import sys
data.to_csv(sys.stdout, sep = '|', na_rep = NULL, index = False, header = False, columns = []) 
sys.stdout 从控制台中输出，不输出到文件中
sep = '|' 控制输出文本的分隔符
na_rep = NULL 控制缺失值的填充
index = False 控制是否输出时同时输出行索引
header = False 控制是否同时输出列名
columns = [] 控制输出的列
```

- Series的to_csv方法
```
dates = pd.date_range('1/1/2021', periods = 7) #这里的
ts = pd.Series(np.arange(7), index = dates)
ts.to_csv(sys.stdout)
2021-01-01,0
2021-01-02,1
2021-01-03,2
2021-01-04,3
2021-01-05,4
2021-01-06,5
2021-01-07,6
```

## 读取Excel文件

- 使用ExcelFile时，通过将xls或xlsx的路径传入，生成实例，再转成DataFrame
```python
xlsx = pd.ExcelFile(path)
data = pd.read_excel(xlsx, 'Sheet1')
也可直接使用read_excel
data = pd.read_excel(path, 'Sheet1')
```

- 输出到Excel文件
```python
data.to_excel(path, 'Sheet1')
```

- JSON格式、HTML格式、二进制格式、HDF5格式及与Web API、数据库交互见《利用Python进行数据分析》P176-P187

# 第七章：数据清洗与准备

## 处理缺失值

- NA处理方法
```python
frame.dropna() #根据是否有缺失值来删去轴
frame.fillna() #用某些值填充NA
frame.isnull() #返回表明哪些值是缺失值的布尔值
```

### 过滤缺失值

- 通过调节dropna的参数来处理NA
```python
dropna(how = 'all') #删去所有值均为NA的行
dropna(axis = 1) #删去包含NA的列，如果不使用参数axis则默认为删去行
```

### 补全缺失值

- 通过字典补全缺失值，字典的键为列索引
```python
df = pd.DataFrame(np.random.randn(6,3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
df.fillna({1: 0.5, 2:0})
            0               1               2
0	-0.267317	0.500000	 0.000000
1	-0.476142	0.500000	 0.000000
2	 0.400210	0.500000	 1.256472
3	-0.731970	0.500000	-0.350872
4	-0.939433      -0.489337	-0.804591
5	-0.212698      -0.339140	 0.312170
```

- fillna的其他参数
```python
method #插值方法，默认为ffill，即按照前一个值插入
axis #需要填充的轴，默认为0
inplace #控制是否替代原数据
limit #用于向前或向后填充时最大的填充范围
```

## 数据转换

### 删除重复值

- 使用drop_duplicates方法删除重复值
```python
data = pd.DataFrame({'k1':['one','two'] * 3 + ['two'], 'k2':[1, 1, 2, 3, 3, 4, 4]})
data.drop_duplicates()
        k1	k2
0	one	1
1	two	1
2	one	2
3	two	3
4	one	3
5	two	4
data.drop_duplicates(['k1']) #根据k1列删去重复值，保留第一个观测到的值
        k1	k2
0	one	1
1	two	1
data.drop_duplicates('k2', keep = 'last') #根据k2列删除重复值，保留最后一个观测到的值
        k1	k2
1	two	1
2	one	2
4	one	3
6	two	4
```

### 使用函数或映射进行数据转换

- 根据输入函数来进行数据转换
```python
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
def meat_to_animal(x):
    if x.lower() in ['bacon', 'pulled pork', 'honey ham']:
        return 'pig'
    elif x.lower() in ['corned beef', 'pastrami']:
        return 'cow'
    else:
        return 'salmon'      
data['animal'] = data['food'].map(meat_to_animal)
data
        food        ounces animal
0	bacon         4.0   pig
1	pulled pork   3.0   pig
2	bacon         12.0  pig
3	Pastrami      6.0   cow
4	corned beef   7.5   cow
5	Bacon         8.0   pig
6	pastrami      3.0   cow
7	honey ham     5.0   pig
8	nova lox      6.0   salmon
```

### 替代值

- 使用replace进行值替换
```python
data = pd.Series([1, -999, 2, -999, 3, -999]) #使用列表
data.replace(-999, 0)
0    1
1    0
2    2
3    0
4    3
5    0
dtype: int64
data.replace({-999: 0, 1: 1000}) #使用字典
0    1000
1       0
2       2
3       0
4       3
5       0
dtype: int64
```

### 重命名轴索引

- 除了使用类似上面的map方法，还可以使用rename方法进行重命名索引
```python
data = pd.DataFrame(np.arange(12).reshape((3,4)),
                    index = ['Ohio', 'Colorado', 'New York'],
                    columns = ['one', 'two', 'three', 'four'])
data.rename(index = str.title, columns = str.upper)
             ONE TWO THREE FOUR
    Ohio      0   1   2     3
Colorado    4   5   6     7
New York    8   9   10    11
data.rename(index = {'Ohio': 'INDIANA'}, columns = {'three': 'peekaboo'}) #字典方法
	   one two peekaboo four
INDIANA     0   1   2        3
Colorado    4   5   6        7
New York    8   9   10       11
```

### 离散化和分箱

- 对连续值进行离散化或分离成箱子处理，可以使用pandas中的cut函数
```
ages = [20, 22, 25, 27, 21, 23, 37, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
pd.cut(ages, bins)
[(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (35, 60], (60, 100], (35, 60], (35, 60], (25, 35]]
Length: 11
Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]
```

- cut函数返回的是一个特殊的Categorical对象，输出了一个表示箱名的字符串数组。对其可以使用categories和codes方法
```
ages = [20, 22, 25, 27, 21, 23, 37, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats.codes
array([0, 0, 0, 1, 0, 0, 2, 3, 2, 2, 1], dtype=int8)
cats.categories
IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]],
              closed='right',
              dtype='interval[int64]')
```

- cut函数的其他参数right和labels
```
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, right = False, labels = group_names) #right=False将区间换成左闭右开，labels设置箱名
['Youth', 'Youth', 'YoungAdult', 'YoungAdult', 'Youth', ..., 'MiddleAged', 'Senior', 'MiddleAged', 'MiddleAged', 'YoungAdult']
Length: 11
Categories (4, object): ['Youth' < 'YoungAdult' < 'MiddleAged' < 'Senior']
```

- 用整数而非列表来进行分箱
```
data = np.random.rand(20)
pd.cut(data, 4, precision = 2) #4表示分成区间等长的四份，precision=2表示将精度设置为两位小数
[(0.58, 0.78], (0.58, 0.78], (0.19, 0.39], (0.78, 0.98], (0.58, 0.78], ..., (0.78, 0.98], (0.19, 0.39], (0.19, 0.39], (0.78, 0.98], (0.78, 0.98]]
Length: 20
Categories (4, interval[float64]): [(0.19, 0.39] < (0.39, 0.58] < (0.58, 0.78] < (0.78, 0.98]]
```

- 使用qcut进行基于分位数的分箱
```
data = np.random.randn(1000)
cats = pd.qcut(data, 4)
pd.value_counts(cats)
(-3.125, -0.71]     250
(-0.71, -0.0381]    250
(-0.0381, 0.661]    250
(0.661, 3.001]      250
dtype: int64
cats = pd.qcut(data, [0, 0.2, 0.45, 0.79, 1]) #自定义分位数，使用列表进行传输
pd.value_counts(cats)
(-0.132, 0.777]     340
(-0.816, -0.132]    250
(0.777, 3.36]       210
(-3.319, -0.816]    200
dtype: int64
```

### 置换和随机取样

- 使用np.random.permutation对DataFrame中的Series或行进行置换（随机重排序），并使用在take函数中
```python
df = pd.DataFrame(np.arange(20).reshape((5,4)))
sampler = np.random.permutation(5) #这里的整数表示想要的轴长度
sampler
array([1, 4, 0, 2, 3])
df.take(sampler)
    0   1   2   3
1   4   5   6   7
4   16  17  18  19
0   0   1   2   3
2   8   9   10  11
3   12  13  14  15
```

- 使用sample方法进行随机取样，使用参数n表示随机取样的数量，使用参数replace来选择放不放回
```python
df = pd.DataFrame(np.arange(20).reshape((5,4)))
df.sample(n = 3)
    0   1   2   3
3   12  13  14  15
1   4   5   6   7
0   0   1   2   3
choices = pd.Series([5, 7, -1, 6, 4])
choices.sample(n = 10, replace = True) 
0    5
3    6
4    4
0    5
2   -1
1    7
4    4
4    4
4    4
0    5
dtype: int64
```

### 计算指标和虚拟变量

- 如果DataFrame的一列有k个不同的值，则可以衍生一个k列的值为1和0的矩阵或DataFrame。使用get_dummies函数来实现
```python
values = np.random.rand(10)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
pd.get_dummies(pd.cut(values, bins, labels = ['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']))

      0~0.2 0.2~0.4 0.4~0.6 0.6~0.8 0.8~1.0
0	0	0	0	1	0
1	0	0	0	0	1
2	0	0	0	1	0
3	0	0	0	1	0
4	0	0	1	0	0
5	0	0	1	0	0
6	0	1	0	0	0
7	1	0	0	0	0
8	0	1	0	0	0
9	1	0	0	0	0
```                                                     

## 字符串操作

### 字符串对象方法

- 使用split方法分隔字符串，并可搭配strip方法去除空格
```python
val = 'a,b,  guide'
val.split(',')
['a', 'b', '  guide']
pieces = [x.strip() for x in val.split(',')]
pieces
['a', 'b', 'guide']
```

- 使用+或者join连接字符串
```python
first, second, third = pieces
first + '::' + second + '::' + third
'a::b::guide'
'::'.join(pieces)
'a::b::guide'
```

- Python内建字符串方法
```python
val.count(',') #返回子字符串在字符串中的非重叠出现次数
val.index(',') #返回子字符串第一次出现的位置索引，如果找不到就报错
val.find(',') #返回子字符串第一次出现的位置索引，如果找不到则返回-1
val.rfind(',') #返回子字符串最后一次出现的位置索引，如果找不到则返回-1
val.replace(',', ':') #使用后一个字符串代替前一个字符串
val.lower() #全部转换为小写字母
val.upper() #全部转换为大写字母
```

### 正则表达式

- re模块的正则表达式，分为匹配、替换和拆分三个主题
```python
import re
text = 'foo   bar\t baz  \tqux'
re.split('\s+', text) # '\s+'是空白字符的正则表达式
['foo', 'bar', 'baz', 'qux']
```

- 我们可以先对字符串进行编译，形成一个可复用的正则表达式对象，这样在处理多个字符串时可以节约CPU；并且可使用findall方法返回一个包含所有正则表达式模式的列表
```python
regex = re.compile('\s+')
regex.split(text)
['foo', 'bar', 'baz', 'qux']
regex.findall(text)
['   ', '\t ', '  \t']
```

- 不同于findall，search只返回第一个匹配项，match只在字符串的起始位置进行匹配
```python
text = '''
Dave dave@google.com
Steve steve@google.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
'''
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.+[A-Z]{2,4}'
regex = re.compile(pattern, flags = re.IGNORECASE) #re.IGNORECASE使正则表达式不区分大小写
regex.findall(text)
['dave@google.com', 'steve@google.com', 'rob@gmail.com', 'ryan@yahoo.com']
regex.search(text)  #或者regex.search(text).group()直接返回字符串
<re.Match object; spasubn=(6, 21), match='dave@google.com'>
text[regex.search(text).start():regex.search(text).end()]
'dave@google.com' #只返回第一个匹配项
print(regex.match(text)) #在起始位置匹配不到则返回None
None
```

- 使用sub方法对所有匹配项进行替换，使用参数n对前n个匹配项进行替换
```python
print(regex.sub('email', text))
Dave email
Steve email
Rob email
Ryan email
print(regex.sub('email', text, 2))
Dave email
Steve email
Rob rob@gmail.com
Ryan ryan@yahoo.com
```

- 使用括号将模式包起来以实现分组，返回元组对象
```python
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.+([A-Z]{2,4})'
regex = re.compile(pattern, flags = re.IGNORECASE)
regex.match('wesm@bright.net').groups()
('wesm', 'bright', 'net')
regex.findall(text)
[('dave', 'google', 'com'),
 ('steve', 'google', 'com'),
 ('rob', 'gmail', 'com'),
 ('ryan', 'yahoo', 'com')]
```

- sub函数结合分组来对不同组的值进行不同的替换
```python
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
Dave Username: dave, Domain: google, Suffix: com
Steve Username: steve, Domain: google, Suffix: com
Rob Username: rob, Domain: gmail, Suffix: com
Ryan Username: ryan, Domain: yahoo, Suffix: com
```

### pandas中的向量化字符串函数

- 使用map方法可以应用函数到每个值，但若有NA则会失败。为了解决这个问题，Series的str属性可以被调用，例如有str.contains方法，用于检验是否包含某个字符
```python
data = pd.Series({'Dave':'dave@google.com',
                 'Steve':'steve@gmail.com',
                 'Rob':'rob@gmail.com',
                 'Wes':np.nan})
data.str.contains('gmail')
Dave     False
Steve     True
Rob       True
Wes        NaN
dtype: object
```

- 当我们使用str属性时，相当于是对Series的每个元素进行操作，这时我们可以在其后衔接其他方法
```python
data.str.cat(others = data1, sep = ',', join = 'right') #根据sep的分隔符按索引连接两个Series（默认分隔符为空格），如果某行Index不相同的话则会在该行返回NaN；join控制哪个是主表，默认为左表；加入不使用others关键字，则会连接该Series中的所有元素；需要所有元素均为字符串
data.str.count('com') #返回Series中每个元素包含字符串的个数
data.str.get(i) #获得Series中每个元素的第i个元素，即字符串的第i个字符
data.str.join(',') #用分隔符连接Series每一行的所有元素，需要为字符串
data.str.replace('com', 'net') #用后面的值替换前面的值
data.str.split('\s+') #以分隔符或其他正则表达式对字符串进行拆分
data.str.strip() #消除字符串两侧的空白
```

# 第八章：数据规整：连接、联合与重塑

## 分层索引

- pandas允许一个轴上有多个索引层级，使我们能在低纬度处理高维数据
```python
data = pd.Series(np.random.randn(9),
                 index = [list('aaabbccdd'), [1, 2, 3, 1, 3, 1, 2, 2, 3]])
a    1    1.331587
     2    0.715279
     3   -1.545400
b    1   -0.008384
     3    0.621336
c    1   -0.720086
     2    0.265512
d    2    0.108549
     3    0.004291
dtype: float64
data['b':'c']
b    1   -0.008384
     3    0.621336
c    1   -0.720086
     2    0.265512
dtype: float64
data.loc[['b', 'd']]
b    1   -0.008384
     3    0.621336
d    2    0.108549
     3    0.004291
dtype: float64
data.loc[:, 2] #这里和DataFrame不同，即两个位置都是对于行的索引，只是不同维度
a    0.715279
c    0.265512
d    0.108549
dtype: float64
```

- 使用unstack和stack方法对多层索引在DataFrame中进行重排列
```python
data.unstack() #unstack函数中有参数可以用来控制排列哪一层的轴，在只有两层索引时可以不考虑
         1          2           3
a    1.331587   0.715279   -1.545400
b   -0.008384     NaN       0.621336
c   -0.720086   0.265512      NaN
d      NaN      0.108549    0.004291
data.unstack().stack()
a    1    1.331587
     2    0.715279
     3   -1.545400
b    1   -0.008384
     3    0.621336
c    1   -0.720086
     2    0.265512
d    2    0.108549
     3    0.004291
dtype: float64
```

- DataFrame中，每个轴都可以有分层索引
```python
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index = [list('aabb'), [1, 2, 1, 2]],
                     columns = [['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame
            Ohio   Colorado
         Green Red  Green
a     1    0	1	2
      2	   3	4	5
b     1    6	7	8
      2    9    10      11
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame
state       Ohio   Colorado
color    Green  Red  Green
key1 key2			
a     1     0	1	2
      2	    3	4	5
b     1     6	7	8
      2     9   10      11
```

### 重排序和层级排序

- 使用swaplevel方法变更层级，靠左或靠上的索引可以理解为更靠外，因此层级更小，也就是说最左边的索引是level=0
```python
frame.swaplevel('key1', 'key2') #等价于frame.swaplevel(0, 1)，表示第一层和第二层索引交换，但使用level时要注意是行轴还是列轴，在swaplevel中使用参数axis=1可以转换为列
state       Ohio   Colorado
color    Green  Red  Green
key2 key1			
1     a     0	1	2
2     a	    3	4	5
1     b     6	7	8
2     b     9   10      11
```

- 使用sort_index方法来选择某一层的索引进行排序，也可以使用参数axis来选择行或列
```python
frame.sort_index(level = 1)
state       Ohio   Colorado
color    Green  Red  Green
key1 key2			
a     1     0	1	2
b     1     6	7	8
a     2	    3	4	5
b     2     9   10      11
```

### 按层级进行汇总统计

- 使用level参数来对某一层进行汇总性统计
```python
frame.sum(level = 1)
state	Ohio	Colorado
color	Green	Red  Green
key2			
1        6       8    10
2        12      14   16
frame.sum(level = 'state', axis = 1)
state     Ohio	Colorado
key1 key2		
a     1	   1      2
      2	   7      5
b     1    13     8
      2    19     11
```

### 使用DataFrame的列进行索引

- 使用set_index、reset_index来实现将DataFrame中的列转换成索引，以及将索引转换成列
```python
frame = pd.DataFrame({'a':range(7),
                      'b':range(7, 0, -1),
                      'c':['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                      'd':[0, 1, 2, 0, 1, 2, 3]})
frame.set_index(['c', 'd']) #在前的为level低的，即外侧；使用参数drop=False可以将转换成索引的列仍留在DataFrame中
		a	b
c	d		
one     0       0	7
        1	1	6
        2	2	5
two     0       3       4
        1	4	3
        2	5	2
        3	6	1
frame.set_index(['c', 'd']).reset_index()
         c	d	a	b
0	one	0	0	7
1	one	1	1	6
2	one	2	2	5
3	two	0	3	4
4	two	1	4	3
5	two	2	5	2
6	two	3	6	1
```



## 联合与合并数据集

### 数据库风格的DataFrame连接

- 使用merge函数来进行DataFrame的join操作
```python
df1 = pd.DataFrame({'key':list('bbacaab'), 'data1':range(7)})
df2 = pd.DataFrame({'key':list('abd'), 'data2':range(3)})
pd.merge(df1, df2, on = 'key') #参数on决定连接键，须为两表都有的列
       key    data1   data2
0	b	0	1
1	b	1	1
2	b	6	1
3	a	2	0
4	a	4	0
5	a	5	0
```

- 连接不同的列名
```python
df3 = pd.DataFrame({'lkey':list('bbacaab'), 'data1':range(7)})
df4 = pd.DataFrame({'rkey':list('abd'), 'data2':range(3)})
pd.merge(df3, df4, left_on = 'lkey', right_on = 'rkey')

       lkey   data1   rkey    data2
0	b	0	b	1
1	b	1	b	1
2	b	6	b	1
3	a	2	a	0
4	a	4	a	0
5	a	5	a	0
```

- merge函数的其他参数
```python
pd.merge(df1, df2, how = 'left') #how决定连接方式，默认为'inner'内连接，可选'right','outer'
pd.merge(df1, df2, left_index = True) #使用left的行索引作为它的连接键，需搭配right_on参数决定right的连接键，或同样使用right_index=True
pd.merge(df1, df2, suffixes = ('_left', '_right')) #决定重复列名的后缀，默认为('_x', '_y')
```

### 沿轴向连接

- 使用concatenate函数对Numpy实现拼接
```python
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis = 1) #axis=1表示增添列
array([[ 0,  1,  2,  3,  0,  1,  2,  3],
       [ 4,  5,  6,  7,  4,  5,  6,  7],
       [ 8,  9, 10, 11,  8,  9, 10, 11]])
```

- 使用concat函数对Series实现拼接
```python
s1 = pd.DataFrame([0, 1], index = list('ab'))
s2 = pd.DataFrame([2, 3, 4], index = list('cde'))
s3 = pd.DataFrame([5, 6], index = list('fg'))
pd.concat([s1, s2, s3]) #默认为沿着axis=0的方向生效，即增添行
        0
a	0
b	1
c	2
d	3
e	4
f	5
g	6
pd.concat([s1, s2], axis = 1) #axis=1，沿列增添
         0	 0
a	0.0	NaN
b	1.0	NaN
c	NaN	2.0
d	NaN	3.0
e	NaN	4.0
s4 = pd.DataFrame([0, 1, 5, 6], index = list('abfg'))
pd.concat([s1, s4], axis = 1, join = 'inner') #join='inner'为内连接，默认为join='outer'，只有这两个
	0	0
a	0	0
b	1	1
pd.concat([s1, s4], keys = ['one', 'two']) #使用keys关键字设置一个多层索引
        0
one   a	0
      b	1
two   a	0
      b	1
      f	5
      g	6 
```

- 在DataFrame上使用相同的逻辑
```python
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)), index = list('abc'), columns = ['one', 'two'])
df2 = pd.DataFrame(np.arange(2).reshape((2, 2)), index = list('ac'), columns = ['three', 'four'])
pd.concat([df1, df2], axis = 1, keys = ['level1', 'level2'])
            level1	       level2
       one     two    three    four     five    six
a	0	1	2	3	0.0	1.0
b	4	5	6	7	NaN	NaN
c	8	9	10      11      2.0     3.0
pd.concat([df1, df2], ignore_index = True) #ignore_index=True表示不沿着连接轴保留索引
    one   two  three    four    five    six
0   0.0	  1.0	2.0	3.0	NaN	NaN
1   4.0	  5.0	6.0	7.0	NaN	NaN
2   8.0	  9.0	10.0    11.0    NaN     NaN
3   NaN	  NaN	NaN	NaN	0.0	1.0
4   NaN	  NaN	NaN	NaN	2.0	3.0
```

### 联合重叠数据

- 使用np.where实现面向数组的if-else操作
```python
a = pd.Series([np.nan, 2.5, 0, 3.5, 4.5, np.nan], index = list('fedcba'))
b = pd.Series([0, np.nan, 2, np.nan, np.nan, 5], index = list('abcdef'))
np.where(pd.isnull(a), b, a)
array([0. , 2.5, 0. , 3.5, 4.5, 5. ])
```

- 使用combine_first在DataFrame中做相同的操作
```python
df1 = pd.DataFrame({'a': [1, np.nan, 5, np.nan],
                   'b': [np.nan, 2, np.nan, 6],
                   'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5, 4, np.nan, 3, 7],
                   'b': [np.nan, 3, 4, 6, 8]})
df1.combine_first(df2) #若有重复保留df1的
         a	 b	 c
0	1.0	NaN	2.0
1	4.0	2.0	6.0
2	5.0	4.0	10.0
3	3.0	6.0	14.0
4	7.0	8.0	NaN
```

## 重塑和透视

### 使用多层索引进行重塑

- unstack（将行中多层索引的数据透视到列）和stack（将列中的数据透视到行形成多层索引）
```python
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index = pd.Index(['Ohio', 'Colorado'], name = 'state'),
                    columns = pd.Index(['one', 'two', 'three'], name = 'number'))
data.stack()
state     number
Ohio      one       0
          two       1
          three     2
Colorado  one       3
          two       4
          three     5
dtype: int32
data.stack().unstack(0) #默认拆最内层，但我们可以输入层级0或者'state'来拆分外层
state   Ohio Colorado
number		
one	  0	3
two	  1	4
three	  2	5
```

- stack的其他参数
```python
data = pd.Series([0, 1, 2, 3, 4, 5, 6], index = [['one', 'one', 'one', 'one', 'two', 'two', 'two'], list('abcdcde')])
data
one     a    0
        b    1
        c    2
        d    3
two     c    4
        d    5
        e    6
dtype: int64
data.unstack().stack(dropna = False) #保留缺失值
one     a    0.0
        b    1.0
        c    2.0
        d    3.0
        e    NaN
two     a    NaN
        b    NaN
        c    4.0
        d    5.0
        e    6.0
dtype: float64
```

### 将长透视为宽

- 

# 第九章：绘图与可视化

## 简明 matplotlib API入门

### 图片与子图

- matplotlib绘制的图位于图片对象中，使用plt.figure可以生成一个新的图片，使用add_subplot可以创建子图
```python
import matplotlib.pyplot as plt
fig = plt.figure() #会出现一个空白的绘图窗口，但Jupyter中没有
ax1 = fig.add_subplot(2, 2, 1) #这里前两个参数表示size，即2*2，后面表示这四个图形中的第一个
ax2 = fig.add_subplot(2, 2, 2) #这里也可以用222代替，如果参数为数字的话一定是三位数
plt.plot(range(10)) #当我们此时使用plt.plot，会在最后一个图片和子图上绘制
```

- 使用subplots函数创建新的图片
```python
fig, axes = plt.subplots(2, 3)  #即创建一个2*3的图片
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True) #表示子图使用相同的x轴刻度，y轴刻度
```

- 使用subplots_adjust调整子图周围的间距
```python
subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = None) #前四个参数调整的是子图的外部和图片边缘的距离，wspace和hspace控制的是图片的宽度和高度百分比，用作子图间的间距；如果我们希望子图间没有间距，则将两者设置为0即可
```

### 颜色、标记和线类型

- plot函数的参数
```python
plt.plot(np.random.randn(10), color = 'k', linestyle = 'dashed', marker = 'o') #color调节颜色，linestyle调整线条模式，marker为连接处标记类型
```

### 刻度、标签和图例

- 设置标题、轴标签、刻度和刻度标签
```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ticks = ax.set_xticks([0, 250, 500, 750, 1000]) #设置刻度值
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation = 30, fontsize = 'small') #label会代替上面的刻度标签，rotation会将轴刻度标签旋转30度
ax.set_title('price plot') #设置标题
ax.set_xlabel('Stages') #设置x轴的名称
```

- 添加图例
```python
plt.plot(np.random.randn(100), color = 'k', label = 'one') #label参数设置标签
plt.legend(loc = 'best') #loc参数设置标签的位置，一般是用'upper','lower'加上'right','left','center'组成，或者使用'best'来选择遮挡图形最少的位置
```


### 将图片保存到文件

- 使用plt.savefig保存图片
```python
plt.savefig('figpath.png', dpi = 400, facecolor = 'w', format = 'png', bbox_inches = 'tight') #第一个参数为图片路径，dpi控制每英寸点数的分辨率；facecolor控制子图之外的图形北京的颜色，默认为白色'w'；format控制文件格式，bbox_inches控制要保存的图片范围，即可以修建实际图形的空白
```

## 使用pandas和seaborn绘图

### 折线图

- Series和DataFrame都有一个plot属性，用于绘制图形，默认情况下为折线图；Series的plot方法参数
```python
label #设置标签
ax #选择作图的子图
style #线的样式，例如'ko--'
alpha #不透明度，从0到
kind #图形类型，如'area','bar','hist','line','pie'
use_index #使用对象索引刻度标签
xticks #用于x刻度的值
xlim #x轴的范围
grid #展示轴网格
```

- DataFrame的plot方法参数
```python
subplots #如果为True，则将DataFrame的每一列绘制在独立的子图中
sharex, sharey #在subplots=True的前途下，如果为True则子图共享轴、刻度和范围
figsize #用于生成图片尺寸的元组
title #设置标题
legend #设置图例
```

### 柱状图

- plot.bar( )和plot.barh( )可以分别绘制垂直和水平的柱状图，其参数和plot方法的参数一致
```python
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random,rand(16), index = list('abcdefghijklmnop'))
data.plot.bar(ax = axes[0], color = 'k', alpha = 0.7)
data.plot.barh(ax = axes[1], color = 'k', aplha = 0.8)
df = pd.DataFrame(np.random.rand(6, 4),
                  index = ['one', 'two', 'three', 'four', 'five', 'six'],
                  columns = pd.Index(['A', 'B', 'C', 'D'], name = 'Genus'))
df.plot.bar(stacked = True) #DataFrame的柱状图将每一行中的值分组到紧挨的柱子中的一组，如果使用参数stacked=True，则紧挨的柱子会堆叠成一根柱子，每个值占柱子的一部分，用不同颜色表示
```

### 直方图和密度图

- 使用plot.hist( )和plot.density( )方法绘制直方图和密度图
```python
data.plot.hist(bins = 100) #类似R中的breaks，参数bins设置数据点的个数
data.plot.density() #生成密度图
```

### 散点图

- 使用seaborn的regplot或plt.scatter()方法绘制散点图
```python
import seaborn as sns
sns.regplot(x, y, data = data) #这里x和y只需给出在data中的列名；会同时生成一个拟合的回归线
sns.pairplot(data, diag_kind = 'kde', plot_kws = {'alpha': 0.2}) #对数据中的每个变量两两绘制散点图，并在对角线上放置每个变量的密度图
plt.scatter(x, y, data = data) 
```

### 分面网格和分类数据

- 使用seaborn的catplot函数实现对多种分组变量分开作图，catplot的函数参数
```python
x, y, hue #x, y为x轴、y轴的变量；hue为变量分类的列，依据其构成不同组紧挨的柱子，每一个组的不同变量都是由hue决定的
row, col #变量分类的列，将分开的变量拆成多个图形
kind #图形的类型
data #数据来源
```

# 第十章：数据聚合与分组操作

## Groupby机制

- 使用groupby函数进行分组，分组键有多种形式
```python
df = pd.DataFrame({'key1': list('aabba'),
                   'key2': ['one', 'two', 'one', 'two', 'one'],
                   'data1': np.random.randn(5),
                   'data2': np.random.randn(5)})
grouped = df['data1'].groupby(df['key1']) #会返回一个Groupby对象，可以用其进行分组后的操作
grouped.mean()
key1
a   -0.190625
b    0.399468
Name: data1, dtype: float64
df['data1'].groupby([df['key1'], df['key2']]).mean() #将多个数组作为列表传入
key1  key2
a     one    -0.903477
        two    -1.222153
b     one    -0.814129
        two     0.521123
Name: data1, dtype: float64
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean() #使用等长度的数组作为分组键
California  2005   -0.512591
              2006    0.067151
Ohio        2005   -0.988471
              2006    0.882941
Name: data1, dtype: float64
df.groupby('key1').mean() #在同一个DataFrame中，可以直接用列名作为分组键
         data1	    data2
key1		
a	0.189742	-0.196091
b	1.160970	-0.474046
```

### 遍历各分组

- Groupby对象支持迭代，会生成一个包含组名和数据块的二维元组序列
```python
for name, group in df.groupby('key1'):
    print(name)
    print(group)
a
  key1 key2     data1     data2
0    a  one  1.358234 -0.551361
1    a  two  2.795955  0.976317
4    a  one -0.152009 -0.989347
b
  key1 key2     data1     data2
2    b  one  0.437124 -2.129031
3    b  two  2.591874 -0.703954
for (k1, k2), group in df.groupby(['key1', 'key2']):  #多个分组键的情况下，元组中的第一个元素是键值的元组
    print((k1, k2))
    print(group)
('a', 'one')
  key1 key2     data1     data2
0    a  one  0.781719 -1.694938
4    a  one  1.996056 -0.467833
('a', 'two')
  key1 key2     data1     data2
1    a  two -1.079675 -1.076447
('b', 'one')
  key1 key2     data1    data2
2    b  one  0.168901 -0.44088
('b', 'two')
  key1 key2     data1     data2
3    b  two  0.273449  0.814665
dict(list(df.groupby('key1'))) #元组转换成字典
{'a':   key1 key2     data1     data2
 0    a  one  1.188776  0.149024
 1    a  two -1.340580  0.408787
 4    a  one -0.248602  0.607185,
 'b':   key1 key2     data1     data2
 2    b  one  2.123337 -0.611105
 3    b  two -0.465233 -0.045758}
```

- 使用参数axis来改变分组的轴
```python
df.dtypes
key1      object
key2      object
data1    float64
data2    float64
dtype: object
for type, group in df.groupby(df.dtypes, axis = 1):  #在列上进行分组
    print(type)
    print(group)
float64
      data1     data2
0 -0.535550  1.396753
1 -1.681759 -1.667218
2  1.664944 -1.382713
3  0.149005 -0.012483
4 -1.865534 -0.813996
object
  key1 key2
0    a  one
1    a  two
2    b  one
3    b  two
4    a  one
```

### 选择一列或所有列的子集

- 使用DataFrame创建的Groupby对象用列名进行索引时，会产生用于聚合的列子集的效果
```python
df['data1'].groupby(df['key1']) 等价于 df.groupby('key1')['data1']
df[['data2']].groupby(df['key1']) 等价于 df.groupby('key1')[['data2']] #注意当传递的索引是列表或数组，即[['data2']]，则返回的对象为DataFrame；如果只有单个列名作为标量传递，如['data1']，则返回Series
```

### 使用字典和Series分组

- 使用字典和Series的索引属性可以对数组进行分组
```python
people = pd.DataFrame(np.random.randn(5, 5), columns = list('abcde'),
                      index = ['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}
people.groupby(mapping, axis = 1).sum() #使用字典分组
            blue      red
Joe     2.432954  0.403291
Steve   0.205590  2.671931
Wes     0.751518  1.138893
Jim    -1.514596  0.246063
Travis  0.992939  2.854520
map_series = pd.Series(mapping)
people.groupby(map_series, axis = 1).count() #使用Series分组
       blue red
Joe    2   3
Steve  2   3
Wes    2   3
Jim    2   3
Travis 2   3
```

### 使用函数分组

- 传递函数作为参数来进行分组，以及将函数与数组、字典等相结合
```python
people.groupby(len).sum()
        a         b         c         d         e
3  -0.941127  1.080025 -1.211739 -5.505950 -1.769787
5   0.145395 -0.418400 -0.559184  0.729773 -0.994581
6   0.817123 -1.467993  1.570138  0.789482  0.780341
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()
             a         b         c         d         e
3   one -0.899791  0.466837  0.694990  0.183670 -0.742098
    two  0.988612  0.478948 -0.635451  0.234414  0.954318
5   one  1.268216  1.312615  1.111863  0.098699  0.978446
6   two  2.021687 -0.790612  1.113606  2.051655 -0.376081
```

### 根据索引层级分组

- 在轴索引的某个层级上进行聚合，使用关键字level
```python
df = pd.DataFrame(np.random.randn(4, 5),
                  columns = [['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]])
df.columns.names = ['city', 'tenor']
df.groupby(level = 'city', axis = 1).count()
city    JP      US
0	2	3
1	2	3
2	2	3
3	2	3
```

## 数据聚合

- 优化的groupby方法
```python
count #分组中非NA值的数量
sum, mean, median, prod #非NA值的和、均值、中位数、积
min, max #非NA值的最小值、最大值
first, last #非NA值的第一个值和最后一个值
```

- 使用自定义的函数，将函数传递给aggregate或agg方法
```python
df = pd.DataFrame({'key1': list('aabba'), 
                   'key2': ['one', 'two', 'one', 'two', 'one'],
                   'data1': np.random.randn(5),
                   'data2': np.random.randn(5)})
def max_to_min(arr):
    return arr.max() - arr.min()
df.groupby('key1').agg(max_to_min)
          data1	         data2
key1		
a	2.041328	2.174891
b	2.013652	0.303047
```

### 逐列及多函数应用

- 用agg方法同时使用多个聚合函数，以字符串形式传递内建函数，自建函数直接传递函数名
```python
tips = pd.DataFrame({'tip': [1.01, 1.66, 3.50, 3.31, 3.61, 4.71],
                     'smoker': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
                     'day': ['Sun', 'Sat', 'Sun', 'Sat', 'Sat', 'Sun'],
                     'size': [2, 3, 3, 2, 4, 4],
                     'tip_pct': [0.0594, 0.1605, 0.1666, 0.1398, 0.1468, 0.1862]})
grouped = tips.groupby(['smoker', 'day'])
def max_to_min(arr):
  return arr.max() - arr.min()
grouped['tip_pct'].agg(['mean', 'sum', max_to_min])
             mean    sum   max_to_min
smoker day			
No     Sat  0.1605  0.1605  0.0000
       Sun  0.1130  0.2260  0.1072
Yes    Sat  0.1433  0.2866  0.0070
       Sun  0.1862  0.1862  0.0000
```

- 使用二维元组的方法对Groupby对象各列的名称进行更改
```python
grouped['tip_pct'].agg([('foo', 'mean'), ('bar', 'sum')])
              foo     bar  
smoker day			
No     Sat  0.1605  0.1605  
       Sun  0.1130  0.2260  
Yes    Sat  0.1433  0.2866  
       Sun  0.1862  0.1862
```

- 将多个函数同时应用到多个列
```python
result = grouped['tip_pct', 'tip'].agg(['count', 'mean', 'max'])
result
	          tip_pct      	        tip
              count      mean	  max	count	mean	max
smoker day						
No     Sat	1	0.1605	0.1605	 1	1.660	1.66
       Sun	2	0.1130	0.1666	 2	2.255	3.50
Yes    Sat	2	0.1433	0.1468	 2	3.460	3.61
       Sun	1	0.1862	0.1862	 1	4.710	4.71
result['tip']
	      count	 mean	 max
smoker  day			
No      Sat	1	1.660	1.66
        Sun	2	2.255	3.50
Yes     Sat	2	3.460	3.61
        Sun	1	4.710	4.71
```

- 使用字典将不同的函数应用在不同列
```python
grouped.agg({'tip': ['min', 'max', 'mean'], 'tip_pct': 'sum'})
                tip                     tip_pct
                min	  max	 mean	 sum
smoker day				
No     Sat	1.66	1.66	1.660	0.1605
       Sun	1.01	3.50	2.255	0.2260
Yes    Sat	3.31	3.61	3.460	0.2866
       Sun	4.71	4.71	4.710	0.1862
```

### 返回不含行索引的聚合数据

- 使用as_index参数来禁用分组键作为索引
```python
tips.groupby(['day', 'smoker'], as_index = False).mean()
	day	smoker   tip    size    tip_pct
0	Sat	No	1.660	3.0	0.1605
1	Sat	Yes     3.460	3.0	0.1433
2	Sun	No	2.255	2.5	0.1130
3	Sun	Yes     4.710	4.0	0.1862
```

## 应用：通用拆分-应用-联合

- 使用apply对分组对象使用函数
```python
def top(df, column = 'tip_pct'):
    return df.sort_values(by = column)
tips.groupby('smoker').apply(top)
                 tip   smoker   day    size    tip_pct
smoker						
No         0	1.01    No	Sun	2	0.0594
           1	1.66	No	Sat	3	0.1605 
           2	3.50	No	Sun	3	0.1666
Yes        3	3.31	Yes	Sat	2	0.1398
           4	3.61	Yes	Sat	4	0.1468
           5	4.71	Yes	Sun	4	0.1862
tips.group('smoker').apply(top, column = 'tip') #也可在apply中传递函数参数
```

### 压缩分组键

- 压缩分组键以禁用分层索引
```python
tips.groupby('smoker', group_keys = False).apply(top)
         tip  Smoker day      size     tip_pct
0	1.01	No   Sun	2	0.0594
1	1.66	No   Sat	3	0.1605
2	3.50	No   Sun	3	0.1666
3	3.31	Yes  Sat	2	0.1398
4	3.61	Yes  Sat	4	0.1468
5	4.71	Yes  Sun	4	0.1862
```

### 分位数与桶分析

- cut、qcut的分箱返回的Categorical对象与groupby结合
```
frame = pd.DataFrame({'data1': np.random.randn(1000), 'data2': np.random.randn(1000)})
quartiles = pd.cut(frame.data1, 4)
grouped = frame.data2.groupby(quartiles)
grouped.agg(['min','max','count','mean'])
                       min         max         count	mean
data1				
(-3.21, -1.733]   -2.049611	1.190694	33   -0.242130
(-1.733, -0.262]  -2.462223	2.793166	356  -0.062656
(-0.262, 1.209]   -3.317669	2.650690	520   0.008558
(1.209, 2.68]     -2.376758	2.456308	91    0.162135
groupping = pd.qcut(frame.data1, 10, labels = False)
grouped = frame.data2.groupby(groupping)
grouped.agg(['min','max','count','mean']) 
          min              max         count       mean
data1				
0	-2.411232	2.793166	100	-0.114321
1	-2.333174	2.448005	100	-0.231702
2	-2.462223	2.397814	100	 0.044576
3	-2.429924	2.105609	100	-0.010368
4	-2.245557	2.494901	100	 0.039768
5	-3.112645	2.324049	100	-0.083543
6	-3.317669	2.317557	100	 0.036278
7	-2.501311	2.650690	100	 0.086418
8	-2.503667	2.216064	100	-0.006586
9	-2.376758	2.456308	100	 0.128563
```

### 示例：使用指定分组值填充缺失值

```python
states = ['Ohio', 'New York', 'Vermont', 'Florida', 'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = pd.Series(np.random.randn(8), index = states)
data[['Vermont', 'Navada', 'Idaho']] = np.nan
data.groupby(group_key).mean()
East    0.679494
West    0.443424
dtype: float64
fill_mean = lambda x: x.fillna(x.mean())
data.groupby(group_key).apply(fill_mean)
Ohio          1.331587
New York      0.715279
Vermont       0.679494
Florida      -0.008384
Oregon        0.621336
Nevada        0.443424
California    0.265512
Idaho         0.443424
dtype: float64
fill_values = {'East': 0.5, 'West': -1}
fill_fun = lambda x: x.fillna(fill_values[x.name])
data.groupby(group_key).apply(fill_fun)
Ohio          1.331587
New York      0.715279
Vermont       0.500000
Florida      -0.008384
Oregon        0.621336
Nevada       -1.000000
California    0.265512
Idaho        -1.000000
dtype: float64
```

### 示例：随机采样与排列

```python
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + list('JQK')
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)
deck = pd.Series(card_val, cards) #deck就是我们想要的牌堆
deck[:13]
AH      1
2H      2
3H      3
4H      4
5H      5
6H      6
7H      7
8H      8
9H      9
10H    10
JH     10
QH     10
KH     10
dtype: int64
deck.sample(5)
KD    10
4H     4
3H     3
7H     7
9S     9
dtype: int64
get_suit = lambda card: card[-1] #last letter is suit
deck.groupby(get_suit).apply(lambda x:x.sample(2))
C  2C     2
   8C     8
D  KD    10
   5D     5
H  2H     2
   7H     7
S  5S     5
   QS    10
dtype: int64
```


### 示例：分组加权平均

```python
df = pd.DataFrame({'category': list('aaaabbbb'), 
                   'data': np.random.randn(8),
                   'weights': np.random.rand(8)})
df.groupby('category').apply(lambda x: np.average(x['data'], weights = x['weights']))
category
a    0.312180
b   -0.536204
dtype: float64
```

## 数据透视表与交叉表

- 使用pandas的pivot_table函数来建立一个透视表
```python
tips = pd.DataFrame({'tip': [1.01, 1.66, 3.50, 3.31, 3.61, 4.71],
                   'smoker': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
                   'day': ['Sun', 'Sat', 'Sun', 'Sat', 'Sat', 'Sun'],
                   'size': [2, 3, 3, 2, 4, 4],
                   'tip_pct': [0.0594, 0.1605, 0.1666, 0.1398, 0.1468, 0.1862]})
tips.pivot_table(index = ['day', 'smoker']) #参数index控制透视表的行，最后结果返回的是分组的平均值
           size  tip  tip_pct
day smoker			
Sat  No    3.0  1.660  0.1605
     Yes   3.0  3.460  0.1433
Sun  No    2.5  2.255  0.1130
     Yes   4.0  4.710  0.1862
tips.pivot_table(['tip_pct', 'size'], index = 'day', columns = 'smoker') #columns控制透视表的列
          size      tip_pct
smoker  No   Yes  No      Yes
day				
Sat     3.0  3.0  0.1605  0.1433
Sun     2.5  4.0  0.1130  0.1862
```

- pivot_table的其他参数
```python
aggfunc #聚合函数，默认为'mean'
fill_value #替换缺失值的值
dropna #如果为True，则不含所有条目为NA的列
margins #添加行/列小计和总计,默认为False
```

### 交叉表：crosstab

- 交叉表是数据透视表中的特殊情况，计算的是分组中的频率
```python
data = pd.DataFrame({'sample': range(1, 11),
                     'Nationality': ['USA', 'Japan', 'USA', 'Japan', 'Japan',
                                     'Japan', 'USA', 'USA', 'Japan', 'USA'],
                     'Handedness': ['Right', 'Left', 'Right', 'Right', 'Left',
                                    'Right', 'Right', 'Left', 'Right', 'Right']})
pd.crosstab(data.Nationality, data.Handedness, margins = True)
Handedness   Left Right All
Nationality			
Japan         2     3	 5
USA           1     4    5
All           3     7    10
```
                     

# 第十一章

# 第十二章：高阶pandas

## 分类数据

- 本节主要介绍Categorical类型

### 背景和目标

- 使用take来转换Series
```python
value = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
dim.take(value)
0     apple
1    orange
0     apple
0     apple
0     apple
1    orange
0     apple
0     apple
dtype: object
```


```python

```


```python

```
