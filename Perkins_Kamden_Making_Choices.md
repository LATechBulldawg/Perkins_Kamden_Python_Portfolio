```python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater than 100')
print('...after conditional')
```

    before conditional...
    ...after conditional



```python
num = 14

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    14 is positive



```python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```

    at least one part is false



```python
if (-1 > 0) or (1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    at least one part is true



```python
import numpy
```


```python
data = numpy.loadtxt(fname='/home/student/Desktop/classroom/myfiles/swc-python/data/inflammation-01.csv', delimiter=',')
```


```python
max_inflammation_0 = numpy.amax(data, axis=0)[0]
max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
```

    Suspicious looking maxima!



```python
max_inflammation_0 = numpy.amax(data, axis=0)[0]
max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')

elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero!')
    
else:
        print('Seems OK!')
```

    Suspicious looking maxima!



```python
data = numpy.loadtxt(fname='/home/student/Desktop/classroom/myfiles/swc-python/data/inflammation-03.csv', delimiter=',')

max_inflammation_0 = numpy.amax(data, axis=0)[0]

max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems OK!')
```

    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!



```python

```
