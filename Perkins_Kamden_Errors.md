```python
#This code has an intentional error. You can type it directly 
#or use it for reference to understand the error message below.

def favorite_ice_cream():
    ice_creams = [
        'chocolate',
        'vanilla',
        'strawberry'
    ]
    print(ice_creams[3])

favorite_ice_cream()
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-1-ee7ec9b51a02> in <module>
         10     print(ice_creams[3])
         11 
    ---> 12 favorite_ice_cream()
    

    <ipython-input-1-ee7ec9b51a02> in favorite_ice_cream()
          8         'strawberry'
          9     ]
    ---> 10     print(ice_creams[3])
         11 
         12 favorite_ice_cream()


    IndexError: list index out of range



```python
def some_function():
    msg = 'hello, world!'
    print(msg)
     return msg
```


      File "<ipython-input-3-b1c076544652>", line 4
        return msg
        ^
    IndentationError: unexpected indent




```python
def some_function():
    msg = 'hello, world!'
    print(msg)
    return msg
```


```python
print(a)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-bca0e2660b9f> in <module>
    ----> 1 print(a)
    

    NameError: name 'a' is not defined



```python
print(hello)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-84af27eb1a31> in <module>
    ----> 1 print(hello)
          2 print("hello")


    NameError: name 'hello' is not defined



```python
print("hello")
```

    hello



```python
for number in range(10):
    count = count + number
print('The count is:', count)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-64cce701dab9> in <module>
          1 for number in range(10):
    ----> 2     count = count + number
          3 print('The count is:', count)


    NameError: name 'count' is not defined



```python
count = 0

for number in range(10):
    count = count + number
print('The count is:', count)
```

    The count is: 45



```python
letters = ['a', 'b', 'c']

print('Letter #1 is', letters[0])
print('Letter #2 is', letters[1])
print('Letter #3 is', letters[2])
print('Letter #4 is', letters[3])
```

    Letter #1 is a
    Letter #2 is b
    Letter #3 is c



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-12-9127bae7a87f> in <module>
          4 print('Letter #2 is', letters[1])
          5 print('Letter #3 is', letters[2])
    ----> 6 print('Letter #4 is', letters[3])
    

    IndexError: list index out of range



```python
letters = ['a', 'b', 'c']

print('Letter #1 is', letters[0])
print('Letter #2 is', letters[1])
print('Letter #3 is', letters[2])
#print('Letter #4 is', letters[3])
```

    Letter #1 is a
    Letter #2 is b
    Letter #3 is c



```python
file_handle = open('myfile.txt', 'r')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-14-080b0e91cc46> in <module>
    ----> 1 file_handle = open('myfile.txt', 'r')
    

    FileNotFoundError: [Errno 2] No such file or directory: 'myfile.txt'



```python
file_handle = open('myfile.txt', 'w')
file_handle.read()
```


    ---------------------------------------------------------------------------

    UnsupportedOperation                      Traceback (most recent call last)

    <ipython-input-15-063a9999adc0> in <module>
          1 file_handle = open('myfile.txt', 'w')
    ----> 2 file_handle.read()
    

    UnsupportedOperation: not readable

