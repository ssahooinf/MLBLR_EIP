
# coding: utf-8

# <!--
# Python:
#   Simple data types
#     integer, float, string
#   Compound data types
#     tuple, list, dictionary, set
#   Flow control
#     if, while, for, try, with
#   Comprehensions, generators
#   Functions
#   Classes
#   Standard library
#     json, collections, itertools
# 
# Numpy
# -->
# 
# This tutorial was contributed by [Justin Johnson](http://cs.stanford.edu/people/jcjohns/).
# 
# We will use the Python programming language for all assignments in this course.
# Python is a great general-purpose programming language on its own, but with the
# help of a few popular libraries (numpy, scipy, matplotlib) it becomes a powerful
# environment for scientific computing.
# 
# We expect that many of you will have some experience with Python and numpy;
# for the rest of you, this section will serve as a quick crash course both on
# the Python programming language and on the use of Python for scientific
# computing.
# 
# Some of you may have previous knowledge in Matlab, in which case we also recommend the [numpy for Matlab users](http://wiki.scipy.org/NumPy_for_Matlab_Users) page.
# 
# You can also find an [IPython notebook version of this tutorial here](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb) created by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) for [CS 228](http://cs.stanford.edu/~ermon/cs228/index.html).
# 
# Table of contents:
# 
# - [Python](#python)
#   - [Basic data types](#python-basic)
#   - [Containers](#python-containers)
#       - [Lists](#python-lists)
#       - [Dictionaries](#python-dicts)
#       - [Sets](#python-sets)
#       - [Tuples](#python-tuples)
#   - [Functions](#python-functions)
#   - [Classes](#python-classes)
# - [Numpy](#numpy)
#   - [Arrays](#numpy-arrays)
#   - [Array indexing](#numpy-array-indexing)
#   - [Datatypes](#numpy-datatypes)
#   - [Array math](#numpy-math)
#   - [Broadcasting](#numpy-broadcasting)
# - [SciPy](#scipy)
#   - [Image operations](#scipy-image)
#   - [MATLAB files](#scipy-matlab)
#   - [Distance between points](#scipy-dist)
# - [Matplotlib](#matplotlib)
#   - [Plotting](#matplotlib-plotting)
#   - [Subplots](#matplotlib-subplots)
#   - [Images](#matplotlib-images)
# 
# <a eip_in='python'></a>
# 
# ## Python
# 
# Python is a high-level, dynamically typed multiparadigm programming language.
# Python code is often said to be almost like pseudocode, since it allows you
# to express very powerful ideas in very few lines of code while being very
# readable. As an example, here is an implementation of the classic quicksort
# algorithm in Python:

# In[1]:


def quicksort(eip_in):
    if len(eip_in) <= 1:
        return eip_in
    eip_list = eip_in[len(eip_in) // 2]
    mlblr_in = [eip for eip in eip_in if eip < eip_list]
    eip_out = [eip for eip in eip_in if eip == eip_list]
    mlblr_out = [eip for eip in eip_in if eip > eip_list]
    return quicksort(mlblr_in) + eip_out + quicksort(mlblr_out)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"


# ### Python versions
# There are currently two different supported versions of Python, 2.7 and 3.5.
# Somewhat confusingly, Python 3.0 introduced many backwards-incompatible changes
# to the language, so code written for 2.7 may not work under 3.5 and vice versa.
# For this class all code will use Python 3.5.
# 
# You can check your Python version at the command line by running
# `python --version`.
# 
# <a eip_in='python-basic'></a>
# 
# ### Basic data types
# 
# Like most languages, Python has a number of basic types including integers,
# floats, booleans, and strings. These data types behave in ways that are
# familiar from other programming languages.
# 
# **Numbers:** Integers and floats work as you would expect from other languages:

# In[2]:


eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Exponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
mlblr = 2.5
print(type(mlblr)) # Prints "<class 'float'>"
print(mlblr, mlblr + 1, mlblr * 2, mlblr ** 2) # Prints "2.5 3.5 5.0 6.25"


# Note that unlike many languages, Python does not have unary increment (`x++`)
# or decrement (`x--`) operators.
# 
# Python also has built-in types for complex numbers;
# you can find all of the details
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex).
# 
# **Booleans:** Python implements all of the usual operators for Boolean logic,
# but uses English words rather than symbols (`&&`, `||`, etc.):

# In[3]:


eip_in = True
eip_out = False
print(type(eip_in)) # Prints "<class 'bool'>"
print(eip_in and eip_out) # Logical AND; prints "False"
print(eip_in or eip_out)  # Logical OR; prints "True"
print(not eip_in)   # Logical NOT; prints "False"
print(eip_in != eip_out)  # Logical XOR; prints "True"


# **Strings:** Python has great support for strings:

# In[4]:


mlblr = 'hello'    # String literals can use single quotes
eip = "world"    # or double quotes; it does not matter.
print(mlblr)       # Prints "hello"
print(len(mlblr))  # String length; prints "5"
eip_in = mlblr + ' ' + eip  # String concatenation
print(eip_in)  # prints "hello world"
eip_out = '%s %s %d' % (mlblr, eip, 12)  # sprintf style string formatting
print(eip_out)  # prints "hello world 12"


# String objects have a bunch of useful methods; for example:

# In[5]:


eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"


# You can find a list of all string methods [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#string-methods).
# 
# <a eip_in='python-containers'></a>
# 
# ### Containers
# Python includes several built-in container types: lists, dictionaries, sets, and tuples.
# 
# <a eip_in='python-lists'></a>
# 
# #### Lists
# A list is the Python equivalent of an array, but is resizeable
# and can contain elements of different types:

# In[6]:


eip = [3, 1, 2]    # Create a list
print(eip, eip[2])  # Prints "[3, 1, 2] 2"
print(eip[-1])     # Negative indices count from the end of the list; prints "2"
eip[2] = 'foo'     # Lists can contain elements of different types
print(eip)         # Prints "[3, 1, 'foo']"
eip.append('bar')  # Add a new element to the end of the list
print(eip)         # Prints "[3, 1, 'foo', 'bar']"
eip = eip.pop()      # Remove and return the last element of the list
print(eip, eip)      # Prints "bar [3, 1, 'foo']"


# As usual, you can find all the gory details about lists
# [in the documentation](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists).
# 
# **Slicing:**
# In addition to accessing list elements one at a time, Python provides
# concise syntax to access sublists; this is known as *slicing*:

# In[7]:


mlblr = list(range(5))     # range is a built-in function that creates a list of integers
print(mlblr)               # Prints "[0, 1, 2, 3, 4]"
print(mlblr[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(mlblr[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(mlblr[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(mlblr[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(mlblr[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
mlblr[2:4] = [8, 9]        # Assign a new sublist to a slice
print(mlblr)               # Prints "[0, 1, 8, 9, 4]"


# We will see slicing again in the context of numpy arrays.
# 
# **Loops:** You can loop over the elements of a list like this:

# In[8]:


mlblr = ['cat', 'dog', 'monkey']
for eip in mlblr:
    print(eip)
# Prints "cat", "dog", "monkey", each on its own line.


# If you want access to the index of each element within the body of a loop,
# use the built-in `enumerate` function:

# In[9]:


mlblr = ['cat', 'dog', 'monkey']
for mlblr_in, eip in enumerate(mlblr):
    print('#%d: %s' % (mlblr_in + 1, eip))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line


# **List comprehensions:**
# When programming, frequently we want to transform one type of data into another.
# As a simple example, consider the following code that computes square numbers:

# In[10]:


mlblr = [0, 1, 2, 3, 4]
squares = []
for eip in mlblr:
    squares.append(eip ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]


# You can make this code simpler using a **list comprehension**:

# In[11]:


mlblr = [0, 1, 2, 3, 4]
squares = [eip ** 2 for eip in mlblr]
print(squares)   # Prints [0, 1, 4, 9, 16]


# List comprehensions can also contain conditions:

# In[12]:


mlblr = [0, 1, 2, 3, 4]
even_squares = [eip ** 2 for eip in mlblr if eip % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"


# <a eip_in='python-dicts'></a>
# 
# #### Dictionaries
# A dictionary stores (key, value) pairs, similar to a `Map` in Java or
# an object in Javascript. You can use it like this:

# In[13]:


eip_dict = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip_dict['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip_dict)     # Check if a dictionary has a given key; prints "True"
eip_dict['fish'] = 'wet'     # Set an entry in a dictionary
print(eip_dict['fish'])      # Prints "wet"
# print(eip_dict['monkey'])  # KeyError: 'monkey' not a key of eip_dict
print(eip_dict.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip_dict.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip_dict['fish']         # Remove an element from a dictionary
print(eip_dict.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# You can find all you need to know about dictionaries
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#dict).
# 
# **Loops:** It is easy to iterate over the keys in a dictionary:

# In[14]:


eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip in eip_dict:
    legs = eip_dict[eip]
    print('A %s has %d legs' % (eip, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"


# If you want access to keys and their corresponding values, use the `items` method:

# In[15]:


eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip, legs in eip_dict.items():
    print('A %s has %d legs' % (eip, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"


# **Dictionary comprehensions:**
# These are similar to list comprehensions, but allow you to easily construct
# dictionaries. For example:

# In[16]:


mlblr = [0, 1, 2, 3, 4]
even_num_to_square = {eip: eip ** 2 for eip in mlblr if eip % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"


# <a eip_in='python-sets'></a>
# 
# #### Sets
# A set is an unordered collection of distinct elements. As a simple example, consider
# the following:

# In[17]:


mlblr = {'cat', 'dog'}
print('cat' in mlblr)   # Check if an element is in a set; prints "True"
print('fish' in mlblr)  # prints "False"
mlblr.add('fish')       # Add an element to a set
print('fish' in mlblr)  # Prints "True"
print(len(mlblr))       # Number of elements in a set; prints "3"
mlblr.add('cat')        # Adding an element that is already in the set does nothing
print(len(mlblr))       # Prints "3"
mlblr.remove('cat')     # Remove an element from a set
print(len(mlblr))       # Prints "2"


# As usual, everything you want to know about sets can be found
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#set).
# 
# 
# **Loops:**
# Iterating over a set has the same syntax as iterating over a list;
# however since sets are unordered, you cannot make assumptions about the order
# in which you visit the elements of the set:

# In[18]:


mlblr = {'cat', 'dog', 'fish'}
for mlblr_in, eip in enumerate(mlblr):
    print('#%d: %s' % (mlblr_in + 1, eip))
# Prints "#1: fish", "#2: dog", "#3: cat"


# **Set comprehensions:**
# Like lists and dictionaries, we can easily construct sets using set comprehensions:

# In[19]:


from math import sqrt
mlblr = {int(sqrt(eip)) for eip in range(30)}
print(mlblr)  # Prints "{0, 1, 2, 3, 4, 5}"


# <a eip_in='python-tuples'></a>
# 
# #### Tuples
# A tuple is an (immutable) ordered list of values.
# A tuple is in many ways similar to a list; one of the most important differences is that
# tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.
# Here is a trivial example:

# In[20]:


eip_dict = {(eip, eip + 1): eip for eip in range(10)}  # Create a dictionary with tuple keys
eip_in = (5, 6)        # Create a tuple
print(type(eip_in))    # Prints "<class 'tuple'>"
print(eip_dict[eip_in])       # Prints "5"
print(eip_dict[(1, 2)])  # Prints "1"


# [The documentation](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences) has more information about tuples.
# 
# <a eip_in='python-functions'></a>
# 
# ### Functions
# Python functions are defined using the `def` keyword. For example:

# In[21]:


def sign(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero'

for eip in [-1, 0, 1]:
    print(sign(eip))
# Prints "negative", "zero", "positive"


# We will often define functions to take optional keyword arguments, like this:

# In[22]:


def mlblr(eip_in, eip_out=False):
    if eip_out:
        print('HELLO, %s!' % eip_in.upper())
    else:
        print('Hello, %s' % eip_in)

mlblr('Bob') # Prints "Hello, Bob"
mlblr('Fred', eip_out=True)  # Prints "HELLO, FRED!"


# There is a lot more information about Python functions
# [in the documentation](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions).
# 
# <a eip_in='python-classes'></a>
# 
# ### Classes
# 
# The syntax for defining classes in Python is straightforward:

# In[23]:


class mlblr(object):

    # Constructor
    def __init__(eip, eip_in):
        eip.eip_in = eip_in  # Create an instance variable

    # Instance method
    def greet(eip, eip_out=False):
        if eip_out:
            print('HELLO, %s!' % eip.eip_in.upper())
        else:
            print('Hello, %s' % eip.eip_in)

mlblr_in = mlblr('Fred')  # Construct an instance of the mlblr class
mlblr_in.greet()            # Call an instance method; prints "Hello, Fred"
mlblr_in.greet(eip_out=True)   # Call an instance method; prints "HELLO, FRED!"


# You can read a lot more about Python classes
# [in the documentation](https://docs.python.org/3.5/tutorial/classes.html).
# 
# <a eip_in='numpy'></a>
# 
# ## Numpy
# 
# [Numpy](http://www.numpy.org/) is the core library for scientific computing in Python.
# It provides a high-performance multidimensional array object, and tools for working with these
# arrays. If you are already familiar with MATLAB, you might find
# [this tutorial useful](http://wiki.scipy.org/NumPy_for_Matlab_Users) to get started with Numpy.
# 
# <a eip_in='numpy-arrays'></a>
# 
# ### Arrays
# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of
# nonnegative integers. The number of dimensions is the *rank* of the array; the *shape*
# of an array is a tuple of integers giving the size of the array along each dimension.
# 
# We can initialize numpy arrays from nested Python lists,
# and access elements using square brackets:

# In[24]:


import numpy as np

mlblr_in = np.array([1, 2, 3])   # Create a rank 1 array
print(type(mlblr_in))            # Prints "<class 'numpy.ndarray'>"
print(mlblr_in.shape)            # Prints "(3,)"
print(mlblr_in[0], mlblr_in[1], mlblr_in[2])   # Prints "1 2 3"
mlblr_in[0] = 5                  # Change an element of the array
print(mlblr_in)                  # Prints "[5, 2, 3]"

mlblr_out = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlblr_out.shape)                     # Prints "(2, 3)"
print(mlblr_out[0, 0], mlblr_out[0, 1], mlblr_out[1, 0])   # Prints "1 2 4"


# Numpy also provides many functions to create arrays:

# In[25]:


import numpy as np

mlblr_in = np.zeros((2,2))   # Create an array of all zeros
print(mlblr_in)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlblr_out = np.ones((1,2))    # Create an array of all ones
print(mlblr_out)              # Prints "[[ 1.  1.]]"

eip_in = np.full((2,2), 7)  # Create a constant array
print(eip_in)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

eip_dict = np.eye(2)         # Create a 2x2 identity matrix
print(eip_dict)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

eip_out = np.random.random((2,2))  # Create an array filled with random values
print(eip_out)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"


# You can read about other methods of array creation
# [in the documentation](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation).
# 
# <a eip_in='numpy-array-indexing'></a>
# 
# ### Array indexing
# Numpy offers several ways to index into arrays.
# 
# **Slicing:**
# Similar to Python lists, numpy arrays can be sliced.
# Since arrays may be multidimensional, you must specify a slice for each dimension
# of the array:

# In[26]:


import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
mlblr_in = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
mlblr_out = mlblr_in[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(mlblr_in[0, 1])   # Prints "2"
mlblr_out[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(mlblr_in[0, 1])   # Prints "77"


# You can also mix integer indexing with slice indexing.
# However, doing so will yield an array of lower rank than the original array.
# Note that this is quite different from the way that MATLAB handles array
# slicing:

# In[27]:


import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
mlblr_in = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
eip = mlblr_in[1, :]    # Rank 1 view of the second row of a
mlblr = mlblr_in[1:2, :]  # Rank 2 view of the second row of a
print(eip, eip.shape)  # Prints "[5 6 7 8] (4,)"
print(mlblr, mlblr.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
eip_in = mlblr_in[:, 1]
eip_out = mlblr_in[:, 1:2]
print(eip_in, eip_in.shape)  # Prints "[ 2  6 10] (3,)"
print(eip_out, eip_out.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"


# **Integer array indexing:**
# When you index into numpy arrays using slicing, the resulting array view
# will always be a subarray of the original array. In contrast, integer array
# indexing allows you to construct arbitrary arrays using the data from another
# array. Here is an example:

# In[28]:


import numpy as np

mlblr_in = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(mlblr_in[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([mlblr_in[0, 0], mlblr_in[1, 1], mlblr_in[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(mlblr_in[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([mlblr_in[0, 1], mlblr_in[0, 1]]))  # Prints "[2 2]"


# One useful trick with integer array indexing is selecting or mutating one
# element from each row of a matrix:

# In[29]:


import numpy as np

# Create a new array from which we will select elements
mlblr_in = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(mlblr_in)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
mlblr_out = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(mlblr_in[np.arange(4), mlblr_out])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
mlblr_in[np.arange(4), mlblr_out] += 10

print(mlblr_in)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])


# **Boolean array indexing:**
# Boolean array indexing lets you pick out arbitrary elements of an array.
# Frequently this type of indexing is used to select the elements of an array
# that satisfy some condition. Here is an example:

# In[30]:


import numpy as np

mlblr_in = np.array([[1,2], [3, 4], [5, 6]])

eip_in = (mlblr_in > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of eip_in tells
                     # whether that element of a is > 2.

print(eip_in)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of eip_in
print(mlblr_in[eip_in])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(mlblr_in[mlblr_in > 2])     # Prints "[3 4 5 6]"


# For brevity we have left out a lot of details about numpy array indexing;
# if you want to know more you should
# [read the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).
# 
# <a eip_in='numpy-datatypes'></a>
# 
# ### Datatypes
# Every numpy array is a grid of elements of the same type.
# Numpy provides a large set of numeric datatypes that you can use to construct arrays.
# Numpy tries to guess a datatype when you create an array, but functions that construct
# arrays usually also include an optional argument to explicitly specify the datatype.
# Here is an example:

# In[31]:


import numpy as np

eip = np.array([1, 2])   # Let numpy choose the datatype
print(eip.dtype)         # Prints "int64"

eip = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(eip.dtype)             # Prints "float64"

eip = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip.dtype)                         # Prints "int64"


# You can read all about numpy datatypes
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).
# 
# <a eip_in='numpy-math'></a>
# 
# ### Array math
# Basic mathematical functions operate elementwise on arrays, and are available
# both as operator overloads and as functions in the numpy module:

# In[32]:


import numpy as np

eip = np.array([[1,2],[3,4]], dtype=np.float64)
mlblr = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip + mlblr)
print(np.add(eip, mlblr))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip - mlblr)
print(np.subtract(eip, mlblr))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip * mlblr)
print(np.multiply(eip, mlblr))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip / mlblr)
print(np.divide(eip, mlblr))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(eip))


# Note that unlike MATLAB, `*` is elementwise multiplication, not matrix
# multiplication. We instead use the `dot` function to compute inner
# products of vectors, to multiply a vector by a matrix, and to
# multiply matrices. `dot` is available both as a function in the numpy
# module and as an instance method of array objects:

# In[33]:


import numpy as np

eip = np.array([[1,2],[3,4]])
mlblr = np.array([[5,6],[7,8]])

eip_in = np.array([9,10])
eip_out = np.array([11, 12])

# Inner product of vectors; both produce 219
print(eip_in.dot(eip_out))
print(np.dot(eip_in, eip_out))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip.dot(eip_in))
print(np.dot(eip, eip_in))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip.dot(mlblr))
print(np.dot(eip, mlblr))


# Numpy provides many useful functions for performing computations on
# arrays; one of the most useful is `sum`:

# In[34]:


import numpy as np

eip = np.array([[1,2],[3,4]])

print(np.sum(eip))  # Compute sum of all elements; prints "10"
print(np.sum(eip, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip, axis=1))  # Compute sum of each row; prints "[3 7]"


# You can find the full list of mathematical functions provided by numpy
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).
# 
# Apart from computing mathematical functions using arrays, we frequently
# need to reshape or otherwise manipulate data in arrays. The simplest example
# of this type of operation is transposing a matrix; to transpose a matrix,
# simply use the `T` attribute of an array object:

# In[35]:


import numpy as np

eip = np.array([[1,2], [3,4]])
print(eip)    # Prints "[[1 2]
            #          [3 4]]"
print(eip.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
eip_in = np.array([1,2,3])
print(eip_in)    # Prints "[1 2 3]"
print(eip_in.T)  # Prints "[1 2 3]"


# Numpy provides many more functions for manipulating arrays; you can see the full list
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html).
# 
# 
# <a eip_in='numpy-broadcasting'></a>
# 
# ### Broadcasting
# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different
# shapes when performing arithmetic operations. Frequently we have a smaller array and a
# larger array, and we want to use the smaller array multiple times to perform some operation
# on the larger array.
# 
# For example, suppose that we want to add a constant vector to each
# row of a matrix. We could do it like this:

# In[36]:


import numpy as np

# We will add the vector eip_in to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_in = np.array([1, 0, 1])
mlblr = np.empty_like(eip)   # Create an empty matrix with the same shape as x

# Add the vector eip_in to each row of the matrix x with an explicit loop
for i in range(4):
    mlblr[i, :] = eip[i, :] + eip_in

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(mlblr)


# This works; however when the matrix `x` is very large, computing an explicit loop
# in Python could be slow. Note that adding the vector `eip_in` to each row of the matrix
# `x` is equivalent to forming a matrix `eip_out` by stacking multiple copies of `eip_in` vertically,
# then performing elementwise summation of `x` and `eip_out`. We could implement this
# approach like this:

# In[37]:


import numpy as np

# We will add the vector eip_in to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_in = np.array([1, 0, 1])
eip_out = np.tile(eip_in, (4, 1))   # Stack 4 copies of eip_in on top of each other
print(eip_out)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
mlblr = eip + eip_out  # Add x and eip_out elementwise
print(mlblr)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


# Numpy broadcasting allows us to perform this computation without actually
# creating multiple copies of `eip_in`. Consider this version, using broadcasting:

# In[38]:


import numpy as np

# We will add the vector eip_in to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_in = np.array([1, 0, 1])
mlblr = eip + eip_in  # Add eip_in to each row of x using broadcasting
print(mlblr)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


# The line `y = x + eip_in` works even though `x` has shape `(4, 3)` and `eip_in` has shape
# `(3,)` due to broadcasting; this line works as if `eip_in` actually had shape `(4, 3)`,
# where each row was a copy of `eip_in`, and the sum was performed elementwise.
# 
# Broadcasting two arrays together follows these rules:
# 
# 1. If the arrays do not have the same rank, prepend the shape of the lower rank array
#    with 1s until both shapes have the same length.
# 2. The two arrays are said to be *compatible* in a dimension if they have the same
#    size in the dimension, or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise
#    maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1,
#    the first array behaves as if it were copied along that dimension
# 
# If this explanation does not make sense, try reading the explanation
# [from the documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
# or [this explanation](http://wiki.scipy.org/EricsBroadcastingDoc).
# 
# Functions that support broadcasting are known as *universal functions*. You can find
# the list of all universal functions
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).
# 
# Here are some applications of broadcasting:

# In[39]:


import numpy as np

# Compute outer product of vectors
eip_in = np.array([1,2,3])  # eip_in has shape (3,)
eip_out = np.array([4,5])    # eip_out has shape (2,)
# To compute an outer product, we first reshape eip_in to be a column
# vector of shape (3, 1); we can then broadcast it against eip_out to yield
# an output of shape (3, 2), which is the outer product of eip_in and eip_out:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(eip_in, (3, 1)) * eip_out)

# Add a vector to each row of a matrix
eip = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and eip_in has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip + eip_in)

# Add a vector to each column of a matrix
# x has shape (2, 3) and eip_out has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against eip_out to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector eip_out added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip.T + eip_out).T)
# Another solution is to reshape eip_out to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(eip + np.reshape(eip_out, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip * 2)


# Broadcasting typically makes your code more concise and faster, so you
# should strive to use it where possible.
# 
# ### Numpy Documentation
# This brief overview has touched on many of the important things that you need to
# know about numpy, but is far from complete. Check out the
# [numpy reference](http://docs.scipy.org/doc/numpy/reference/)
# to find out much more about numpy.
# 
# <a eip_in='scipy'></a>
# 
# ## SciPy
# Numpy provides a high-performance multidimensional array and basic tools to
# compute with and manipulate these arrays.
# [SciPy](http://docs.scipy.org/doc/scipy/reference/)
# builds on this, and provides
# a large number of functions that operate on numpy arrays and are useful for
# different types of scientific and engineering applications.
# 
# The best way to get familiar with SciPy is to
# [browse the documentation](http://docs.scipy.org/doc/scipy/reference/index.html).
# We will highlight some parts of SciPy that you might find useful for this class.
# 
# <a eip_in='scipy-image'></a>
# 
# ### Image operations
# SciPy provides some basic functions to work with images.
# For example, it has functions to read images from disk into numpy arrays,
# to write numpy arrays to disk as images, and to resize images.
# Here is a simple example that showcases these functions:

# In[40]:


from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
eip = imread('assets/cat.jpg')
print(eip.dtype, eip.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
mlblr = eip * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
mlblr = imresize(mlblr, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', mlblr)


# <div class='fig figcenter fighighlight'>
#   <eip src='assets/cat.jpg'>
#   <eip src='assets/cat_tinted.jpg'>
#   <div class='figcaption'>
#     Left: The original image.
#     Right: The tinted and resized image.
#   </div>
# </div>
# 
# <a eip_in='scipy-matlab'></a>
# 
# ### MATLAB files
# The functions `scipy.io.loadmat` and `scipy.io.savemat` allow you to read and
# write MATLAB files. You can read about them
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/io.html).
# 
# <a eip_in='scipy-dist'></a>
# 
# ### Distance between points
# SciPy defines some useful functions for computing distances between sets of points.
# 
# The function `scipy.spatial.distance.pdist` computes the distance between all pairs
# of points in a given set:

# In[ ]:


import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
eip = np.array([[0, 1], [1, 0], [2, 0]])
print(eip)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
eip_dict = squareform(pdist(eip, 'euclidean'))
print(eip_dict)


# You can read all the details about this function
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
# 
# A similar function (`scipy.spatial.distance.cdist`) computes the distance between all pairs
# across two sets of points; you can read about it
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
# 
# <a eip_in='matplotlib'></a>
# 
# ## Matplotlib
# [Matplotlib](http://matplotlib.org/) is a plotting library.
# In this section give a brief introduction to the `matplotlib.pyplot` module,
# which provides a plotting system similar to that of MATLAB.
# 
# <a eip_in='matplotlib-plot'></a>
# 
# ### Plotting
# The most important function in matplotlib is `plot`,
# which allows you to plot 2D data. Here is a simple example:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip = np.arange(0, 3 * np.pi, 0.1)
mlblr = np.sin(eip)

# Plot the points using matplotlib
plt.plot(eip, mlblr)
plt.show()  # You must call plt.show() to make graphics appear.


# Running this code produces the following plot:
# 
# <div class='fig figcenter fighighlight'>
#   <eip src='assets/sine.png'>
# </div>
# 
# With just a little bit of extra work we can easily plot multiple lines
# at once, and add a title, legend, and axis labels:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_in = np.sin(eip)
eip_out = np.cos(eip)

# Plot the points using matplotlib
plt.plot(eip, eip_in)
plt.plot(eip, eip_out)
plt.xlabel('eip axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# <div class='fig figcenter fighighlight'>
#   <eip src='assets/sine_cosine.png'>
# </div>
# 
# You can read much more about the `plot` function
# [in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).
# 
# <a eip_in='matplotlib-subplots'></a>
# 
# ### Subplots
# You can plot different things in the same figure using the `subplot` function.
# Here is an example:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_in = np.sin(eip)
eip_out = np.cos(eip)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(eip, eip_in)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(eip, eip_out)
plt.title('Cosine')

# Show the figure.
plt.show()


# <div class='fig figcenter fighighlight'>
#   <eip src='assets/sine_cosine_subplot.png'>
# </div>
# 
# You can read much more about the `subplot` function
# [in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).
# 
# <a eip_in='matplotlib-images'></a>
# 
# ### Images
# You can use the `imshow` function to show images. Here is an example:

# In[ ]:


import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

eip = imread('assets/cat.jpg')
mlblr = eip * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(eip)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(mlblr))
plt.show()


# <div class='fig figcenter fighighlight'>
#   <eip src='assets/cat_tinted_imshow.png'>
# </div>
