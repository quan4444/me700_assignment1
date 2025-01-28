## Bisection method

The bisection  method is a root-finding method that applies to any continuous function for which one knows two values with opposite signs [1]. While the method is simple, it comes with some disadvantages - slow convergence, requires bracketing, not suitable for multiple roots, limited precision, and requires evaluation of the function in question. Here, we will go through some examples to demonstrate the functionality and weaknesses of the bisection method.

[1] https://en.wikipedia.org/wiki/Bisection_method

### Example 1 - Warming up

Let's start of with an easy example:

Find the root of $y=(0.5x)^3-2$.


```python
import numpy as np
import bisection as bs
from typing import Union

def my_function(x:Union[int,float])->Union[int,float]:
    return (0.5*x)**3 - 2

a_guess = -2
b_guess = 4
ans=bs.bisection_method(my_function,a_guess,b_guess)
print(f'found answer = {np.round(ans,4)}')
print(f'known answer = {2.5198}')
```

    found answer = 2.5198
    known answer = 2.5198


### Example 2 - Are you good at guessing?

Example 1 provides a function with 1 root. What happen when we're trying to solve a function with multiple roots?

Find the root of $y=(0.2x)^2-1$.


```python
import numpy as np
import bisection as bs
from typing import Union

def my_function(x:Union[int,float])->Union[int,float]:
    return (0.2*x)**2-1

a_guess = -6
b_guess = 1
ans=bs.bisection_method(my_function,a_guess,b_guess)
print('initial guess 1 --------------------------')
print(f'initial guess a={a_guess} and b={b_guess}')
print(f'found answer = {np.round(ans,4)}')
print(f'known answers = ({-5.0},{5.0})')

a_guess2 = 0
b_guess2 = 5
ans=bs.bisection_method(my_function,a_guess2,b_guess2)
print('\ninitial guess 2 --------------------------')
print(f'initial guess a={a_guess2} and b={b_guess2}')
print(f'found answer = {np.round(ans,4)}')
print(f'known answers = ({-5.0},{5.0})')
print('\n As you can see, the bisection method is very sensitive to the initial guess.')
```

    initial guess 1 --------------------------
    initial guess a=-6 and b=1
    found answer = -5.0
    known answers = (-5.0,5.0)
    
    initial guess 2 --------------------------
    initial guess a=0 and b=5
    found answer = 5.0
    known answers = (-5.0,5.0)
    
     As you can see, the bisection method is very sensitive to the initial guess.


### Example 3 - Limited precision

Can you guess what's happening in this example? (The first code block is supposed to fail)

Find the root of $y=(0.1x)^5-0.5x-2$.


```python
import numpy as np
import bisection as bs
from typing import Union

def my_function(x:Union[int,float])->Union[int,float]:
    return (0.1*x)**5-0.5*x-2

a_guess = 15
b_guess = 16
ans=bs.bisection_method(my_function,a_guess,b_guess)
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Cell In[3], line 10
          8 a_guess = 15
          9 b_guess = 16
    ---> 10 ans=bs.bisection_method(my_function,a_guess,b_guess)


    [... skipping some code blocks for conciseness ...]

    File /projectnb/me700/students/quan/assignments/me700_assignment1/src/bisection.py:24, in bisection_method(func, a, b, tol, recursion_count, max_iter)
         21 # if the function has been called recursively max_iter times,
         22 # bisection method is not suitable to find root.
         23 if recursion_count > max_iter:
    ---> 24     raise Exception(f'Bisection method cannot find a root within tolerance after {max_iter} iterations.')
         25 else:
         26     recursion_count+=1


    Exception: Bisection method cannot find a root within tolerance after 1000 iterations.


Here, the bisection method cannot find a solution after iterating 1000 times. We can either (i) lower the tolerance, (ii) increase the number of iterations, or (iii) select a different solver. We can first lower the tolerance since we don't have to be super precised in this problem.


```python
a_guess = 15
b_guess = 16
ans=bs.bisection_method(my_function,a_guess,b_guess,tol=1e-8)
print(f'found answer = {np.round(ans,2)}')
print(f'known answers = {15.82}')
```

    found answer = 15.82
    known answers = 15.82


### Example 4 - Vroom vroom

Stacy is at a stoplight in her mom's car. As the light turns green, Stacy accelerates at a rate of $5.50m/s^2$ before having to hard break after traveling $15m$ due to a deer running by. Assuming that Stacy has not reached top speed during the duration, how long did it take for Stacy to travel the distance before stopping?

Ans: We can use the simple kinematic equation $\Delta x=v_it+\frac{1}{2}at^2$


```python
import numpy as np
import bisection as bs
from typing import Union

def my_function(x:Union[int,float])->Union[int,float]:
    return 0.5*3.50*x**2-15

a_guess = 0
b_guess = 5
ans=bs.bisection_method(my_function,a_guess,b_guess)
print(f'found answer = {np.round(ans,4)}')
print(f'known answers = {2.9277}')
```

    found answer = 2.9277
    known answers = 2.9277


### Example 5 - Oscillation

A mass $m$ is attached to the end of a spring with $k=1$ that follows Hooke's law. When pulled, the mass takes $5s$ to travel up and down. Assuming negligible friction and constant amplitude of oscillation, what is the mass $m$?

Ans: $f=\frac{1}{T}=\frac{1}{2\pi}\sqrt{\frac{k}{m}}\Rightarrow \frac{1}{5}=\frac{1}{2\pi}\sqrt{\frac{1}{m}}$


```python
import numpy as np
import bisection as bs
from typing import Union

def my_function(x:Union[int,float])->Union[int,float]:
    return 5/(2*np.pi)*np.sqrt(1/x)-1

a_guess = 0.1
b_guess = 5
ans=bs.bisection_method(my_function,a_guess,b_guess)
print(f'found answer = {np.round(ans,4)}')
print(f'known answers = {0.6333}')
```

    found answer = 0.6333
    known answers = 0.6333

