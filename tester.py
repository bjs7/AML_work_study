


from itertools import count, cycle, islice

# Infinite counter
for i in islice(count(10), 5):  # [10, 11, 12, 13, 14]
    print(i)

# Cycling through a sequence
from itertools import cycle
colors = cycle(['red', 'green', 'blue'])
for _ in range(4):
    print(next(colors))  # red green blue red





squares = (x * x for x in range(10))
print(next(squares))  # prints 0


from itertools import groupby

data = [('fruit', 'apple'), ('fruit', 'banana'), ('veg', 'carrot'), ('veg', 'spinach')]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))



def gen():
    yield 1
    yield 2
    yield 3

test123 = gen()
next(test123)

for num in gen():
    print(num)


def handler(func, *args, **kwargs):
    print("Before function call")
    result = func(*args, **kwargs)
    print("After function call")
    #return result

def greet(name, age):
    print(f"Hello {name}, age {age}")

test = handler(greet, "Alice", age=30)


def greet(name: str) -> str:
    return f"Hello, {name}!"

age: int = 30



from typing import Optional

def get_name(id: int) -> Optional[str]:
    if id == 0:
        return None
    return "Alice"

get_name


from typing import List, Dict, Tuple

names: List[str] = ["Alice", "Bob"]

names1 = ["Alice", "Bob"]


from typing import TypedDict

class Movie(TypedDict):
    title: str
    year: int
    rating: float

movie: Movie = {"title": "Inception", "year": 2010, "rating": 'test'}


from dataclasses import dataclass

@dataclass
class Movie:
    title: str
    year: int
    rating: float

movie = Movie("Inception", 2010, 8.8)





from itertools import product

colors = ['red', 'green']
sizes = ['S', 'M']
print(list(product(colors, sizes)))

test123 = product(colors, sizes)
next(test123)


from itertools import count

test123 = count(1)
next(test123)


import unittest

def add(a, b):
    return a + b

class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, -1), -2)

if __name__ == '__main__':
    unittest.main()

class JsonMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class Person(JsonMixin):
    def __init__(self, name, age, tester):
        self.name = name
        self.age = age
        self.tester = tester

p = Person("Alice", 30, '123')
print(p.to_json())  # {"name": "Alice", "age": 30}






from abc import ABC, abstractmethod

# 1. Define Strategy Interface
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

# 2. Concrete Strategies
class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} with credit card.")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using PayPal.")

# 3. Context Class uses strategy
class ShoppingCart:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def checkout(self, amount):
        self.strategy.pay(amount)

# Usage:
cart = ShoppingCart(PayPalPayment())

test123 = PayPalPayment()
test123.pay(1)

cart.checkout(100)  # Paid 100 using PayPal

cart = ShoppingCart(CreditCardPayment())
cart.checkout(50)   # Paid 50 with credit card




class Expr:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Expr(f"({self.value} + {other.value})")

    def __str__(self):
        return self.value

a = Expr("x")
b = Expr("y")
c = a + b
print(c)  # (x + y)



import inspect

def foo(a, b=42, c=10, booli = True): 
    pass

sig = inspect.signature(foo)
print(sig.parameters)  # OrderedDict([('a', ...), ('b', ...)])







