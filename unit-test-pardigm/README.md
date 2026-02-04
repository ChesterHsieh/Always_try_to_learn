# Intro

It's snippet code for collecting the best practices of writing testable code, following the clean architecture, and come
out some anti-patterns.

# How to run

On purpose, In this repo only have the backend code, and pure built-in python, so you can run it simply by run the *
*main.py**

# Unitest

1. Always try to figure out which part is most frequently changed, and write the test for that part.
2. Always try to write the test for the edge cases.
3. Always try to write the test for most import part. aka use case in this case.

## Cover Topic

- mock object. variety way to inject
- Setup/teardown
- Command to control the test scale

# Regard to some clean architecture (some thought from my own)

Even after implementing a whole fake app, the importance of some layers still need to think twice,

## presenter

Do we need presenter?
=> I guess, it's not. Don't overly optimize the code. If in our case, the output work is compartively simple. we
shouldn't tackle that part.

## What’s the difference between a repository and a DAO?

Imagine you’re a person representing the repository layer. Do you know about the existence of the database? No, you
don’t. Your job is to interact with the use cases, without concerning yourself with how or where the data is stored.

Now, if you represent the DAO (Data Access Object), you have some awareness of a physical place where data is kept. The
DAO acts like instructions to someone (or something) responsible for writing down information and retrieving old records
from storage.

# Anti-pattern (or maybe some reasonable modification)

## Story I

Description
As an emergency maintenance engineer, I need to directly modify the order status in the database to quickly resolve
customer complaints without going through the existing business logic and use cases.
Acceptance Criteria

	1.	In the event of an order status error, I can directly connect to the database and change the order status without using the application layer or business logic.
	2.	The updated order status should take effect immediately and be reflected in the front-end interface.
	3.	The modification should not trigger any business logic or event handling (such as notifying the user or updating inventory).

## Story II

ORM argument. [ThePrimeTime comment](https://www.youtube.com/watch?v=bpGvVI7NM_k&t=630s&ab_channel=ThePrimeTime).
Mostly agree to the reason to remvoe the ORM from our case. So what to do? Is that okay to remove all the ORM and keep
some type of query builder? or raw sql

## Story III
Ok. Fastapi is a good framework. How do we adapt this framework to our case? Which part should we keep and which part.
