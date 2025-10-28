import mlrun

@mlrun.handler(outputs = {"return454": "hello"}) 
def helloWorld():
    """
    A function which prints "Hello World!" and returns it
    """
    print("Hello World!")

    return "Hello World!"