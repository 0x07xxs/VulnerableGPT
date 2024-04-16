try:
    user_input = input("Enter a Python expression: ")
    result = eval(user_input)
    print("Result:", result)
except Exception as e:
    print("Error:", e)