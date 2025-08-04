class some_class:
  def __init__(self, name, age):
    self.name = name
    self.age = age
  
  def print_data(self):
    print(f"My name is {self.name} and I am {self.age} years old")



dillon = some_class("dillon", 32)
dillon.print_data()


