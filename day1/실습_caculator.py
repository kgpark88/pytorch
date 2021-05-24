class Calculator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def add(self):
        result = self.x + self.y
        return result
    
    def sub(self):
        result = self.x - self.y
        return result
    
    def mul(self):
        result = self.x * self.y
        return result
    
    def div(self):
        result = self.x / self.y
        return result


if __name__ == '__main__':
    x = 3
    y = 7
    cal = Calculator(x, y)
    print(cal.add())
    print(cal.sub())
    print(cal.mul())
    print(cal.div())
