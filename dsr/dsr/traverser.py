import numpy as np
from fractions import Fraction

from dsr.library import Token, PlaceholderConstant

GAMMA = 0.57721566490153286060651209008240243104215933593992

"""Define custom unprotected operators"""
def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))

def expneg(x1):
    return np.exp(-x1)

def n3(x1):
    return np.power(x1, 3)

def n4(x1):
    return np.power(x1, 4)

def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))

def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val)+1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)

unprotected_ops = [
    # Binary operators
    Token(np.add, "add", arity=2, complexity=1),
    Token(np.subtract, "sub", arity=2, complexity=1),
    Token(np.multiply, "mul", arity=2, complexity=1),
    Token(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    Token(np.sin, "sin", arity=1, complexity=3),
    Token(np.cos, "cos", arity=1, complexity=3),
    Token(np.tan, "tan", arity=1, complexity=4),
    Token(np.exp, "exp", arity=1, complexity=4),
    Token(np.log, "log", arity=1, complexity=4),
    Token(np.sqrt, "sqrt", arity=1, complexity=4),
    Token(np.square, "n2", arity=1, complexity=2),
    Token(np.negative, "neg", arity=1, complexity=1),
    Token(np.abs, "abs", arity=1, complexity=2),
    Token(np.maximum, "max", arity=1, complexity=4),
    Token(np.minimum, "min", arity=1, complexity=4),
    Token(np.tanh, "tanh", arity=1, complexity=4),
    Token(np.reciprocal, "inv", arity=1, complexity=2),

    # Custom unary operators
    Token(logabs, "logabs", arity=1, complexity=4),
    Token(expneg, "expneg", arity=1, complexity=4),
    Token(n3, "n3", arity=1, complexity=3),
    Token(n4, "n4", arity=1, complexity=3),
    Token(sigmoid, "sigmoid", arity=1, complexity=4),
    Token(harmonic, "harmonic", arity=1, complexity=4)
]

function_map = {
    op.name : op for op in unprotected_ops
    }

# Defining the Binary Tree

class Node:
     
    def __init__(self):
         
        self.data = None
        self.left = None
        self.right = None 

    def PrintTree(self):
      if self.left:
         self.left.PrintTree()
      print(self.data)
      if self.right:
         self.right.PrintTree()
    
'''
1. We take the first operator
2. We create a Node with that operator. We remove that operator from the array.
3. If that operator is a variable, we attach it to the previous node and return one up to the previous node. 
    4. If that node takes another argument, we recurse down a step to construct the next node. Otherwise, we return this node up to the previous node.
3. If that operator is a function, then we construct a new left node by starting with (1).
''' # [mul, add, x, x, x]
def build(root):
    if len(operators) == 0: return None
    if operators[0] == 'x1': 
        root.data = x[:,0]
        operators.pop(0)
        return root
    if operators[0] == 'x2':
        root.data = x[:,1]
        operators.pop(0)
        return root
    root.data = function_map[operators[0]]
    operators.pop(0)
    if root.data.arity == 1:
        tmp = Node()
        root.left = build(tmp)
        return root
    if root.data.arity == 2:
        tmp1 = Node()
        root.left = build(tmp1)
        tmp2 = Node()
        root.right = build(tmp2)
        return root


'''
1. If root is an x, we return x
2. If root has arity 1, we return applying the root to the root's left node 
3. If root has arity 2, we return applying the root to the root's left and right nodes
'''
def tree_search(root):
    if x.shape[1] == 2:
        if np.array_equal(root.data, x[:, 0]) or np.array_equal(root.data, x[:, 1]): return root.data
    else:
        if np.array_equal(root.data, x[:, 0]): return root.data
    if root.data.arity == 1:
        return root.data.function(tree_search(root.left))
    if root.data.arity == 2:
        return root.data.function(tree_search(root.left), tree_search(root.right))
    return "We have encountered a grave issue"


def evaluator(ops, xs, y):
    global operators, x
    operators, x = ops, xs
    plantling = Node()
    build(plantling)
    return y - tree_search(plantling)


############# <--- Previous Attempts ---> ##############

#[mult, add, add, x1, x2, x1, sin, x1, ]

""" def traverse(operators, xs):
    if len(operators) == 0: return
    if operators[0] == 'x1' or operators[0] == 'x2':
        return xs
    t = unprotected_ops[operators[0]]
    if t.arity == 1:
        return t.function(xs) --> return t.function(traverse(operators[1:], xs))
    else:
        remaining = 2
        i = 1
        while remaining > 0:
            tmp = unprotected_ops[operators[i]]
            if operators[i] == 'x1' or operators[i] == 'x2':
                remaining -= 1
            if tmp.arity == 1:
                remaining -=2
                i += 1
            else:
                remaining += 2
            i += 1
        return t.function(traverse(operators[1:], xs), traverse(operators[i:], xs)) """


#[mult, add, add, x1, x2, x1, sin, x1, ]
""" def traverse(operators, xs, index):
    if operators[index] == 'x1' or operators[index] == 'x2': return xs, index #fix later
    t = function_map[operators[index]]
    if t.arity == 1:
        return t.function(traverse(operators, xs, index+1)[0]), index
    if t.arity == 2:
        val1, new_index = traverse(operators, xs, index+1)
        if operators[new_index] == 'x1' or operators[new_index] == 'x2' or function_map[operators[new_index]].arity == 1:
            return t.function(val1, traverse(operators, xs, new_index+2)[0]), index
        else: return t.function(val1, traverse(operators, xs, new_index+1)[0]), index """

""" def evaluate(tarray, index, xs):
    if tarray[index] == 'x1' or tarray[index] == 'x2':
        tarray[index] = xs
        return  (tarray[index], index + 1) 
    else:
        opt = function_map[tarray[index]] #save the operator
        paras = opt.arity
        parasArray = []
        for i in range(paras):
            value, newIndex = evaluate(tarray, index + 1, xs)
            parasArray.append(value)
            index = newIndex
        return opt(*parasArray) """

# print(evaluate(ops, 0, x))

# [mult, add, add, x1, x2, x1, sin, x1, ] = sin(x1) * x1 + x2 + x1

# mult
# sin add
# x1   add x1
#       x1 x2
