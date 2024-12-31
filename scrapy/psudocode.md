i want to created an autograd engine but first what is autograd engine:
well to pu it lightly an autograd engine stands for automatic gradient it will compute the gradient for every value in the equation 
for example if we have 2 + 3 = 5 if we want to compute the grad with respect to 3 it would be 5 * 2 we multiply it by 5 because of the chain rule to implement this we need a class the have two things:
1, a bidirectional graph
2, a collection of operateres we need   