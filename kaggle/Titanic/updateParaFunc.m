function theta = updateParaFunc(initial_theta, grad, lr)

theta = initial_theta - lr*grad;

end