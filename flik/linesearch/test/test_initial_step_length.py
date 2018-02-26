"""Tests for flik.linesearch.conditions"""

import numpy as np

from nose.tools import assert_raises

from initial_step import initial_alpha_grad_based

def test_initial_alpha_grad_based():
    """Tests for selecting the initial alpha value (gradient-based)"""

    #Set parameters

    def grad(current_point):
        """Gradient of function=current_point**2"""
        return 2*current_point

    current_point = np.array(1.8)
    previous_point = np.array(8.4)
    current_step = np.array(-0.6)
    previous_step = np.array(-0.4)
    previous_alpha = 0.8

    #Checking input quality

    grad_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_grad_based, grad_w, current_point, previous_point,
                  current_step, previous_step, previous_alpha)
    current_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                   current_step, previous_step, previous_alpha)
    previous_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point_w,
                   current_step, previous_step, previous_alpha)
    current_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step_w, previous_step, previous_alpha)
    previous_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step, previous_step_w, previous_alpha)
    current_point_w = np.array(10)
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                   current_step, previous_step, previous_alpha)
    previous_point_w = np.array(11)
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point_w,
                   current_step, previous_step, previous_alpha)
    current_step_w = np.array(12)
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step_w, previous_step, previous_alpha)
    previous_step_w = np.array(13)
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step, previous_step_w, previous_alpha)
    previous_alpha_w = 2
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step, previous_step, previous_alpha_w)
    previous_alpha_w = 1.4142
    assert_raises(ValueError, initial_alpha_grad_based, grad, current_point, previous_point,
                   current_step, previous_step, previous_alpha_w)
    current_point_w = np.array([1., 0.])
    current_step_w = np.array([0., 1.])
    assert_raises(ValueError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                   current_step_w, previous_step, previous_alpha)

    #Checking return value

    #Checking output type
    assert isinstance(initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step, previous_step, previous_alpha), float)
    #Checking output range
    assert 0 < initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step, previous_step, previous_alpha) <= 1

def test_initial_alpha_func_based():
    """Tests for selecting the initial alpha value (function-based)"""

    #Set parameters

    def func(current_point):
        """Objective function"""
        return current_point**2

    def grad(current_point):
        """Gradient of function=current_point**2"""
        return 2*current_point

    current_point = np.array(1.8)
    previous_point = np.array(1.9)
    current_step = np.array(-0.6)

    #Checking input quality

    func_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_func_based, func_w, grad, current_point, previous_point, current_step)
    grad_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_func_based, func, grad_w, current_point, previous_point, current_step)
    current_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point_w, previous_point, current_step)
    previous_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point, previous_point_w, current_step)
    current_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point, previous_point, current_step_w)
    current_point_w = np.array(10)
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point_w, previous_point, current_step)
    previous_point_w = np.array(11)
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point, previous_point_w, current_step)
    current_step_w = np.array(12)
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point, previous_point, current_step_w)
    current_point_w = np.array([1., 0.])
    current_step_w = np.array([0., 1.])
    assert_raises(ValueError, initial_alpha_func_based, func, grad, current_point_w, previous_point, current_step_w)

    #Checking return value

    #Checking output type
    assert isinstance(initial_alpha_func_based(func, grad, current_point, previous_point, current_step), float)
    #Checking output range
    assert 0 < initial_alpha_func_based(func, grad, current_point, previous_point, current_step) <= 1
