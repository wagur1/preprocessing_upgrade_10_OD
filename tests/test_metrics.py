import math
from vcm_preprocess.metrics import bd_rate, bd_quality

def test_bd_rate_identical():
    points = [(100, .5), (200, .6), (400, .7), (800, .8)]
    assert abs(bd_rate(points, points)) < 1e-8

def test_bd_rate_saves_bits():
    ref = [(100, .5), (200, .6), (400, .7), (800, .8)]
    test = [(80, .5), (160, .6), (320, .7), (640, .8)]
    assert bd_rate(ref, test) < -15

def test_bd_quality_identical():
    points = [(100, .5), (200, .6), (400, .7), (800, .8)]
    assert abs(bd_quality(points, points)) < 1e-8
