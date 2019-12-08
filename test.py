import superautodiff as sad

a = sad.AutoDiff('a', 2)
b = sad.AutoDiff('b', 3)

obj = [a, 2]

c = sad.AutoDiffVector(obj)
