# 4.2 손실함수
# 4.2.1 오차제곱합(sum squares error)
>>> import numpy as np
>>> y = np.array( [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] )
>>> t = np.array( [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] )		# 2가 정답
>>> sum_squares_error(y, t)
0.09750000000000003		# 오차가 작다.
>>> y = np.array( [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] )
>>> sum_squares_error(y, t)
0.5975		# 오차가 크다
# 따라서 y[2] 0.6 인 출려값을 가진 벡터가 정답일 확률이 높다.

# 4.2.2 교차 엔트로피 오차(cross entropy error, CEE)
# 자연상수 e = 2.718281828459045
>>> def cross_entropy_error(y, t) :
	delta = 1e-7 		# log(0) = inf 가 되는 것을 방지하기 위해 쓴다
	return -np.sum(t * np.log(y + delta))
>>> t = np.array( [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] )		# 2가 정답
>>> y = np.array( [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] )
>>> cross_entropy_error(y, t)
0.510825457099338
>>> y = np.array([0.1 , 0.05, 0.1 , 0.  , 0.05, 0.1 , 0.  , 0.6 , 0.  , 0.  ])
>>> cross_entropy_error(y, t)
2.302584092994546
