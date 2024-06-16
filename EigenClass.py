import numpy as np

class EigenClass:
  @staticmethod
  def get_eigen(matrix: np.array, showEquality=False):
    evalues, evectors = np.linalg.eig(matrix)

    if showEquality:
      for idx, evalue in enumerate(evalues):
        # Av (matrix*vector) = Lv (lambda*vector)

        matrix_vector = np.dot(matrix, evectors[:, idx].transpose())
        lambda_vector = evalue * evectors[:, idx]
        
        print(f'{idx+1}: {matrix_vector} == {lambda_vector}')

        # does not work, most probably cause of too big float not storing correctly
        # print(np.array_equal(matrix_vector, lambda_vector))
    
    return evalues, evectors