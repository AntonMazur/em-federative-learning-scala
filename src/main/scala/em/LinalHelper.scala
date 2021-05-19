package em

import breeze.linalg.DenseVector

trait LinalHelper {
  implicit class DoubleVectorOps(vector: DenseVector[Double]) {
    def scalarProduct(that: DenseVector[Double]): Double = {
      (vector.toDenseMatrix * that).apply(0)
    }
  }

  implicit class IntVectorOps(vector: DenseVector[Int]) {
    def scalarProduct(that: DenseVector[Int]): Int = {
      (vector.toDenseMatrix * that).apply(0)
    }
  }
}
