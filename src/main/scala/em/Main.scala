package em

import breeze.linalg._

object Main extends App with LinalHelper {

//  val model = new EMModel(Seq(DenseVector(1.0, 2.0)), Seq(DenseMatrix((3.0, 1.0), (3.0, 1.0))));

  val vector = DenseVector(1, 2)
  val matrix = DenseMatrix((1, 0), (0, 1))

  println(vector * 2)
  matrix(0,0) = 2

  println(vector * vector.t)

  vector + vector
  println(vector.scalarProduct(vector))
  val a = vector.t * matrix
  println(vector * vector)
  val res = vector.scalarProduct((matrix * vector).toDenseVector)
  println(res)
}
