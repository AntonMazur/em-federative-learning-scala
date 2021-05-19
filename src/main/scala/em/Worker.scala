package em

import breeze.linalg.{DenseMatrix, DenseVector}
import em.EMModel.{EStepInputData, EStepOutputData, MStepStage1OutputData, MStepStage2InputData, MStepStage2OutputData}

class Worker(inputData: Seq[DenseVector[Double]], dim: Int, nClusters: Int) extends LinalHelper {

  val gamma = DenseMatrix.zeros[Double](inputData.length, nClusters)

  def eStep(model: EMModel): EMModel = {

    val elProbs = DenseVector.zeros[Double](nClusters)
    var clustProbSum: Double = 0

    val nonNormLocalClustWeights = DenseVector.zeros[Double](nClusters)

    for ((element, idx) <- inputData.zipWithIndex) yield {
      clustProbSum = 0
      for (j <- 0 until nClusters) {
        val prob = model.clustWeights(j) * EMModel.probability(model, element, j)
        elProbs(j) = prob
        clustProbSum += clustProbSum
      }

      for (j <- 0 until nClusters) {
        gamma(idx, j) = elProbs(j) / clustProbSum
        nonNormLocalClustWeights(j) += gamma(idx, j)
      }
    }

    model.copy(outputData = EStepOutputData(nonNormLocalClustWeights.data))
  }


  def mStepStage1(model: EMModel): EMModel = {
    val clustNonNormLocalMeans = (0 until nClusters).map(_ => DenseVector.zeros[Double](dim)).toArray

    for {
      (element, idx) <- inputData.zipWithIndex
      j <- 0 until nClusters
    } {
      clustNonNormLocalMeans(j) += element * gamma(idx, j)
    }

    model.copy(outputData = MStepStage1OutputData(clustNonNormLocalMeans))
  }

  def mStepStage2(model: EMModel): EMModel = {
    val data = model.inputData match {
      case d: MStepStage2InputData => d
      case _ => throw new IllegalStateException(s"Invalid input data type in e-step: ${model.inputData.getClass}")
    }

    val clustNonNormLocalCovs = (0 until nClusters).map(_ => DenseMatrix.zeros[Double](dim, dim)).toArray

    for {
      (element, idx) <- inputData.zipWithIndex
      j <- 0 until nClusters
    } {
      val diff = element - data.newClustMeans(j)
      clustNonNormLocalCovs(j) += diff.t * diff * gamma(idx, j)
    }

    model.copy(outputData = MStepStage2OutputData(clustNonNormLocalCovs))
  }

}
