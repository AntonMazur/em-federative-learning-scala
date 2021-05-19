package em

import breeze.linalg._
import em.EMModel.{EmptyData, InputData, OutputData}


case class EMModel(
    means: Seq[DenseVector[Double]],
    covs: Seq[DenseMatrix[Double]],
    covsDet: Seq[Double],
    covsInv: Seq[DenseMatrix[Double]],
    clustWeights: Seq[Double],
    inputData: InputData = EmptyData,
    outputData: OutputData = EmptyData
)

object EMModel extends LinalHelper {

  def mergeOutputModels(model1: EMModel, model2: EMModel, nonNormClustProbability: Seq[Double]): EMModel = {
    model1.copy(outputData = mergeOutputData(model1.outputData, model2.outputData) match {
      case data: MStepStage1OutputData =>
        MStepStage1OutputData(data.nonNormLocalClustMeans
          .zipWithIndex
          .map { case (nonNormClustMean, idx) => nonNormClustMean / nonNormClustProbability(idx) }
        )
      case data: MStepStage2OutputData =>
        MStepStage2OutputData(data.nonNormLocalClustCovs
          .zipWithIndex
          .map { case (nonNormClustCov, idx) => nonNormClustCov / nonNormClustProbability(idx) }
        )
      case data => data
    })


  }


  def mergeOutputData(outputData1: OutputData, outputData2: OutputData): OutputData = {
    require(outputData1.getClass == outputData2.getClass)

    (outputData1, outputData2) match {
      case (EStepOutputData(nonNormLocalClustersWeights1), EStepOutputData(nonNormLocalClustersWeights2)) =>
        EStepOutputData(
          nonNormLocalClustersWeights1
            .zip(nonNormLocalClustersWeights2)
            .map { case (p1, p2) => p1 + p2 })
      case (MStepStage1OutputData(nonNormLocalClustMeans1), MStepStage1OutputData(nonNormLocalClustMeans2)) =>
        MStepStage1OutputData(
          nonNormLocalClustMeans1
            .zip(nonNormLocalClustMeans2)
            .map { case (p1, p2) => p1 + p2 })
      case (MStepStage2OutputData(nonNormLocalClustCovs1), MStepStage2OutputData(nonNormLocalClustCovs2)) =>
        MStepStage2OutputData(
          nonNormLocalClustCovs1
            .zip(nonNormLocalClustCovs2)
            .map { case (p1, p2) => p1 + p2 })
      case _ => throw new IllegalArgumentException()
    }
  }

  sealed trait InputData

  case class EStepInputData() extends InputData

  case class MStepStage1InputData() extends InputData

  case class MStepStage2InputData(newClustMeans: Seq[DenseVector[Double]]) extends InputData

  sealed trait OutputData

  case class EStepOutputData(nonNormLocalClustWeights: Seq[Double]) extends OutputData

  case class MStepStage1OutputData(nonNormLocalClustMeans: Seq[DenseVector[Double]]) extends OutputData

  case class MStepStage2OutputData(nonNormLocalClustCovs: Seq[DenseMatrix[Double]]) extends OutputData

  object EmptyData extends InputData with OutputData

  def apply(means: Seq[DenseVector[Double]], covs: Seq[DenseMatrix[Double]], clustWeights: Seq[Double]): EMModel = {
    val covsDet = covs.map(det(_))
    val covsInv = covs.map(inv(_))

    EMModel(means, covs, covsDet, covsInv, clustWeights)
  }

  def probability(model: EMModel, vector: DenseVector[Double], clusterIndex: Int): Double = {
    val prob = Math.pow(2 * Math.PI, -0.5 * vector.length) * Math.pow(model.covsDet(clusterIndex), -0.5)
    val mahaDist = vector scalarProduct (model.covsInv(clusterIndex) * vector).toDenseVector
    val expTerm = Math.pow(Math.E, -0.5 * mahaDist)
    prob * expTerm
  }

}
