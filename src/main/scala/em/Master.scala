package em

import em.EMModel.{EStepOutputData, MStepStage1OutputData, MStepStage2InputData, MStepStage2OutputData}

class Master {

  val worker: Seq[Worker] = Seq()

  def iterate(model: EMModel, n: Int): EMModel = {

    val modelAfterEStep = worker.map(_.eStep(model))
      .reduce(EMModel.mergeOutputModels(_, _, Seq.empty))

    val nonNormClustWeights = modelAfterEStep.outputData match {
      case EStepOutputData(weights) => weights
      case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
    }

    val modelAfterMStepStage1 = worker.map(_.mStepStage1(modelAfterEStep.copy()))
      .reduce(EMModel.mergeOutputModels(_, _, nonNormClustWeights))

    val newClustMeans = modelAfterMStepStage1.outputData match {
      case MStepStage1OutputData(clustMeans) => clustMeans
      case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
    }

    val modelBeforeMStepStage2 = modelAfterMStepStage1.copy(inputData = MStepStage2InputData(newClustMeans))

    val modelAfterMStepStage2 = worker.map(_.mStepStage2(modelBeforeMStepStage2.copy()))
      .reduce(EMModel.mergeOutputModels(_, _, nonNormClustWeights))

    val newClustCovs = modelAfterMStepStage2.outputData match {
      case MStepStage2OutputData(clustCov) => clustCov
      case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
    }

    val newClustWeights = nonNormClustWeights.map(_ / n)

    EMModel(newClustMeans, newClustCovs, newClustWeights)
  }

  def run() = {
    var model: EMModel = null
    var n = 0

    for (i <- 0 to 100) {
      model = iterate(model, 0)
    }
  }

}
