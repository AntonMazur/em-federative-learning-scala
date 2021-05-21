package em

import com.twitter.util.Future
import em.Model.{EStepOutputData, MStepStage1OutputData, MStepStage2InputData, MStepStage2OutputData}


class Master(workers: Seq[Worker], n: Int) {


  private def iterate(model: Model, nIterations: Int): Future[Model] = {

    if (nIterations == 0) {
      Future(model)
    } else {
      val newModel = for {
        modelAfterEStep <- Future.collect(workers.map(_.eStep(model)))
          .map(_.reduce(Model.mergeOutputModels(_, _, Seq.empty)))

        nonNormClustWeights = modelAfterEStep.outputData match {
          case EStepOutputData(weights) => weights
          case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
        }

        modelAfterMStepStage1 <- Future.collect(workers.map(_.mStepStage1(modelAfterEStep.copy())))
          .map(_.reduce(Model.mergeOutputModels(_, _, nonNormClustWeights)))

        newClustMeans = modelAfterMStepStage1.outputData match {
          case MStepStage1OutputData(clustMeans) => clustMeans
          case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
        }

        modelBeforeMStepStage2 = modelAfterMStepStage1.copy(inputData = MStepStage2InputData(newClustMeans))

        modelAfterMStepStage2 <- Future.collect(workers.map(_.mStepStage2(modelBeforeMStepStage2.copy())))
          .map(_.reduce(Model.mergeOutputModels(_, _, nonNormClustWeights)))

        newClustCovs = modelAfterMStepStage2.outputData match {
          case MStepStage2OutputData(clustCov) => clustCov
          case data => throw new IllegalStateException(s"Data has invalid type: ${data.getClass}")
        }

        newClustWeights = nonNormClustWeights.map(_ / n)
      } yield {
        Model(newClustMeans, newClustCovs, newClustWeights)
      }

      newModel.flatMap(iterate(_, nIterations - 1))
    }

  }

  def run() = {
    var model: Model = null
    var n = 0

    iterate(model, 100)
  }

}

object Master {
  object Config {
    trait StopCriteria

    case object NUM_ITERATIONS extends StopCriteria

    case object LIKEHOODCHANGE extends StopCriteria

    case object ONE_OF extends StopCriteria
  }

  case class Config(
      stopCriteria: Config.StopCriteria,
      nClusters: Int,
      maxIter: Option[Int] = Some(500),
      likehoodDelta: Option[Double] = Some(0.001)
  )
}
