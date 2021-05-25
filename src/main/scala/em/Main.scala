package em

import breeze.linalg._
import com.twitter.util.{Await, FuturePool}
import org.apache.commons.math3.distribution.{MixtureMultivariateNormalDistribution, MultivariateNormalDistribution}
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization

import java.util.Scanner
import scala.collection.convert.{DecorateAsJava, DecorateAsScala}
import scala.io.Source
import scala.util.Random

object Main extends App with LinalHelper with DecorateAsJava with DecorateAsScala {


  val random = new Random();

  val nClusters = 5
  val dim = 5
  val dataLength = 100000

  def generateMultivariateNormalDistribuitionMixtureData(nDistribuitions: Int, nData: Int, dim: Int) = {
    def genMean(): Array[Double] = {
      val mean = new Array[Double](dim)
      for (i <- 0 until dim) {
        mean(i) = random.nextInt(50)
      }

      mean
    }

    val covMatrix = (() => {
      val result = Array.ofDim[Double](dim, dim)
      for (i <- 0 until dim) {
        result(i)(i) = 1
//        for (j <- 0 until dim - i) {
//          result(i)(j) = random.nextInt()
//          result(j)(i) = result(i)(j)
//        }
      }

      result
    })

    val distributions = for (i <- 0 until nDistribuitions) yield {
      new MultivariateNormalDistribution(genMean(), covMatrix())
    }

    (distributions, (for (i <- 0 until nData) yield {
      DenseVector[Double](distributions(random.nextInt(nDistribuitions)).sample())
    }))
  }

  def genInitDists(nDistribuitions: Int, dim: Int) = {
    def genMean(): Array[Double] = {
      val mean = new Array[Double](dim)
      for (i <- 0 until dim) {
        mean(i) = random.nextInt(1000)
      }

      mean
    }

    val covMatrix: Array[Array[Double]] = {
      val result = Array.ofDim[Double](dim, dim)
      for (i <- 0 until dim) {
        result(i)(i) = 1
      }

      result
    }

    for (i <- 0 until nDistribuitions) yield {
      new MultivariateNormalDistribution(genMean(), covMatrix)
    }
  }

  val (dists, data) = generateMultivariateNormalDistribuitionMixtureData(nClusters, dataLength, dim)

  val doubleMatrix = data.map(_.data).toArray

  val distsList = genInitDists(nClusters, dim).map(dist => new org.apache.commons.math3.util.Pair(Double.box(1d / nClusters), dist)).asJava

  val mixture = new MixtureMultivariateNormalDistribution(distsList)
  val model = new MultivariateNormalMixtureExpectationMaximization(doubleMatrix)
  while (true) {
    val estDist = MultivariateNormalMixtureExpectationMaximization.estimate(doubleMatrix, nClusters)

    val startTime = System.nanoTime();

//    print("Threshold: ")
//    val threshold = scala.io.StdIn.readInt();
    model.fit(estDist, 100000, 1e-5)
    model.getLogLikelihood
    val endTime = System.nanoTime()
    val time: Double = (endTime - startTime) / 1000000000d

    println(s"Time: $time")

    println(estimatePrecision(model.getFittedModel, dists))
  }


  val divNumber = random.nextInt(dataLength / 6) + dataLength / 2

  val workers = Seq(
    new Worker(data.take(divNumber), dim, nClusters, FuturePool.unboundedPool),
    new Worker(data.take(divNumber), dim, nClusters, FuturePool.unboundedPool)
  )

  val master = new Master(workers, dataLength)

  def estimatePrecision(model: MixtureMultivariateNormalDistribution, initialDists: Seq[MultivariateNormalDistribution]): Double = {
    val mappingOldToNew = model.getComponents.asScala.zipWithIndex.map{ case (p, newDIdx) => {
      val means = p.getSecond.getMeans
      val mappingMeansIdx = initialDists.zipWithIndex.map { case (dist, idx) =>
        val otherMeans = dist.getMeans

        val s = (for (i <- 0 until means.length) yield {
          Math.pow(means(i) - otherMeans(i), 2)
        }).sum

        (s, idx)
      }.minBy(_._1)._2

      initialDists(mappingMeansIdx) -> p.getSecond
    }}.toMap

    var correct = 0d
    var incorrect = 0d

    for (i <- 0 until 1000000) {
      val originDist = initialDists(random.nextInt(initialDists.length))

      val sample = originDist.sample()

      val predDistDens = model.getComponents.asScala.map(_.getSecond)
        .map(d => d -> d.density(sample))

      val predDist = predDistDens.maxBy(_._2)._1

      if (mappingOldToNew.get(originDist).contains(predDist)) {
        correct += 1
      } else {
        incorrect += 1
      }


    }


    correct / (correct + incorrect)
  }

  //  val resultModel = Await.result(master.run())
  //
  //  println(resultModel)

}
