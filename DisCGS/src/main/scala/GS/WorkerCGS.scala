package GS

import Common.Tools.{mean, normalizeLogProbability, partitionToOrderedCount, sample}
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.log
import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

class WorkerCGS(workerID: Int,
                            indices: List[Int],
                            data: List[DenseVector[Double]],
                            prior: NormalInverseWishart = new NormalInverseWishart(),
                            alpha:Double,
                            initByUserPartition: Option[List[Int]] = None
                           ) extends Serializable {

  val n: Int = data.length
  val priorPredictive: List[Double] = data.map(prior.predictive)
  var globalPartition: ListBuffer[Int] = List.fill(n)(0).to(ListBuffer)
  var partition: List[Int] = initByUserPartition match {
    case Some(m) =>
      require(m.length == data.length)
      m
    case None => List.fill(data.length)(0)}

  var countCluster: ListBuffer[Int] = partitionToOrderedCount(partition).to(ListBuffer)

  var NIWParams: ListBuffer[NormalInverseWishart] = (data zip partition).groupBy(_._2).values.map(e => {
    val dataPerCluster = e.map(_._1)
    val clusterIdx = e.head._2
    (clusterIdx, prior.update(dataPerCluster))
  }).toList.sortBy(_._1).map(_._2).to(ListBuffer)

    var actualAlpha:Double = alpha
  def computeClusterPartitionProbabilities(idx: Int,
                                           verbose: Boolean=false): List[Double] = {
    countCluster.indices.map(k => {
      (k, log(countCluster(k).toDouble) + posteriorPredictive(data(idx), k))
    }).toList.sortBy(_._1).map(_._2)
  }

  def drawMembership(i: Int): Unit = {
    val probPartition = computeClusterPartitionProbabilities(i)
    val probPartitionNewCluster = log(actualAlpha) + priorPredictive(i)
    val normalizedProbs = normalizeLogProbability(probPartition :+ probPartitionNewCluster)
    val newPartition = sample(normalizedProbs)
    partition = partition.updated(i, newPartition)
  }

  def removeElementFromCluster(idx: Int): Unit = {
    val currentPartition = partition(idx)
    if (countCluster(currentPartition) == 1) {
      countCluster.remove(currentPartition)
      partition = partition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countCluster.update(currentPartition, countCluster.apply(currentPartition) - 1)
    }
  }

  def addElementToCluster(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ListBuffer(1)
    } else {
      countCluster.update(newPartition, countCluster.apply(newPartition) + 1)
    }
  }


  def posteriorPredictive(observation: DenseVector[Double], cluster: Int): Double = {
    NIWParams(cluster).predictive(observation)
  }

  def removeElementFromNIW(idx: Int): Unit = {
    val currentPartition =  partition(idx)
    if (countCluster(currentPartition) == 1) {
      NIWParams.remove(currentPartition)
    } else {
      val updatedNIWParams = NIWParams(currentPartition).removeObservations(List(data(idx)))
      NIWParams.update(currentPartition, updatedNIWParams)
    }
  }

  def addElementToNIW(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      val newNIWparam = this.prior.update(List(data(idx)))
      NIWParams = NIWParams ++ ListBuffer(newNIWparam)
    } else {
      val updatedNIWParams = NIWParams(newPartition).update(List(data(idx)))
      NIWParams.update(newPartition, updatedNIWParams)
    }
  }

  def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromNIW(i)
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToNIW(i)
      addElementToCluster(i)
    }
  }

  def computeMatrix(dataInCluster: List[DenseVector[Double]], m: DenseVector[Double]): DenseMatrix[Double] = {
    sum(dataInCluster.map(x => (x - m) * (x - m).t))
  }

  def computeSufficientStatistics(membership: List[Int]): ListBuffer[(DenseVector[Double], DenseMatrix[Double])] = {

    val dataPerCluster = (data zip membership).groupBy(_._2).values.map(e => {
      val dataPerCluster = e.map(_._1)
      val clusterIdx = e.head._2
      (clusterIdx, dataPerCluster)
    }).toList.sortBy(_._1).map(_._2).to(ListBuffer)

    val T1 = dataPerCluster.map(x=>mean(x))
    val T2 = (dataPerCluster zip T1).map(x => computeMatrix(x._1,x._2))
    T1 zip T2
  }

  def updateDPMWithMasterResults(masterResults:List[(Int,Int, NormalInverseWishart, Int)]) :Unit = {
    var results = masterResults.filter(_._1 ==this.workerID)
    results = results.sortBy(_._2)
    require(results.length==NIWParams.length,"results length is not equal to NIWParams length")	
    NIWParams = results.map(_._3).to(ListBuffer)
    for(i<-0 until n){
      globalPartition.update(i, results(partition(i))._4)
    }
  }

  def getGlobalResults(): List[(Int,Int)] = indices zip globalPartition

  ///////////////////////

  def run(maxIter: Int = 1,
          maxIterBurnin: Int = 0,
          verbose: Boolean = false): (Int, List[Int], List[(DenseVector[Double], DenseMatrix[Double])], List[Int], List[NormalInverseWishart]) = {

    var membershipEveryIteration = List(partition)
    var ssEveryIteration = List(computeSufficientStatistics(partition))
    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        println("\u03B1 = " + actualAlpha.toString)
        println("Cluster sizes: "+ countCluster.mkString(" "))
      }

      if (iter <= (maxIter + maxIterBurnin)) {

        updatePartition()

       val ss = computeSufficientStatistics(partition)

        membershipEveryIteration = membershipEveryIteration :+ partition
        ssEveryIteration = ssEveryIteration :+ ss
        go(iter + 1)
      }
    }
    go(1)
    (workerID, membershipEveryIteration.last, ssEveryIteration.last.toList, countCluster.toList, NIWParams.toList)
  }
}
